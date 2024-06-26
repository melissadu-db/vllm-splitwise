import argparse
import json
import os
from typing import Optional, Callable

from vllm.model_executor.layers.fused_moe import quant_fused_moe, get_config_file_name
import torch
import torch.nn.functional as F
import triton


def main(
    gpu_idx: int, 
    gpu_ids: list[int],
    batch_sizes: Optional[list[int]] = 
    None, tp_size: Optional[int] = 8
):
    # Set the GPU device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_idx])
    method = quant_fused_moe

    if batch_sizes is None:
        batch_sizes = [
            1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 
            256, 512, 1024, 1536, 2048, 3072, 4096
        ]

    # Distribute work based on the GPU ID
    local_batch_sizes = [
        bs for idx, bs in enumerate(batch_sizes) 
        if idx % len(gpu_ids) == gpu_idx
    ]
    print(f"GPU {gpu_idx}: Running work with batch sizes {local_batch_sizes}")

    for bs in local_batch_sizes:
        run_grid(bs, gpu_idx, method=method, tp_size=tp_size)

    print(f"GPU {gpu_idx}: Completed work with batch sizes {local_batch_sizes}")


def run_grid(bs: int, gpu_idx: int, method: Callable, tp_size: Optional[int] = 8):
    d_model = 6144
    num_total_experts = 16
    top_k = 4
    model_intermediate_size = 10752
    num_layers = 40
    num_calls = 5

    num_warmup_trials = 1
    num_trials = 2

    configs = []
    if bs <= 16:
        BLOCK_SIZES_M = [16]
    elif bs <= 32:
        BLOCK_SIZES_M = [16, 32]
    elif bs <= 64:
        BLOCK_SIZES_M = [16, 32, 64]
    elif bs <= 128:
        BLOCK_SIZES_M = [16, 32, 64, 128]
    else:
        BLOCK_SIZES_M = [16, 32, 64, 128, 256]

    for block_size_n in [32, 64, 128, 256]:
        for block_size_m in BLOCK_SIZES_M:
            for block_size_k in [32, 64, 128, 256]:
                for group_size_m in [1, 16, 32, 64]:
                    for num_warps in [4, 8]:
                        configs.append({
                            "BLOCK_SIZE_M": block_size_m,
                            "BLOCK_SIZE_N": block_size_n,
                            "BLOCK_SIZE_K": block_size_k,
                            "GROUP_SIZE_M": group_size_m,
                            "num_warps": num_warps,
                            "num_stages": 4,
                        })

    best_config = None
    best_time_us = 1e20

    for config in configs:
        # warmup
        print('warming up')
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    method=method,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        print('benchmarking')
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                method=method,
                config=config,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

            print(f'{kernel_dur_us=:.1f} {model_dur_ms=:.1f}'
                  f' {bs=} {tp_size=} {top_k=} {num_total_experts=} '
                  f'{d_model=} {model_intermediate_size=} {num_layers=}')

    print("best_time_us", best_time_us)
    print("best_config", best_config)

    print(f"Best time for batch size {bs}: {best_time_us} μs with config {best_config}")

    filename = get_config_file_name(num_total_experts,
                                    model_intermediate_size // tp_size,
                                    quant=True)
    filename += f"_{gpu_idx}"
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")

def create_qweight(E, K, N):
    # Generate a tensor of shape (E, K, N, 4) with random uint8 values
    random_uint8s = torch.randint(0, 256, (E, K, N, 4), dtype=torch.uint8)
    
    # Convert uint8 values to int32 for bitwise operations
    random_uint32s = random_uint8s.to(torch.int32)
    
    # Shift bits into their correct positions
    int32_tensor = (random_uint32s[..., 0] << 24) | (random_uint32s[..., 1] << 16) | \
                   (random_uint32s[..., 2] << 8) | random_uint32s[..., 3]

    return int32_tensor

def load_inputs(
    num_experts: int, 
    d_model: int,
    intermediate_size: int, 
    num_bits: int,
    tp_size: int
):
    numel_per_i32 = 32 // num_bits
    
    # w1 is the up projection, with w1 and v1 concatenated
    qweight1 = create_qweight(
        num_experts, 
        d_model // numel_per_i32, 
        intermediate_size * 2 // tp_size
    ).to("cuda:0")
    scales1 = torch.full(
        (num_experts, 1, intermediate_size * 2 // tp_size), 
        0.0003, 
        dtype=torch.float16
    ).to("cuda:0")
    zeros1 = torch.full(
        (num_experts, 1, d_model // numel_per_i32), 
        2139062143, 
        dtype=torch.int32
    ).to("cuda:0")
    g_idx1 = torch.zeros((num_experts, d_model)).to("cuda:0")
    # w2 is the down projection down to d_model
    qweight2 = create_qweight(
        num_experts,
        intermediate_size * 2 // tp_size // numel_per_i32, 
        d_model
    ).to("cuda:0")
    scales2 = torch.full((num_experts, 1, d_model), 0.0003, dtype=torch.float16).to("cuda:0")
    zeros2 = torch.full((num_experts, 1, d_model // tp_size), 2139062143, dtype=torch.int32).to("cuda:0")
    g_idx2 = torch.zeros((num_experts, intermediate_size // tp_size)).to("cuda:0")
    return (qweight1, scales1, zeros1, g_idx1), (qweight2, scales2, zeros2, g_idx2)


def run_timing(num_calls: int, bs: int, d_model: int, num_total_experts: int,
               top_k: int, tp_size: int, model_intermediate_size: int, method,
               config) -> float:
    # shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=torch.float16,
    )
    # TODO: add packing factor
    # qweight_1 is column parallel, without permutation
    weights1, weights2 = load_inputs(
        num_experts=num_total_experts,
        d_model=d_model,
        intermediate_size=model_intermediate_size,
        num_bits=8,
        tp_size=tp_size,
    )
    qweight1, scales1, zeros1, g_idx1 = weights1
    qweight2, scales2, zeros2, g_idx2 = weights2

    gating_output = F.softmax(torch.rand(
        (num_calls, bs, num_total_experts),
        device=hidden_states.device,
        dtype=torch.float32,
    ),
    dim=-1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = method(
            hidden_states=hidden_states,
            qweights_1=qweight1, qscales_1=scales1, qzeros_1=zeros1, g_idx_1=g_idx1,
            qweights_2=qweight2, qscales_2=scales2, qzeros_2=zeros2, g_idx_2=g_idx2,
            gating_output=gating_output[i],
            topk=top_k,
            renormalize=True,
            inplace=True,
            override_config=config,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


# Set up argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark GPTQ across multiple GPUs",
        epilog="Specify GPU ID and optionally provide batch sizes and tp_size."
    )
    
    # Mandatory GPU ID
    parser.add_argument(
        "gpu_idx",
        type=int,
        help="The local idx of the GPU to use, starting from 0"
    )

    parser.add_argument(
        "gpus",
        type=str,
        help="Comma-separated list of GPU IDs to use."
    )

    # Optional batch sizes as a comma-separated list
    parser.add_argument(
        "batch_sizes",
        type=str,
        nargs="?",
        help="Comma-separated list of batch sizes to benchmark."
    )

    # Optional tp_size with a default value
    parser.add_argument(
        "tp_size",
        type=int,
        nargs="?",
        default=8,
        help="Tensor parallelism size (default: 8)."
    )

    return parser.parse_args()

# Entry point with argument parsing
if __name__ == "__main__":
    args = parse_arguments()

    # Parse the batch sizes from the comma-separated string
    if args.batch_sizes:
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(",") if bs]
    else:
        batch_sizes = []
    
    gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(",") if gpu_id]

    # Call the main function with parsed arguments
    main(args.gpu_idx, gpu_ids, batch_sizes, args.tp_size)
