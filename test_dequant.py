import torch
from vllm._C import ops
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe as fused_moe_vllm
from vllm.model_executor.layers.fused_moe.quant_fused_moe import fused_moe as fused_moe_vllm_quant
import numpy as np
import torch.nn.functional as F

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


def test_full_fused_moe():
    torch.manual_seed(1)
    num_bits = 8
    seq_len = 128
    num_experts = 16
    topk = 4
    d_model = 6144
    intermediate_size = int(6144 * 1.75)

    weights1, weights2 = load_inputs(num_experts, d_model, intermediate_size, num_bits, 4)
    qweight1, scales1, zeros1, g_idx1 = weights1
    qweight2, scales2, zeros2, g_idx2 = weights2

    dtype = scales1.dtype
    x = torch.rand((seq_len, 6144), dtype=dtype).to(qweight1.device)
    gating_output = F.softmax(torch.rand(
        (seq_len, num_experts),
        device=x.device,
        dtype=torch.float32,
    ),
    dim=-1)

    dequant_w1 = ops.dequant_gptq(
        qweight1, zeros1, scales1, torch.zeros_like(g_idx1), num_bits, False
    ).permute(0, 2, 1)
    dequant_w2 = ops.dequant_gptq(
        qweight2, zeros2, scales2, torch.zeros_like(g_idx2), num_bits, False
    ).permute(0, 2, 1)
    ref_output = fused_moe_vllm(x, dequant_w1, dequant_w2, gating_output, topk, True)

    quant_output = fused_moe_vllm_quant(
        x, 
        qweight1, scales1, zeros1, torch.zeros_like(g_idx1), 
        qweight2, scales2, zeros2, torch.zeros_like(g_idx2), 
        gating_output, topk, True, num_bits=num_bits
    )

    diffs = torch.count_nonzero(ref_output - quant_output)

    assert diffs == 0, f"Found {100 * diffs / ref_output.numel():.2f}% differences between ref and quant fused_moe"
    print("full_fused_moe test passed!")


test_full_fused_moe()
