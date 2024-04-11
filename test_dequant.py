import time
import torch
from vllm._C import ops
from dequant_baseline import dequant248 as dequant_ref
from dequant_sandbox import dequant248 as dequant_custom, fused_moe as fused_moe_custom
from fused_moe_baseline import fused_moe as fused_moe_ref
import numpy as np

def load_inputs():
    qweight = torch.load("./quantized_weights/0_qweight.pt")
    scales = torch.load("./quantized_weights/0_scales.pt")
    zeros = torch.load("./quantized_weights/0_qzeros.pt")
    g_idx = torch.load("./quantized_weights/0_g_idx.pt")
    return qweight, scales, zeros, g_idx


def test_custom_dequant():
    qweight, scales, zeros, g_idx = load_inputs()
    autogptq = dequant_ref(qweight[1], scales[1], zeros[1], torch.zeros_like(g_idx[1]), 8)
    custom = dequant_custom(qweight, scales, zeros, torch.zeros_like(g_idx), 8)
    diff = autogptq - custom
    assert torch.count_nonzero(diff) == 0

def test_fused_moe():
    torch.manual_seed(1)
    num_bits = 8
    seq_len = 128
    num_experts = 16
    topk = 4


    qweight, scales, zeros, g_idx = load_inputs()

    dtype = scales.dtype
    x = torch.rand((seq_len, 6144), dtype=dtype).to(qweight.device)
    gating_output = torch.rand((seq_len, num_experts), dtype=dtype).to(qweight.device)

    for _ in range(10):
        tick = time.time()
        dequant_w1 = ops.dequant_gptq(
            qweight, zeros, scales, torch.zeros_like(g_idx), num_bits, False
        ).permute(0, 2, 1)
        ref_output = fused_moe_ref(x, dequant_w1, gating_output, topk, True)
        tock = time.time()
        ref_time = tock - tick

    for _ in range(10):
        tick = time.time()
        fused_output = fused_moe_custom(x, qweight, scales, zeros, torch.zeros_like(g_idx), gating_output, topk, True, num_bits)
        tock = time.time()
        custom_time = tock - tick
    diff = ref_output - fused_output
    assert torch.count_nonzero(diff) == 0
    print(f"Speedup: {ref_time / custom_time:.2f}x")
    

def test_dequant():
    qweight, scales, zeros, g_idx = load_inputs()
    quant_bits = 8

    gptq_ref = ops.dequant_gptq(qweight, zeros, scales, torch.zeros_like(g_idx), quant_bits, False).to(torch.float32)
    exllama_ref = ops.dequant_gptq(qweight, zeros, scales, g_idx, quant_bits, True).to(torch.float32)
    diff = gptq_ref - exllama_ref
    print(f"Found {np.count_nonzero(diff.cpu().numpy())} total differences between exllama and gptq")
    autogptq = dequant_ref(qweight[0], scales[0], zeros[0], torch.zeros_like(g_idx[0]), quant_bits)
    diff_autogptq = gptq_ref[0] - autogptq
    print(f"Found {np.count_nonzero(diff_autogptq.cpu().numpy())} total differences between ref and autogptq")
    print("Test passed!")

# test_dequant()
# test_custom_dequant()
test_fused_moe()

    # model params
    # num_experts = 16
    # in_dim = 6144
    # out_dim = 6144 * 1.75
    # tp_size = 4
    # quant_bits = 8
    # packing_factor = 32 // quant_bits

    # qweight = torch.randint(0, (1<<31) - 1, (num_experts, in_dim // tp_size, 2 * out_dim // packing_factor), dtype=torch.int32)
    # scales = torch.rand((num_experts, 1, 2 * out_dim // packing_factor), dtype=torch.float16)
    # zeros = torch.randint(0, (1<<31) - 1, (num_experts, 1, in_dim // tp_size), dtype=torch.int32)
    # g_idx = torch.arange(0, in_dim).repeat(num_experts, 1)