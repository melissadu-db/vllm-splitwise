import torch
from vllm._C import ops
from dequant_baseline import dequant248 as dequant_ref
from dequant_sandbox import dequant248 as dequant_custom, fused_moe as fused_moe_custom
from fused_moe_baseline import fused_moe as fused_moe_ref
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe as fused_moe_vllm
from vllm.model_executor.layers.fused_moe.quant_fused_moe import fused_moe as fused_moe_vllm_quant
import numpy as np

def load_inputs():
    # w1 is the up projection, with w1 and v1 concatenated
    qweight1 = torch.load("./quant_weights_new/w1_qweight.pt")
    scales1 = torch.load("./quant_weights_new/w1_scales.pt")
    zeros1 = torch.load("./quant_weights_new/w1_qzeros.pt")
    g_idx1 = torch.load("./quant_weights_new/w1_g_idx.pt")
    # w2 is the down projection down to d_model
    qweight2 = torch.load("./quant_weights_new/w2_qweight.pt")
    scales2 = torch.load("./quant_weights_new/w2_scales.pt")
    zeros2 = torch.load("./quant_weights_new/w2_qzeros.pt")
    g_idx2 = torch.load("./quant_weights_new/w2_g_idx.pt")
    return (qweight1, scales1, zeros1, g_idx1), (qweight2, scales2, zeros2, g_idx2)


def test_custom_dequant():
    qweight, scales, zeros, g_idx = load_inputs()
    autogptq = dequant_ref(qweight[1], scales[1], zeros[1], torch.zeros_like(g_idx[1]), 8)
    custom = dequant_custom(qweight, scales, zeros, torch.zeros_like(g_idx), 8)
    diff = autogptq - custom
    assert torch.count_nonzero(diff) == 0

def test_full_fused_moe():
    torch.manual_seed(1)
    weights1, weights2 = load_inputs()
    qweight1, scales1, zeros1, g_idx1 = weights1
    qweight2, scales2, zeros2, g_idx2 = weights2
    num_bits = 8
    seq_len = 128
    num_experts = 16
    topk = 4

    dtype = scales1.dtype
    x = torch.rand((seq_len, 6144), dtype=dtype).to(qweight1.device)
    gating_output = torch.rand((seq_len, num_experts), dtype=dtype).to(qweight1.device)

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


def test_fused_moe():
    torch.manual_seed(1)
    num_bits = 8
    seq_len = 128
    num_experts = 16
    topk = 4

    weights1, weights2 = load_inputs()
    qweight1, scales1, zeros1, g_idx1 = weights1
    # qweight2, scales2, zeros2, g_idx2 = weights2
    
    dtype = scales1.dtype

    x = torch.rand((seq_len, 6144), dtype=dtype).to(qweight1.device)
    gating_output = torch.rand((seq_len, num_experts), dtype=dtype).to(qweight1.device)

    fused_output = fused_moe_custom(x, qweight1, scales1, zeros1, torch.zeros_like(g_idx1), gating_output, topk, True, num_bits)

    dequant_w1 = ops.dequant_gptq(
        qweight1, zeros1, scales1, torch.zeros_like(g_idx1), num_bits, False
    ).permute(0, 2, 1)
    ref_output = fused_moe_ref(x, dequant_w1, gating_output, topk, True)
    print(ref_output)

    diff = ref_output - fused_output
    num_diffs = torch.count_nonzero(diff)
    assert num_diffs == 0, f"Found {num_diffs / diff.numel()}% differences w1 between ref and custom fused_moe"
    

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
# test_fused_moe()
test_full_fused_moe()
