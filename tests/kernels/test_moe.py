"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import tempfile

import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from vllm._C import ops

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.quantization.awq import (AWQConfig,
                                                         AWQLinearMethod)
from vllm.model_executor.layers.quantization.gptq import (ExllamaState,
                                                          GPTQConfig,
                                                          GPTQLinearMethod)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel, initialize_model_parallel)


def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a,
                              w1,
                              w2,
                              score,
                              topk,
                              renormalize=False,
                              inplace=False)
    torch_output = torch_moe(a, w1, w2, score, topk)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""
    # Initialize dist environment
    if not torch.distributed.is_initialized():
        temp_file = tempfile.mkstemp()[1]
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            init_method=f"file://{temp_file}",
        )
    initialize_model_parallel()
    torch.set_default_dtype(dtype)

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        tp_size=1,
    ).cuda()

    # Load the weights
    vllm_moe.gate.linear_weights["weight"][:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.ws.weight[i][:] = torch.cat(weights, dim=0)
        vllm_moe.w2s.weight[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(inputs)
    vllm_states = vllm_moe.forward(inputs)

    # destroy dist environment
    destroy_model_parallel()

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    assert torch.allclose(hf_states,
                          vllm_states,
                          rtol=mixtral_moe_tol[dtype],
                          atol=mixtral_moe_tol[dtype])


def torch_moe_gptq(a, w1, w1_gidx, w1_scale, w1_zero, w2, w2_gidx, w2_scale,
                   w2_zero, score, topk, bits):
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    (B, D) = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[2],
                      dtype=a.dtype,
                      device=a.device)
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            dw1 = ops.dequant_gptq(w1[i], w1_zero[i], w1_scale[i], w1_gidx[i],
                                   bits, False)
            dw2 = ops.dequant_gptq(w2[i], w2_zero[i], w2_scale[i], w2_gidx[i],
                                   bits, False)
            r1 = SiluAndMul()(torch.matmul(a[mask], dw1))
            out[mask] = torch.matmul(r1, dw2)
    return (out.view(B, -1, w2.shape[2]) *
            topk_weight.view(B, -1, 1)).sum(dim=1).half()


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("exstate",
                         [ExllamaState.UNINITIALIZED, ExllamaState.UNUSED])
@pytest.mark.parametrize("groupsize", [-1, 128])
@pytest.mark.parametrize("actorder", [True, False])
@pytest.mark.parametrize("bits", [4, 8])
def test_fused_moe_gptq(m: int, n: int, k: int, e: int, topk: int,
                        exstate: ExllamaState, groupsize: int, actorder: bool,
                        bits: int):
    if bits == 8 and exstate == ExllamaState.UNUSED:
        pytest.skip(
            '8-bit group_gptq_gemm are not supported for exstate=UNUSED')

    RANGE = 1000000000
    a = torch.randn((m, k), device='cuda', dtype=torch.half) / 100
    qw1 = torch.randint(-RANGE,
                        RANGE, (e, (k // 32) * bits, n * 2),
                        dtype=torch.int,
                        device='cuda')
    qw2 = torch.randint(-RANGE,
                        RANGE, (e, (n // 32) * bits, k),
                        dtype=torch.int,
                        device='cuda')

    groupsize1 = groupsize if groupsize != -1 else k
    groupsize2 = groupsize if groupsize != -1 else n
    gidx1 = torch.tensor([i // groupsize1 for i in range(k)],
                         dtype=torch.int32,
                         device='cuda').unsqueeze(0).expand(e, k).contiguous()
    gidx2 = torch.tensor([i // groupsize2 for i in range(n)],
                         dtype=torch.int32,
                         device='cuda').unsqueeze(0).expand(e, n).contiguous()

    scale1 = torch.randn(
        (e, k // groupsize1, n * 2), dtype=torch.half, device='cuda') / 1000
    scale2 = torch.randn(
        (e, n // groupsize2, k), dtype=torch.half, device='cuda') / 1000

    zero1 = torch.randint(-RANGE,
                          RANGE, (e, k // groupsize1, (n * 2 // 32) * bits),
                          dtype=torch.int32,
                          device='cuda')
    zero2 = torch.randint(-RANGE,
                          RANGE, (e, n // groupsize2, (k // 32) * bits),
                          dtype=torch.int32,
                          device='cuda')
    w1 = {
        "qweight": qw1,
        "g_idx": gidx1,
        "scales": scale1,
        "qzeros": zero1,
        "exllama_state": exstate
    }
    w2 = {
        "qweight": qw2,
        "g_idx": gidx2,
        "scales": scale2,
        "qzeros": zero2,
        "exllama_state": exstate
    }

    score = torch.randn((m, e), device='cuda', dtype=torch.half)

    gptq_method = GPTQLinearMethod(GPTQConfig(bits, groupsize, actorder))
    torch_output = torch_moe_gptq(a, qw1, gidx1, scale1, zero1, qw2, gidx2,
                                  scale2, zero2, score, topk, bits)
    cuda_output = gptq_method.apply_moe_weights(w1, w2, a, score, topk, False)
    # gptq kernels have large variance in output
    assert torch.allclose(cuda_output, torch_output, atol=5e-2, rtol=0)


def torch_moe_awq(a, w1, w1_scale, w1_zero, w2, w2_scale, w2_zero, score,
                  topk):
    score = torch.softmax(score.float(), dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)
    (B, D) = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[2] * 8,
                      dtype=a.dtype,
                      device=a.device)
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            dw1 = ops.awq_dequantize(w1[i], w1_scale[i], w1_zero[i], 0, 0, 0)
            dw2 = ops.awq_dequantize(w2[i], w2_scale[i], w2_zero[i], 0, 0, 0)
            r1 = SiluAndMul()(torch.matmul(a[mask].half(), dw1))
            out[mask] = torch.matmul(r1, dw2).to(out.dtype)
    return (out.view(B, -1, w2.shape[2] * 8) *
            topk_weight.view(B, -1, 1)).sum(dim=1).half()


@pytest.mark.parametrize("m", [1024, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 512, 1024])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2, 6])
def test_fused_moe_awq(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):
    # awq requires minimum capability 75
    if torch.version.hip is not None:
        return
    capability = torch.cuda.get_device_capability()
    capability = capability[0] * 10 + capability[1]
    if capability < 75:
        return

    RANGE = 1000000000
    groupsize = 128
    a = torch.randn((m, k), device='cuda', dtype=torch.half) / 10
    qw1 = torch.randint(-RANGE,
                        RANGE, (e, k, n * 2 // 8),
                        dtype=torch.int,
                        device='cuda')
    qw2 = torch.randint(-RANGE,
                        RANGE, (e, n, k // 8),
                        dtype=torch.int,
                        device='cuda')

    scale1 = torch.randn(
        (e, k // groupsize, n * 2), dtype=torch.half, device='cuda') / 50
    scale2 = torch.randn(
        (e, n // groupsize, k), dtype=torch.half, device='cuda') / 50

    zero1 = torch.randint(-RANGE,
                          RANGE, (e, k // groupsize, (n * 2 // 32) * 4),
                          dtype=torch.int32,
                          device='cuda')
    zero2 = torch.randint(-RANGE,
                          RANGE, (e, n // groupsize, (k // 32) * 4),
                          dtype=torch.int32,
                          device='cuda')
    w1 = {"qweight": qw1, "scales": scale1, "qzeros": zero1}
    w2 = {"qweight": qw2, "scales": scale2, "qzeros": zero2}

    score = torch.randn((m, e), device='cuda', dtype=torch.half)

    awq_method = AWQLinearMethod(AWQConfig(4, groupsize, False))
    torch_output = torch_moe_awq(a, qw1, scale1, zero1, qw2, scale2, zero2,
                                 score, topk)
    cuda_output = awq_method.apply_moe_weights(w1, w2, a, score, topk, False)
    assert torch.allclose(cuda_output, torch_output, atol=1e-2, rtol=0)
