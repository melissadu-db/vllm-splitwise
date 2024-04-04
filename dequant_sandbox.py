import torch
from vllm.model_executor.layers.fused_moe import moe_align_block_size, fused_topk
from typing import Optional, Dict, Any
import triton
import triton.language as tl

# state = {}
# state_dir = "/mnt/workdisk/linden/vllm-private/quantized_weights"
# for file in os.listdir(state_dir):
#     if file.startswith("0") and "output" not in file:
#         state[file] = torch.load(os.path.join(state_dir, file))

@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    qweight_ptr, # B
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # pointers to quantization state
    zeros_ptr,
    scales_ptr,
    g_idx_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ze,
    stride_zn,
    stride_se,
    stride_sn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # TODO: we can add tl.max_contiguous(tl.multiple_of) for funsies later
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    off_experts = tl.load(expert_ids_ptr + pid_m)

    # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    qweight_ptrs = qweight_ptr + off_experts * stride_be + (
        offs_k[None, :] // 4 * stride_bk 
        + offs_bn[:, None] * stride_bn 
    )

    # (BLOCK_SIZE_N, )
    zeros_ptrs = zeros_ptr + off_experts * stride_ze + (offs_bn * stride_zn // 4)
    # (BLOCK_SIZE_N, )
    scales_ptrs = scales_ptr + off_experts * stride_se + (offs_bn * stride_sn)

    shifter = (offs_k % 4) * 8
    zeros_shifter = (offs_bn % 4) * 8
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] + k * BLOCK_SIZE_K < K),
                    other=0.0)
                    
        # (BLOCK_SIZE_N, BLOCK_SIZE_K)
        qweight = tl.load(qweight_ptrs,
                    mask=(offs_bn[:, None] * stride_bn & 
                          offs_k[None, :] + k * BLOCK_SIZE_K < K),
                    other=0.0)

        # weight_q = scale * (weight_8bit - zero_point)
        # zeros = tl.load(zeros_ptrs, mask=offs_bn * stride_zn < N // 4, other=0.0)
        scales = tl.load(scales_ptrs, mask=offs_bn * stride_sn < N, other=0.0)
        # zeros = (zeros >> zeros_shifter) & 0xFF
        # zeros = (zeros + 1) * scales

        b = (qweight >> shifter[None, :]) & 0xFF
        b = (b - 127) * scales[:, None]
        # if pid == 0 and k == 0:
            # tl.device_print("a: ", a)
            # tl.device_print("b: ", b)
        
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, tl.trans(b))

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        qweight_ptrs += BLOCK_SIZE_K // 4 * stride_bk
        
        # zeros_ptrs += BLOCK_SIZE_N // 4 * stride_zn
        # scales_ptrs += BLOCK_SIZE_N * stride_sn

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def dequant_kernel(
    qweight_ptr, out_ptr, scales_ptr,
    # Matrix dimensions
    E, K, N,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_qe,
    stride_qk,
    stride_qn,
    stride_se,
    stride_sn,
    stride_oe,
    stride_ok,
    stride_on,
    # Meta-parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    # if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        # return

    offs_bn = tl.arange(0, BLOCK_SIZE_N) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K) % K

    off_experts = 0

    # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    n_idxs = offs_bn + pid * BLOCK_SIZE_N
    # n_idxs = (offs_bn[None, :] + pid * BLOCK_SIZE_N)
    qweight_ptrs = qweight_ptr + off_experts * stride_qe + (
        offs_k[:, None] * stride_qk
        + n_idxs[None, :] * stride_qn
    )

    scale_offsets = off_experts * stride_se + (offs_bn * stride_sn)
    scales_ptrs = scales_ptr + scale_offsets
    # shifter = (offs_bn % 4) * 8
    
    # TODO: add mask

    scales = tl.load(scales_ptrs, mask=n_idxs < N, other=0.0)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        qweight = tl.load(qweight_ptrs)
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
        # b = (qweight >> shifter[:, None]) & 0xFF
        # tl.device_print("b: ", b)
        # b = (b - 127) * scales[None, :]
        b = qweight * scales[None, :]
        # tl.device_print('block k: ', k, ' b: ', b)
        # tl.device_print("scales: ", scales)
        acc += b  # TODO explicit cast to bf16 instead of using accumulator
        # out_ptrs = out_ptr + offs_bn[:, None] * stride_on + offs_k[None, :] * stride_ok
        k_idxs = offs_k[:, None] + k * BLOCK_SIZE_K
        offsets = (n_idxs[None, :] * stride_on) + (k_idxs * stride_ok)
        tl.store(out_ptr + offsets, acc)
        # tl.store(out_ptrs, offsets)
        
        # Update qweight_ptrs
        # qweight_ptrs += BLOCK_SIZE_K // 4 * stride_qk
        qweight_ptrs += BLOCK_SIZE_K * stride_qk  # TODO just use k_idxs and n_idxs


    # acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float16)

    # b = (qweight >> shifter[None, :]) & 0xFF
    # b = (b - 128) * scales[:, None]
    # acc += b

    # out_ptrs = out_ptr + offs_bn[:, None] * stride_on + offs_k[None, :] * stride_ok
    # tl.store(out_ptrs, acc)

def dequantize(qweights_EKN, scales):
    # qweights_EKN = qweights_EKN.view(torch.uint8).permute(0, 2, 1)
    E, K, N = qweights_EKN.shape
    print(f"original shape: {qweights_EKN.shape}")
    qweights_EKN = qweights_EKN.reshape((E, N, K)).view(dtype=torch.uint8).reshape((E, K * 4, N))
    print(f"Shifted shape: {qweights_EKN.shape}")
    BLOCK_SIZE_K = 4
    BLOCK_SIZE_N = 8
    E, K, N = qweights_EKN.shape
    # K = int(K * 4)
    # N = N // 4
    out_buf = torch.empty(
        # (K, BLOCK_SIZE_N), 
        (E, K, N),
        dtype=torch.float32,
        device=qweights_EKN.device,
    )
    dequant_kernel[(1, 1, 1)](
        qweights_EKN, out_buf, scales,
        E, K, N,
        qweights_EKN.stride(0),
        qweights_EKN.stride(1),
        qweights_EKN.stride(2),
        scales.stride(0),
        scales.stride(2),
        out_buf.stride(0),
        out_buf.stride(1),
        out_buf.stride(2),
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
    )
    return out_buf
    

def fused_moe(
    hidden_states: torch.Tensor,
    # w1 quant state
    w1_qweight: torch.Tensor,
    w1_qzeros: torch.Tensor,
    w1_scales: torch.Tensor,
    w1_g_idx: torch.Tensor,
    # w2 quant state
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = True,
    override_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    # assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    # assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    # assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    M, _ = hidden_states.shape
    E, N, _ = w1_qweight.shape

    topk_weights, topk_ids = fused_topk(gating_output, topk, renormalize)

    # if override_config:
    #     config = override_config
    # else:
        # # First try to load optimal config from the file
        # configs = get_moe_configs(E, w2.shape[2])

        # if configs:
        #     # If an optimal configuration map has been found, look up the
        #     # optimal config
        #     config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        # else:
            # Else use the default config
    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }

    if M <= E:
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
        }

    print(f"launch config: {config}")
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)

    # we need to unsqueeze and permute here
    invoke_fused_moe_kernel(hidden_states, w1_qweight.permute(0, 2, 1), intermediate_cache1,
                            w1_scales.squeeze(1), w1_qzeros.squeeze(1), g_idx,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False,
                            topk_ids.shape[1], config)
    print("-" * 80)
    print("Result:")
    print(intermediate_cache1)
    with open("./quantized_weights/0_output.pt", "wb") as f:
        torch.save(intermediate_cache1, f)




def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any]) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
    
    print(f"{B.shape=}")
    print(f"{A.shape=}")
    print(f"{B.stride()=}")
    print(f"stride_bn: {B.stride(1)}")
    print(f"stride_bk: {B.stride(2)}")
    print(f"{topk_ids=}")
    print(f"{expert_ids=}")

    
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        zeros,
        scales,
        g_idx,
        B.shape[1], # N
        B.shape[2], # K
        sorted_token_ids.shape[0], # EM
        topk_ids.numel(),
        A.stride(0), # stride_am
        A.stride(1), # stride_ak
        B.stride(0), # stride_be
        B.stride(2), # stride_bk
        B.stride(1), # stride_bn
        C.stride(1), # stride_cm
        C.stride(2), # stride_cn
        zeros.stride(0), # stride_ze
        zeros.stride(1), # stride_zn
        scales.stride(0), # stride_se
        scales.stride(1), # stride_sn
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        **config,
    )

# top_k = 4
# E = 16

# torch.manual_seed(1)

# qweight = state['0_qweight.pt']
# qzeros = state['0_qzeros.pt']
# scales = state["0_scales.pt"]
# g_idx = state['0_g_idx.pt']
# dtype = scales.dtype
# print(f'Running with {dtype=}')
# T = 2
# E = 16

# x = torch.rand((T, 6144), dtype=dtype).to(qweight.device)
# gating_output = torch.rand((T, E), dtype=dtype).to(x.device)
# dequantized = dequantize(qweight, scales)
# torch.set_printoptions(profile="full")


# fused_moe(
#     x,
#     qweight,
#     qzeros,
#     scales,
#     g_idx,
#     gating_output,
#     topk=top_k,
#     renormalize=True,
#     inplace=True,
# )
