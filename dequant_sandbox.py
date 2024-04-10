import torch
from vllm.model_executor.layers.fused_moe import moe_align_block_size, fused_topk
from typing import Optional, Dict, Any
import triton
import triton.language as tl

@triton.jit
def dequant_kernel_248(
    g_idx_ptr, scales_ptr, qweight_ptr, qzeros_ptr, expert_ids_ptr, out_ptr,
    maxq: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    EM, N, K,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # strides
    stride_qe: tl.constexpr, stride_qk: tl.constexpr, stride_qn: tl.constexpr,
    stride_se: tl.constexpr,
    stride_ok: tl.constexpr, stride_on: tl.constexpr,
    num_bits: int,
):
    # Block indexing from triton matmul tutorial for 
    # L2 cache reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m != 0:
        return

    numel_per_i32 = 32 // num_bits

    # Offsets of the block for the output dimension
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # Offsets of k relative to the entire input dimension (not packed)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    q_offs_k = offs_k // numel_per_i32
    qweight_shifts = (offs_k % numel_per_i32) * num_bits 
    qzero_shifts = (offs_bn % numel_per_i32) * num_bits

    # off_experts = tl.load(expert_ids_ptr + pid_m)
    off_experts = 0  # for now, dequantize expert 0
    qweight_ptrs = qweight_ptr + off_experts * stride_qe + (q_offs_k[:, None] * stride_qk +
                                                offs_bn[None, :] * stride_qn)

    row_idxs = q_offs_k

    # (BLOCK_SIZE_K, )
    # g_idx = tl.load(g_idx_ptr + row_idxs, None, eviction_policy="evict_last")


    # tmp1 = g_idx + num_groups
    # tmp2 = g_idx < 0
    # tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    # groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + off_experts * stride_se + offs_bn, None).to(
        tl.float32
    )

    qzeros = tl.load(
        # qzeros_ptr + ((qzero_ncols * groups) + (offs_bn // numel_per_i32)),
        # assuming there's one group?
        qzeros_ptr + (offs_bn // numel_per_i32),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> qzero_shifts
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1
    # Unpack zeros
    qzero_ncols: tl.constexpr = outfeatures // numel_per_i32
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Unpack weights
        qweights = tl.load(qweight_ptrs, None)
        weights = qweights >> qweight_shifts[:, None]  # bit shift qweight

        weights = weights & maxq


        weights = weights - zeros
        weights = scales * weights
        weights = weights.to(tl.float16)
        tl.static_print(weights.dtype)
        tl.dot(weights, tl.trans(weights))

        tl.store(out_ptr + (
            (offs_k[:, None] + k * BLOCK_SIZE_K) * stride_ok + 
            offs_bn[None, :] * stride_on
        ), weights)
        
        qweight_ptrs += BLOCK_SIZE_K // numel_per_i32 * stride_qk



def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """

    # permute, similar to how we do in the fused_moe kernel
    # the shape is (E, N, K) but since this is a torch permute,
    # the data layout is still (E, K, N)
    print(f"incoming qweight shape: {qweight.shape}")
    qweight = qweight.permute(0, 2, 1) 

    num_groups = scales.shape[0]
    outfeatures = scales.shape[2]
    infeatures = g_idx.shape[1]
    num_tokens = 128

    out = torch.empty((infeatures, outfeatures), device="cuda", dtype=torch.float16)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    # grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731
    grid = lambda META: (triton.cdiv(num_tokens, META[
        'BLOCK_SIZE_M']) * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']), )
    

    expert_ids = None
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_M = 64
    GROUP_SIZE_M = 1
    dequant_kernel_248[grid](
        g_idx, scales, qweight, qzeros, expert_ids, out,
        maxq,
        outfeatures,
        num_groups,
        4, outfeatures, infeatures,
        BLOCK_SIZE_N, BLOCK_SIZE_K, BLOCK_SIZE_M, GROUP_SIZE_M,
        qweight.stride(0), qweight.stride(2), qweight.stride(1),
        scales.stride(0),
        out.stride(0), out.stride(1),
        bits
    )
    return out
    

@triton.jit()
def fused_moe_kernel(
    # Pointers to matrices
    x_ptr, qweight_ptr, qscales_ptr, qzeros_ptr, g_idx_ptr, out_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    # Matrix dimensions
    N, K, EM,
    num_valid_tokens,
    # Strides
    stride_xm, stride_xk,
    stride_qe, stride_qk, stride_qn, # strides for qweight
    stride_om, stride_on,
    stride_se,
    # Meta-parameters
    num_bits: int,
    maxq: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Adjust offsets for packed tensors: qweight, qzero
    numel_per_i32 = 32 // num_bits
    q_offs_k = offs_k // numel_per_i32
    qweight_shifts = (offs_k % numel_per_i32) * num_bits
    qzero_shifts = (offs_bn % numel_per_i32) * num_bits

    x_ptrs = x_ptr + (offs_token[:, None] // top_k * stride_xm +
                      offs_k[None, :] * stride_xk)

    off_experts = tl.load(expert_ids_ptr + pid_m)
    qweight_ptrs = qweight_ptr + off_experts * stride_qe + (q_offs_k[:, None] * stride_qk +
                                                offs_bn[None, :] * stride_qn)

    # qscales and qzeros only have a dependence on the n dimension, so we only have to 
    # do a single I/O
    scales = tl.load(qscales_ptr + off_experts * stride_se + offs_bn, None)#.to(tl.float16)
    qzeros = tl.load(
        # qzeros_ptr + ((qzero_ncols * groups) + (offs_bn // numel_per_i32)),
        # assuming there's one group?
        qzeros_ptr + (offs_bn // numel_per_i32),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> qzero_shifts
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1


    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        x = tl.load(x_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        qweights = tl.load(
            qweight_ptrs, 
            mask=offs_k[:, None] + k * BLOCK_SIZE_K // 4 < K // 4,
            other=0.0
        )
        weights = qweights >> qweight_shifts[:, None]  # bit shift qweight
        weights = weights & maxq

        weights = weights - zeros
        weights = scales * weights
        weights = weights.to(tl.float16)
        x = x.to(tl.float16)
        # tl.static_print(scales.dtype)
        # tl.device_print("x dtype: ", x.dtype)


        # b = tl.load(b_ptrs,
                    # mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    # other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(x, weights)
        # tl.device_print("accumulator dtype: ", accumulator.dtype)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        qweight_ptrs += BLOCK_SIZE_K // numel_per_i32 * stride_qk


    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_token[:, None] + stride_on * offs_on[
        None, :]
    c_mask = token_mask[:, None] & (offs_on[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    x: torch.Tensor, 
    qweights: torch.Tensor, 
    qscales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    out: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool, top_k: int, num_bits: int,
    config: Dict[str, Any]) -> None:

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    infeatures, outfeatures = g_idx.shape[1], qscales.shape[2]
    print(f"{infeatures=}, {outfeatures=}")

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(outfeatures, META['BLOCK_SIZE_N']), )
    maxq = 2**num_bits - 1

    # Both of these are the *unpacked* dimensions that are necessary for pointer
    # arithmetic to be computed properly
    fused_moe_kernel[grid](
        x, qweights, qscales, qzeros, g_idx, out,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        outfeatures, infeatures, sorted_token_ids.shape[0],
        topk_ids.numel(),
        x.stride(0), x.stride(1),
        qweights.stride(0), qweights.stride(2), qweights.stride(1),
        out.stride(0), out.stride(1),
        qscales.stride(0),
        num_bits=num_bits,
        maxq=maxq,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        **config,
    )

def fused_moe(
    hidden_states: torch.Tensor,
    qweights: torch.Tensor, qscales: torch.Tensor, qzeros: torch.Tensor, g_idx: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_bits: int = 8
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
    assert gating_output.shape[1] == qweights.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    # assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    # assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]

    # permute qweights
    qweights = qweights.permute(0, 2, 1)
    print(f"qweights: {qweights.shape} {qweights.dtype} {qweights.device}")

    M, _ = hidden_states.shape
    E, N, _ = qweights.shape

    topk_weights, topk_ids = fused_topk(gating_output, topk, renormalize)

    # if override_config:
    #     config = override_config
    # else:
    #     # First try to load optimal config from the file
    #     configs = get_moe_configs(E, w2.shape[2])

    #     if configs:
    #         # If an optimal configuration map has been found, look up the
    #         # optimal config
    #         config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    #     else:
    #         # Else use the default config
    #         config = {
    #             'BLOCK_SIZE_M': 64,
    #             'BLOCK_SIZE_N': 64,
    #             'BLOCK_SIZE_K': 32,
    #             'GROUP_SIZE_M': 8
    #         }

    #         if M <= E:
    config = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 16,
        'BLOCK_SIZE_K': 16,
        'GROUP_SIZE_M': 1
    }

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)

    invoke_fused_moe_kernel(hidden_states, qweights, qscales, qzeros, g_idx, intermediate_cache1,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False,
                            topk_ids.shape[1], num_bits, config)

    return intermediate_cache1

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
