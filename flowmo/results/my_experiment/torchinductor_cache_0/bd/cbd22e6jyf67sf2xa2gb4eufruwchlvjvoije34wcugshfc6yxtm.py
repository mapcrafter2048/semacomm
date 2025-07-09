
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__log_softmax_backward_data__softmax__softmax_backward_data__to_copy_add_div_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp9 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = 0.0025
    tmp3 = tmp1 * tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.001953125
    tmp7 = tmp5 * tmp6
    tmp8 = -tmp7
    tmp10 = 2.0
    tmp11 = tmp9 * tmp10
    tmp12 = 10.0
    tmp13 = tmp11 * tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 - tmp19
    tmp21 = tmp8 * tmp20
    tmp22 = -tmp3
    tmp23 = tmp22 * tmp4
    tmp24 = -tmp23
    tmp26 = tmp25 + tmp14
    tmp27 = tl_math.log(tmp26)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp24 * tmp25
    tmp30 = (tmp29 / tmp26)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31 * tmp6
    tmp33 = tmp21 + tmp32
    tmp34 = tmp11.to(tl.float32)
    tmp35 = tmp34 * tmp4
    tmp37 = tmp35 - tmp36
    tmp38 = tmp37 * tmp12
    tmp39 = tl_math.exp(tmp38)
    tmp41 = (tmp39 / tmp40)
    tmp42 = tmp8 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [R0_BLOCK])
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp46 = tmp33 * tmp41
    tmp47 = tl.broadcast_to(tmp46, [R0_BLOCK])
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp47, 0))
    tmp50 = tl_math.exp(tmp20)
    tmp51 = tmp50 * tmp45
    tmp52 = tmp42 - tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 * tmp12
    tmp55 = -tmp41
    tmp56 = libdevice.fma(tmp55, tmp49, tmp46)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57 * tmp12
    tmp59 = tmp54 + tmp58
    tmp60 = tmp59 * tmp10
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp60, None)
