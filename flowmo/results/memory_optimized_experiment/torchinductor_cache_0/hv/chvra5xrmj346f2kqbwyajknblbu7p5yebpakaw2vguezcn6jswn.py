
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 7, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__softmax__to_copy_add_div_mul_sum_4(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 10.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp10, 0))
    tmp13 = tmp8 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = tl_math.log(tmp17)
    tmp19 = tmp2.to(tl.float32)
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [R0_BLOCK])
    tmp24 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp22, 0))
    tmp25 = tmp21 - tmp24
    tmp26 = tmp25 * tmp3
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tl.broadcast_to(tmp27, [R0_BLOCK])
    tmp30 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = (tmp27 / tmp30)
    tmp32 = tmp7 - tmp12
    tmp33 = tmp32 - tmp18
    tmp34 = tmp31 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [R0_BLOCK])
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (x0), tmp24, None)
    tl.store(out_ptr2 + (x0), tmp30, None)
    tl.store(out_ptr3 + (x0), tmp37, None)
