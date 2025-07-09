
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 4352)
    x0 = (xindex % 3)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 4352, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = x0
    tmp13 = tl.full([1], 2, tl.int32)
    tmp14 = tmp12 == tmp13
    tmp15 = tmp13 == tmp13
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp13 == tmp16
    tmp18 = tmp16 == tmp16
    tmp19 = ((((-256) + x1) // 64) % 64)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 0.0
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tl.where(tmp17, tmp20, tmp21)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = (((-256) + x1) % 64)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tl.where(tmp15, tmp27, tmp24)
    tmp29 = tmp12 == tmp16
    tmp30 = tl.where(tmp29, tmp20, tmp21)
    tmp31 = tl.where(tmp29, tmp22, tmp30)
    tmp32 = tl.where(tmp14, tmp27, tmp31)
    tmp33 = tl.where(tmp14, tmp28, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp9, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp8, tmp35)
    tl.store(out_ptr0 + (x3), tmp36, xmask)
