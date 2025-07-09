
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': '6769567BF9CB28FC4140E0A692BA44D0F6558D1D18952489FF33DAF4A85D2E95', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6684672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex // 3342336
    x6 = (xindex % 278528)
    x2 = ((xindex // 64) % 4352)
    x1 = ((xindex // 2) % 32)
    x3 = ((xindex // 278528) % 12)
    x8 = xindex // 278528
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x6 + 557056*x4), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (1 + 2*x6 + 557056*x4), None, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (256*x8 + (x2)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tmp1 >= tmp4
    tmp18 = tl.full([1], 4352, tl.int64)
    tmp19 = tmp1 < tmp18
    tmp20 = tl.load(in_ptr4 + (2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp5, tmp16, tmp20)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp0 * tmp22
    tmp25 = tl.load(in_ptr1 + (1 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp8
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (1 + 2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp5, tmp32, tmp33)
    tmp35 = tl.load(in_ptr4 + (1 + 2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp5, tmp34, tmp35)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp24 * tmp37
    tmp39 = tmp23 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tl.load(in_ptr1 + (768 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.load(in_ptr5 + (256*x8 + (x2)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.load(in_ptr6 + (2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp5, tmp49, tmp50)
    tmp52 = tl.load(in_ptr7 + (2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp53 = tl.where(tmp5, tmp51, tmp52)
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp0 * tmp54
    tmp56 = tl.load(in_ptr1 + (769 + 2*x1 + 64*x3 + 2304*(x2) + 589824*x4), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp57 * tmp43
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tl.load(in_ptr6 + (1 + 2*x1), tmp5, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 * tmp61
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp5, tmp63, tmp64)
    tmp66 = tl.load(in_ptr7 + (1 + 2*x1 + 64*x3 + 768*((-256) + x2) + 3145728*x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp67 = tl.where(tmp5, tmp65, tmp66)
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp24 * tmp68
    tmp70 = tmp55 + tmp69
    tmp71 = tmp70.to(tl.float32)
    tl.store(out_ptr1 + (x5), tmp40, None)
    tl.store(out_ptr3 + (x5), tmp71, None)
