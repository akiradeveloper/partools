//! Small workgroup controls shared by composed algorithms.

use cubecl::prelude::*;

/// Stable rank, count, and first lane among equal eight-bit digits in a
/// subgroup. Ballot words beyond the actual subgroup width are not consumed.
#[cubecl::cube]
pub(crate) fn plane_digit_rank(digit: u32, valid: bool) -> (u32, u32, u32) {
    let mut peers = plane_ballot(valid);
    #[unroll]
    for bit in 0u32..8u32 {
        let set = (digit & (1u32 << bit)) != 0u32;
        let ones = plane_ballot(set);
        if set {
            peers &= ones;
        } else {
            peers &= !ones;
        }
    }
    let rank = RuntimeCell::<u32>::new(0u32);
    let count = RuntimeCell::<u32>::new(0u32);
    let first = RuntimeCell::<u32>::new(0u32);
    #[unroll]
    for word in 0usize..4usize {
        if word * 32usize < PLANE_DIM as usize {
            let mask = peers.extract(word);
            if count.read() == 0u32 && mask != 0u32 {
                first.store(word as u32 * 32u32 + mask.trailing_zeros());
            }
            count.store(count.read() + mask.count_ones());
            if word < UNIT_POS_PLANE as usize / 32usize {
                rank.store(rank.read() + mask.count_ones());
            } else if word == UNIT_POS_PLANE as usize / 32usize {
                let lower = (1u32 << (UNIT_POS_PLANE % 32u32)) - 1u32;
                rank.store(rank.read() + (mask & lower).count_ones());
            }
        }
    }
    (
        rank.read(),
        if valid { count.read() } else { 0u32 },
        first.read(),
    )
}

/// Exclusive prefix and total for at most 256 lanes. Subgroups reduce the
/// values in parallel; only their totals cross workgroup memory. A subgroup
/// leader also handles a partially occupied final subgroup.
#[cubecl::cube]
pub(crate) fn block_exclusive_sum(value: u32) -> (u32, u32) {
    let local_prefix = plane_exclusive_sum(value);
    let local_total = plane_sum(value);
    let mut prefixes = Shared::<[u32]>::new_slice(257usize);
    if UNIT_POS_PLANE == 0u32 {
        prefixes[PLANE_POS as usize] = local_total;
    }
    sync_cube();
    let planes = CUBE_DIM.div_ceil(PLANE_DIM);
    if UNIT_POS == 0u32 {
        let prefix = RuntimeCell::<u32>::new(0u32);
        let plane = RuntimeCell::<u32>::new(0u32);
        while plane.read() < planes {
            let count = prefixes[plane.read() as usize];
            prefixes[plane.read() as usize] = prefix.read();
            prefix.store(prefix.read() + count);
            plane.store(plane.read() + 1u32);
        }
        prefixes[planes as usize] = prefix.read();
    }
    sync_cube();
    (
        local_prefix + prefixes[PLANE_POS as usize],
        prefixes[planes as usize],
    )
}

/// Scans one contiguous row per workgroup and records its total. Each lane
/// owns a contiguous chunk, so only lane totals need a workgroup scan.
#[cubecl::cube]
pub(crate) fn row_exclusive_sum(
    input: &[u32],
    width: usize,
    output: &mut [u32],
    totals: &mut [u32],
) {
    let row = CUBE_POS as usize;
    let chunk = width.div_ceil(CUBE_DIM as usize);
    let start = usize::min(UNIT_POS as usize * chunk, width);
    let end = usize::min(start + chunk, width);
    let sum = RuntimeCell::<u32>::new(0u32);
    let index = RuntimeCell::<usize>::new(start);
    while index.read() < end {
        sum.store(sum.read() + input[row * width + index.read()]);
        index.store(index.read() + 1usize);
    }
    let (prefix, total) = block_exclusive_sum(sum.read());
    if UNIT_POS == 0u32 {
        totals[row] = total;
    }
    let current = RuntimeCell::<u32>::new(prefix);
    index.store(start);
    while index.read() < end {
        output[row * width + index.read()] = current.read();
        current.store(current.read() + input[row * width + index.read()]);
        index.store(index.read() + 1usize);
    }
}
