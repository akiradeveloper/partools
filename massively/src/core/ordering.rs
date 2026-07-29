//! Ordering controls derived from adjacent semantic items.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::core::arg_reduce::{ArgReduceDispatch, ArgReductionOp, arg_reduce};
use crate::core::arity::{A13, Dispatch};
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::launch::cube_count_1d;
use crate::core::op::{BinaryPredicateOp, ReductionOp};
use crate::core::read::{
    AdjacentIndexedTransform, Env0, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots,
    ReadExpression, StageRead,
};
use crate::core::reduce::{ReduceDispatch, reduce};
use crate::core::value::MStorageElement;
use crate::op::IndexedBinaryOp;
use crate::{DeviceVec, Error, Executor, MIndex, MVec};

const BLOCK_SIZE: u32 = 256;
const SORT_BLOCK_ITEMS: usize = 256;
const SORT_BLOCK_SIZE: u32 = SORT_BLOCK_ITEMS as u32;
const SORT_MERGE_SIZE: usize = 64;
const SORT_MERGE_ITEMS: usize = 64;

#[cubecl::cube]
pub(crate) fn binary_predicate<Item, Op>(lhs: Item, rhs: Item) -> bool
where
    Item: CubeType + 'static,
    Op: BinaryPredicateOp<Item>,
{
    crate::flag::is_set(Op::apply(lhs, rhs))
}

#[cubecl::cube]
trait AdjacentFlagOp<Item: CubeType>: 'static + Send + Sync {
    fn first() -> u32;
    fn apply(previous: Item, current: Item) -> u32;
}

struct ArgMinFirst<Less>(PhantomData<fn() -> Less>);
struct ArgMinLast<Less>(PhantomData<fn() -> Less>);
struct ArgMaxFirst<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMinFirst<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        crate::core::ordering::binary_predicate::<Item, Less>(rhs, lhs)
    }
}

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMinLast<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        !crate::core::ordering::binary_predicate::<Item, Less>(lhs, rhs)
    }
}

#[cubecl::cube]
impl<Item, Less> ArgReductionOp<Item> for ArgMaxFirst<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn rhs_wins(lhs: Item, rhs: Item) -> bool {
        crate::core::ordering::binary_predicate::<Item, Less>(lhs, rhs)
    }
}

struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

struct FirstAdjacentMatch<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> IndexedBinaryOp<Item> for FirstAdjacentMatch<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    type Output = u32;

    fn apply(previous: Item, current: Item, index: u32) -> u32 {
        if index != 0u32
            && crate::core::ordering::binary_predicate::<Item, Equal>(previous, current)
        {
            index - 1u32
        } else {
            4_294_967_295u32
        }
    }
}

struct FirstSortedBreak<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> IndexedBinaryOp<Item> for FirstSortedBreak<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    type Output = u32;

    fn apply(previous: Item, current: Item, index: u32) -> u32 {
        if index != 0u32 && crate::core::ordering::binary_predicate::<Item, Less>(current, previous)
        {
            index
        } else {
            4_294_967_295u32
        }
    }
}

pub(crate) struct UniqueHead<Equal>(PhantomData<fn() -> Equal>);

#[cubecl::cube]
impl<Item, Equal> AdjacentFlagOp<Item> for UniqueHead<Equal>
where
    Item: CubeType + 'static,
    Equal: BinaryPredicateOp<Item>,
{
    fn first() -> u32 {
        1u32
    }

    fn apply(previous: Item, current: Item) -> u32 {
        if crate::core::ordering::binary_predicate::<Item, Equal>(previous, current) {
            0u32
        } else {
            1u32
        }
    }
}

pub(crate) struct SortedBreak<Less>(PhantomData<fn() -> Less>);

#[cubecl::cube]
impl<Item, Less> AdjacentFlagOp<Item> for SortedBreak<Less>
where
    Item: CubeType + 'static,
    Less: BinaryPredicateOp<Item>,
{
    fn first() -> u32 {
        0u32
    }

    fn apply(previous: Item, current: Item) -> u32 {
        if crate::core::ordering::binary_predicate::<Item, Less>(current, previous) {
            1u32
        } else {
            0u32
        }
    }
}

macro_rules! define_adjacent_flags_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: AdjacentFlagOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            order: &[u32],
            use_order: &[u32],
            len: &[u32],
            flags: &mut [u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                flags[index] = if index == 0usize {
                    Op::first()
                } else {
                    let previous = if use_order[0] != 0u32 {
                        order[index - 1usize] as usize
                    } else {
                        index - 1usize
                    };
                    let current = if use_order[0] != 0u32 {
                        order[index] as usize
                    } else {
                        index
                    };
                    Op::apply(
                        Expr::$method($( $slot, )+ offsets, previous),
                        Expr::$method($( $slot, )+ offsets, current),
                    )
                };
            }
        }
    };
}

define_adjacent_flags_kernel!(adjacent_flags_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);

trait AdjacentFlagDispatch<R, Input, Item, Slots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        order: Option<&DeviceVec<R, u32>>,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_adjacent_flags_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident; [$( $leaf:ident:$index:literal ),+],$env:ty) => {
        impl<R, Input, Item, Op, $( $leaf ),+>
            AdjacentFlagDispatch<R, Input, Item, $env, Op> for Dispatch<$arity, crate::core::storage::S1>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Op: AdjacentFlagOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(
                exec: &Executor<R>,
                input: &Input,
                order: Option<&DeviceVec<R, u32>>,
            ) -> Result<DeviceVec<R, u32>, Error> {
                let input_capacity = input.physical_len()?;
                let input_extent = input.logical_extent()?;
                let (capacity, extent) = if let Some(order) = order {
                    let order_extent = order.logical_extent();
                    let extent = input_extent.zipped(&order_extent)?;
                    (order.capacity(), extent)
                } else {
                    (input_capacity, input_extent)
                };
                let mut flags = exec.alloc_row::<u32>(capacity);
                flags.set_logical_extent(extent.clone());
                if capacity == 0 {
                    return Ok(flags);
                }
                let reads = Bindings::read(exec, input)?;
                let offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let len_handle = extent.materialize(exec)?;
                let use_order = exec
                    .client()
                    .create_from_slice(u32::as_bytes(&[u32::from(order.is_some())]));
                let (order, order_len) = order
                    .map(|order| (order.handle.clone(), order.capacity()))
                    .unwrap_or_else(|| {
                        (
                            exec.client().create_from_slice(u32::as_bytes(&[0u32])),
                            1,
                        )
                    });
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        exec.client(),
                        cube_count_1d(capacity.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(order, order_len),
                        BufferArg::from_raw_parts(use_order, 1),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(flags.handle.clone(), flags.capacity()),
                    );
                }
                Ok(flags)
            }
        }
    };
}

impl_adjacent_flags_dispatch!(A13,Eval13,adjacent_flags_a13; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12],Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>);

macro_rules! define_block_sort_permutation_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            len: &[u32],
            output: &mut [u32],
        ) {
            let local = UNIT_POS as usize;
            let tile_start = (CUBE_POS as usize) * SORT_BLOCK_ITEMS;
            let logical_len = len[0] as usize;
            let tile_len = if logical_len > tile_start {
                let remaining = logical_len - tile_start;
                if remaining < SORT_BLOCK_ITEMS {
                    remaining
                } else {
                    SORT_BLOCK_ITEMS
                }
            } else {
                0usize
            };

            let mut indices_a = Shared::<[u32]>::new_slice(SORT_BLOCK_ITEMS);
            let mut indices_b = Shared::<[u32]>::new_slice(SORT_BLOCK_ITEMS);
            if local < tile_len {
                indices_a[local] = (tile_start + local) as u32;
            }
            sync_cube();

            let width = RuntimeCell::<usize>::new(1usize);
            let source_a = RuntimeCell::<u32>::new(1u32);
            while width.read() < SORT_BLOCK_ITEMS {
                if local < tile_len {
                    let pair_width = width.read() * 2usize;
                    let base = (local / pair_width) * pair_width;
                    let left_remaining = tile_len - base;
                    let left_len = if left_remaining < width.read() {
                        left_remaining
                    } else {
                        width.read()
                    };
                    let right_start = base + left_len;
                    let right_remaining = tile_len - right_start;
                    let right_len = if right_remaining < width.read() {
                        right_remaining
                    } else {
                        width.read()
                    };
                    let rank = local - base;
                    let low_init = if rank > right_len { rank - right_len } else { 0usize };
                    let high_init = if rank < left_len { rank } else { left_len };
                    let low = RuntimeCell::<usize>::new(low_init);
                    let high = RuntimeCell::<usize>::new(high_init);

                    while low.read() < high.read() {
                        let left_rank = (low.read() + high.read()) / 2usize;
                        let right_rank = rank - left_rank;
                        if left_rank < left_len && right_rank > 0usize {
                            let left_index = if source_a.read() != 0u32 {
                                indices_a[base + left_rank]
                            } else {
                                indices_b[base + left_rank]
                            };
                            let right_index = if source_a.read() != 0u32 {
                                indices_a[right_start + right_rank - 1usize]
                            } else {
                                indices_b[right_start + right_rank - 1usize]
                            };
                            if !crate::core::ordering::binary_predicate::<Item, Less>(
                                Expr::$method($( $slot, )+ offsets, right_index as usize),
                                Expr::$method($( $slot, )+ offsets, left_index as usize),
                            ) {
                                low.store(left_rank + 1usize);
                            } else {
                                high.store(left_rank);
                            }
                        } else {
                            high.store(left_rank);
                        }
                    }

                    let left_rank = low.read();
                    let right_rank = rank - left_rank;
                    let selected = if source_a.read() != 0u32 {
                        if left_rank < left_len {
                            let left_index = indices_a[base + left_rank];
                            if right_rank >= right_len {
                                left_index
                            } else {
                                let right_index = indices_a[right_start + right_rank];
                                if !crate::core::ordering::binary_predicate::<Item, Less>(
                                    Expr::$method($( $slot, )+ offsets, right_index as usize),
                                    Expr::$method($( $slot, )+ offsets, left_index as usize),
                                ) {
                                    left_index
                                } else {
                                    right_index
                                }
                            }
                        } else {
                            indices_a[right_start + right_rank]
                        }
                    } else if left_rank < left_len {
                        let left_index = indices_b[base + left_rank];
                        if right_rank >= right_len {
                            left_index
                        } else {
                            let right_index = indices_b[right_start + right_rank];
                            if !crate::core::ordering::binary_predicate::<Item, Less>(
                                Expr::$method($( $slot, )+ offsets, right_index as usize),
                                Expr::$method($( $slot, )+ offsets, left_index as usize),
                            ) {
                                left_index
                            } else {
                                right_index
                            }
                        }
                    } else {
                        indices_b[right_start + right_rank]
                    };

                    if source_a.read() != 0u32 {
                        indices_b[local] = selected;
                    } else {
                        indices_a[local] = selected;
                    }
                }
                sync_cube();
                source_a.store(1u32 - source_a.read());
                width.store(width.read() * 2usize);
            }

            if local < tile_len {
                output[tile_start + local] = if source_a.read() != 0u32 {
                    indices_a[local]
                } else {
                    indices_b[local]
                };
            }
        }
    };
}

define_block_sort_permutation_kernel!(block_sort_permutation_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);

macro_rules! define_merge_permutation_kernel {
    ($name:ident,$eval:ident,$method:ident; $( $leaf:ident:$slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            input: &[u32],
            output: &mut [u32],
            len: &[u32],
            width: &[u32],
        ) {
            let logical_len = len[0] as usize;
            let run_width = width[0] as usize;
            let pair_width = if logical_len == 0usize {
                1usize
            } else if logical_len <= run_width
                || run_width > logical_len - run_width
            {
                logical_len
            } else {
                run_width * 2usize
            };
            let merge_tile_items = SORT_MERGE_SIZE * SORT_MERGE_ITEMS;
            let tiles_per_pair = pair_width.div_ceil(merge_tile_items);
            let pair = (CUBE_POS as usize) / tiles_per_pair;
            let tile = (CUBE_POS as usize) % tiles_per_pair;
            let base = pair * pair_width;

            if base < logical_len {
                let pair_len = if logical_len - base < pair_width {
                    logical_len - base
                } else {
                    pair_width
                };
                let left_len = if pair_len < run_width { pair_len } else { run_width };
                let right_start = base + left_len;
                let right_len = pair_len - left_len;
                let tile_rank_start = tile * merge_tile_items;

                if tile_rank_start < pair_len {
                    let tile_rank_end = if tile_rank_start + merge_tile_items < pair_len {
                        tile_rank_start + merge_tile_items
                    } else {
                        pair_len
                    };

                    let mut partition = Shared::<[u32]>::new_slice(5usize);
                    if UNIT_POS == 0u32 {
                        let pair_ordered = right_len == 0usize
                            || !crate::core::ordering::binary_predicate::<Item, Less>(
                                Expr::$method(
                                    $( $slot, )+
                                    offsets,
                                    input[right_start] as usize,
                                ),
                                Expr::$method(
                                    $( $slot, )+
                                    offsets,
                                    input[right_start - 1usize] as usize,
                                ),
                            );
                        partition[4] = if pair_ordered { 1u32 } else { 0u32 };
                        if !pair_ordered {
                            let begin_low_init = if tile_rank_start > right_len {
                                tile_rank_start - right_len
                            } else {
                                0usize
                            };
                            let begin_high_init = if tile_rank_start < left_len {
                                tile_rank_start
                            } else {
                                left_len
                            };
                            let begin_low = RuntimeCell::<usize>::new(begin_low_init);
                            let begin_high = RuntimeCell::<usize>::new(begin_high_init);
                            while begin_low.read() < begin_high.read() {
                                let left_rank = (begin_low.read() + begin_high.read()) / 2usize;
                                let right_rank = tile_rank_start - left_rank;
                                if left_rank < left_len && right_rank > 0usize {
                                    let left_index = input[base + left_rank];
                                    let right_index = input[right_start + right_rank - 1usize];
                                    if !crate::core::ordering::binary_predicate::<Item, Less>(
                                        Expr::$method($( $slot, )+ offsets, right_index as usize),
                                        Expr::$method($( $slot, )+ offsets, left_index as usize),
                                    ) {
                                        begin_low.store(left_rank + 1usize);
                                    } else {
                                        begin_high.store(left_rank);
                                    }
                                } else {
                                    begin_high.store(left_rank);
                                }
                            }

                            let left_begin = begin_low.read();
                            partition[0] = left_begin as u32;
                            partition[1] = (tile_rank_start - left_begin) as u32;
                        }
                    }
                    if UNIT_POS == 1u32 {
                        let pair_ordered = right_len == 0usize
                            || !crate::core::ordering::binary_predicate::<Item, Less>(
                                Expr::$method(
                                    $( $slot, )+
                                    offsets,
                                    input[right_start] as usize,
                                ),
                                Expr::$method(
                                    $( $slot, )+
                                    offsets,
                                    input[right_start - 1usize] as usize,
                                ),
                            );
                        if !pair_ordered {
                            let end_low_init = if tile_rank_end > right_len {
                                tile_rank_end - right_len
                            } else {
                                0usize
                            };
                            let end_high_init = if tile_rank_end < left_len {
                                tile_rank_end
                            } else {
                                left_len
                            };
                            let end_low = RuntimeCell::<usize>::new(end_low_init);
                            let end_high = RuntimeCell::<usize>::new(end_high_init);
                            while end_low.read() < end_high.read() {
                                let left_rank = (end_low.read() + end_high.read()) / 2usize;
                                let right_rank = tile_rank_end - left_rank;
                                if left_rank < left_len && right_rank > 0usize {
                                    let left_index = input[base + left_rank];
                                    let right_index = input[right_start + right_rank - 1usize];
                                    if !crate::core::ordering::binary_predicate::<Item, Less>(
                                        Expr::$method($( $slot, )+ offsets, right_index as usize),
                                        Expr::$method($( $slot, )+ offsets, left_index as usize),
                                    ) {
                                        end_low.store(left_rank + 1usize);
                                    } else {
                                        end_high.store(left_rank);
                                    }
                                } else {
                                    end_high.store(left_rank);
                                }
                            }

                            let left_end = end_low.read();
                            partition[2] = left_end as u32;
                            partition[3] = (tile_rank_end - left_end) as u32;
                        }
                    }
                    sync_cube();

                    if partition[4] != 0u32 {
                        let copy_position = RuntimeCell::<usize>::new(UNIT_POS as usize);
                        let tile_len = tile_rank_end - tile_rank_start;
                        while copy_position.read() < tile_len {
                            let position = base + tile_rank_start + copy_position.read();
                            output[position] = input[position];
                            copy_position.store(copy_position.read() + SORT_MERGE_SIZE);
                        }
                    } else {
                        let left_begin = partition[0] as usize;
                        let right_begin = partition[1] as usize;
                        let left_count = partition[2] as usize - left_begin;
                        let right_count = partition[3] as usize - right_begin;
                        let tile_len = left_count + right_count;
                        let mut shared_indices =
                            Shared::<[u32]>::new_slice(SORT_MERGE_SIZE * SORT_MERGE_ITEMS);
                        let load_pos = RuntimeCell::<usize>::new(UNIT_POS as usize);
                        while load_pos.read() < tile_len {
                            let source = if load_pos.read() < left_count {
                                base + left_begin + load_pos.read()
                            } else {
                                right_start + right_begin + load_pos.read() - left_count
                            };
                            shared_indices[load_pos.read()] = input[source];
                            load_pos.store(load_pos.read() + SORT_MERGE_SIZE);
                        }
                        sync_cube();

                        let local_start = (UNIT_POS as usize) * SORT_MERGE_ITEMS;
                        if local_start < tile_len {
                            let local_end = if local_start + SORT_MERGE_ITEMS < tile_len {
                                local_start + SORT_MERGE_ITEMS
                            } else {
                                tile_len
                            };
                            let local_low_init = if local_start > right_count {
                                local_start - right_count
                            } else {
                                0usize
                            };
                            let local_high_init = if local_start < left_count {
                                local_start
                            } else {
                                left_count
                            };
                            let local_low = RuntimeCell::<usize>::new(local_low_init);
                            let local_high = RuntimeCell::<usize>::new(local_high_init);
                            while local_low.read() < local_high.read() {
                                let left_rank = (local_low.read() + local_high.read()) / 2usize;
                                let right_rank = local_start - left_rank;
                                if left_rank < left_count && right_rank > 0usize {
                                    let left_index = shared_indices[left_rank];
                                    let right_index =
                                        shared_indices[left_count + right_rank - 1usize];
                                    if !crate::core::ordering::binary_predicate::<Item, Less>(
                                        Expr::$method($( $slot, )+ offsets, right_index as usize),
                                        Expr::$method($( $slot, )+ offsets, left_index as usize),
                                    ) {
                                        local_low.store(left_rank + 1usize);
                                    } else {
                                        local_high.store(left_rank);
                                    }
                                } else {
                                    local_high.store(left_rank);
                                }
                            }

                            let left_rank = RuntimeCell::<usize>::new(local_low.read());
                            let right_rank =
                                RuntimeCell::<usize>::new(local_start - local_low.read());
                            let cursor = RuntimeCell::<usize>::new(local_start);
                            while cursor.read() < local_end {
                                let out_index = base + tile_rank_start + cursor.read();
                                if left_rank.read() < left_count {
                                    let left_index = shared_indices[left_rank.read()];
                                    if right_rank.read() >= right_count {
                                        output[out_index] = left_index;
                                        left_rank.store(left_rank.read() + 1usize);
                                    } else {
                                        let right_index =
                                            shared_indices[left_count + right_rank.read()];
                                        if !crate::core::ordering::binary_predicate::<Item, Less>(
                                            Expr::$method(
                                                $( $slot, )+
                                                offsets,
                                                right_index as usize,
                                            ),
                                            Expr::$method(
                                                $( $slot, )+
                                                offsets,
                                                left_index as usize,
                                            ),
                                        ) {
                                            output[out_index] = left_index;
                                            left_rank.store(left_rank.read() + 1usize);
                                        } else {
                                            output[out_index] = right_index;
                                            right_rank.store(right_rank.read() + 1usize);
                                        }
                                    }
                                } else {
                                    output[out_index] =
                                        shared_indices[left_count + right_rank.read()];
                                    right_rank.store(right_rank.read() + 1usize);
                                }
                                cursor.store(cursor.read() + 1usize);
                            }
                        }
                    }
                }
            }
        }
    };
}

define_merge_permutation_kernel!(merge_permutation_a13,Eval13,eval13; L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12);

trait SortControlDispatch<R, Input, Item, Slots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error>;
}

macro_rules! impl_sort_control_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident; [$( $leaf:ident:$index:literal ),+],$env:ty) => {
        impl<R, Input, Item, Less, $( $leaf ),+>
            SortControlDispatch<R, Input, Item, $env, Less> for Dispatch<$arity, crate::core::storage::S1>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn run(exec: &Executor<R>, input: &Input) -> Result<DeviceVec<R, u32>, Error> {
                let capacity = input.physical_len()?;
                let extent = input.logical_extent()?;
                let mut current = exec.alloc_row::<u32>(capacity);
                current.set_logical_extent(extent.clone());
                if capacity == 0 {
                    return Ok(current);
                }
                let len_handle = extent.materialize(exec)?;
                let reads = Bindings::read(exec, input)?;
                let offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                unsafe {
                    block_sort_permutation_a13::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        Input::DeviceExpr,
                        Less,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(capacity.div_ceil(SORT_BLOCK_ITEMS))?,
                        CubeDim::new_1d(SORT_BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets.clone(), reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(current.handle.clone(), current.capacity()),
                    );
                }
                if capacity <= SORT_BLOCK_ITEMS {
                    return Ok(current);
                }
                let mut next = exec.alloc_row::<u32>(capacity);
                next.set_logical_extent(extent);
                let mut width = SORT_BLOCK_ITEMS;
                while width < capacity {
                    let width_u32 = u32::try_from(width)
                        .map_err(|_| Error::LengthTooLarge { len: width })?;
                    let width_handle = exec.client().create_from_slice(u32::as_bytes(&[width_u32]));
                    let pair_width = width.saturating_mul(2).min(capacity);
                    let pairs = capacity.div_ceil(pair_width);
                    let tiles_per_pair =
                        pair_width.div_ceil(SORT_MERGE_SIZE * SORT_MERGE_ITEMS);
                    let count = cube_count_1d(pairs.saturating_mul(tiles_per_pair).max(1))?;
                    unsafe {
                        $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Less, R>(
                            exec.client(),
                            count.clone(),
                            CubeDim::new_1d(SORT_MERGE_SIZE as u32),
                            $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                            BufferArg::from_raw_parts(offsets.clone(), reads.offsets.len()),
                            BufferArg::from_raw_parts(current.handle.clone(), current.capacity()),
                            BufferArg::from_raw_parts(next.handle.clone(), next.capacity()),
                            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                            BufferArg::from_raw_parts(width_handle, 1),
                        );
                    }
                    core::mem::swap(&mut current, &mut next);
                    width = width.saturating_mul(2);
                }
                Ok(current)
            }
        }
    };
}

impl_sort_control_dispatch!(A13,Eval13,merge_permutation_a13; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12],Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>);

/// Generates a stable ordering permutation from one read expression.
#[doc(hidden)]
pub trait SortControlInput<R: Runtime, Less>: ReadExpression + Sized {
    fn sort_control(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> SortControlInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Less: BinaryPredicateOp<Input::Item>,
    Dispatch<A13, crate::core::storage::S1>:
        SortControlDispatch<R, Input, Input::Item, KernelReadSlots<Input::Slots>, Less>,
{
    fn sort_control(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::core::storage::S1> as SortControlDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Less,
        >>::run(exec, &self)
    }
}

pub(crate) fn unique_head_flags_ordered<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    order: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, UniqueHead<Equal>>,
{
    input.adjacent_flags_ordered(exec, order)
}

pub(crate) fn sort_control_with<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: SortControlInput<R, Less>,
{
    input.sort_control(exec)
}

pub(crate) trait AdjacentFlagInput<R: Runtime, Op>: ReadExpression + Sized {
    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn adjacent_flags_ordered(
        self,
        exec: &Executor<R>,
        order: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Op> AdjacentFlagInput<R, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: AdjacentFlagOp<Input::Item>,
    Dispatch<A13, crate::core::storage::S1>:
        AdjacentFlagDispatch<R, Input, Input::Item, KernelReadSlots<Input::Slots>, Op>,
{
    fn adjacent_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::core::storage::S1> as AdjacentFlagDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Op,
        >>::run(exec, &self, None)
    }

    fn adjacent_flags_ordered(
        self,
        exec: &Executor<R>,
        order: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        <Dispatch<A13, crate::core::storage::S1> as AdjacentFlagDispatch<
            R,
            Input,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            Op,
        >>::run(exec, &self, Some(order))
    }
}

pub(crate) fn unique_head_flags<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, UniqueHead<Equal>>,
{
    input.adjacent_flags(exec)
}

pub(crate) fn sorted_break_flags<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: AdjacentFlagInput<R, SortedBreak<Less>>,
{
    input.adjacent_flags(exec)
}

/// Internal public-API capability for adjacent pair search.
#[doc(hidden)]
pub trait AdjacentFindInput<R: Runtime, Equal>: ReadExpression + Sized {
    fn first_adjacent_match(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Equal> AdjacentFindInput<R, Equal> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Equal: BinaryPredicateOp<Input::Item>,
    AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::core::arity::A13, crate::core::storage::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>>,
            u32,
            MinU32,
            crate::core::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstAdjacentMatch<Equal>> as LowerReadExpression>::Slots,
            >,
            Storage = DeviceVec<R, u32>,
        >,
{
    fn first_adjacent_match(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            AdjacentIndexedTransform::new(self, FirstAdjacentMatch::<Equal>(PhantomData)),
            exec.to_device(&[u32::MAX]),
            MinU32,
        )
    }
}

/// Finds the first element of the first adjacent pair, or returns a sentinel.
pub(crate) fn adjacent_find<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    _equal: Equal,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: AdjacentFindInput<R, Equal>,
{
    input.first_adjacent_match(exec)
}

/// Internal public-API capability for stable adjacent deduplication.
#[doc(hidden)]
pub trait UniqueInput<R: Runtime, Equal, Output>: ReadExpression + Sized {
    fn unique_into(self, exec: &Executor<R>, output: Output) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Equal, Output> UniqueInput<R, Equal, Output> for Input
where
    R: Runtime,
    Input: ReadExpression
        + Clone
        + AdjacentFlagInput<R, UniqueHead<Equal>>
        + crate::core::selection::CopySelected<R, Output>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    fn unique_into(self, exec: &Executor<R>, output: Output) -> Result<DeviceVec<R, u32>, Error> {
        let flags = unique_head_flags::<R, _, Equal>(exec, self.clone())?;
        let control = crate::core::selection::SelectionControl::from_flags(exec, flags)?;
        self.copy_selected(exec, &control, output)
    }
}

/// Removes consecutive duplicates, keeping the first item in each run.
pub(crate) fn unique<R, Input, Equal, Output>(
    exec: &Executor<R>,
    input: Input,
    _equal: Equal,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: UniqueInput<R, Equal, Output>,
{
    input.unique_into(exec, output)
}

/// Internal public-API capability hiding the concrete adjacent-control
/// dispatch from the function signature.
#[doc(hidden)]
pub trait SortedInput<R: Runtime, Less>: ReadExpression + Sized {
    fn first_sorted_break(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> SortedInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Less: BinaryPredicateOp<Input::Item>,
    AdjacentIndexedTransform<Input, FirstSortedBreak<Less>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::core::arity::A13, crate::core::storage::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstSortedBreak<Less>>,
            u32,
            MinU32,
            crate::core::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstSortedBreak<Less>> as LowerReadExpression>::Slots,
            >,
            Storage = DeviceVec<R, u32>,
        >,
{
    fn first_sorted_break(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            AdjacentIndexedTransform::new(self, FirstSortedBreak::<Less>(PhantomData)),
            exec.to_device(&[u32::MAX]),
            MinU32,
        )
    }
}

/// Returns the first index at which the input ceases to be sorted, or a sentinel.
pub(crate) fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: SortedInput<R, Less>,
{
    input.first_sorted_break(exec)
}

/// Returns the first sorted break, or a sentinel when the input is sorted.
pub(crate) fn is_sorted<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Input: SortedInput<R, Less>,
{
    input.first_sorted_break(exec)
}

/// Internal public-API capability for extremum queries.
#[doc(hidden)]
pub trait ExtremumInput<R: Runtime, Less>: ReadExpression + Sized {
    fn first_minimum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn last_minimum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn first_maximum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Less> ExtremumInput<R, Less> for Input
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0> + Clone,
    Less: BinaryPredicateOp<Input::Item>,
    Dispatch<crate::core::arity::A13, crate::core::storage::S1>: ArgReduceDispatch<
            R,
            Input,
            ArgMinFirst<Less>,
            crate::core::read::KernelReadSlots<Input::Slots>,
        > + ArgReduceDispatch<
            R,
            Input,
            ArgMinLast<Less>,
            crate::core::read::KernelReadSlots<Input::Slots>,
        > + ArgReduceDispatch<
            R,
            Input,
            ArgMaxFirst<Less>,
            crate::core::read::KernelReadSlots<Input::Slots>,
        >,
{
    fn first_minimum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMinFirst::<Less>(PhantomData))
    }

    fn last_minimum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMinLast::<Less>(PhantomData))
    }

    fn first_maximum(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        arg_reduce(exec, self.clone(), ArgMaxFirst::<Less>(PhantomData))
    }
}

/// Returns the first minimum element index.
pub(crate) fn min_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    input.first_minimum(exec)
}

/// Returns the first maximum element index.
pub(crate) fn max_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    input.first_maximum(exec)
}

/// Returns the last minimum and first maximum indices.
pub(crate) fn minmax_element<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    _less: Less,
) -> Result<(MVec<R, MIndex>, MVec<R, MIndex>), Error>
where
    R: Runtime,
    Input: ExtremumInput<R, Less>,
{
    let min = input.last_minimum(exec)?;
    let max = input.first_maximum(exec)?;
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MFlag;
    use crate::core::allocation::RowStorage;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct LexicographicLess;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for LexicographicLess {
        fn apply(lhs: Seven, rhs: Seven) -> MFlag {
            crate::flag::from_bool(lhs.0 < rhs.0)
        }
    }

    struct EqualSeven;

    #[cubecl::cube]
    impl BinaryPredicateOp<Seven> for EqualSeven {
        fn apply(lhs: Seven, rhs: Seven) -> MFlag {
            crate::flag::from_bool(
                lhs.0 == rhs.0
                    && lhs.1 == rhs.1
                    && lhs.2 == rhs.2
                    && lhs.3 == rhs.3
                    && lhs.4 == rhs.4
                    && lhs.5 == rhs.5
                    && lhs.6 == rhs.6,
            )
        }
    }

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> MFlag {
            crate::flag::from_bool(lhs < rhs)
        }
    }

    #[test]
    fn sorted_queries_use_padded_fixed_evaluator_with_flat_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let first = exec.to_device(&[1_u32, 1, 2, 2]);
        let second = exec.to_device(&[0_u32, 1, 0, 1]);
        let zeros: Vec<_> = (0..5).map(|_| exec.to_device(&[0_u32; 4])).collect();
        let make_input = || {
            let seven = Zip::new(
                first.column(),
                Zip::new(
                    second.column(),
                    Zip::new(
                        zeros[0].column(),
                        Zip::new(
                            zeros[1].column(),
                            Zip::new(
                                zeros[2].column(),
                                Zip::new(zeros[3].column(), zeros[4].column()),
                            ),
                        ),
                    ),
                ),
            );
            Permute::new(seven, Counting::new(0, 4))
        };
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(
                &exec,
                &is_sorted(&exec, make_input(), LexicographicLess).unwrap(),
            )
            .unwrap(),
            u32::MAX
        );

        let bad_first = exec.to_device(&[1_u32, 2, 1, 3]);
        let bad_input = Permute::new(
            Zip::new(
                bad_first.column(),
                Zip::new(
                    second.column(),
                    Zip::new(
                        zeros[0].column(),
                        Zip::new(
                            zeros[1].column(),
                            Zip::new(
                                zeros[2].column(),
                                Zip::new(zeros[3].column(), zeros[4].column()),
                            ),
                        ),
                    ),
                ),
            ),
            Counting::new(0, 4),
        );
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(
                &exec,
                &is_sorted_until(&exec, bad_input, LexicographicLess).unwrap(),
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn generated_sort_control_stages_fit_the_binding_budget() {
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Block = block_sort_permutation_a13::BlockSortPermutationA13<
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            ScalarExpr,
            LessU32,
            WgpuRuntime,
        >;
        type Merge = merge_permutation_a13::MergePermutationA13<
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            ScalarExpr,
            LessU32,
            WgpuRuntime,
        >;

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let settings = KernelSettings::new(
            CubeDim::new_1d(SORT_BLOCK_SIZE).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        let handle = exec.client().empty(core::mem::size_of::<u32>());
        let arg = unsafe {
            <[u32] as LaunchArg>::register(BufferArg::from_raw_parts(handle, 1), &mut launcher)
        };

        let block = Block::new(
            settings.clone(),
            exec.client().clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
        );
        crate::core::launch::assert_binding_budget("block sort control", &block);

        let merge = Merge::new(
            settings,
            exec.client().clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg,
        );
        crate::core::launch::assert_binding_budget("merge sort control", &merge);
    }

    #[test]
    fn sort_control_then_gather_orders_eight_read_slots_into_storage7() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 513usize;
        let keys: Vec<u32> = (0..len).map(|index| (index as u32 * 37) % 23).collect();
        let rows: Vec<u32> = (0..len as u32).collect();
        let key_column = exec.to_device(&keys);
        let row_column = exec.to_device(&rows);
        let payloads: Vec<_> = (2_u32..7)
            .map(|column| {
                let values: Vec<u32> = rows.iter().map(|row| row + column * 1_000).collect();
                exec.to_device(&values)
            })
            .collect();
        let seven = Zip::new(
            key_column.column(),
            Zip::new(
                row_column.column(),
                Zip::new(
                    payloads[0].column(),
                    Zip::new(
                        payloads[1].column(),
                        Zip::new(
                            payloads[2].column(),
                            Zip::new(payloads[3].column(), payloads[4].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, len));
        let ordering = sort_control_with(&exec, input.clone(), LexicographicLess).unwrap();
        let output = exec.alloc_row::<Seven>(len);
        crate::core::indexed::gather_direct(&exec, input, ordering.column(), output.write())
            .unwrap();

        let (keys, rows, payload2, _, _, _, payload6) = crate::MStorage::into_columns(output);
        let sorted_keys = exec.to_host(&keys).unwrap();
        let sorted_rows = exec.to_host(&rows).unwrap();
        for index in 1..len {
            assert!(sorted_keys[index - 1] <= sorted_keys[index]);
            if sorted_keys[index - 1] == sorted_keys[index] {
                assert!(sorted_rows[index - 1] < sorted_rows[index]);
            }
        }
        let sorted_payload2 = exec.to_host(&payload2).unwrap();
        let sorted_payload6 = exec.to_host(&payload6).unwrap();
        for index in 0..len {
            assert_eq!(sorted_payload2[index], sorted_rows[index] + 2_000);
            assert_eq!(sorted_payload6[index], sorted_rows[index] + 6_000);
        }
    }

    #[test]
    fn adjacent_find_and_unique_separate_eight_slot_control_from_storage7_apply() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[1_u32, 1, 2, 2, 3]);
        let copies: Vec<_> = (0..6)
            .map(|_| exec.to_device(&[9_u32, 9, 8, 8, 7]))
            .collect();
        let make_input = || {
            Permute::new(
                Zip::new(
                    values.column(),
                    Zip::new(
                        copies[0].column(),
                        Zip::new(
                            copies[1].column(),
                            Zip::new(
                                copies[2].column(),
                                Zip::new(
                                    copies[3].column(),
                                    Zip::new(copies[4].column(), copies[5].column()),
                                ),
                            ),
                        ),
                    ),
                ),
                Counting::new(0, 5),
            )
        };

        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(
                &exec,
                &adjacent_find(&exec, make_input(), EqualSeven).unwrap(),
            )
            .unwrap(),
            0
        );
        let output = exec.alloc_row::<Seven>(5);
        let count = unique(&exec, make_input(), EqualSeven, output.write()).unwrap();
        let count = exec.to_host(&count).unwrap()[0];
        let (first, _, _, _, _, _, _) = crate::MStorage::into_columns(output);
        assert_eq!(count, 3);
        assert_eq!(exec.to_host(&first.slice(..count)).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn extremum_queries_preserve_oracle_tie_breaking() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[2_u32, 1, 3, 1, 3]);
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(
                &exec,
                &min_element(&exec, input.column(), LessU32).unwrap(),
            )
            .unwrap(),
            1
        );
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(
                &exec,
                &max_element(&exec, input.column(), LessU32).unwrap(),
            )
            .unwrap(),
            2
        );
        let (minimum, maximum) = minmax_element(&exec, input.column(), LessU32).unwrap();
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(&exec, &minimum).unwrap(),
            3
        );
        assert_eq!(
            crate::api::value::read::<WgpuRuntime, MIndex>(&exec, &maximum).unwrap(),
            2
        );
    }
}
