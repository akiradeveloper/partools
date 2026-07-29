//! Stable two-range merge control and arity-independent payload application.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::core::allocation::{RowStorage, ScratchStorage};
use crate::core::arity::A13;
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13At;
use crate::core::op::BinaryPredicateOp;
use crate::core::output::{LowerOutputExpression, OutputExpression, StageOutput};
use crate::core::read::{Env0, Env13, LowerReadExpression, ReadExpression, StageRead};
use crate::core::storage::{
    Recompose, SharedLeaves, StorageLayout, StorePadded12, StorePadded12Expand,
};
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor};

const MERGE_SIZE: u32 = 64;
const MERGE_ITEMS: usize = 4;
const MERGE_TILE: usize = MERGE_SIZE as usize * MERGE_ITEMS;
const MERGE_CONTROL_WORDS: usize = 3;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn prepare_merge_control(
    left_length: &[u32],
    right_length: &[u32],
    parameters: &[u32],
    control: &mut [u32],
) {
    if ABSOLUTE_POS == 0usize {
        let metadata = control.len() - MERGE_CONTROL_WORDS;
        control[metadata] = left_length[0];
        control[metadata + 1usize] = right_length[0];
        control[metadata + 2usize] = parameters[0];
    }
}

macro_rules! define_merge_control_kernel {
    ($name:ident; [$( $left_leaf:ident:$left_slot:ident:$right_leaf:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $left_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            $( $right_leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            Left: Eval13At<Item, $( $left_leaf ),+>,
            Right: Eval13At<Item, $( $right_leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $left_slot: &[$left_leaf], )+
            $( $right_slot: &[$right_leaf], )+
            offsets: &[u32],
            control: &mut [u32],
        ) {
            let metadata = control.len() - MERGE_CONTROL_WORDS;
            let left_len = control[metadata] as usize;
            let right_len = control[metadata + 1usize] as usize;
            let right_base = control[metadata + 2usize];
            let total = left_len + right_len;
            let tile_start = (CUBE_POS as usize) * MERGE_TILE;
            if tile_start < total {
                let tile_end = if tile_start + MERGE_TILE < total {
                    tile_start + MERGE_TILE
                } else {
                    total
                };
                let mut partition = Shared::<[u32]>::new_slice(4usize);
                if UNIT_POS == 0u32 {
                    let begin_low_init = if tile_start > right_len {
                        tile_start - right_len
                    } else {
                        0usize
                    };
                    let begin_high_init = if tile_start < left_len {
                        tile_start
                    } else {
                        left_len
                    };
                    let begin_low = RuntimeCell::<usize>::new(begin_low_init);
                    let begin_high = RuntimeCell::<usize>::new(begin_high_init);
                    while begin_low.read() < begin_high.read() {
                        let left_rank = (begin_low.read() + begin_high.read()) / 2usize;
                        let right_rank = tile_start - left_rank;
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::core::ordering::binary_predicate::<Item, Less>(
                                Right::eval13_at(
                                    $( $right_slot, )+
                                    offsets,
                                    13usize,
                                    right_rank - 1usize,
                                ),
                                Left::eval13_at($( $left_slot, )+ offsets, 0usize, left_rank),
                            )
                        {
                            begin_low.store(left_rank + 1usize);
                        } else {
                            begin_high.store(left_rank);
                        }
                    }

                    let end_low_init = if tile_end > right_len {
                        tile_end - right_len
                    } else {
                        0usize
                    };
                    let end_high_init = if tile_end < left_len {
                        tile_end
                    } else {
                        left_len
                    };
                    let end_low = RuntimeCell::<usize>::new(end_low_init);
                    let end_high = RuntimeCell::<usize>::new(end_high_init);
                    while end_low.read() < end_high.read() {
                        let left_rank = (end_low.read() + end_high.read()) / 2usize;
                        let right_rank = tile_end - left_rank;
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::core::ordering::binary_predicate::<Item, Less>(
                                Right::eval13_at(
                                    $( $right_slot, )+
                                    offsets,
                                    13usize,
                                    right_rank - 1usize,
                                ),
                                Left::eval13_at($( $left_slot, )+ offsets, 0usize, left_rank),
                            )
                        {
                            end_low.store(left_rank + 1usize);
                        } else {
                            end_high.store(left_rank);
                        }
                    }

                    let left_begin = begin_low.read();
                    let right_begin = tile_start - left_begin;
                    partition[0] = left_begin as u32;
                    partition[1] = right_begin as u32;
                    partition[2] = (end_low.read() - left_begin) as u32;
                    partition[3] = ((tile_end - end_low.read()) - right_begin) as u32;
                }
                sync_cube();

                let left_begin = partition[0] as usize;
                let right_begin = partition[1] as usize;
                let left_count = partition[2] as usize;
                let right_count = partition[3] as usize;
                let tile_len = left_count + right_count;
                let local_start = UNIT_POS as usize * MERGE_ITEMS;
                if local_start < tile_len {
                    let local_end = if local_start + MERGE_ITEMS < tile_len {
                        local_start + MERGE_ITEMS
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
                        if left_rank < left_count
                            && right_rank > 0usize
                            && !crate::core::ordering::binary_predicate::<Item, Less>(
                                Right::eval13_at(
                                    $( $right_slot, )+
                                    offsets,
                                    13usize,
                                    right_begin + right_rank - 1usize,
                                ),
                                Left::eval13_at(
                                    $( $left_slot, )+
                                    offsets,
                                    0usize,
                                    left_begin + left_rank,
                                ),
                            )
                        {
                            local_low.store(left_rank + 1usize);
                        } else {
                            local_high.store(left_rank);
                        }
                    }

                    let left_rank = RuntimeCell::<usize>::new(local_low.read());
                    let right_rank = RuntimeCell::<usize>::new(local_start - local_low.read());
                    let cursor = RuntimeCell::<usize>::new(local_start);
                    while cursor.read() < local_end {
                        let take_left = left_rank.read() < left_count
                            && (right_rank.read() >= right_count
                                || !crate::core::ordering::binary_predicate::<Item, Less>(
                                    Right::eval13_at(
                                        $( $right_slot, )+
                                        offsets,
                                        13usize,
                                        right_begin + right_rank.read(),
                                    ),
                                    Left::eval13_at(
                                        $( $left_slot, )+
                                        offsets,
                                        0usize,
                                        left_begin + left_rank.read(),
                                    ),
                                ));
                        let encoded = if take_left {
                            let encoded = (left_begin + left_rank.read()) as u32;
                            left_rank.store(left_rank.read() + 1usize);
                            encoded
                        } else {
                            let encoded = right_base + (right_begin + right_rank.read()) as u32;
                            right_rank.store(right_rank.read() + 1usize);
                            encoded
                        };
                        control[tile_start + cursor.read()] = encoded;
                        cursor.store(cursor.read() + 1usize);
                    }
                }
            }
        }
    };
}

define_merge_control_kernel!(merge_control_a13; [LL0:left0:RL0:right0,LL1:left1:RL1:right1,LL2:left2:RL2:right2,LL3:left3:RL3:right3,LL4:left4:RL4:right4,LL5:left5:RL5:right5,LL6:left6:RL6:right6,LL7:left7:RL7:right7,LL8:left8:RL8:right8,LL9:left9:RL9:right9,LL10:left10:RL10:right10,LL11:left11:RL11:right11,LL12:left12:RL12:right12]);

pub(crate) trait MergeDirectInput<R: Runtime, Right, Output, Less>: ReadExpression {
    fn merge_direct(
        &self,
        exec: &Executor<R>,
        right: &Right,
        right_positions: Option<&DeviceVec<R, u32>>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Left, Right, Output, Less> MergeDirectInput<R, Right, Output, Less> for Left
where
    R: Runtime,
    Left: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0> + Clone,
    Right: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0> + Clone,
    Less: BinaryPredicateOp<Output::Item>,
    Output::Item: ScratchStorage<R>,
    <Output::Item as StorageLayout>::StorageLeaves: SharedLeaves + StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    <Output::Item as StorageLayout>::DeviceLayout:
        Recompose<Output::Item, Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: crate::core::output::PaddedOutputSlots<
            Leaves = <Output::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn merge_direct(
        &self,
        exec: &Executor<R>,
        right: &Right,
        right_positions: Option<&DeviceVec<R, u32>>,
        less: Less,
        output: Output,
    ) -> Result<(), Error> {
        if let Some(positions) = right_positions {
            let mut selected =
                <Output::Item as ScratchStorage<R>>::alloc_scratch(exec, positions.capacity());
            selected.set_logical_extent(positions.logical_extent());
            crate::core::indexed::PermutationCopyInput::permutation_copy(
                right.clone(),
                exec,
                positions.column(),
                None,
                selected.write(),
            )?;
            let control = merge_control_fixed(
                exec,
                crate::core::read::FixedRead::new(self.clone()),
                crate::core::read::FixedRead::new(selected.read()),
                less,
            )?;
            apply_fixed(
                exec,
                crate::core::read::FixedRead::new(self.clone()),
                crate::core::read::FixedRead::new(selected.read()),
                &control,
                output,
            )
        } else {
            let control = merge_control_fixed(
                exec,
                crate::core::read::FixedRead::new(self.clone()),
                crate::core::read::FixedRead::new(right.clone()),
                less,
            )?;
            apply_fixed(
                exec,
                crate::core::read::FixedRead::new(self.clone()),
                crate::core::read::FixedRead::new(right.clone()),
                &control,
                output,
            )
        }
    }
}
pub(crate) fn merge_direct<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeDirectInput<R, Right, Output, Less>,
{
    left.merge_direct(exec, &right, None, less, output)
}

pub(crate) fn merge_direct_selected_right<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    right_positions: &DeviceVec<R, u32>,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeDirectInput<R, Right, Output, Less>,
{
    left.merge_direct(exec, &right, Some(right_positions), less, output)
}

pub(crate) trait MergeApplyInput<R: Runtime, Right, Output>: ReadExpression {
    fn merge_apply(
        &self,
        exec: &Executor<R>,
        right: &Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Left, Right, Output> MergeApplyInput<R, Right, Output> for Left
where
    R: Runtime,
    Left: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output::Item: ScratchStorage<R>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: crate::core::output::PaddedOutputSlots<
            Leaves = <Output::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn merge_apply(
        &self,
        exec: &Executor<R>,
        right: &Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error> {
        let operation_len = control.permutation.capacity();
        let active_extent = control.permutation.logical_extent();
        if output.physical_len()? < active_extent.upper_bound() {
            return Err(Error::OutputTooShort {
                input: active_extent.upper_bound(),
                output: output.physical_len()?,
            });
        }
        if operation_len == 0 {
            return Ok(());
        }

        let mut combined = <Output::Item as ScratchStorage<R>>::alloc_scratch(exec, operation_len);
        combined.set_logical_extent(active_extent.clone());
        crate::core::transform::materialize_fixed(
            exec,
            self,
            &combined.slice_mut(0..control.left_capacity),
        )?;
        crate::core::transform::materialize_fixed(
            exec,
            right,
            &combined.slice_mut(control.left_capacity..operation_len),
        )?;
        crate::core::indexed::PermutationCopyInput::permutation_copy(
            combined.read(),
            exec,
            control.permutation.column(),
            None,
            output,
        )
    }
}

pub(crate) struct MergeDispatch<Storage>(PhantomData<fn() -> Storage>);

pub(crate) trait MergeControlDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error>;
}

macro_rules! impl_merge_control_dispatch {
    ($storage:ty,$arity:ty,$kernel:ident; [$( $left_leaf:ident:$left_index:literal:$right_leaf:ident:$right_index:literal ),+]) => {
        impl<R, Left, Right, Item, Less, $( $left_leaf, )+ $( $right_leaf ),+>
            MergeControlDispatch<
                R,
                Left,
                Right,
                Item,
                Env13<$( $left_leaf ),+>,
                Env13<$( $right_leaf ),+>,
                Less,
            >
            for MergeDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $left_leaf: MStorageElement, )+
            $( $right_leaf: MStorageElement, )+
            Left: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $left_leaf ),+>>
                + StageRead<R, Env0>,
            Right: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $right_leaf ),+>>
                + StageRead<R, Env0>,
            Left::DeviceExpr: Eval13At<Item, $( $left_leaf ),+>,
            Right::DeviceExpr: Eval13At<Item, $( $right_leaf ),+>,
        {
            fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error> {
                let left_capacity = left.physical_len()?;
                let right_capacity = right.physical_len()?;
                let total_capacity = left_capacity.checked_add(right_capacity).ok_or(Error::LengthTooLarge { len: usize::MAX })?;
                let left_extent = left.logical_extent()?;
                let right_extent = right.logical_extent()?;
                let total_extent = crate::core::extent::LogicalExtent::add(
                    exec,
                    &left_extent,
                    &right_extent,
                    total_capacity,
                )?;
                if total_capacity == 0 {
                    let mut permutation = exec.alloc_row::<u32>(0);
                    permutation.set_logical_extent(total_extent);
                    return Ok(MergeControl {
                        permutation,
                        left_capacity,
                        right_capacity,
                        left_extent,
                        right_extent,
                    });
                }
                let left_bindings = Bindings::read(exec, left)?;
                let right_bindings = Bindings::read(exec, right)?;
                let mut combined_offsets = left_bindings.offsets.clone();
                combined_offsets.extend_from_slice(&right_bindings.offsets);
                let offsets = exec.client().create_from_slice(u32::as_bytes(&combined_offsets));
                let left_length = left_extent.materialize(exec)?;
                let right_length = right_extent.materialize(exec)?;
                let right_base = u32::try_from(left_capacity)
                    .map_err(|_| Error::LengthTooLarge { len: left_capacity })?;
                let parameters = exec.client().create_from_slice(u32::as_bytes(&[right_base]));
                let control_len = total_capacity
                    .checked_add(MERGE_CONTROL_WORDS)
                    .ok_or(Error::LengthTooLarge { len: total_capacity })?;
                let control = exec.alloc_column::<u32>(control_len);
                let mut permutation = exec.column_from_handle::<u32>(
                    control.handle.clone(),
                    total_capacity,
                );
                permutation.set_logical_extent(total_extent);
                unsafe {
                    prepare_merge_control::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(left_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(right_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(parameters, 1),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                    $kernel::launch_unchecked::<Item, $( $left_leaf, )+ $( $right_leaf, )+ Left::DeviceExpr, Right::DeviceExpr, Less, R>(
                        exec.client(),
                        crate::core::launch::cube_count_1d(total_capacity.div_ceil(MERGE_TILE))?,
                        CubeDim::new_1d(MERGE_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$left_index].0.clone(), left_bindings.slots[$left_index].1), )+
                        $( BufferArg::from_raw_parts(right_bindings.slots[$right_index].0.clone(), right_bindings.slots[$right_index].1), )+
                        BufferArg::from_raw_parts(offsets, combined_offsets.len()),
                        BufferArg::from_raw_parts(control.handle.clone(), control_len),
                    );
                }
                Ok(MergeControl {
                    permutation,
                    left_capacity,
                    right_capacity,
                    left_extent,
                    right_extent,
                })
            }
        }
    };
}

impl_merge_control_dispatch!(crate::core::storage::S12,A13,merge_control_a13; [LL0:0:RL0:0,LL1:1:RL1:1,LL2:2:RL2:2,LL3:3:RL3:3,LL4:4:RL4:4,LL5:5:RL5:5,LL6:6:RL6:6,LL7:7:RL7:7,LL8:8:RL8:8,LL9:9:RL9:9,LL10:10:RL10:10,LL11:11:RL11:11,LL12:12:RL12:12]);

/// Stable merge permutation over a conceptual `left || right` payload.
#[doc(hidden)]
pub struct MergeControl<R: Runtime> {
    pub(crate) permutation: DeviceVec<R, u32>,
    pub(crate) left_capacity: usize,
    pub(crate) right_capacity: usize,
    pub(crate) left_extent: crate::core::extent::LogicalExtent,
    pub(crate) right_extent: crate::core::extent::LogicalExtent,
}

pub(crate) fn merge_control_fixed<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<MergeControl<R>, Error>
where
    R: Runtime,
    Left: ReadExpression<ReadArity = A13> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item, ReadArity = A13>
        + LowerReadExpression
        + StageRead<R, Env0>,
    MergeDispatch<crate::core::storage::S12>:
        MergeControlDispatch<R, Left, Right, Left::Item, Left::Slots, Right::Slots, Less>,
{
    <MergeDispatch<crate::core::storage::S12> as MergeControlDispatch<
        R,
        Left,
        Right,
        Left::Item,
        Left::Slots,
        Right::Slots,
        Less,
    >>::run(exec, &left, &right)
}

/// Normalizes two payload expressions, then gathers them through a reusable
/// merge permutation.
pub(crate) fn apply_fixed<R, Left, Right, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    control: &MergeControl<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeApplyInput<R, Right, Output> + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item> + StageRead<R, Env0>,
{
    let left_capacity = left.physical_len()?;
    let right_capacity = right.physical_len()?;
    if left_capacity != control.left_capacity || right_capacity != control.right_capacity {
        return Err(Error::LengthMismatch {
            left: left_capacity + right_capacity,
            right: control.left_capacity + control.right_capacity,
        });
    }
    left.logical_extent()?.zipped(&control.left_extent)?;
    right.logical_extent()?.zipped(&control.right_extent)?;
    left.merge_apply(exec, &right, control, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs < rhs)
        }
    }

    #[test]
    fn generated_merge_control_fits_the_binding_budget() {
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Kernel = merge_control_a13::MergeControlA13<
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
            ScalarExpr,
            LessU32,
            WgpuRuntime,
        >;

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let settings = KernelSettings::new(
            CubeDim::new_1d(MERGE_SIZE).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        let handle = exec.client().empty(core::mem::size_of::<u32>());
        let arg = unsafe {
            <[u32] as LaunchArg>::register(BufferArg::from_raw_parts(handle, 1), &mut launcher)
        };
        let kernel = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            Kernel,
            settings,
            exec.client().clone(),
            arg
        );
        crate::core::launch::assert_binding_budget("merge control", &kernel);
    }

    #[test]
    fn merge_is_stable_and_payloads_reuse_one_control() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 2, 5]);
        let right = exec.to_device(&[2_u32, 3, 4]);
        let output = exec.to_device(&[0_u32; 7]);
        merge_direct(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            LessU32,
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);

        let left_values = exec.to_device(&[10_u32, 20, 21, 50]);
        let right_values = exec.to_device(&[200_u32, 300, 400]);
        let out_keys = exec.to_device(&[0_u32; 7]);
        let out_values = exec.to_device(&[0_u32; 7]);
        let control = merge_control_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            LessU32,
        )
        .unwrap();
        apply_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            &control,
            out_keys.slice_mut(..),
        )
        .unwrap();
        apply_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left_values.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right_values.column()),
            &control,
            out_values.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);
        assert_eq!(
            exec.to_host(&out_values).unwrap(),
            vec![10, 20, 21, 200, 300, 400, 50]
        );

        // Keep a binary output in the monomorphization surface as well.
        let pair_out = Zip::new(
            exec.to_device(&[0_u32; 7]).slice_mut(..),
            exec.to_device(&[0_u32; 7]).slice_mut(..),
        );
        let pair_left = Zip::new(left.column(), left_values.column());
        let pair_right = Zip::new(right.column(), right_values.column());
        struct LessPair;
        #[cubecl::cube]
        impl BinaryPredicateOp<(u32, u32)> for LessPair {
            fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> crate::MFlag {
                crate::flag::from_bool(lhs.0 < rhs.0)
            }
        }
        merge_direct(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(pair_left),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(pair_right),
            LessPair,
            pair_out,
        )
        .unwrap();
    }
}
