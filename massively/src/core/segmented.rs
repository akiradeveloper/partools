//! Segmented scan and reduction schedules over flat-row storage.

use cubecl::prelude::*;

use crate::core::allocation::{CopyStorage, RowStorage, ScratchStorage};
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::op::ReductionOp;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
};
use crate::core::read::{Env0, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead};
use crate::core::storage::{
    Decompose, LoadMutPadded12, LoadPadded12, MutableLeaves, PlaneShuffleLeaves, Recompose,
    SharedLeaves, SharedLeavesExpand, StorePadded12, StorePadded12Expand,
};
use crate::{DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;

type FixedStorage<R, Item> = <Item as ScratchStorage<R>>::Storage;

mod kernels;
use kernels::*;

pub(crate) fn segmented_inclusive<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    flags: &DeviceVec<R, u32>,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval13<
            Item,
            <Input::Slots as PaddedReadSlots>::L0,
            <Input::Slots as PaddedReadSlots>::L1,
            <Input::Slots as PaddedReadSlots>::L2,
            <Input::Slots as PaddedReadSlots>::L3,
            <Input::Slots as PaddedReadSlots>::L4,
            <Input::Slots as PaddedReadSlots>::L5,
            <Input::Slots as PaddedReadSlots>::L6,
            <Input::Slots as PaddedReadSlots>::L7,
            <Input::Slots as PaddedReadSlots>::L8,
            <Input::Slots as PaddedReadSlots>::L9,
            <Input::Slots as PaddedReadSlots>::L10,
            <Input::Slots as PaddedReadSlots>::L11,
            <Input::Slots as PaddedReadSlots>::L12,
        >,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    let len = input.physical_len()?;
    let output_len = output.physical_len()?;
    if flags.capacity() != len || output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: flags.capacity().min(output_len),
        });
    }
    if len == 0 {
        return Ok(());
    }
    let extent = input.logical_extent()?.zipped(&flags.logical_extent())?;
    let len_handle = extent.materialize(exec)?;
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let block_extent = extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?;

    let control_len = len.checked_add(1).ok_or(Error::LengthTooLarge { len })?;
    let control = exec.alloc_column::<u32>(control_len);
    let continuation = exec.alloc_column::<u32>(control_len);
    let mut local_values = Item::alloc_scratch(exec, len);
    local_values.set_logical_extent(extent.clone());
    let mut block_sums = Item::alloc_scratch(exec, blocks);
    block_sums.set_logical_extent(block_extent.clone());
    let mut block_flags = exec.alloc_column::<u32>(blocks);
    block_flags.set_logical_extent(block_extent);

    let input_bindings = Bindings::read(exec, input)?;
    let local_write = local_values.write();
    let local_output_bindings = Bindings::write(exec, &local_write)?;
    let input_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&input_bindings.offsets));
    let local_output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&local_output_bindings.offsets));

    unsafe {
        prepare_segment_control::launch_unchecked::<R>(
            exec.client(),
            crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(control.handle.clone(), control_len),
        );
        segmented_local_scan_a13::launch_unchecked::<
            Item,
            <Input::Slots as PaddedReadSlots>::L0,
            <Input::Slots as PaddedReadSlots>::L1,
            <Input::Slots as PaddedReadSlots>::L2,
            <Input::Slots as PaddedReadSlots>::L3,
            <Input::Slots as PaddedReadSlots>::L4,
            <Input::Slots as PaddedReadSlots>::L5,
            <Input::Slots as PaddedReadSlots>::L6,
            <Input::Slots as PaddedReadSlots>::L7,
            <Input::Slots as PaddedReadSlots>::L8,
            <Input::Slots as PaddedReadSlots>::L9,
            <Input::Slots as PaddedReadSlots>::L10,
            <Input::Slots as PaddedReadSlots>::L11,
            <Input::Slots as PaddedReadSlots>::L12,
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            Input::DeviceExpr,
            Item::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            crate::core::launch::cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input_bindings.slots[0].0.clone(), input_bindings.slots[0].1),
            BufferArg::from_raw_parts(input_bindings.slots[1].0.clone(), input_bindings.slots[1].1),
            BufferArg::from_raw_parts(input_bindings.slots[2].0.clone(), input_bindings.slots[2].1),
            BufferArg::from_raw_parts(input_bindings.slots[3].0.clone(), input_bindings.slots[3].1),
            BufferArg::from_raw_parts(input_bindings.slots[4].0.clone(), input_bindings.slots[4].1),
            BufferArg::from_raw_parts(input_bindings.slots[5].0.clone(), input_bindings.slots[5].1),
            BufferArg::from_raw_parts(input_bindings.slots[6].0.clone(), input_bindings.slots[6].1),
            BufferArg::from_raw_parts(input_bindings.slots[7].0.clone(), input_bindings.slots[7].1),
            BufferArg::from_raw_parts(input_bindings.slots[8].0.clone(), input_bindings.slots[8].1),
            BufferArg::from_raw_parts(input_bindings.slots[9].0.clone(), input_bindings.slots[9].1),
            BufferArg::from_raw_parts(
                input_bindings.slots[10].0.clone(),
                input_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                input_bindings.slots[11].0.clone(),
                input_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(
                input_bindings.slots[12].0.clone(),
                input_bindings.slots[12].1,
            ),
            BufferArg::from_raw_parts(control.handle.clone(), control_len),
            BufferArg::from_raw_parts(input_offsets, input_bindings.offsets.len()),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[0].0.clone(),
                local_output_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[1].0.clone(),
                local_output_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[2].0.clone(),
                local_output_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[3].0.clone(),
                local_output_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[4].0.clone(),
                local_output_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[5].0.clone(),
                local_output_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[6].0.clone(),
                local_output_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[7].0.clone(),
                local_output_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[8].0.clone(),
                local_output_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[9].0.clone(),
                local_output_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[10].0.clone(),
                local_output_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                local_output_bindings.slots[11].0.clone(),
                local_output_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(local_output_offsets, local_output_bindings.offsets.len()),
            crate::core::launch::plane_count_bound(exec, BLOCK_SIZE),
        );
        segment_continuation_kernel::launch_unchecked::<R>(
            exec.client(),
            crate::core::launch::cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(control.handle.clone(), control_len),
            BufferArg::from_raw_parts(continuation.handle.clone(), control_len),
        );
    }

    let local_read = local_values.read();
    let local_input_bindings = Bindings::read(exec, &local_read)?;
    let sum_write = block_sums.write();
    let sum_bindings = Bindings::write(exec, &sum_write)?;
    let local_input_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&local_input_bindings.offsets));
    let sum_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&sum_bindings.offsets));
    unsafe {
        summarize_segment_blocks_padded12::launch_unchecked::<
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            R,
        >(
            exec.client(),
            crate::core::launch::cube_count_1d(blocks)?,
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[0].0.clone(),
                local_input_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[1].0.clone(),
                local_input_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[2].0.clone(),
                local_input_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[3].0.clone(),
                local_input_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[4].0.clone(),
                local_input_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[5].0.clone(),
                local_input_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[6].0.clone(),
                local_input_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[7].0.clone(),
                local_input_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[8].0.clone(),
                local_input_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[9].0.clone(),
                local_input_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[10].0.clone(),
                local_input_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                local_input_bindings.slots[11].0.clone(),
                local_input_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(local_input_offsets, local_input_bindings.offsets.len()),
            BufferArg::from_raw_parts(continuation.handle.clone(), control_len),
            BufferArg::from_raw_parts(sum_bindings.slots[0].0.clone(), sum_bindings.slots[0].1),
            BufferArg::from_raw_parts(sum_bindings.slots[1].0.clone(), sum_bindings.slots[1].1),
            BufferArg::from_raw_parts(sum_bindings.slots[2].0.clone(), sum_bindings.slots[2].1),
            BufferArg::from_raw_parts(sum_bindings.slots[3].0.clone(), sum_bindings.slots[3].1),
            BufferArg::from_raw_parts(sum_bindings.slots[4].0.clone(), sum_bindings.slots[4].1),
            BufferArg::from_raw_parts(sum_bindings.slots[5].0.clone(), sum_bindings.slots[5].1),
            BufferArg::from_raw_parts(sum_bindings.slots[6].0.clone(), sum_bindings.slots[6].1),
            BufferArg::from_raw_parts(sum_bindings.slots[7].0.clone(), sum_bindings.slots[7].1),
            BufferArg::from_raw_parts(sum_bindings.slots[8].0.clone(), sum_bindings.slots[8].1),
            BufferArg::from_raw_parts(sum_bindings.slots[9].0.clone(), sum_bindings.slots[9].1),
            BufferArg::from_raw_parts(sum_bindings.slots[10].0.clone(), sum_bindings.slots[10].1),
            BufferArg::from_raw_parts(sum_bindings.slots[11].0.clone(), sum_bindings.slots[11].1),
            BufferArg::from_raw_parts(sum_offsets, sum_bindings.offsets.len()),
            BufferArg::from_raw_parts(block_flags.handle.clone(), blocks),
        );
    }

    if blocks > 1 {
        let mut prefixes = Item::alloc_scratch(exec, blocks);
        prefixes.set_logical_extent(block_sums.logical_extent());
        let block_read = crate::core::read::FixedRead::new(block_sums.read());
        let prefix_write = prefixes.write();
        segmented_inclusive::<R, _, _, Item, Op>(exec, &block_read, &block_flags, &prefix_write)?;
        let prefix_read = prefixes.read();
        let prefix_bindings = Bindings::read(exec, &prefix_read)?;
        let local_write = local_values.write();
        let local_bindings = Bindings::write(exec, &local_write)?;
        let prefix_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&prefix_bindings.offsets));
        let local_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&local_bindings.offsets));
        unsafe {
            segmented_prefix_padded12::launch_unchecked::<
                Item,
                <Item::StorageLeaves as StorePadded12>::O0,
                <Item::StorageLeaves as StorePadded12>::O1,
                <Item::StorageLeaves as StorePadded12>::O2,
                <Item::StorageLeaves as StorePadded12>::O3,
                <Item::StorageLeaves as StorePadded12>::O4,
                <Item::StorageLeaves as StorePadded12>::O5,
                <Item::StorageLeaves as StorePadded12>::O6,
                <Item::StorageLeaves as StorePadded12>::O7,
                <Item::StorageLeaves as StorePadded12>::O8,
                <Item::StorageLeaves as StorePadded12>::O9,
                <Item::StorageLeaves as StorePadded12>::O10,
                <Item::StorageLeaves as StorePadded12>::O11,
                Item::StorageLeaves,
                Item::DeviceLayout,
                Op,
                R,
            >(
                exec.client(),
                crate::core::launch::cube_count_1d(blocks)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[0].0.clone(),
                    prefix_bindings.slots[0].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[1].0.clone(),
                    prefix_bindings.slots[1].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[2].0.clone(),
                    prefix_bindings.slots[2].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[3].0.clone(),
                    prefix_bindings.slots[3].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[4].0.clone(),
                    prefix_bindings.slots[4].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[5].0.clone(),
                    prefix_bindings.slots[5].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[6].0.clone(),
                    prefix_bindings.slots[6].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[7].0.clone(),
                    prefix_bindings.slots[7].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[8].0.clone(),
                    prefix_bindings.slots[8].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[9].0.clone(),
                    prefix_bindings.slots[9].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[10].0.clone(),
                    prefix_bindings.slots[10].1,
                ),
                BufferArg::from_raw_parts(
                    prefix_bindings.slots[11].0.clone(),
                    prefix_bindings.slots[11].1,
                ),
                BufferArg::from_raw_parts(continuation.handle.clone(), control_len),
                BufferArg::from_raw_parts(prefix_offsets, prefix_bindings.offsets.len()),
                BufferArg::from_raw_parts(local_offsets, local_bindings.offsets.len()),
                BufferArg::from_raw_parts(
                    local_bindings.slots[0].0.clone(),
                    local_bindings.slots[0].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[1].0.clone(),
                    local_bindings.slots[1].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[2].0.clone(),
                    local_bindings.slots[2].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[3].0.clone(),
                    local_bindings.slots[3].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[4].0.clone(),
                    local_bindings.slots[4].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[5].0.clone(),
                    local_bindings.slots[5].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[6].0.clone(),
                    local_bindings.slots[6].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[7].0.clone(),
                    local_bindings.slots[7].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[8].0.clone(),
                    local_bindings.slots[8].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[9].0.clone(),
                    local_bindings.slots[9].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[10].0.clone(),
                    local_bindings.slots[10].1,
                ),
                BufferArg::from_raw_parts(
                    local_bindings.slots[11].0.clone(),
                    local_bindings.slots[11].1,
                ),
            );
        }
    }

    crate::core::transform::materialize_fixed(exec, &local_values.read(), output)
}
pub(crate) fn segmented_exclusive<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    flags: &DeviceVec<R, u32>,
    init: FixedStorage<R, Item>,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval13<
            Item,
            <Input::Slots as PaddedReadSlots>::L0,
            <Input::Slots as PaddedReadSlots>::L1,
            <Input::Slots as PaddedReadSlots>::L2,
            <Input::Slots as PaddedReadSlots>::L3,
            <Input::Slots as PaddedReadSlots>::L4,
            <Input::Slots as PaddedReadSlots>::L5,
            <Input::Slots as PaddedReadSlots>::L6,
            <Input::Slots as PaddedReadSlots>::L7,
            <Input::Slots as PaddedReadSlots>::L8,
            <Input::Slots as PaddedReadSlots>::L9,
            <Input::Slots as PaddedReadSlots>::L10,
            <Input::Slots as PaddedReadSlots>::L11,
            <Input::Slots as PaddedReadSlots>::L12,
        >,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    let len = input.physical_len()?;
    if flags.capacity() != len || output.physical_len()? != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: flags.capacity().min(output.physical_len()?),
        });
    }
    if len == 0 {
        return Ok(());
    }
    if init.len()? != 1 {
        return Err(Error::LengthMismatch {
            left: init.len()?,
            right: 1,
        });
    }
    let extent = input.logical_extent()?.zipped(&flags.logical_extent())?;
    let mut inclusive = Item::alloc_scratch(exec, len);
    inclusive.set_logical_extent(extent.clone());
    let inclusive_write = inclusive.write();
    segmented_inclusive::<R, _, _, Item, Op>(exec, input, flags, &inclusive_write)?;
    let len_handle = extent.materialize(exec)?;
    let seeded_len = len.checked_add(1).ok_or(Error::LengthTooLarge { len })?;
    let mut seeded = Item::alloc_scratch(exec, seeded_len);
    init.copy_storage(exec, seeded.slice_mut(..1))?;
    inclusive.copy_storage(exec, seeded.slice_mut(1..))?;
    seeded.set_logical_extent(crate::core::extent::LogicalExtent::add(
        exec,
        &crate::core::extent::LogicalExtent::fixed(1),
        &extent,
        seeded_len,
    )?);
    let seeded_read = seeded.read();
    let seeded_bindings = Bindings::read(exec, &seeded_read)?;
    let output_bindings = Bindings::write(exec, output)?;
    let seeded_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&seeded_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    unsafe {
        segmented_exclusive_padded12::launch_unchecked::<
            Item,
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            Item::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[0].0.clone(),
                seeded_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[1].0.clone(),
                seeded_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[2].0.clone(),
                seeded_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[3].0.clone(),
                seeded_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[4].0.clone(),
                seeded_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[5].0.clone(),
                seeded_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[6].0.clone(),
                seeded_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[7].0.clone(),
                seeded_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[8].0.clone(),
                seeded_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[9].0.clone(),
                seeded_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[10].0.clone(),
                seeded_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[11].0.clone(),
                seeded_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(seeded_offsets, seeded_bindings.offsets.len()),
            BufferArg::from_raw_parts(output_offsets, output_bindings.offsets.len()),
            BufferArg::from_raw_parts(
                output_bindings.slots[0].0.clone(),
                output_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[1].0.clone(),
                output_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[2].0.clone(),
                output_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[3].0.clone(),
                output_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[4].0.clone(),
                output_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[5].0.clone(),
                output_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[6].0.clone(),
                output_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[7].0.clone(),
                output_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[8].0.clone(),
                output_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[9].0.clone(),
                output_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[10].0.clone(),
                output_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[11].0.clone(),
                output_bindings.slots[11].1,
            ),
        );
    }
    Ok(())
}

fn launch_reduce_selected<R, Item, Op, Output>(
    exec: &Executor<R>,
    inclusive: &FixedStorage<R, Item>,
    init: &FixedStorage<R, Item>,
    positions: &DeviceVec<R, u32>,
    count: &DeviceVec<R, u32>,
    source_extent: &crate::core::extent::LogicalExtent,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    // The caller may provide one output slot per possible key rather than one
    // per input item (segment reductions commonly do this).  The selected
    // count is device-resident, so cap the dispatch by the physical output
    // bound just as the previous selected-copy path did.
    let len = positions.capacity().min(output.physical_len()?);
    if len == 0 {
        return Ok(());
    }
    let source_len = source_extent.materialize(exec)?;
    let inclusive_len = inclusive.len()?;
    let seeded_len = inclusive_len
        .checked_add(1)
        .ok_or(Error::LengthTooLarge { len: inclusive_len })?;
    let mut seeded = Item::alloc_scratch(exec, seeded_len);
    init.copy_storage(exec, seeded.slice_mut(..1))?;
    inclusive.copy_storage(exec, seeded.slice_mut(1..))?;
    seeded.set_logical_extent(crate::core::extent::LogicalExtent::add(
        exec,
        &crate::core::extent::LogicalExtent::fixed(1),
        source_extent,
        seeded_len,
    )?);
    let resolved_len = len;
    let resolved = exec.alloc_column::<u32>(resolved_len);
    let seeded_read = seeded.read();
    let seeded_bindings = Bindings::read(exec, &seeded_read)?;
    let output_bindings = Bindings::write(exec, output)?;
    let seeded_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&seeded_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    unsafe {
        resolve_segment_tails::launch_unchecked::<R>(
            exec.client(),
            crate::core::launch::cube_count_1d(positions.capacity().div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
            BufferArg::from_raw_parts(count.handle.clone(), 1),
            BufferArg::from_raw_parts(source_len.handle.clone(), 1),
            BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
        );
        segmented_reduce_selected_padded12::launch_unchecked::<
            Item,
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            Item::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[0].0.clone(),
                seeded_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[1].0.clone(),
                seeded_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[2].0.clone(),
                seeded_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[3].0.clone(),
                seeded_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[4].0.clone(),
                seeded_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[5].0.clone(),
                seeded_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[6].0.clone(),
                seeded_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[7].0.clone(),
                seeded_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[8].0.clone(),
                seeded_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[9].0.clone(),
                seeded_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[10].0.clone(),
                seeded_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                seeded_bindings.slots[11].0.clone(),
                seeded_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
            BufferArg::from_raw_parts(count.handle.clone(), 1),
            BufferArg::from_raw_parts(seeded_offsets, seeded_bindings.offsets.len()),
            BufferArg::from_raw_parts(output_offsets, output_bindings.offsets.len()),
            BufferArg::from_raw_parts(
                output_bindings.slots[0].0.clone(),
                output_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[1].0.clone(),
                output_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[2].0.clone(),
                output_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[3].0.clone(),
                output_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[4].0.clone(),
                output_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[5].0.clone(),
                output_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[6].0.clone(),
                output_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[7].0.clone(),
                output_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[8].0.clone(),
                output_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[9].0.clone(),
                output_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[10].0.clone(),
                output_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[11].0.clone(),
                output_bindings.slots[11].1,
            ),
        );
    }
    Ok(())
}

pub(crate) fn segmented_reduce<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    heads: &DeviceVec<R, u32>,
    positions: &DeviceVec<R, u32>,
    count: &DeviceVec<R, u32>,
    init: FixedStorage<R, Item>,
    _op: Op,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval13<
            Item,
            <Input::Slots as PaddedReadSlots>::L0,
            <Input::Slots as PaddedReadSlots>::L1,
            <Input::Slots as PaddedReadSlots>::L2,
            <Input::Slots as PaddedReadSlots>::L3,
            <Input::Slots as PaddedReadSlots>::L4,
            <Input::Slots as PaddedReadSlots>::L5,
            <Input::Slots as PaddedReadSlots>::L6,
            <Input::Slots as PaddedReadSlots>::L7,
            <Input::Slots as PaddedReadSlots>::L8,
            <Input::Slots as PaddedReadSlots>::L9,
            <Input::Slots as PaddedReadSlots>::L10,
            <Input::Slots as PaddedReadSlots>::L11,
            <Input::Slots as PaddedReadSlots>::L12,
        >,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    let len = input.physical_len()?;
    if heads.capacity() != len || positions.capacity() != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: heads.capacity().min(positions.capacity()),
        });
    }
    let extent = input
        .logical_extent()?
        .zipped(&heads.logical_extent())?
        .zipped(&positions.logical_extent())?;
    if len == 0 {
        return Ok(());
    }
    if init.len()? != 1 {
        return Err(Error::LengthMismatch {
            left: init.len()?,
            right: 1,
        });
    }
    let mut inclusive = Item::alloc_scratch(exec, len);
    inclusive.set_logical_extent(extent.clone());
    let inclusive_write = inclusive.write();
    segmented_inclusive::<R, _, _, Item, Op>(exec, input, heads, &inclusive_write)?;
    launch_reduce_selected::<R, Item, Op, _>(
        exec, &inclusive, &init, positions, count, &extent, output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::StorageLayout;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    #[test]
    fn segment_continuation_is_derived_without_mutating_head_control() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 260usize;
        let mut heads = vec![0u32; len + 1usize];
        heads[2] = 1u32;
        heads[255] = 2u32;
        heads[258] = 4u32;
        heads[len] = len as u32;
        let heads = exec.to_device(&heads);
        let continuation = exec.alloc_column::<u32>(len + 1usize);

        unsafe {
            segment_continuation_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize)).unwrap(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(heads.handle.clone(), len + 1usize),
                BufferArg::from_raw_parts(continuation.handle.clone(), len + 1usize),
            );
        }

        let actual = exec.to_host(&continuation).unwrap();
        assert_eq!(&actual[..2], &[0, 0]);
        assert!(actual[2..255].iter().all(|&value| value == 1));
        assert_eq!(actual[255], 3);
        assert_eq!(&actual[256..258], &[0, 0]);
        assert_eq!(&actual[258..260], &[4, 4]);
        assert_eq!(actual[len], len as u32);
        assert_eq!(exec.to_host(&heads).unwrap()[255], 2u32);
    }

    #[test]
    fn generated_segmented_stages_fit_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type LocalKernel = segmented_local_scan_a13::SegmentedLocalScanA13<
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
            ScalarLeaves,
            ScalarExpr,
            ScalarLayout,
            Sum,
            WgpuRuntime,
        >;
        type SummaryKernel = summarize_segment_blocks_padded12::SummarizeSegmentBlocksPadded12<
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
            ScalarLeaves,
            WgpuRuntime,
        >;
        type ExclusiveKernel = segmented_exclusive_padded12::SegmentedExclusivePadded12<
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
            ScalarLeaves,
            ScalarLayout,
            Sum,
            WgpuRuntime,
        >;

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let settings = KernelSettings::new(
            CubeDim::new_1d(BLOCK_SIZE).into(),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        let handle = exec.client().empty(core::mem::size_of::<u32>());
        let arg = unsafe {
            <[u32] as LaunchArg>::register(BufferArg::from_raw_parts(handle, 1), &mut launcher)
        };

        let local = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            LocalKernel,
            settings.clone(),
            exec.client().clone(),
            arg.clone(),
            crate::core::launch::plane_count_bound(&exec, BLOCK_SIZE)
        );
        crate::core::launch::assert_binding_budget("segmented local scan", &local);
        let summary = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            SummaryKernel,
            settings.clone(),
            exec.client().clone(),
            arg.clone()
        );
        crate::core::launch::assert_binding_budget("segmented block summary", &summary);
        let exclusive = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            ExclusiveKernel,
            settings,
            exec.client().clone(),
            arg
        );
        crate::core::launch::assert_binding_budget("segmented exclusive apply", &exclusive);
    }

    #[test]
    fn segmented_algorithms_preserve_boundaries_across_blocks() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 600usize;
        let heads = [0usize, 17, 255, 256, 300, 511];
        let values: Vec<u32> = (0..len).map(|index| index as u32 % 7 + 1).collect();
        let mut flags = vec![0u32; len];
        for &head in &heads {
            flags[head] = 1;
        }

        let input = exec.to_device(&values);
        let flags_device = exec.to_device(&flags);
        let inclusive_output = exec.to_device(&vec![0u32; len]);
        segmented_inclusive::<WgpuRuntime, _, _, u32, Sum>(
            &exec,
            &input.column(),
            &flags_device,
            &inclusive_output.slice_mut(..),
        )
        .unwrap();

        let mut expected_inclusive = Vec::with_capacity(len);
        let mut running = 0u32;
        for (index, &value) in values.iter().enumerate() {
            if flags[index] != 0 {
                running = value;
            } else {
                running += value;
            }
            expected_inclusive.push(running);
        }
        assert_eq!(exec.to_host(&inclusive_output).unwrap(), expected_inclusive);

        let initial = 10u32;
        let exclusive_output = exec.to_device(&vec![0u32; len]);
        segmented_exclusive::<WgpuRuntime, _, _, u32, Sum>(
            &exec,
            &input.column(),
            &flags_device,
            exec.to_device(&[initial]),
            &exclusive_output.slice_mut(..),
        )
        .unwrap();
        let expected_exclusive: Vec<_> = (0..len)
            .map(|index| {
                if index == 0 || flags[index] != 0 {
                    initial
                } else {
                    initial + expected_inclusive[index - 1]
                }
            })
            .collect();
        assert_eq!(exec.to_host(&exclusive_output).unwrap(), expected_exclusive);

        let mut rank = 0u32;
        let positions: Vec<_> = flags
            .iter()
            .map(|&head| {
                rank += u32::from(head != 0);
                rank
            })
            .collect();
        let positions = exec.to_device(&positions);
        let count = exec.to_device(&[heads.len() as u32]);
        let reduced_output = exec.to_device(&vec![0u32; heads.len()]);
        segmented_reduce::<WgpuRuntime, _, _, u32, Sum>(
            &exec,
            &input.column(),
            &flags_device,
            &positions,
            &count,
            exec.to_device(&[initial]),
            Sum,
            &reduced_output.slice_mut(..),
        )
        .unwrap();
        let expected_reduced: Vec<_> = heads
            .iter()
            .enumerate()
            .map(|(rank, &head)| {
                let tail = heads.get(rank + 1).copied().unwrap_or(len);
                initial + values[head..tail].iter().sum::<u32>()
            })
            .collect();
        assert_eq!(exec.to_host(&reduced_output).unwrap(), expected_reduced);
    }
}
