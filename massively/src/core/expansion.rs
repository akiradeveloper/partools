//! Variable-cardinality output generation through the fixed row ABI.

use cubecl::prelude::*;

use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
};
use crate::core::read::{Env0, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead};
use crate::core::storage::{Decompose, StorageLayout, StorePadded12, StorePadded12Expand};
use crate::op::ExpandOp;
use crate::{DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;
// Each expanded output stores `[input_index, local_index]`.
const RESOLVED_EXPANSION_WORDS: usize = 2;

/// Turns prefix offsets and owners into the canonical expansion mapping.
#[cubecl::cube(launch_unchecked, explicit_define)]
fn resolve_expansion(element_offsets: &[u32], owners: &[u32], resolved: &mut [u32]) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < owners.len() {
        let input_index = (owners[output_index] - 1u32) as usize;
        let base = output_index * RESOLVED_EXPANSION_WORDS;
        resolved[base] = input_index as u32;
        resolved[base + 1usize] = output_index as u32 - element_offsets[input_index];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn generate_a13<
    InputItem: CubeType + Send + Sync + 'static,
    OutputItem: CubeType + Send + Sync + 'static,
    L0: CubePrimitive + cubecl::frontend::Scalar,
    L1: CubePrimitive + cubecl::frontend::Scalar,
    L2: CubePrimitive + cubecl::frontend::Scalar,
    L3: CubePrimitive + cubecl::frontend::Scalar,
    L4: CubePrimitive + cubecl::frontend::Scalar,
    L5: CubePrimitive + cubecl::frontend::Scalar,
    L6: CubePrimitive + cubecl::frontend::Scalar,
    L7: CubePrimitive + cubecl::frontend::Scalar,
    L8: CubePrimitive + cubecl::frontend::Scalar,
    L9: CubePrimitive + cubecl::frontend::Scalar,
    L10: CubePrimitive + cubecl::frontend::Scalar,
    L11: CubePrimitive + cubecl::frontend::Scalar,
    L12: CubePrimitive + cubecl::frontend::Scalar,
    O0: CubePrimitive + cubecl::frontend::Scalar,
    O1: CubePrimitive + cubecl::frontend::Scalar,
    O2: CubePrimitive + cubecl::frontend::Scalar,
    O3: CubePrimitive + cubecl::frontend::Scalar,
    O4: CubePrimitive + cubecl::frontend::Scalar,
    O5: CubePrimitive + cubecl::frontend::Scalar,
    O6: CubePrimitive + cubecl::frontend::Scalar,
    O7: CubePrimitive + cubecl::frontend::Scalar,
    O8: CubePrimitive + cubecl::frontend::Scalar,
    O9: CubePrimitive + cubecl::frontend::Scalar,
    O10: CubePrimitive + cubecl::frontend::Scalar,
    O11: CubePrimitive + cubecl::frontend::Scalar,
    Leaves: CubeType
        + Send
        + Sync
        + 'static
        + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        >,
    Expr: Eval13<InputItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<OutputItem, Leaves = Leaves>,
    Op: ExpandOp<InputItem, Output = OutputItem>,
>(
    slot0: &[L0],
    slot1: &[L1],
    slot2: &[L2],
    slot3: &[L3],
    slot4: &[L4],
    slot5: &[L5],
    slot6: &[L6],
    slot7: &[L7],
    slot8: &[L8],
    slot9: &[L9],
    slot10: &[L10],
    slot11: &[L11],
    slot12: &[L12],
    read_offsets: &[u32],
    resolved: &[u32],
    out0: &mut [O0],
    out1: &mut [O1],
    out2: &mut [O2],
    out3: &mut [O3],
    out4: &mut [O4],
    out5: &mut [O5],
    out6: &mut [O6],
    out7: &mut [O7],
    out8: &mut [O8],
    out9: &mut [O9],
    out10: &mut [O10],
    out11: &mut [O11],
    write_offsets: &[u32],
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < resolved.len() / RESOLVED_EXPANSION_WORDS {
        let base = output_index * RESOLVED_EXPANSION_WORDS;
        let input_index = resolved[base] as usize;
        let local_index = resolved[base + 1usize];
        let input = Expr::eval13(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot7,
            slot8,
            slot9,
            slot10,
            slot11,
            slot12,
            read_offsets,
            input_index,
        );
        Layout::decompose(Op::generate(input, local_index)).store_padded(
            out0,
            out1,
            out2,
            out3,
            out4,
            out5,
            out6,
            out7,
            out8,
            out9,
            out10,
            out11,
            write_offsets,
            output_index,
        );
    }
}

/// Generates one output row for every expanded output position.
pub(crate) fn generate<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: &Input,
    element_offsets: &DeviceVec<R, u32>,
    owners: &DeviceVec<R, u32>,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Op: ExpandOp<Input::Item, Output = Output::Item>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
{
    let input_len = input.physical_len()?;
    let expected_offsets = input_len
        .checked_add(1)
        .ok_or(Error::LengthTooLarge { len: input_len })?;
    if element_offsets.capacity() != expected_offsets {
        return Err(Error::LengthMismatch {
            left: element_offsets.capacity(),
            right: expected_offsets,
        });
    }

    let output_len = output.physical_len()?;
    if owners.capacity() != output_len {
        return Err(Error::LengthMismatch {
            left: owners.capacity(),
            right: output_len,
        });
    }
    if output_len == 0 {
        return Ok(());
    }

    let reads = Bindings::read(exec, input)?;
    let writes = Bindings::write(exec, output)?;
    let read_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&reads.offsets));
    let write_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&writes.offsets));
    let cubes = crate::core::launch::cube_count_1d(output_len.div_ceil(BLOCK_SIZE as usize))?;
    let resolved_len = output_len
        .checked_mul(RESOLVED_EXPANSION_WORDS)
        .ok_or(Error::LengthTooLarge { len: output_len })?;
    let resolved = exec.alloc_column::<u32>(resolved_len);

    unsafe {
        resolve_expansion::launch_unchecked::<R>(
            exec.client(),
            cubes.clone(),
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(element_offsets.handle.clone(), element_offsets.capacity()),
            BufferArg::from_raw_parts(owners.handle.clone(), owners.capacity()),
            BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
        );
        generate_a13::launch_unchecked::<
            Input::Item,
            Output::Item,
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
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
            <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
            <Output::Item as StorageLayout>::StorageLeaves,
            Input::DeviceExpr,
            <Output::Item as StorageLayout>::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            cubes,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(reads.slots[0].0.clone(), reads.slots[0].1),
            BufferArg::from_raw_parts(reads.slots[1].0.clone(), reads.slots[1].1),
            BufferArg::from_raw_parts(reads.slots[2].0.clone(), reads.slots[2].1),
            BufferArg::from_raw_parts(reads.slots[3].0.clone(), reads.slots[3].1),
            BufferArg::from_raw_parts(reads.slots[4].0.clone(), reads.slots[4].1),
            BufferArg::from_raw_parts(reads.slots[5].0.clone(), reads.slots[5].1),
            BufferArg::from_raw_parts(reads.slots[6].0.clone(), reads.slots[6].1),
            BufferArg::from_raw_parts(reads.slots[7].0.clone(), reads.slots[7].1),
            BufferArg::from_raw_parts(reads.slots[8].0.clone(), reads.slots[8].1),
            BufferArg::from_raw_parts(reads.slots[9].0.clone(), reads.slots[9].1),
            BufferArg::from_raw_parts(reads.slots[10].0.clone(), reads.slots[10].1),
            BufferArg::from_raw_parts(reads.slots[11].0.clone(), reads.slots[11].1),
            BufferArg::from_raw_parts(reads.slots[12].0.clone(), reads.slots[12].1),
            BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
            BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
            BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
            BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
            BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
            BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
            BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
            BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
            BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
            BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
            BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
            BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
            BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
            BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
            BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Repeat;

    #[cubecl::cube]
    impl ExpandOp<u32> for Repeat {
        type Output = u32;

        fn count(input: u32) -> u32 {
            input
        }

        fn generate(input: u32, local_index: u32) -> u32 {
            input + local_index
        }
    }

    #[test]
    fn generated_expansion_apply_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Kernel = generate_a13::GenerateA13<
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
            ScalarLeaves,
            ScalarExpr,
            ScalarLayout,
            Repeat,
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
        let kernel = crate::core::launch::kernel_with_max_explicit_storage_bindings!(
            Kernel,
            settings,
            exec.client().clone(),
            arg
        );
        crate::core::launch::assert_binding_budget("expansion apply", &kernel);
    }
}
