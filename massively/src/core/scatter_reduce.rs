//! Conflict-free application of already reduced logical scatter proposals.

use cubecl::prelude::*;

use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::read::{Env0, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead};
use crate::core::storage::{
    Decompose, LoadMutPadded12, Recompose, StorageLayout, StorePadded12, StorePadded12Expand,
};
use crate::op::ReductionOp;
use crate::{Error, Executor};

const BLOCK_SIZE: u32 = 256;
// Each resolved proposal stores `[valid, destination]`.
const RESOLVED_SCATTER_WORDS: usize = 2;

/// Resolves the sorted proposal order and lazy destination expression without
/// binding the proposal values or destination storage.
#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn resolve_scatter_a13<
    I0: CubePrimitive + cubecl::frontend::Scalar,
    I1: CubePrimitive + cubecl::frontend::Scalar,
    I2: CubePrimitive + cubecl::frontend::Scalar,
    I3: CubePrimitive + cubecl::frontend::Scalar,
    I4: CubePrimitive + cubecl::frontend::Scalar,
    I5: CubePrimitive + cubecl::frontend::Scalar,
    I6: CubePrimitive + cubecl::frontend::Scalar,
    I7: CubePrimitive + cubecl::frontend::Scalar,
    I8: CubePrimitive + cubecl::frontend::Scalar,
    I9: CubePrimitive + cubecl::frontend::Scalar,
    I10: CubePrimitive + cubecl::frontend::Scalar,
    I11: CubePrimitive + cubecl::frontend::Scalar,
    I12: CubePrimitive + cubecl::frontend::Scalar,
    IndexExpr: Eval13<crate::MIndex, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12>,
>(
    index0: &[I0],
    index1: &[I1],
    index2: &[I2],
    index3: &[I3],
    index4: &[I4],
    index5: &[I5],
    index6: &[I6],
    index7: &[I7],
    index8: &[I8],
    index9: &[I9],
    index10: &[I10],
    index11: &[I11],
    index12: &[I12],
    index_offsets: &[u32],
    positions: &[u32],
    order: &[u32],
    len: &[u32],
    active_len: &[u32],
    resolved: &mut [u32],
) {
    let position = ABSOLUTE_POS as usize;
    if position < resolved.len() / RESOLVED_SCATTER_WORDS {
        let base = position * RESOLVED_SCATTER_WORDS;
        if position < len[0] as usize && position < active_len[0] as usize {
            let index_position = order[positions[position] as usize] as usize;
            resolved[base] = 1u32;
            resolved[base + 1usize] = IndexExpr::eval13(
                index0,
                index1,
                index2,
                index3,
                index4,
                index5,
                index6,
                index7,
                index8,
                index9,
                index10,
                index11,
                index12,
                index_offsets,
                index_position,
            );
        } else {
            resolved[base] = 0u32;
        }
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn scatter_combine_a13<
    Item: CubeType + Send + Sync + 'static,
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
    Leaves: LoadMutPadded12<
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
        > + Send
        + Sync
        + 'static,
    SourceExpr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    source0: &[L0],
    source1: &[L1],
    source2: &[L2],
    source3: &[L3],
    source4: &[L4],
    source5: &[L5],
    source6: &[L6],
    source7: &[L7],
    source8: &[L8],
    source9: &[L9],
    source10: &[L10],
    source11: &[L11],
    source12: &[L12],
    source_offsets: &[u32],
    resolved: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
    output_offsets: &[u32],
) {
    let position = ABSOLUTE_POS as usize;
    if position < resolved.len() / RESOLVED_SCATTER_WORDS {
        let base = position * RESOLVED_SCATTER_WORDS;
        if resolved[base] != 0u32 {
            let destination = resolved[base + 1usize];
            let proposal = SourceExpr::eval13(
                source0,
                source1,
                source2,
                source3,
                source4,
                source5,
                source6,
                source7,
                source8,
                source9,
                source10,
                source11,
                source12,
                source_offsets,
                position,
            );
            let previous = Layout::recompose(Leaves::load_mut_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                destination as usize,
            ));
            Layout::decompose(Op::apply(previous, proposal)).store_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                destination as usize,
            );
        }
    }
}

/// Applies one conflict-free proposal per selected logical destination.
#[doc(hidden)]
pub trait ScatterCombineInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: Indices,
        positions: &crate::DeviceVec<R, u32>,
        order: &crate::DeviceVec<R, u32>,
        active_len: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>;
}

impl<R, Source, Indices, Output> ScatterCombineInput<R, Indices, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Indices: ReadExpression<Item = crate::MIndex> + LowerReadExpression + StageRead<R, Env0>,
    Output: crate::core::facade::KernelOutput<R>
        + crate::core::output::OutputExpression<Item = Source::Item>,
    Source::Item: StorageLayout,
    <Source::Item as StorageLayout>::StorageLeaves: LoadMutPadded12,
    <Source::Item as StorageLayout>::DeviceLayout: Decompose<Source::Item, Leaves = <Source::Item as StorageLayout>::StorageLeaves>
        + Recompose<Source::Item, Leaves = <Source::Item as StorageLayout>::StorageLeaves>,
{
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: Indices,
        positions: &crate::DeviceVec<R, u32>,
        order: &crate::DeviceVec<R, u32>,
        active_len: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>,
    {
        let len = self.physical_len()?;
        if len != positions.capacity() {
            return Err(Error::LengthMismatch {
                left: len,
                right: positions.capacity(),
            });
        }
        if len != order.capacity() {
            return Err(Error::LengthMismatch {
                left: len,
                right: order.capacity(),
            });
        }
        if len == 0 {
            return Ok(());
        }

        let extent = self.logical_extent()?.zipped(&positions.logical_extent())?;
        let source_reads = Bindings::read(exec, &self)?;
        let index_reads = Bindings::read(exec, &indices)?;
        let writes = Bindings::write(exec, &output)?;

        let source_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&source_reads.offsets));
        let index_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&index_reads.offsets));
        let output_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let len_handle = extent.materialize(exec)?;
        let resolved_len = len
            .checked_mul(RESOLVED_SCATTER_WORDS)
            .ok_or(Error::LengthTooLarge { len })?;
        let resolved = exec.alloc_column::<u32>(resolved_len);

        // Resolve destinations independently from applying the row proposals.
        // This keeps both kernels below the fixed-slot binding budget.
        unsafe {
            resolve_scatter_a13::launch_unchecked::<
                <Indices::Slots as PaddedReadSlots>::L0,
                <Indices::Slots as PaddedReadSlots>::L1,
                <Indices::Slots as PaddedReadSlots>::L2,
                <Indices::Slots as PaddedReadSlots>::L3,
                <Indices::Slots as PaddedReadSlots>::L4,
                <Indices::Slots as PaddedReadSlots>::L5,
                <Indices::Slots as PaddedReadSlots>::L6,
                <Indices::Slots as PaddedReadSlots>::L7,
                <Indices::Slots as PaddedReadSlots>::L8,
                <Indices::Slots as PaddedReadSlots>::L9,
                <Indices::Slots as PaddedReadSlots>::L10,
                <Indices::Slots as PaddedReadSlots>::L11,
                <Indices::Slots as PaddedReadSlots>::L12,
                Indices::DeviceExpr,
                R,
            >(
                exec.client(),
                crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(index_reads.slots[0].0.clone(), index_reads.slots[0].1),
                BufferArg::from_raw_parts(index_reads.slots[1].0.clone(), index_reads.slots[1].1),
                BufferArg::from_raw_parts(index_reads.slots[2].0.clone(), index_reads.slots[2].1),
                BufferArg::from_raw_parts(index_reads.slots[3].0.clone(), index_reads.slots[3].1),
                BufferArg::from_raw_parts(index_reads.slots[4].0.clone(), index_reads.slots[4].1),
                BufferArg::from_raw_parts(index_reads.slots[5].0.clone(), index_reads.slots[5].1),
                BufferArg::from_raw_parts(index_reads.slots[6].0.clone(), index_reads.slots[6].1),
                BufferArg::from_raw_parts(index_reads.slots[7].0.clone(), index_reads.slots[7].1),
                BufferArg::from_raw_parts(index_reads.slots[8].0.clone(), index_reads.slots[8].1),
                BufferArg::from_raw_parts(index_reads.slots[9].0.clone(), index_reads.slots[9].1),
                BufferArg::from_raw_parts(index_reads.slots[10].0.clone(), index_reads.slots[10].1),
                BufferArg::from_raw_parts(index_reads.slots[11].0.clone(), index_reads.slots[11].1),
                BufferArg::from_raw_parts(index_reads.slots[12].0.clone(), index_reads.slots[12].1),
                BufferArg::from_raw_parts(index_offsets, index_reads.offsets.len()),
                BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                BufferArg::from_raw_parts(order.handle.clone(), order.capacity()),
                BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                BufferArg::from_raw_parts(active_len.handle.clone(), 1),
                BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
            );
            scatter_combine_a13::launch_unchecked::<
                Source::Item,
                <Source::Slots as PaddedReadSlots>::L0,
                <Source::Slots as PaddedReadSlots>::L1,
                <Source::Slots as PaddedReadSlots>::L2,
                <Source::Slots as PaddedReadSlots>::L3,
                <Source::Slots as PaddedReadSlots>::L4,
                <Source::Slots as PaddedReadSlots>::L5,
                <Source::Slots as PaddedReadSlots>::L6,
                <Source::Slots as PaddedReadSlots>::L7,
                <Source::Slots as PaddedReadSlots>::L8,
                <Source::Slots as PaddedReadSlots>::L9,
                <Source::Slots as PaddedReadSlots>::L10,
                <Source::Slots as PaddedReadSlots>::L11,
                <Source::Slots as PaddedReadSlots>::L12,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
                <Source::Item as StorageLayout>::StorageLeaves,
                Source::DeviceExpr,
                <Source::Item as StorageLayout>::DeviceLayout,
                Op,
                R,
            >(
                exec.client(),
                crate::core::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(source_reads.slots[0].0.clone(), source_reads.slots[0].1),
                BufferArg::from_raw_parts(source_reads.slots[1].0.clone(), source_reads.slots[1].1),
                BufferArg::from_raw_parts(source_reads.slots[2].0.clone(), source_reads.slots[2].1),
                BufferArg::from_raw_parts(source_reads.slots[3].0.clone(), source_reads.slots[3].1),
                BufferArg::from_raw_parts(source_reads.slots[4].0.clone(), source_reads.slots[4].1),
                BufferArg::from_raw_parts(source_reads.slots[5].0.clone(), source_reads.slots[5].1),
                BufferArg::from_raw_parts(source_reads.slots[6].0.clone(), source_reads.slots[6].1),
                BufferArg::from_raw_parts(source_reads.slots[7].0.clone(), source_reads.slots[7].1),
                BufferArg::from_raw_parts(source_reads.slots[8].0.clone(), source_reads.slots[8].1),
                BufferArg::from_raw_parts(source_reads.slots[9].0.clone(), source_reads.slots[9].1),
                BufferArg::from_raw_parts(
                    source_reads.slots[10].0.clone(),
                    source_reads.slots[10].1,
                ),
                BufferArg::from_raw_parts(
                    source_reads.slots[11].0.clone(),
                    source_reads.slots[11].1,
                ),
                BufferArg::from_raw_parts(
                    source_reads.slots[12].0.clone(),
                    source_reads.slots[12].1,
                ),
                BufferArg::from_raw_parts(source_offsets, source_reads.offsets.len()),
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
                BufferArg::from_raw_parts(output_offsets, writes.offsets.len()),
            );
        }
        Ok(())
    }
}

pub(crate) fn apply<R, Source, Indices, Output, Op>(
    exec: &Executor<R>,
    source: Source,
    indices: Indices,
    positions: &crate::DeviceVec<R, u32>,
    order: &crate::DeviceVec<R, u32>,
    active_len: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: ScatterCombineInput<R, Indices, Output>,
    Op: ReductionOp<Source::Item>,
{
    source.scatter_combine::<Op>(exec, indices, positions, order, active_len, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    #[test]
    fn generated_scatter_apply_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Kernel = scatter_combine_a13::ScatterCombineA13<
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
        crate::core::launch::assert_binding_budget("scatter apply", &kernel);
    }
}
