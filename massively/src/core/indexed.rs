//! Indexed copy primitives with independently evaluated value and index inputs.

use cubecl::prelude::*;

use crate::core::bindings::{Bindings, TRANSFER_PARAMETERS_OFFSET, TRANSFER_READ_OFFSET};
use crate::core::eval::{Eval13, Eval13At};
use crate::core::read::{Env0, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead};
use crate::core::storage::{Decompose, StorageLayout, StorePadded12, StorePadded12Expand};
use crate::{Error, Executor};

const BLOCK_SIZE: u32 = 256;
// The first word stores the active entry count.  Gather entries then store
// only `source_position`, because their output position is the entry index;
// scatter entries store `[source_position, output_position]`.  Entries outside
// the active prefix do not need to be initialized.  This keeps the resolver
// and transfer stages independent without carrying redundant per-row fields.
const RESOLVED_INDEX_HEADER_WORDS: usize = 1;
const RESOLVED_GATHER_ENTRY_WORDS: usize = 1;
const RESOLVED_SCATTER_ENTRY_WORDS: usize = 2;

#[derive(Clone, Copy)]
pub(crate) enum IndexSelection<'a, R: Runtime> {
    /// Block routing control shared with selection transfers.
    Routing {
        control: &'a crate::core::selection::SelectionControl<R>,
        retain_output_position: bool,
    },
}

/// Resolves lazy index and selection expressions into one canonical mapping
/// buffer before the value-transfer stage binds its fixed source/output slots.
#[cubecl::cube(launch_unchecked, explicit_define)]
fn resolve_index_a13<
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
    index_slot0: &[I0],
    index_slot1: &[I1],
    index_slot2: &[I2],
    index_slot3: &[I3],
    index_slot4: &[I4],
    index_slot5: &[I5],
    index_slot6: &[I6],
    index_slot7: &[I7],
    index_slot8: &[I8],
    index_slot9: &[I9],
    index_slot10: &[I10],
    index_slot11: &[I11],
    index_slot12: &[I12],
    index_offsets: &[u32],
    selection_route: &[u32],
    selection_block_prefixes: &[u32],
    mode: &[u32],
    active_len: &[u32],
    source_len: &[u32],
    resolved: &mut [u32],
) {
    let dispatch_position = ABSOLUTE_POS as usize;
    let ranked = mode[0] != 0u32;
    // Operation 0 is a compact gather, operation 1 is a scatter, and
    // operation 2 is a gather retaining the proposal's output position.
    let operation = mode[1];
    let compact_gather = operation == 0u32;
    let entry_words = if compact_gather {
        RESOLVED_GATHER_ENTRY_WORDS
    } else {
        RESOLVED_SCATTER_ENTRY_WORDS
    };
    let entry_capacity = (resolved.len() - RESOLVED_INDEX_HEADER_WORDS) / entry_words;
    if dispatch_position == 0usize {
        resolved[0] = active_len[0];
    }

    let compact_position = RuntimeCell::<usize>::new(dispatch_position);
    let active = RuntimeCell::<bool>::new(dispatch_position < active_len[0] as usize);
    if ranked {
        if dispatch_position < source_len[0] as usize {
            let (rank, selected) = crate::core::selection::route_rank_before(
                selection_route,
                selection_block_prefixes,
                dispatch_position,
            );
            active.store(selected);
            if active.read() {
                compact_position.store(rank as usize);
            }
        } else {
            active.store(false);
        }
    }

    if active.read() && compact_position.read() < entry_capacity {
        let source = dispatch_position;
        let indexed = IndexExpr::eval13(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot4,
            index_slot5,
            index_slot6,
            index_slot7,
            index_slot8,
            index_slot9,
            index_slot10,
            index_slot11,
            index_slot12,
            index_offsets,
            source,
        );
        let base = RESOLVED_INDEX_HEADER_WORDS + compact_position.read() * entry_words;
        resolved[base] = if operation == 1u32 {
            source as u32
        } else {
            indexed
        };
        if !compact_gather {
            resolved[base + 1usize] = if operation == 1u32 {
                indexed
            } else {
                source as u32
            };
        }
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn indexed_copy_a13<
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
    SourceExpr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<Item, Leaves = Leaves>,
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
    #[comptime] gather: bool,
) {
    let compact_position = ABSOLUTE_POS as usize;
    let entry_words = if gather {
        RESOLVED_GATHER_ENTRY_WORDS
    } else {
        RESOLVED_SCATTER_ENTRY_WORDS
    };
    let entry_capacity = (resolved.len() - RESOLVED_INDEX_HEADER_WORDS) / entry_words;
    if compact_position < entry_capacity && compact_position < resolved[0] as usize {
        let base = RESOLVED_INDEX_HEADER_WORDS + compact_position * entry_words;
        let source_position = resolved[base] as usize;
        let output_position = if gather {
            compact_position
        } else {
            resolved[base + 1usize] as usize
        };
        let source = SourceExpr::eval13(
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
            source_position,
        );
        Layout::decompose(source).store_padded(
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
            output_position,
        );
    }
}

/// Applies an already materialized physical permutation directly.  Unlike a
/// general gather, this path does not resolve the index expression into a
/// second element-sized buffer before transferring the payload.
#[cubecl::cube(launch_unchecked, explicit_define)]
fn permutation_copy_a13<
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
    SourceExpr: Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<Item, Leaves = Leaves>,
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
    permutation: &[u32],
    active_len: &[u32],
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
    metadata: &[u32],
) {
    let output_position = ABSOLUTE_POS as usize;
    if output_position < active_len[0] as usize {
        let source_position =
            permutation[metadata[TRANSFER_PARAMETERS_OFFSET] as usize + output_position] as usize;
        let source = SourceExpr::eval13_at(
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
            metadata,
            TRANSFER_READ_OFFSET,
            source_position,
        );
        Layout::decompose(source).store_padded(
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
            metadata,
            output_position,
        );
    }
}

/// Direct transfer capability for a physical permutation control.
#[doc(hidden)]
pub(crate) trait PermutationCopyInput<R: Runtime, Output>: ReadExpression + Sized {
    fn permutation_copy(
        self,
        exec: &Executor<R>,
        permutation: crate::core::read::Column<u32>,
        active_len: Option<&crate::DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Output> PermutationCopyInput<R, Output> for Input
where
    R: Runtime,
    Input: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: crate::core::output::OutputExpression
        + crate::core::output::LowerOutputExpression
        + crate::core::output::StageOutput<R, Env0>,
    Output::Slots: crate::core::output::PaddedOutputSlots<
            Leaves = <Output::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn permutation_copy(
        self,
        exec: &Executor<R>,
        permutation: crate::core::read::Column<u32>,
        active_len: Option<&crate::DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error> {
        let operation_capacity = permutation.len;
        let output_len = output.physical_len()?;
        let active_extent = active_len
            .map(|active_len| {
                crate::core::extent::LogicalExtent::from_device(active_len, operation_capacity)
            })
            .unwrap_or_else(|| permutation.extent.clone());
        if active_len.is_none() && output_len < active_extent.upper_bound() {
            return Err(Error::OutputTooShort {
                input: active_extent.upper_bound(),
                output: output_len,
            });
        }
        let operation_len = if active_len.is_some() {
            operation_capacity.min(output_len)
        } else {
            operation_capacity
        };
        if operation_len == 0 {
            return Ok(());
        }

        let reads = Bindings::read(exec, &self)?;
        let mut permutation_read = Bindings::new();
        <crate::core::read::Column<u32> as StageRead<R, Env0>>::stage_at(
            &permutation,
            exec.client(),
            exec.id(),
            &mut permutation_read,
        )?;
        let writes = Bindings::write(exec, &output)?;

        let active_len = active_extent.materialize(exec)?;
        let metadata = Bindings::transfer_metadata(&reads, &writes, &[permutation_read.offsets[0]]);
        let metadata_handle = exec.client().create_from_slice(u32::as_bytes(&metadata));

        unsafe {
            permutation_copy_a13::launch_unchecked::<
                Input::Item,
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
                R,
            >(
                exec.client(),
                crate::core::launch::cube_count_1d(operation_len.div_ceil(BLOCK_SIZE as usize))?,
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
                BufferArg::from_raw_parts(
                    permutation_read.slots[0].0.clone(),
                    permutation_read.slots[0].1,
                ),
                BufferArg::from_raw_parts(active_len.handle.clone(), 1),
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
                BufferArg::from_raw_parts(metadata_handle, metadata.len()),
            );
        }
        Ok(())
    }
}

/// Internal capability for an indexed gather or scatter.
#[doc(hidden)]
pub trait IndexedCopyInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn indexed_copy(
        self,
        exec: &Executor<R>,
        indices: Indices,
        gather: bool,
        output: Output,
    ) -> Result<(), Error> {
        self.indexed_copy_selected(exec, indices, None, None, gather, output)
    }

    fn indexed_copy_selected(
        self,
        exec: &Executor<R>,
        indices: Indices,
        selection: Option<IndexSelection<'_, R>>,
        active_len: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> IndexedCopyInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Indices: ReadExpression<Item = crate::MIndex> + LowerReadExpression + StageRead<R, Env0>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: crate::core::output::OutputExpression
        + crate::core::output::LowerOutputExpression
        + crate::core::output::StageOutput<R, Env0>,
    Output::Slots: crate::core::output::PaddedOutputSlots<
            Leaves = <Output::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn indexed_copy_selected(
        self,
        exec: &Executor<R>,
        indices: Indices,
        selection: Option<IndexSelection<'_, R>>,
        active_len: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        output: Output,
    ) -> Result<(), Error> {
        let values_len = self.physical_len()?;
        let indices_len = indices.physical_len()?;
        let values_extent = self.logical_extent()?;
        let indices_extent = indices.logical_extent()?;
        let inferred_extent = if gather {
            indices_extent.clone()
        } else {
            values_extent.zipped(&indices_extent)?
        };
        let unselected_len = if gather { indices_len } else { values_len };
        if !gather && values_len != indices_len {
            return Err(Error::LengthMismatch {
                left: values_len,
                right: indices_len,
            });
        }
        let operation_capacity = match selection {
            Some(IndexSelection::Routing { control, .. }) => {
                if control.len() != indices_len {
                    return Err(Error::LengthMismatch {
                        left: indices_len,
                        right: control.len(),
                    });
                }
                indices_extent.zipped(&control.source_extent())?;
                control.len()
            }
            None => unselected_len,
        };
        let output_len = output.physical_len()?;
        let active_extent = active_len
            .map(|active_len| {
                crate::core::extent::LogicalExtent::from_device(active_len, operation_capacity)
            })
            .unwrap_or(inferred_extent);
        if gather && active_len.is_none() && output_len < active_extent.upper_bound() {
            return Err(Error::OutputTooShort {
                input: active_extent.upper_bound(),
                output: output_len,
            });
        }
        // A device-resident active length cannot be used for a host-side exact
        // capacity check.  For gathers, the compact position is also the
        // output position, so limiting the over-dispatch to the output
        // capacity is sufficient to keep every write in bounds.  The kernel
        // additionally guards it with `active_len`.
        //
        // Scatter writes use the evaluated index as the output position; the
        // number of proposals is therefore unrelated to the output length.
        let operation_len = if gather && active_len.is_some() {
            operation_capacity.min(output_len)
        } else {
            operation_capacity
        };
        if operation_len == 0 {
            return Ok(());
        }

        let transfer_gather = gather
            && !matches!(
                selection,
                Some(IndexSelection::Routing {
                    retain_output_position: true,
                    ..
                })
            );
        let resolved_entry_words = if transfer_gather {
            RESOLVED_GATHER_ENTRY_WORDS
        } else {
            RESOLVED_SCATTER_ENTRY_WORDS
        };
        let resolved_len = operation_len
            .checked_mul(resolved_entry_words)
            .and_then(|entries| entries.checked_add(RESOLVED_INDEX_HEADER_WORDS))
            .ok_or(Error::LengthTooLarge { len: operation_len })?;
        let _: u32 = resolved_len
            .try_into()
            .map_err(|_| Error::LengthTooLarge { len: resolved_len })?;
        let reads = Bindings::read(exec, &self)?;
        let index_reads = Bindings::read(exec, &indices)?;
        let writes = Bindings::write(exec, &output)?;

        let read_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&reads.offsets));
        let index_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&index_reads.offsets));
        let write_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let selection_kind = match selection {
            None => 0u32,
            Some(IndexSelection::Routing { .. }) => 1u32,
        };
        let operation = if transfer_gather {
            0u32
        } else if gather {
            2u32
        } else {
            1u32
        };
        let mode = exec
            .client()
            .create_from_slice(u32::as_bytes(&[selection_kind, operation]));
        let (
            selection_route,
            selection_route_capacity,
            selection_block_prefixes,
            selection_block_prefix_capacity,
        ) = match selection.as_ref() {
            Some(IndexSelection::Routing { control, .. }) => (
                control.route().handle.clone(),
                control.route().capacity(),
                control.block_prefixes().handle.clone(),
                control.block_prefixes().capacity(),
            ),
            _ => {
                let empty = exec.client().create_from_slice(u32::as_bytes(&[0u32]));
                (empty.clone(), 1, empty, 1)
            }
        };
        let active_len = active_extent.materialize(exec)?;
        let source_len = indices_extent.materialize(exec)?;
        let resolved = exec.alloc_column::<u32>(resolved_len);

        // Index evaluation and value transfer are separate semantic stages so
        // neither kernel binds two fixed thirteen-read groups at once.
        unsafe {
            resolve_index_a13::launch_unchecked::<
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
                crate::core::launch::cube_count_1d(
                    (if selection_kind != 0 {
                        operation_capacity
                    } else {
                        operation_len
                    })
                    .div_ceil(BLOCK_SIZE as usize),
                )?,
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
                BufferArg::from_raw_parts(selection_route, selection_route_capacity),
                BufferArg::from_raw_parts(
                    selection_block_prefixes,
                    selection_block_prefix_capacity,
                ),
                BufferArg::from_raw_parts(mode, 2),
                BufferArg::from_raw_parts(active_len.handle.clone(), 1),
                BufferArg::from_raw_parts(source_len.handle.clone(), 1),
                BufferArg::from_raw_parts(resolved.handle.clone(), resolved_len),
            );
            indexed_copy_a13::launch_unchecked::<
                Values::Item,
                <Values::Slots as PaddedReadSlots>::L0,
                <Values::Slots as PaddedReadSlots>::L1,
                <Values::Slots as PaddedReadSlots>::L2,
                <Values::Slots as PaddedReadSlots>::L3,
                <Values::Slots as PaddedReadSlots>::L4,
                <Values::Slots as PaddedReadSlots>::L5,
                <Values::Slots as PaddedReadSlots>::L6,
                <Values::Slots as PaddedReadSlots>::L7,
                <Values::Slots as PaddedReadSlots>::L8,
                <Values::Slots as PaddedReadSlots>::L9,
                <Values::Slots as PaddedReadSlots>::L10,
                <Values::Slots as PaddedReadSlots>::L11,
                <Values::Slots as PaddedReadSlots>::L12,
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
                Values::DeviceExpr,
                <Output::Item as StorageLayout>::DeviceLayout,
                R,
            >(
                exec.client(),
                crate::core::launch::cube_count_1d(operation_len.div_ceil(BLOCK_SIZE as usize))?,
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
                transfer_gather,
            );
        }
        Ok(())
    }
}

/// Internal gather capability retained for selection and ordering controls.
#[doc(hidden)]
pub trait GatherInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: IndexedCopyInput<R, Indices, Output>,
{
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error> {
        self.indexed_copy(exec, indices, true, output)
    }
}

pub(crate) fn gather_direct<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherInput<R, Indices, Output>,
{
    values.gather(exec, indices, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::allocation::RowStorage;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, FixedRead, Permute, ReverseCounting};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn generated_permutation_transfer_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Kernel = permutation_copy_a13::PermutationCopyA13<
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
        crate::core::launch::assert_binding_budget("permutation_transfer", &kernel);
    }

    #[test]
    fn generated_indexed_stages_fit_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;

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

        let resolve = resolve_index_a13::ResolveIndexA13::<
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
            WgpuRuntime,
        >::new(
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
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
        );
        crate::core::launch::assert_binding_budget("index resolution", &resolve);

        let transfer = indexed_copy_a13::IndexedCopyA13::<
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
            WgpuRuntime,
        >::new(
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
            true,
        );
        crate::core::launch::assert_binding_budget("indexed transfer", &transfer);
    }

    #[test]
    fn gather_seven_columns_uses_independent_index_expression() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 2])).collect();
        let indices = exec.to_device(&[2_u32, 0]);
        let values = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        gather_direct(&exec, FixedRead::new(values), indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 3, column as u32 * 10 + 1]
            );
        }
    }

    #[test]
    fn gather_accepts_logical_usize_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let encoded = exec.to_device(&[2_u32, 0]);
        let indices = encoded.column();
        let output = exec.alloc::<u32>(2);

        gather_direct(&exec, values.column(), indices, output.slice_mut(..)).unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10]);
    }

    #[test]
    fn gather_accepts_lazy_indices_independently() {
        type Seven = (u32, u32, u32, u32, u32, u32, u32);
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let seven = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let values = Permute::new(seven, Counting::new(0, 3));
        let raw_indices = exec.to_device(&[2_u32, 0]);
        let indices = Permute::new(raw_indices.column(), Counting::new(0, 2));
        let output = exec.alloc_row::<Seven>(2);

        gather_direct(&exec, FixedRead::new(values), indices, output.write()).unwrap();
        let (first, _, _, _, _, _, last) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&first).unwrap(), vec![3, 1]);
        assert_eq!(exec.to_host(&last).unwrap(), vec![63, 61]);
    }

    #[test]
    fn reverse_counting_remains_a_valid_internal_index() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let output = exec.alloc::<u32>(3);

        gather_direct(
            &exec,
            values.column(),
            ReverseCounting::new(3),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20, 10]);
    }
}
