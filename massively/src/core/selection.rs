//! Stable selection composed from block routing control and payload transfer.

use cubecl::prelude::*;

use crate::core::allocation::RowStorage;
use crate::core::arity::A1;
use crate::core::bindings::{Bindings, TRANSFER_PARAMETERS_OFFSET, TRANSFER_READ_OFFSET};
use crate::core::eval::{Eval13, Eval13At};
use crate::core::iter::Zip;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, SliceOutput, StageOutput,
};
use crate::core::read::{
    Constant, Env0, Env1, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead,
    Transform,
};
use crate::core::runtime::ColumnMut;
use crate::core::scan::{inclusive_scan_u32, last_u32};
use crate::core::storage::{
    Concat, Decompose, S1, StorageLayout, StorePadded12, StorePadded12Expand,
};
use crate::core::transform::materialize;
use crate::op::UnaryOp;
use crate::{DeviceVec, Error, Executor, MFlag};

const BLOCK_SIZE: u32 = 256;
const ROUTE_WORD_BITS: usize = 32;
const ROUTE_WORDS_PER_BLOCK: usize = BLOCK_SIZE as usize / ROUTE_WORD_BITS;
const ROUTE_WORD_STRIDE: usize = 2;
const ROUTE_BLOCK_STRIDE: usize = ROUTE_WORDS_PER_BLOCK * ROUTE_WORD_STRIDE;
const ROUTE_MASK_OFFSET: usize = 0;
const ROUTE_PREFIX_OFFSET: usize = 1;

/// Returns the number of selected rows before `index` and whether `index`
/// itself is selected.
#[cubecl::cube]
pub(crate) fn route_rank_before(
    route: &[u32],
    block_prefixes: &[u32],
    index: usize,
) -> (u32, bool) {
    let block = index / BLOCK_SIZE as usize;
    let local = index - block * BLOCK_SIZE as usize;
    let word = local / ROUTE_WORD_BITS;
    let bit = local - word * ROUTE_WORD_BITS;
    let route_index = block * ROUTE_BLOCK_STRIDE + word * ROUTE_WORD_STRIDE;
    let mask = route[route_index + ROUTE_MASK_OFFSET];
    let lower_mask = if bit == 0usize {
        0u32
    } else {
        (1u32 << bit as u32) - 1u32
    };
    let block_prefix = if block == 0usize {
        0u32
    } else {
        block_prefixes[block - 1usize]
    };
    let rank =
        block_prefix + route[route_index + ROUTE_PREFIX_OFFSET] + (mask & lower_mask).count_ones();
    let selected = (mask & (1u32 << bit as u32)) != 0u32;
    (rank, selected)
}

/// Packs subgroup ballots into the fixed 32-bit route representation. The
/// representation is independent of the device's subgroup width, including
/// subgroups smaller than a route word and subgroups spanning several words.
#[cubecl::cube]
fn write_block_route(selected: bool, route: &mut [u32], block_counts: &mut [u32]) {
    let lane = UNIT_POS as usize;
    let block = CUBE_POS as usize;
    if block >= block_counts.len() {
        terminate!();
    }
    let ballot = plane_ballot(selected);
    let mut fragments = Shared::<[u32]>::new_slice(BLOCK_SIZE as usize);
    if UNIT_POS_PLANE == 0u32 {
        #[unroll]
        for word in 0usize..4usize {
            if word * ROUTE_WORD_BITS < PLANE_DIM as usize {
                fragments[lane + word * ROUTE_WORD_BITS] = ballot.extract(word);
            }
        }
    }
    sync_cube();

    let mut counts = Shared::<[u32]>::new_slice(ROUTE_WORDS_PER_BLOCK);
    if lane < ROUTE_WORDS_PER_BLOCK {
        let mask = RuntimeCell::<u32>::new(0u32);
        let bit = RuntimeCell::<usize>::new(0usize);
        let step = usize::min(PLANE_DIM as usize, ROUTE_WORD_BITS);
        while bit.read() < ROUTE_WORD_BITS {
            mask.store(
                mask.read() | (fragments[lane * ROUTE_WORD_BITS + bit.read()] << bit.read() as u32),
            );
            bit.store(bit.read() + step);
        }
        route[block * ROUTE_BLOCK_STRIDE + lane * ROUTE_WORD_STRIDE + ROUTE_MASK_OFFSET] =
            mask.read();
        counts[lane] = mask.read().count_ones();
    }
    sync_cube();
    if lane == 0usize {
        let prefix = RuntimeCell::<u32>::new(0u32);
        let word = RuntimeCell::<usize>::new(0usize);
        while word.read() < ROUTE_WORDS_PER_BLOCK {
            route[block * ROUTE_BLOCK_STRIDE
                + word.read() * ROUTE_WORD_STRIDE
                + ROUTE_PREFIX_OFFSET] = prefix.read();
            prefix.store(prefix.read() + counts[word.read()]);
            word.store(word.read() + 1usize);
        }
        block_counts[block] = prefix.read();
    }
}

/// Evaluates a logical stencil once and records a compact route.  Each block
/// emits eight masks, the exclusive selected count before each mask, and one
/// block total.  Only the much smaller block totals are globally scanned.
#[cubecl::cube(launch_unchecked, explicit_define)]
fn selection_rank_a13<
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
    StencilExpr: Eval13<MFlag, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
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
    source_len: &[u32],
    parameters: &[u32],
    route: &mut [u32],
    block_counts: &mut [u32],
) {
    let block = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let index = block * BLOCK_SIZE as usize + lane;
    let mut selected = false;
    if index < source_len[0] as usize {
        selected = crate::flag::is_set(StencilExpr::eval13(
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
            index,
        ));
        if parameters[0] != 0u32 {
            selected = !selected;
        }
    }

    write_block_route(selected, route, block_counts);
    if ABSOLUTE_POS == 0usize {
        route[route.len() - 1usize] = source_len[0];
    }
}

/// Converts an already materialized inclusive rank into the compact routing
/// representation.  This is retained for segmented algorithms that need the
/// dense prefix for a separate boundary query.
#[cubecl::cube(launch_unchecked)]
fn positions_to_route_kernel(
    positions: &[u32],
    source_len: &[u32],
    route: &mut [u32],
    block_counts: &mut [u32],
) {
    let block = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let index = block * BLOCK_SIZE as usize + lane;
    let selected = index < source_len[0] as usize
        && positions[index]
            != if index == 0usize {
                0u32
            } else {
                positions[index - 1usize]
            };

    write_block_route(selected, route, block_counts);
    if ABSOLUTE_POS == 0usize {
        route[route.len() - 1usize] = source_len[0];
    }
}

#[cubecl::cube(launch_unchecked)]
fn selected_indices_kernel(
    route: &[u32],
    block_prefixes: &[u32],
    len: &[u32],
    indices: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let (rank, selected) = route_rank_before(route, block_prefixes, index);
        if selected {
            indices[rank as usize] = index as u32;
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn materialize_positions_kernel(
    route: &[u32],
    block_prefixes: &[u32],
    len: &[u32],
    positions: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let (rank, selected) = route_rank_before(route, block_prefixes, index);
        positions[index] = rank + if selected { 1u32 } else { 0u32 };
    }
}

// Runtime destinations used by the common rank -> transfer stage.  Keeping
// this as data avoids multiplying every input/output kernel specialization by
// the number of selection APIs.
const TRANSFER_COMPACT: u32 = 0;
const TRANSFER_PARTITION: u32 = 1;
const TRANSFER_RETAINED: u32 = 2;

/// Transfers rows directly from their source position using block routing
/// control.  The compact destination is reconstructed from a block prefix, a
/// word prefix, and a mask population count, so no element-sized rank or
/// permutation is read from global memory.
#[cubecl::cube(launch_unchecked, explicit_define)]
fn ranked_transfer_a13<
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
    route: &[u32],
    block_prefixes: &[u32],
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
    let index = ABSOLUTE_POS as usize;
    let source_len = route[route.len() - 1usize] as usize;
    if index < source_len {
        let (rank, selected) = route_rank_before(route, block_prefixes, index);
        let mode = metadata[TRANSFER_PARAMETERS_OFFSET];
        if selected || mode == TRANSFER_PARTITION {
            let destination = if mode == TRANSFER_RETAINED {
                index
            } else if selected {
                rank as usize
            } else {
                let block_count =
                    crate::core::launch::logical_block_count(source_len, BLOCK_SIZE as usize);
                let selected_count = if block_count == 0usize {
                    0u32
                } else {
                    block_prefixes[block_count - 1usize]
                };
                (selected_count + index as u32 - rank) as usize
            };
            if destination < metadata[TRANSFER_PARAMETERS_OFFSET + 1usize] as usize {
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
                    index,
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
                    destination,
                );
            }
        }
    }
}

/// Recursive fill over output leaves.
#[doc(hidden)]
pub trait FillOutput<R: Runtime>: OutputExpression + Sized {
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error>;
}

impl<R, T> FillOutput<R> for ColumnMut<T>
where
    R: Runtime,
    T: crate::core::value::MStorageElement + StorageLayout<StorageArity = S1>,
    Constant<T>: ReadExpression<Item = T, ReadArity = A1>
        + LowerReadExpression<Slots = Env1<T>>
        + StageRead<R, Env0>,
    ColumnMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = Env1<T>>
        + StageOutput<R, Env0>,
{
    fn fill_output(self, exec: &Executor<R>, value: T) -> Result<(), Error> {
        let len = self.capacity();
        materialize(exec, Constant::new(value, len), self)
    }
}

impl<R, Left, Right> FillOutput<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: FillOutput<R>,
    Right: FillOutput<R>,
    Zip<Left, Right>: OutputExpression,
    <Left::Item as StorageLayout>::StorageLeaves: Concat<
            <Right::Item as StorageLayout>::StorageLeaves,
            Output = <<Zip<Left, Right> as OutputExpression>::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        let left_len = self.0.physical_len()?;
        let right_len = self.1.physical_len()?;
        if left_len != right_len {
            return Err(Error::LengthMismatch {
                left: left_len,
                right: right_len,
            });
        }
        let (left, right) =
            <Left::Item as StorageLayout>::StorageLeaves::split(value.into_storage_leaves());
        self.0
            .fill_output(exec, Left::Item::from_storage_leaves(left))?;
        self.1
            .fill_output(exec, Right::Item::from_storage_leaves(right))
    }
}

impl<R, Output> FillOutput<R> for crate::core::output::Slice<R, Output>
where
    R: Runtime,
    Output: FillOutput<R>,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        self.into_inner().fill_output(exec, value)
    }
}

/// Internal capability to consume a logical stencil expression.
#[doc(hidden)]
pub trait FlagInput<R: Runtime>: ReadExpression<Item = MFlag> + Sized {
    fn flag_len(&self) -> Result<usize, Error>;
    fn flag_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error>;
    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
}

impl<R, Stencil> FlagInput<R> for Stencil
where
    R: Runtime,
    Stencil: ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
{
    fn flag_len(&self) -> Result<usize, Error> {
        self.physical_len()
    }

    fn flag_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error> {
        self.logical_extent()
    }

    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        SelectionControl::from_stencil(exec, self, false)
    }

    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        SelectionControl::from_stencil(exec, self, true)
    }
}

/// Stable selected-row control shared by every payload that uses the same
/// flags.  Keeping this separate from the payload prevents by-key algorithms
/// from coupling key arity to value arity.
#[doc(hidden)]
pub struct SelectionControl<R: Runtime> {
    len: usize,
    source_extent: crate::core::extent::LogicalExtent,
    route: DeviceVec<R, u32>,
    block_prefixes: DeviceVec<R, u32>,
    count: DeviceVec<R, u32>,
}

impl<R: Runtime> SelectionControl<R> {
    pub(crate) fn from_flags(exec: &Executor<R>, flags: DeviceVec<R, u32>) -> Result<Self, Error> {
        Self::from_stencil(exec, flags.column(), false)
    }

    fn from_stencil<Stencil>(
        exec: &Executor<R>,
        stencil: Stencil,
        invert: bool,
    ) -> Result<Self, Error>
    where
        Stencil: ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
    {
        let len = stencil.physical_len()?;
        let source_extent = stencil.logical_extent()?;
        let blocks = len.div_ceil(BLOCK_SIZE as usize);
        let route_len = blocks
            .checked_mul(ROUTE_BLOCK_STRIDE)
            .and_then(|words| words.checked_add(1))
            .ok_or(Error::LengthTooLarge { len })?;
        let route = exec.alloc_row::<u32>(route_len);
        let mut block_counts = exec.alloc_row::<u32>(blocks);
        block_counts.set_logical_extent(source_extent.ceil_div(
            exec,
            BLOCK_SIZE as usize,
            blocks,
        )?);

        if blocks != 0 {
            let reads = Bindings::read(exec, &stencil)?;
            let read_offsets = exec
                .client()
                .create_from_slice(u32::as_bytes(&reads.offsets));
            let source_len = source_extent.materialize(exec)?;
            let parameters = exec.to_device(&[u32::from(invert)]);
            unsafe {
                selection_rank_a13::launch_unchecked::<
                    <Stencil::Slots as PaddedReadSlots>::L0,
                    <Stencil::Slots as PaddedReadSlots>::L1,
                    <Stencil::Slots as PaddedReadSlots>::L2,
                    <Stencil::Slots as PaddedReadSlots>::L3,
                    <Stencil::Slots as PaddedReadSlots>::L4,
                    <Stencil::Slots as PaddedReadSlots>::L5,
                    <Stencil::Slots as PaddedReadSlots>::L6,
                    <Stencil::Slots as PaddedReadSlots>::L7,
                    <Stencil::Slots as PaddedReadSlots>::L8,
                    <Stencil::Slots as PaddedReadSlots>::L9,
                    <Stencil::Slots as PaddedReadSlots>::L10,
                    <Stencil::Slots as PaddedReadSlots>::L11,
                    <Stencil::Slots as PaddedReadSlots>::L12,
                    Stencil::DeviceExpr,
                    R,
                >(
                    exec.client(),
                    crate::core::launch::cube_count_1d(blocks)?,
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
                    BufferArg::from_raw_parts(source_len.handle.clone(), 1),
                    BufferArg::from_raw_parts(parameters.handle.clone(), 1),
                    BufferArg::from_raw_parts(route.handle.clone(), route.capacity()),
                    BufferArg::from_raw_parts(block_counts.handle.clone(), block_counts.capacity()),
                );
            }
        }

        let block_prefixes = inclusive_scan_u32(exec, &block_counts)?;
        let count = last_u32(exec, &block_prefixes)?;
        Ok(Self {
            len,
            source_extent,
            route,
            block_prefixes,
            count,
        })
    }

    pub(crate) fn from_positions(
        exec: &Executor<R>,
        positions: DeviceVec<R, u32>,
    ) -> Result<Self, Error> {
        let source_extent = positions.logical_extent();
        let len = positions.capacity();
        let blocks = len.div_ceil(BLOCK_SIZE as usize);
        let route_len = blocks
            .checked_mul(ROUTE_BLOCK_STRIDE)
            .and_then(|words| words.checked_add(1))
            .ok_or(Error::LengthTooLarge { len })?;
        let route = exec.alloc_row::<u32>(route_len);
        let mut block_counts = exec.alloc_row::<u32>(blocks);
        block_counts.set_logical_extent(source_extent.ceil_div(
            exec,
            BLOCK_SIZE as usize,
            blocks,
        )?);
        if blocks != 0 {
            let source_len = source_extent.materialize(exec)?;
            unsafe {
                positions_to_route_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(blocks)?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                    BufferArg::from_raw_parts(source_len.handle.clone(), 1),
                    BufferArg::from_raw_parts(route.handle.clone(), route.capacity()),
                    BufferArg::from_raw_parts(block_counts.handle.clone(), block_counts.capacity()),
                );
            }
        }
        let block_prefixes = inclusive_scan_u32(exec, &block_counts)?;
        let count = last_u32(exec, &block_prefixes)?;
        Ok(Self {
            len,
            source_extent,
            route,
            block_prefixes,
            count,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn count(&self) -> &DeviceVec<R, u32> {
        &self.count
    }

    pub(crate) fn selected_extent(&self) -> crate::core::extent::LogicalExtent {
        crate::core::extent::LogicalExtent::from_device(&self.count, self.len)
    }

    pub(crate) fn route(&self) -> &DeviceVec<R, u32> {
        &self.route
    }

    pub(crate) fn block_prefixes(&self) -> &DeviceVec<R, u32> {
        &self.block_prefixes
    }

    /// Materializes a dense inclusive prefix only for algorithms that need
    /// arbitrary segment-boundary lookup.
    pub(crate) fn materialize_positions(
        &self,
        exec: &Executor<R>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        let mut positions = exec.alloc_row::<u32>(self.len);
        positions.set_logical_extent(self.source_extent.clone());
        if self.len != 0 {
            let len_handle = self.source_extent.materialize(exec)?;
            unsafe {
                materialize_positions_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(self.len.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.route.handle.clone(), self.route.capacity()),
                    BufferArg::from_raw_parts(
                        self.block_prefixes.handle.clone(),
                        self.block_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                );
            }
        }
        Ok(positions)
    }

    /// Materializes compact source positions only for algorithms that truly
    /// need random access to segment boundaries.  Ordinary selection and
    /// partition transfers consume block routing directly.
    pub(crate) fn materialize_indices(
        &self,
        exec: &Executor<R>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        let mut indices = exec.alloc_row::<u32>(self.len);
        if self.len != 0 {
            let len_handle = self.source_extent.materialize(exec)?;
            unsafe {
                selected_indices_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(self.len.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.route.handle.clone(), self.route.capacity()),
                    BufferArg::from_raw_parts(
                        self.block_prefixes.handle.clone(),
                        self.block_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                    BufferArg::from_raw_parts(indices.handle.clone(), indices.capacity()),
                );
            }
        }
        indices.set_logical_extent(crate::core::extent::LogicalExtent::from_device(
            &self.count,
            self.len,
        ));
        Ok(indices)
    }

    pub(crate) fn source_extent(&self) -> crate::core::extent::LogicalExtent {
        self.source_extent.clone()
    }
}

/// Arity-independent application of an already computed stable rank.
#[doc(hidden)]
pub trait RankedTransferInput<R: Runtime, Output>:
    ReadExpression + StageRead<R, Env0> + Sized
{
    fn ranked_transfer(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        mode: u32,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Output> RankedTransferInput<R, Output> for Input
where
    R: Runtime,
    Input: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
{
    fn ranked_transfer(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        mode: u32,
        output: Output,
    ) -> Result<(), Error> {
        let source_len = self.physical_len()?;
        if source_len != control.len() {
            return Err(Error::LengthMismatch {
                left: source_len,
                right: control.len(),
            });
        }
        self.logical_extent()?.zipped(&control.source_extent())?;
        let output_len = output.physical_len()?;
        if mode != TRANSFER_COMPACT && output_len < source_len {
            return Err(Error::OutputTooShort {
                input: source_len,
                output: output_len,
            });
        }
        if source_len == 0 {
            return Ok(());
        }

        let reads = Bindings::read(exec, &self)?;
        let writes = Bindings::write(exec, &output)?;

        let output_len: u32 = output_len
            .try_into()
            .map_err(|_| Error::LengthTooLarge { len: output_len })?;
        let metadata = Bindings::transfer_metadata(&reads, &writes, &[mode, output_len]);
        let metadata_handle = exec.client().create_from_slice(u32::as_bytes(&metadata));

        unsafe {
            ranked_transfer_a13::launch_unchecked::<
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
                crate::core::launch::cube_count_1d(source_len.div_ceil(BLOCK_SIZE as usize))?,
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
                    control.route().handle.clone(),
                    control.route().capacity(),
                ),
                BufferArg::from_raw_parts(
                    control.block_prefixes().handle.clone(),
                    control.block_prefixes().capacity(),
                ),
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

/// Internal capability for copying selected rows from a readable expression.
#[doc(hidden)]
pub trait CopySelected<R: Runtime, Output>: ReadExpression + Sized {
    fn source_len(&self) -> Result<usize, Error>;
    fn source_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error>;
    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Output> CopySelected<R, Output> for Input
where
    R: Runtime,
    Input: RankedTransferInput<R, Output>,
    Output: OutputExpression,
{
    fn source_len(&self) -> Result<usize, Error> {
        self.physical_len()
    }

    fn source_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error> {
        self.logical_extent()
    }

    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<DeviceVec<R, u32>, Error> {
        self.ranked_transfer(exec, control, TRANSFER_COMPACT, output)?;
        Ok(control.count().clone())
    }
}

/// Stably copies values whose stencil is nonzero.
pub(crate) fn copy_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    input.source_extent()?.zipped(&stencil.flag_extent()?)?;
    let control = stencil.selected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably copies values whose stencil is zero.
pub(crate) fn remove_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    input.source_extent()?.zipped(&stencil.flag_extent()?)?;
    let control = stencil.rejected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably partitions passing values before failing values.
pub(crate) fn partition<R, Input, Pred, Output>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: Clone + crate::core::predicate::PredicateInput<R, Pred> + RankedTransferInput<R, Output>,
    Output: SliceOutput,
{
    let len = input.source_len()?;
    let output_len = output.physical_len()?;
    if output_len < len {
        return Err(Error::OutputTooShort {
            input: len,
            output: output_len,
        });
    }
    let control = input.clone().predicate_control(exec)?;
    let passing = control.count().clone();
    input.ranked_transfer(exec, &control, TRANSFER_PARTITION, output)?;
    Ok(passing)
}

/// Replaces items whose logical stencil is true.
pub(crate) fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    value: <Output::Item as crate::core::allocation::ScratchStorage<R>>::Storage,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
    Output::Item: crate::core::allocation::ScratchStorage<R>,
    <Output::Item as crate::core::allocation::ScratchStorage<R>>::Storage: RowStorage<R>,
    crate::core::read::Repeat<
        <<Output::Item as crate::core::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Read,
    >: RankedTransferInput<R, Output>,
{
    let stencil_len = stencil.flag_len()?;
    let output_len = output.physical_len()?;
    if stencil_len != output_len {
        return Err(Error::LengthMismatch {
            left: stencil_len,
            right: output_len,
        });
    }
    let control = stencil.selected_control(exec)?;
    RankedTransferInput::ranked_transfer(
        crate::core::read::Repeat::new(value.read(), control.len()),
        exec,
        &control,
        TRANSFER_RETAINED,
        output,
    )
}

/// Internal capability for a transform selected by a logical stencil.
#[doc(hidden)]
pub trait TransformWhereInput<R: Runtime, Stencil, Output, Op>: ReadExpression + Sized {
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Stencil, Output, Op> TransformWhereInput<R, Stencil, Output, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: ReadExpression<Item = Op::Output>,
    Transform<Input, Op>: RankedTransferInput<R, Output>,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
{
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error> {
        let input_len = self.physical_len()?;
        let stencil_len = stencil.flag_len()?;
        let output_len = output.physical_len()?;
        if input_len != stencil_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: stencil_len,
            });
        }
        if input_len != output_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: output_len,
            });
        }
        self.logical_extent()?.zipped(&stencil.flag_extent()?)?;
        let control = stencil.selected_control(exec)?;
        RankedTransferInput::ranked_transfer(
            Transform::new(self, op),
            exec,
            &control,
            TRANSFER_RETAINED,
            output,
        )
    }
}

/// Applies `op` only where the logical stencil is true.
pub(crate) fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: TransformWhereInput<R, Stencil, Output, Op>,
{
    input.transform_where(exec, op, stencil, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn generated_ranked_transfer_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type ScalarExpr = <crate::core::read::Column<u32> as LowerReadExpression>::DeviceExpr;
        type Kernel = ranked_transfer_a13::RankedTransferA13<
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
        crate::core::launch::assert_binding_budget("ranked_transfer", &kernel);
    }

    #[test]
    fn copy_where_preserves_flat_three_column_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0, 40.0]);
        let c = exec.to_device(&[100_i32, 200, 300, 400]);
        let flags = exec.to_device(&[0_u32, 1, 1, 0]);
        let out_a = exec.to_device(&[0_u32; 4]);
        let out_b = exec.to_device(&[0_f32; 4]);
        let out_c = exec.to_device(&[0_i32; 4]);
        let input = Zip::new(a.column(), Zip::new(b.column(), c.column()));
        let output = Zip::new(
            Zip::new(out_a.slice_mut(..), out_b.slice_mut(..)),
            out_c.slice_mut(..),
        );

        let count = copy_where(&exec, input, flags.column(), output).unwrap();
        let count = exec.to_host(&count).unwrap()[0];
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&out_a.slice(..count)).unwrap(), vec![2, 3]);
        assert_eq!(
            exec.to_host(&out_b.slice(..count)).unwrap(),
            vec![20.0, 30.0]
        );
        assert_eq!(exec.to_host(&out_c.slice(..count)).unwrap(), vec![200, 300]);
    }

    #[test]
    fn block_route_reconstructs_binary_positions() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let flags = exec.to_device(&[0_u32, 7, 3, 0]);
        let control = flags.column().selected_control(&exec).unwrap();
        let positions = control.materialize_positions(&exec).unwrap();
        assert_eq!(exec.to_host(&positions).unwrap(), vec![0, 1, 2, 2]);
        assert_eq!(exec.to_host(control.count()).unwrap(), vec![2]);
    }

    #[test]
    fn block_route_crosses_multiple_control_blocks() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = BLOCK_SIZE as usize * 3usize + 17usize;
        let values: Vec<u32> = (0..len as u32).collect();
        let flags: Vec<u32> = (0..len)
            .map(|index| u32::from(index % 7usize == 0usize || index % 31usize == 3usize))
            .collect();
        let expected: Vec<u32> = values
            .iter()
            .copied()
            .zip(flags.iter().copied())
            .filter_map(|(value, flag)| (flag != 0).then_some(value))
            .collect();
        let values = exec.to_device(&values);
        let flags = exec.to_device(&flags);
        let output = exec.to_device(&vec![0u32; len]);

        let count =
            copy_where(&exec, values.column(), flags.column(), output.slice_mut(..)).unwrap();
        assert_eq!(exec.to_host(&count).unwrap(), vec![expected.len() as u32]);
        assert_eq!(
            exec.to_host(&output.slice(..expected.len() as u32))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn copy_where_on_seven_leaves_accepts_an_eighth_permutation_slot() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let flags = exec.to_device(&[1_u32, 0, 1]);
        let input = Zip::new(
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

        let count = copy_where(&exec, input, flags.column(), output).unwrap();
        assert_eq!(exec.to_host(&count).unwrap(), vec![2]);
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(&output.slice(..2)).unwrap(),
                vec![column as u32 * 10 + 1, column as u32 * 10 + 3]
            );
        }
    }

    #[test]
    fn remove_where_inverts_nonzero_stencil() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[10_u32, 20, 30, 40]);
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        let output = exec.to_device(&[0_u32; 4]);
        let count =
            remove_where(&exec, input.column(), flags.column(), output.slice_mut(..)).unwrap();
        let count = exec.to_host(&count).unwrap()[0];
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&output.slice(..2)).unwrap(), vec![10, 30]);
    }

    struct IsEven;

    #[cubecl::cube]
    impl crate::op::PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> MFlag {
            crate::flag::from_bool(input % 2u32 == 0u32)
        }
    }

    #[test]
    fn partition_is_stable_on_both_sides() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[3_u32, 2, 4, 1, 6, 5]);
        let output = exec.to_device(&[0_u32; 6]);
        let split = partition(&exec, input.column(), IsEven, output.slice_mut_usize(..)).unwrap();
        let split = exec.to_host(&split).unwrap()[0];
        assert_eq!(split, 3);
        assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 3, 1, 5]);
    }

    #[test]
    fn fill_and_replace_where_recurse_over_binary_output_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[0_u32; 4]);
        let b = exec.to_device(&[0_f32; 4]);
        let c = exec.to_device(&[0_i32; 4]);
        let output = || {
            Zip::new(
                Zip::new(a.slice_mut_usize(..), b.slice_mut_usize(..)),
                c.slice_mut_usize(..),
            )
        };
        output()
            .fill_output(&exec, (7_u32, 2.5_f32, -3_i32))
            .unwrap();
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        let value = crate::api::value::store(&exec, (9_u32, 4.5_f32, -8_i32)).unwrap();
        replace_where(
            &exec,
            crate::api::value::into_scratch::<WgpuRuntime, (u32, f32, i32)>(value),
            flags.column(),
            output(),
        )
        .unwrap();
        assert_eq!(exec.to_host(&a).unwrap(), vec![7, 9, 7, 9]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![2.5, 4.5, 2.5, 4.5]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![-3, -8, -3, -8]);
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    struct IncrementSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for IncrementSeven {
        type Output = Seven;
        fn apply(input: Seven) -> Seven {
            (
                input.0 + 1u32,
                input.1 + 1u32,
                input.2 + 1u32,
                input.3 + 1u32,
                input.4 + 1u32,
                input.5 + 1u32,
                input.6 + 1u32,
            )
        }
    }

    #[test]
    fn transform_where_separates_eight_slot_input_from_storage7_copy() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[100_u32; 3])).collect();
        let stencil = exec.to_device(&[1_u32, 0, 1]);
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
        let input = Permute::new(seven, Counting::new(0, 3));
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

        transform_where(&exec, input, IncrementSeven, stencil.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 2, 100, column as u32 * 10 + 4]
            );
        }
    }
}
