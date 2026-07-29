//! Reusable prefix-scan control primitives.

use cubecl::prelude::*;

use crate::core::allocation::{CopyStorage, RowStorage};
use crate::core::arity::{A13, Dispatch};
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::launch::cube_count_1d;
use crate::core::op::ReductionOp;
use crate::core::output::{
    LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
};
use crate::core::read::{
    Adjacent, Env0, Env12, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots,
    ReadExpression, StageRead,
};
use crate::core::storage::{
    Decompose, LoadMutPadded12, LoadPadded12, MutableLeaves, PlaneShuffleLeaves, Recompose, S12,
    SharedLeaves, StorageLayout, StorePadded12, StorePadded12Expand,
};
use crate::core::transform::materialize;
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;

type FixedScanStorage<R, Item> = <Item as crate::core::allocation::ScratchStorage<R>>::Storage;
type FixedScanRead<R, Item> =
    crate::core::read::FixedRead<<FixedScanStorage<R, Item> as RowStorage<R>>::Read>;
type FixedScanOutput<R, Item> = <FixedScanStorage<R, Item> as RowStorage<R>>::Write;

mod kernels;
use kernels::*;

#[doc(hidden)]
pub trait ScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
    Item: crate::core::allocation::ScratchStorage<R>,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        op: Op,
        output: &Output,
        init: Option<&FixedScanStorage<R, Item>>,
    ) -> Result<(), Error>;
}

#[doc(hidden)]
pub trait ScanPassDispatch<R, Input, Output, Partials, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run_pass(
        exec: &Executor<R>,
        input: &Input,
        output: &Output,
        partials: &Partials,
        exclusive: bool,
    ) -> Result<(), Error>;
}

macro_rules! impl_padded_scan_dispatch {
    (
        $arity:ty, $eval:ident, $kernel:ident, $env:ty;
        [$( $leaf:ident:$index:literal ),+]
    ) => {
        impl<
            R, Input, Output, Partials, Item, Op,
            O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
            $( $leaf ),+
        > ScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            $env,
            Env12<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>,
            Op,
        >
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            Item: StorageLayout + Send + Sync + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = Item::StorageLeaves>,
            Op: ReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            O0: MStorageElement,
            O1: MStorageElement,
            O2: MStorageElement,
            O3: MStorageElement,
            O4: MStorageElement,
            O5: MStorageElement,
            O6: MStorageElement,
            O7: MStorageElement,
            O8: MStorageElement,
            O9: MStorageElement,
            O10: MStorageElement,
            O11: MStorageElement,
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression<Item = Item>
                + LowerOutputExpression
                + StageOutput<R, Env0>,
            Output::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Partials: OutputExpression<Item = Item>
                + LowerOutputExpression
                + StageOutput<R, Env0>,
            Partials::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Item::StorageLeaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + LoadPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
        {
            fn run_pass(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
                partials: &Partials,
                exclusive: bool,
            ) -> Result<(), Error> {
                let len = input.physical_len()?;
                let output_len = output.physical_len()?;
                if output_len != len {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let blocks = len.div_ceil(BLOCK_SIZE as usize);
                let partial_len = partials.physical_len()?;
                if partial_len != blocks {
                    return Err(Error::LengthMismatch { left: blocks, right: partial_len });
                }
                let reads = Bindings::read(exec, input)?;
                let writes = Bindings::write(exec, output)?;
                let partial_bindings = Bindings::write(exec, partials)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let zero_values = [0u32; 12];
                let zero_offsets = exec.client().create_from_slice(u32::as_bytes(&zero_values));
                let len_handle = input.logical_extent()?.materialize(exec)?;
                // Both scan forms share the same local inclusive scan. For
                // an exclusive scan, partials[0] already holds the initial
                // row and block tails are shifted by one control entry.
                unsafe {
                    $kernel::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                        Item::StorageLeaves,
                        Item::DeviceLayout,
                        Input::DeviceExpr,
                        Op,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets.clone(), reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        BufferArg::from_raw_parts(write_offsets.clone(), writes.offsets.len()),
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
                        exclusive,
                        crate::core::launch::plane_count_bound(exec, BLOCK_SIZE),
                    );
                    if blocks > 1 {
                        padded_block_tails12::launch_unchecked::<
                            O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                            Item::StorageLeaves,
                            R,
                        >(
                            exec.client(),
                            cube_count_1d(blocks.div_ceil(BLOCK_SIZE as usize))?,
                            CubeDim::new_1d(BLOCK_SIZE),
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
                            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                            exclusive,
                            BufferArg::from_raw_parts(partial_bindings.slots[0].0.clone(), partial_bindings.slots[0].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[1].0.clone(), partial_bindings.slots[1].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[2].0.clone(), partial_bindings.slots[2].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[3].0.clone(), partial_bindings.slots[3].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[4].0.clone(), partial_bindings.slots[4].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[5].0.clone(), partial_bindings.slots[5].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[6].0.clone(), partial_bindings.slots[6].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[7].0.clone(), partial_bindings.slots[7].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[8].0.clone(), partial_bindings.slots[8].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[9].0.clone(), partial_bindings.slots[9].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[10].0.clone(), partial_bindings.slots[10].1),
                            BufferArg::from_raw_parts(partial_bindings.slots[11].0.clone(), partial_bindings.slots[11].1),
                            BufferArg::from_raw_parts(zero_offsets, 12),
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

impl_padded_scan_dispatch!(A13,Eval13,padded_scan_a13,Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]);

fn scan_pass<R, Input, Output, Partials, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
    partials: &Partials,
    exclusive: bool,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots>
        + StageOutput<R, Env0>,
    Partials: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots>
        + StageOutput<R, Env0>,
    Item: StorageLayout,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::core::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as ScanPassDispatch<
        R,
        Input,
        Output,
        Partials,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::core::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run_pass(exec, input, output, partials, exclusive)
}

fn add_fixed_prefixes<R, Output, Item, Op>(
    exec: &Executor<R>,
    prefixes: &FixedScanStorage<R, Item>,
    output: &Output,
    len: usize,
    extent: &crate::core::extent::LogicalExtent,
    exclusive: bool,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    if len == 0 {
        return Ok(());
    }
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let prefix_len = RowStorage::len(prefixes)?;
    if prefix_len != blocks {
        return Err(Error::LengthMismatch {
            left: blocks,
            right: prefix_len,
        });
    }

    let prefix_read = prefixes.read();
    let prefix_bindings = Bindings::read(exec, &prefix_read)?;
    let output_bindings = Bindings::write(exec, output)?;

    let prefix_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&prefix_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    let len_handle = extent.materialize(exec)?;

    unsafe {
        add_block_prefix_padded12::launch_unchecked::<
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
            cube_count_1d(blocks)?,
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
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            exclusive,
            BufferArg::from_raw_parts(prefix_offsets, prefix_bindings.offsets.len()),
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

fn scan_fixed_storage<R, Item, Op>(
    exec: &Executor<R>,
    input: &FixedScanStorage<R, Item>,
    output: &mut FixedScanStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::core::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    let len = RowStorage::len(input)?;
    let output_len = RowStorage::len(output)?;
    if output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: output_len,
        });
    }
    if len == 0 {
        return Ok(());
    }

    let extent = RowStorage::logical_extent(input);
    RowStorage::set_logical_extent(output, extent.clone());
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let mut partials = Item::alloc_scratch(exec, blocks);
    RowStorage::set_logical_extent(
        &mut partials,
        extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?,
    );
    let input_read = FixedScanRead::<R, Item>::new(input.read());
    let output_write = output.write();
    let partial_write = partials.write();
    scan_pass::<R, _, _, _, Item, Op>(exec, &input_read, &output_write, &partial_write, false)?;

    if blocks > 1 {
        let mut prefixes = Item::alloc_scratch(exec, blocks);
        scan_fixed_storage::<R, Item, Op>(exec, &partials, &mut prefixes)?;
        add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, &output_write, len, &extent, false)?;
    }
    Ok(())
}

impl<R, Input, Output, Item, Op, ReadSlots, WriteSlots>
    ScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op> for Dispatch<A13, S12>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        > + ScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::core::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        _op: Op,
        output: &Output,
        init: Option<&FixedScanStorage<R, Item>>,
    ) -> Result<(), Error> {
        let len = input.physical_len()?;
        let output_len = output.physical_len()?;
        if output_len != len {
            return Err(Error::LengthMismatch {
                left: len,
                right: output_len,
            });
        }
        if len == 0 {
            return Ok(());
        }

        let extent = input.logical_extent()?;
        let blocks = len.div_ceil(BLOCK_SIZE as usize);
        let mut partials = Item::alloc_scratch(exec, blocks);
        RowStorage::set_logical_extent(
            &mut partials,
            extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?,
        );
        let exclusive = init.is_some();
        if let Some(initial) = init {
            initial.copy_storage(exec, partials.slice_mut(..1))?;
        }
        let partial_write = partials.write();
        <Dispatch<A13, S12> as ScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        >>::run_pass(exec, input, output, &partial_write, exclusive)?;

        if blocks > 1 {
            let mut prefixes = Item::alloc_scratch(exec, blocks);
            scan_fixed_storage::<R, Item, Op>(exec, &partials, &mut prefixes)?;
            add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, output, len, &extent, exclusive)?;
        } else if exclusive {
            add_fixed_prefixes::<R, _, Item, Op>(exec, &partials, output, len, &extent, true)?;
        }
        Ok(())
    }
}

/// Computes an inclusive scan into preallocated output storage.
pub(crate) fn inclusive_scan<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Op: ReductionOp<Item>,
    Item: crate::core::allocation::ScratchStorage<R>,
    Output: OutputExpression<Item = Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
    Dispatch<A13, S12>: ScanDispatch<
            R,
            Input,
            Output,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::core::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as ScanDispatch<
        R,
        Input,
        Output,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::core::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run(exec, &input, op, &output, None)
}

/// Computes adjacent reductions while preserving the first input item.
pub(crate) fn adjacent_difference<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Output::Item>,
    Op: ReductionOp<Output::Item>,
    Adjacent<Input, Op>:
        ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
{
    materialize(exec, Adjacent::new(input, op), output)
}

/// Computes an exclusive scan into preallocated output storage.
pub(crate) fn exclusive_scan<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    init: FixedScanStorage<R, Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Output: OutputExpression<Item = Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
    Dispatch<A13, S12>: ScanDispatch<
            R,
            Input,
            Output,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::core::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as ScanDispatch<
        R,
        Input,
        Output,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::core::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run(exec, &input, op, &output, Some(&init))
}

/// Computes independent inclusive scans for each fixed-size block.
pub(crate) fn local_inclusive_scan_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    if input.capacity() == 0 {
        let mut output = exec.alloc_row::<u32>(0);
        output.set_logical_extent(input.logical_extent());
        return Ok(output);
    }
    let len = input.capacity();
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let extent = input.logical_extent();
    let mut output = exec.alloc_row::<u32>(len);
    output.set_logical_extent(extent.clone());
    let len_handle = extent.materialize(exec)?;
    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            exec.client(),
            cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), len),
        );
    }
    Ok(output)
}

/// Extracts one reduction value per block from a block-local scan.
pub(crate) fn block_tails_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    let len = input.capacity();
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let extent = input.logical_extent();
    let mut block_tails = exec.alloc_row::<u32>(blocks);
    block_tails.set_logical_extent(extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?);
    if blocks == 0 {
        return Ok(block_tails);
    }
    let len_handle = extent.materialize(exec)?;
    unsafe {
        u32_block_tails_kernel::launch_unchecked::<R>(
            exec.client(),
            cube_count_1d(blocks.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(block_tails.handle.clone(), blocks),
        );
    }
    Ok(block_tails)
}

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    let len = input.capacity();
    if len == 0 {
        return local_inclusive_scan_u32(exec, input);
    }
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let extent = input.logical_extent();
    let output = local_inclusive_scan_u32(exec, input)?;
    if blocks > 1 {
        let block_sums = block_tails_u32(exec, &output)?;
        let prefixes = inclusive_scan_u32(exec, &block_sums)?;
        let len_handle = extent.materialize(exec)?;
        unsafe {
            u32_add_block_prefix_kernel::launch_unchecked::<R>(
                exec.client(),
                cube_count_1d(blocks)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(prefixes.handle.clone(), blocks),
                BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
    }
    Ok(output)
}

pub(crate) fn last_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    let output = exec.alloc_row::<u32>(1);
    let len = input.logical_extent().materialize(exec)?;
    unsafe {
        copy_last_kernel::launch_unchecked::<R>(
            exec.client(),
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input.handle.clone(), input.capacity()),
            BufferArg::from_raw_parts(len.handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), 1),
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::allocation::RowStorage;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute, Transform};
    use crate::op::UnaryOp;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn inclusive_u32_scan_crosses_block_and_recursive_boundaries() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&vec![1_u32; 70_001]);
        let output = inclusive_scan_u32(&exec, &input).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 1);
        assert_eq!(actual[255], 256);
        assert_eq!(actual[256], 257);
        assert_eq!(actual[65_535], 65_536);
        assert_eq!(actual[70_000], 70_001);
        assert_eq!(
            exec.to_host(&last_u32(&exec, &output).unwrap()).unwrap(),
            vec![70_001]
        );
    }

    #[test]
    fn local_u32_scan_and_block_tails_are_independent_controls() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&vec![1_u32; 300]);

        let local = local_inclusive_scan_u32(&exec, &input).unwrap();
        let actual = exec.to_host(&local).unwrap();
        assert_eq!(actual[0], 1);
        assert_eq!(actual[255], 256);
        assert_eq!(actual[256], 1);
        assert_eq!(actual[299], 44);

        let tails = block_tails_u32(&exec, &local).unwrap();
        assert_eq!(exec.to_host(&tails).unwrap(), vec![256, 44]);
    }

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    #[test]
    fn generated_scan_stages_fit_the_binding_budget() {
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

        let local_scan = padded_scan_a13::PaddedScanA13::<
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
            ScalarLayout,
            ScalarExpr,
            Sum,
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
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            arg.clone(),
            true,
            crate::core::launch::plane_count_bound(&exec, BLOCK_SIZE),
        );
        crate::core::launch::assert_binding_budget("local scan", &local_scan);

        let add_prefix = add_block_prefix_padded12::AddBlockPrefixPadded12::<
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
            arg,
            false,
        );
        crate::core::launch::assert_binding_budget("block prefix propagation", &add_prefix);
    }

    struct TakeLeft;

    #[cubecl::cube]
    impl ReductionOp<u32> for TakeLeft {
        fn apply(lhs: u32, _rhs: u32) -> u32 {
            lhs
        }
    }

    struct SumPair;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for SumPair {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    struct ComposeAffine;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for ComposeAffine {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (
                (rhs.0 * lhs.0) % 65_521u32,
                (rhs.0 * lhs.1 + rhs.1) % 65_521u32,
            )
        }
    }

    #[test]
    fn block_reduction_preserves_non_commutative_scan_order() {
        const MODULUS: u32 = 65_521;
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 600usize;
        let multipliers: Vec<_> = (0..len).map(|index| index as u32 % 5 + 1).collect();
        let offsets: Vec<_> = (0..len).map(|index| index as u32 * 7 % 13).collect();
        let multiplier_input = exec.to_device(&multipliers);
        let offset_input = exec.to_device(&offsets);
        let output = exec.alloc_row::<(u32, u32)>(len);

        inclusive_scan(
            &exec,
            Zip::new(multiplier_input.column(), offset_input.column()),
            ComposeAffine,
            output.write(),
        )
        .unwrap();

        let actual_multipliers = exec.to_host(&output.0).unwrap();
        let actual_offsets = exec.to_host(&output.1).unwrap();
        let mut expected = (1u32, 0u32);
        for index in 0..len {
            let rhs = (multipliers[index], offsets[index]);
            expected = (
                (rhs.0 * expected.0) % MODULUS,
                (rhs.0 * expected.1 + rhs.1) % MODULUS,
            );
            assert_eq!((actual_multipliers[index], actual_offsets[index]), expected,);
        }
    }

    #[test]
    fn split_scan_preserves_in_place_input_before_block_reduction() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&vec![1u32; 600]);

        inclusive_scan(&exec, values.column(), Sum, values.slice_mut(..)).unwrap();

        let actual = exec.to_host(&values).unwrap();
        assert_eq!(actual[255], 256);
        assert_eq!(actual[256], 257);
        assert_eq!(actual[599], 600);
    }

    #[test]
    fn exclusive_scan_preserves_in_place_rows_and_slice_sentinels() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        for len in [1usize, 255, 256, 257, 1_025, 70_001] {
            let input: Vec<_> = (0..len).map(|i| i as u32 % 17 + 1).collect();
            let mut padded = vec![91u32];
            padded.extend(&input);
            padded.push(93);
            let values = exec.to_device(&padded);
            exclusive_scan(
                &exec,
                values.slice_usize(1..len + 1),
                exec.to_device(&[19u32]),
                Sum,
                values.slice_mut_usize(1..len + 1),
            )
            .unwrap();
            let mut sum = 19u32;
            for (destination, value) in padded[1..len + 1].iter_mut().zip(input) {
                *destination = sum;
                sum += value;
            }
            assert_eq!(exec.to_host(&values).unwrap(), padded, "length {len}");
        }
    }

    #[test]
    fn inclusive_pair_scan_crosses_recursive_block_boundary() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 70_001;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let output = exec.alloc_row::<(u32, u32)>(len);

        inclusive_scan(
            &exec,
            Zip::new(left.column(), right.column()),
            SumPair,
            output.write(),
        )
        .unwrap();

        let actual_left = exec.to_host(&output.0).unwrap();
        let actual_right = exec.to_host(&output.1).unwrap();
        for &index in &[0, 255, 256, 65_535, 70_000] {
            assert_eq!(actual_left[index], index as u32 + 1);
            assert_eq!(actual_right[index], 2 * (index as u32 + 1));
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct SumSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for SumSeven {
        type Output = u32;
        fn apply(input: Seven) -> u32 {
            input.0 + input.1 + input.2 + input.3 + input.4 + input.5 + input.6
        }
    }

    struct SumSevenItems;

    #[cubecl::cube]
    impl ReductionOp<Seven> for SumSevenItems {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                lhs.1 + rhs.1,
                lhs.2 + rhs.2,
                lhs.3 + rhs.3,
                lhs.4 + rhs.4,
                lhs.5 + rhs.5,
                lhs.6 + rhs.6,
            )
        }
    }

    #[test]
    fn inclusive_storage7_scan_accepts_eight_read_slots() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; 600]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 600));
        let output = exec.alloc_row::<Seven>(600);

        inclusive_scan(&exec, input, SumSevenItems, output.write()).unwrap();

        let (a, b, c, d, e, f, g) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&a).unwrap()[599], 600);
        assert_eq!(exec.to_host(&b).unwrap()[599], 1_200);
        assert_eq!(exec.to_host(&c).unwrap()[599], 1_800);
        assert_eq!(exec.to_host(&d).unwrap()[599], 2_400);
        assert_eq!(exec.to_host(&e).unwrap()[599], 3_000);
        assert_eq!(exec.to_host(&f).unwrap()[599], 3_600);
        assert_eq!(exec.to_host(&g).unwrap()[599], 4_200);
    }

    #[test]
    fn inclusive_scalar_scan_uses_padded_fixed_evaluator() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0..7).map(|_| exec.to_device(&[1_u32; 600])).collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Transform::new(Permute::new(seven, Counting::new(0, 600)), SumSeven);
        let output = exec.to_device(&[0_u32; 600]);
        inclusive_scan(&exec, input, Sum, output.slice_mut(..)).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 7);
        assert_eq!(actual[255], 7 * 256);
        assert_eq!(actual[256], 7 * 257);
        assert_eq!(actual[599], 7 * 600);
    }

    #[test]
    fn exclusive_scalar_scan_applies_init_once() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        let output = exec.to_device(&[99_u32; 6]);
        let init = crate::api::value::store(&exec, 10_u32).unwrap();
        exclusive_scan(
            &exec,
            input.column(),
            crate::api::value::into_scratch::<WgpuRuntime, u32>(init),
            Sum,
            output.slice_mut_usize(1..5),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 10, 11, 13, 16, 99]);
    }

    #[test]
    fn exclusive_scan_crosses_recursive_boundaries_in_operand_order() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 70_001usize;
        let input = exec.to_device(&vec![1_u32; len]);
        let output = exec.to_device(&vec![0_u32; len]);
        let init = crate::api::value::store(&exec, 10_u32).unwrap();
        exclusive_scan(
            &exec,
            input.column(),
            crate::api::value::into_scratch::<WgpuRuntime, u32>(init),
            Sum,
            output.slice_mut_usize(..),
        )
        .unwrap();
        let actual = exec.to_host(&output).unwrap();
        for &index in &[0, 255, 256, 65_535, 70_000] {
            assert_eq!(actual[index], 10 + index as u32);
        }

        let ordered = exec.to_device(&vec![0_u32; 600]);
        let init = crate::api::value::store(&exec, 42_u32).unwrap();
        exclusive_scan(
            &exec,
            input.slice_usize(..600),
            crate::api::value::into_scratch::<WgpuRuntime, u32>(init),
            TakeLeft,
            ordered.slice_mut_usize(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&ordered).unwrap(), vec![42; 600]);
    }

    #[test]
    fn exclusive_scan_preserves_non_commutative_block_totals_in_order() {
        const MODULUS: u32 = 65_521;
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 600usize;
        let multipliers: Vec<_> = (0..len).map(|index| index as u32 % 5 + 1).collect();
        let offsets: Vec<_> = (0..len).map(|index| index as u32 * 7 % 13).collect();
        let multiplier_input = exec.to_device(&multipliers);
        let offset_input = exec.to_device(&offsets);
        let output = exec.alloc_row::<(u32, u32)>(len);
        let init_value = (3u32, 5u32);
        let init = crate::api::value::store(&exec, init_value).unwrap();

        exclusive_scan(
            &exec,
            Zip::new(multiplier_input.column(), offset_input.column()),
            crate::api::value::into_scratch::<WgpuRuntime, (u32, u32)>(init),
            ComposeAffine,
            output.write(),
        )
        .unwrap();

        let actual_multipliers = exec.to_host(&output.0).unwrap();
        let actual_offsets = exec.to_host(&output.1).unwrap();
        let mut expected = init_value;
        for index in 0..len {
            assert_eq!((actual_multipliers[index], actual_offsets[index]), expected);
            let rhs = (multipliers[index], offsets[index]);
            expected = (
                (rhs.0 * expected.0) % MODULUS,
                (rhs.0 * expected.1 + rhs.1) % MODULUS,
            );
        }
    }

    #[test]
    fn adjacent_difference_is_a_regular_fused_read_expression() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 3, 6, 10]);
        let output = exec.to_device(&[0_u32; 4]);
        adjacent_difference(&exec, input.column(), Sum, output.slice_mut(..)).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 4, 9, 16]);
    }

    #[test]
    fn exclusive_storage7_accepts_eight_read_slots_and_preserves_semantic_init() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&[value; 4]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 4));
        let output = exec.alloc_row::<Seven>(4);
        let init: Seven = (10, 20, 30, 40, 50, 60, 70);
        let init = crate::api::value::store(&exec, init).unwrap();
        exclusive_scan(
            &exec,
            input,
            crate::api::value::into_scratch::<WgpuRuntime, Seven>(init),
            SumSevenItems,
            output.write(),
        )
        .unwrap();

        let (first, _, _, _, _, _, last) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&first).unwrap(), vec![10, 11, 12, 13]);
        assert_eq!(exec.to_host(&last).unwrap(), vec![70, 77, 84, 91]);
    }

    #[test]
    fn scan_rejects_mismatched_output_tree_and_foreign_storage() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[4_u32, 5, 6]);
        let out_left = exec.to_device(&[0_u32; 3]);
        let out_right = exec.to_device(&[0_u32; 2]);
        assert_eq!(
            inclusive_scan(
                &exec,
                Zip::new(left.column(), right.column()),
                SumPair,
                Zip::new(out_left.slice_mut(..), out_right.slice_mut(..)),
            ),
            Err(Error::LengthMismatch { left: 3, right: 2 })
        );

        let other = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let foreign_output = other.to_device(&[0_u32; 3]);
        assert_eq!(
            inclusive_scan(&exec, left.column(), Sum, foreign_output.slice_mut(..)),
            Err(Error::ForeignExecutor)
        );
    }
}
