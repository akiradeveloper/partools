//! Read-arity and storage-arity indexed reductions.

#![allow(private_interfaces)]

use cubecl::prelude::*;

use crate::core::allocation::RowStorage;
use crate::core::arity::{A13, Dispatch};
use crate::core::bindings::Bindings;
use crate::core::eval::Eval13;
use crate::core::launch::cube_count_1d;
use crate::core::op::ReductionOp;
use crate::core::read::{
    Env0, Env12, Env13, LowerReadExpression, PaddedReadSlots, ReadExpression, StageRead,
};
use crate::core::storage::{
    Decompose, LoadMutPadded12, MutableLeaves, MutableLeavesExpand, PlaneShuffleLeaves, Recompose,
    S12, SharedLeaves, SharedLeavesExpand, StorageLayout, StorePadded12, StorePadded12Expand,
};
use crate::core::value::MStorageElement;
use crate::{Error, Executor};

type FixedReduceStorage<R, Item> = <Item as crate::core::allocation::ScratchStorage<R>>::Storage;
type FixedReduceRead<R, Item> = crate::core::read::FixedRead<
    <FixedReduceStorage<R, Item> as crate::core::allocation::RowStorage<R>>::Read,
>;
type FixedReduceOutput<R, Item> =
    <FixedReduceStorage<R, Item> as crate::core::allocation::RowStorage<R>>::Write;

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_UNIT: usize = 32;
const TILE_SIZE: usize = BLOCK_SIZE as usize * ITEMS_PER_UNIT;

#[cubecl::cube]
fn accumulate_register<Item, Leaves, Layout, Op>(cells: &Leaves::Cells, value: Item)
where
    Item: CubeType,
    Leaves: MutableLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let accumulated = Op::apply(Layout::recompose(Leaves::read(cells)), value);
    Leaves::store(cells, Layout::decompose(accumulated));
}

#[cubecl::cube]
fn reduce_plane_full<Item, Leaves, Layout, Op>(value: Item) -> Leaves
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        if UNIT_POS_PLANE < offset.read() {
            let combined = Op::apply(
                Layout::recompose(Leaves::read(&cells)),
                Layout::recompose(right),
            );
            Leaves::store(&cells, Layout::decompose(combined));
        }
        offset.store(offset.read() / 2u32);
    }
    Leaves::read(&cells)
}

#[cubecl::cube]
fn reduce_plane_valid<Item, Leaves, Layout, Op>(value: Item, valid: u32) -> (Leaves, u32)
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let cell_valid = RuntimeCell::<u32>::new(valid);
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        let right_cells = right.into_cells();
        let right_valid = plane_shuffle_down(cell_valid.read(), offset.read());
        if UNIT_POS_PLANE < offset.read() && right_valid != 0u32 {
            if cell_valid.read() != 0u32 {
                let combined = Op::apply(
                    Layout::recompose(Leaves::read(&cells)),
                    Layout::recompose(Leaves::read(&right_cells)),
                );
                Leaves::store(&cells, Layout::decompose(combined));
            } else {
                Leaves::store(&cells, Leaves::read(&right_cells));
                cell_valid.store(1u32);
            }
        }
        offset.store(offset.read() / 2u32);
    }
    (Leaves::read(&cells), cell_valid.read())
}

#[cubecl::cube]
fn combine_plane_results<Item, Leaves, Layout, Op>(
    value: Leaves,
    value_valid: u32,
    #[comptime] plane_capacity: usize,
) -> (Leaves, u32)
where
    Item: CubeType + Send + Sync + 'static,
    Leaves: SharedLeaves + MutableLeaves + PlaneShuffleLeaves + Send + Sync + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let mut shared = Leaves::new_shared(plane_capacity);
    let mut plane_valid = Shared::<[u32]>::new_slice(plane_capacity);
    let result = value.into_cells();
    let result_valid = RuntimeCell::<u32>::new(0u32);
    if UNIT_POS_PLANE == 0u32 {
        Leaves::read(&result).store_shared(&mut shared, PLANE_POS as usize);
        plane_valid[PLANE_POS as usize] = value_valid;
    }
    sync_cube();
    if PLANE_POS == 0u32 {
        let plane_count = CUBE_DIM.div_ceil(PLANE_DIM);
        let source = if UNIT_POS_PLANE < plane_count {
            UNIT_POS_PLANE as usize
        } else {
            0usize
        };
        let source_valid = if UNIT_POS_PLANE < plane_count {
            plane_valid[source]
        } else {
            0u32
        };
        let accumulator =
            Layout::decompose(Layout::recompose(Leaves::load_shared(&shared, source))).into_cells();
        let accumulator_valid = RuntimeCell::<u32>::new(source_valid);
        let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
        while cursor.read() < plane_count {
            let index = cursor.read() as usize;
            if plane_valid[index] != 0u32 {
                let next = Leaves::load_shared(&shared, index).into_cells();
                if accumulator_valid.read() != 0u32 {
                    accumulate_register::<Item, Leaves, Layout, Op>(
                        &accumulator,
                        Layout::recompose(Leaves::read(&next)),
                    );
                } else {
                    Leaves::store(&accumulator, Leaves::read(&next));
                    accumulator_valid.store(1u32);
                }
            }
            cursor.store(cursor.read() + PLANE_DIM);
        }
        let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
            Layout::recompose(Leaves::read(&accumulator)),
            accumulator_valid.read(),
        );
        if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
            Leaves::store(&result, block_result.0);
            result_valid.store(1u32);
        }
    }
    (Leaves::read(&result), result_valid.read())
}

#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
fn finish_reduce_value_padded12<
    Item,
    O0,
    O1,
    O2,
    O3,
    O4,
    O5,
    O6,
    O7,
    O8,
    O9,
    O10,
    O11,
    Leaves,
    Layout,
    Op,
>(
    value: Leaves,
    value_valid: u32,
    #[comptime] with_init: bool,
    zero_offsets: &[u32],
    partial0: &mut [O0],
    partial1: &mut [O1],
    partial2: &mut [O2],
    partial3: &mut [O3],
    partial4: &mut [O4],
    partial5: &mut [O5],
    partial6: &mut [O6],
    partial7: &mut [O7],
    partial8: &mut [O8],
    partial9: &mut [O9],
    partial10: &mut [O10],
    partial11: &mut [O11],
    #[comptime] plane_capacity: usize,
) where
    Item: CubeType + Send + Sync + 'static,
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
    Leaves: SharedLeaves
        + MutableLeaves
        + PlaneShuffleLeaves
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
        > + LoadMutPadded12<
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
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let block_result =
        combine_plane_results::<Item, Leaves, Layout, Op>(value, value_valid, plane_capacity);
    if block_result.1 != 0u32 {
        let result = Leaves::into_cells(block_result.0);
        if with_init {
            let initial = Leaves::load_mut_padded(
                partial0,
                partial1,
                partial2,
                partial3,
                partial4,
                partial5,
                partial6,
                partial7,
                partial8,
                partial9,
                partial10,
                partial11,
                zero_offsets,
                0usize,
            );
            Leaves::store(
                &result,
                Layout::decompose(Op::apply(
                    Layout::recompose(initial),
                    Layout::recompose(Leaves::read(&result)),
                )),
            );
        }
        Leaves::read(&result).store_padded(
            partial0,
            partial1,
            partial2,
            partial3,
            partial4,
            partial5,
            partial6,
            partial7,
            partial8,
            partial9,
            partial10,
            partial11,
            zero_offsets,
            CUBE_POS as usize,
        );
    }
}

macro_rules! define_padded_reduce_eval_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            O0: CubePrimitive + cubecl::frontend::Scalar, O1: CubePrimitive + cubecl::frontend::Scalar, O2: CubePrimitive + cubecl::frontend::Scalar, O3: CubePrimitive + cubecl::frontend::Scalar,
            O4: CubePrimitive + cubecl::frontend::Scalar, O5: CubePrimitive + cubecl::frontend::Scalar, O6: CubePrimitive + cubecl::frontend::Scalar, O7: CubePrimitive + cubecl::frontend::Scalar,
            O8: CubePrimitive + cubecl::frontend::Scalar, O9: CubePrimitive + cubecl::frontend::Scalar, O10: CubePrimitive + cubecl::frontend::Scalar, O11: CubePrimitive + cubecl::frontend::Scalar,
            Leaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >                + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            #[comptime] with_init: bool,
            zero_offsets: &[u32],
            partial0: &mut [O0], partial1: &mut [O1], partial2: &mut [O2],
            partial3: &mut [O3], partial4: &mut [O4], partial5: &mut [O5],
            partial6: &mut [O6], partial7: &mut [O7], partial8: &mut [O8],
            partial9: &mut [O9], partial10: &mut [O10], partial11: &mut [O11],
            #[comptime] plane_capacity: usize,
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            if CUBE_POS as usize >= crate::core::launch::logical_block_count(logical_len, TILE_SIZE) {
                terminate!();
            }
            let tile_start = (CUBE_POS as usize) * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = usize::min(first_index, logical_len - 1usize);
            let accumulator = Layout::decompose(
                Expr::$method($( $slot, )+ read_offsets, safe_index),
            ).into_cells();

            if logical_len - tile_start >= TILE_SIZE {
                for item in 1usize..ITEMS_PER_UNIT {
                    let value = Expr::$method(
                        $( $slot, )+ read_offsets, first_index + item * cube_dim,
                    );
                    accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                }
                let result = reduce_plane_full::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                );
                finish_reduce_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                    result, 1u32, with_init, zero_offsets,
                    partial0, partial1, partial2, partial3, partial4, partial5,
                    partial6, partial7, partial8, partial9, partial10, partial11, plane_capacity,
                );
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let index = first_index + item * cube_dim;
                    if index < logical_len {
                        let value = Expr::$method($( $slot, )+ read_offsets, index);
                        accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                    }
                }
                let result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                    if first_index < logical_len { 1u32 } else { 0u32 },
                );
                finish_reduce_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                    result.0, result.1, with_init, zero_offsets,
                    partial0, partial1, partial2, partial3, partial4, partial5,
                    partial6, partial7, partial8, partial9, partial10, partial11, plane_capacity,
                );
            }
        }
    };
}

define_padded_reduce_eval_kernel!(padded_reduce_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

fn pass_block_count(len: usize) -> usize {
    len.div_ceil(TILE_SIZE).max(1)
}

/// Consumer-specific reduction dispatch.
#[doc(hidden)]
pub(crate) trait ReduceDispatch<R: Runtime, Input, Item, Op, Slots> {
    type Storage;

    fn execute(
        exec: &Executor<R>,
        input: &Input,
        init: Self::Storage,
    ) -> Result<Self::Storage, Error>;
}

/// One fixed-ABI reduction pass. The output contains one partial per tile.
#[doc(hidden)]
pub trait ReducePassDispatch<R, Input, Output, Item, Op, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn execute_pass(
        exec: &Executor<R>,
        input: &Input,
        output: &Output,
        with_init: bool,
    ) -> Result<(), Error>;
}

macro_rules! impl_padded_reduce_pass_dispatch {
    ($arity:ty,$eval:ident,$kernel:ident,$read_env:ty,$write_env:ty; [$( $leaf:ident:$index:literal ),+]) => {
        impl<R, Input, Output, Item, Op, $( $leaf, )+ O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>
            ReducePassDispatch<R, Input, Output, Item, Op, $read_env, $write_env>
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            Item: StorageLayout + Send + Sync + 'static,
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
            Op: ReductionOp<Item>,
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: crate::core::output::OutputExpression<Item = Item>
                + crate::core::output::LowerOutputExpression
                + crate::core::output::StageOutput<R, Env0>,
            Output::Slots: crate::core::output::PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Item::StorageLeaves: StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                > + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                > + SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = Item::StorageLeaves>
                + Recompose<Item, Leaves = Item::StorageLeaves>,
        {
            fn execute_pass(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
                with_init: bool,
            ) -> Result<(), Error> {
                let len = input.physical_len()?;
                debug_assert!(len != 0);
                let blocks = pass_block_count(len);
                let bindings = Bindings::read(exec, input)?;
                let output_bindings = Bindings::write(exec, output)?;
                let offsets = exec.client().create_from_slice(u32::as_bytes(&bindings.offsets));
                let zero_values = [0u32; 12];
                let zero_offsets = exec.client().create_from_slice(u32::as_bytes(&zero_values));
                let len_handle = input.logical_extent()?.materialize(exec)?;
                unsafe {
                    $kernel::launch_unchecked::<
                        Item, $( $leaf, )+
                        O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                        Item::StorageLeaves, Item::DeviceLayout, Input::DeviceExpr, Op, R,
                    >(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(bindings.slots[$index].0.clone(), bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        with_init,
                        BufferArg::from_raw_parts(zero_offsets, 12),
                        BufferArg::from_raw_parts(output_bindings.slots[0].0.clone(), output_bindings.slots[0].1),
                        BufferArg::from_raw_parts(output_bindings.slots[1].0.clone(), output_bindings.slots[1].1),
                        BufferArg::from_raw_parts(output_bindings.slots[2].0.clone(), output_bindings.slots[2].1),
                        BufferArg::from_raw_parts(output_bindings.slots[3].0.clone(), output_bindings.slots[3].1),
                        BufferArg::from_raw_parts(output_bindings.slots[4].0.clone(), output_bindings.slots[4].1),
                        BufferArg::from_raw_parts(output_bindings.slots[5].0.clone(), output_bindings.slots[5].1),
                        BufferArg::from_raw_parts(output_bindings.slots[6].0.clone(), output_bindings.slots[6].1),
                        BufferArg::from_raw_parts(output_bindings.slots[7].0.clone(), output_bindings.slots[7].1),
                        BufferArg::from_raw_parts(output_bindings.slots[8].0.clone(), output_bindings.slots[8].1),
                        BufferArg::from_raw_parts(output_bindings.slots[9].0.clone(), output_bindings.slots[9].1),
                        BufferArg::from_raw_parts(output_bindings.slots[10].0.clone(), output_bindings.slots[10].1),
                        BufferArg::from_raw_parts(output_bindings.slots[11].0.clone(), output_bindings.slots[11].1),
                        crate::core::launch::plane_count_bound(exec, BLOCK_SIZE),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_padded_reduce_pass_dispatch!(
    A13,
    Eval13,
    padded_reduce_a13,
    Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>,
    Env12<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11>;
    [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]
);

fn reduce_pass<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
    with_init: bool,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: crate::core::read::PaddedReadSlots>
        + StageRead<R, Env0>,
    Output: crate::core::output::OutputExpression<Item = Item>
        + crate::core::output::LowerOutputExpression<Slots: crate::core::output::PaddedOutputSlots>
        + crate::core::output::StageOutput<R, Env0>,
    Item: StorageLayout,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            Input,
            Output,
            Item,
            Op,
            crate::core::read::KernelReadSlots<Input::Slots>,
            crate::core::output::KernelOutputSlots<Output::Slots>,
        >,
{
    <Dispatch<A13, S12> as ReducePassDispatch<
        R,
        Input,
        Output,
        Item,
        Op,
        crate::core::read::KernelReadSlots<Input::Slots>,
        crate::core::output::KernelOutputSlots<Output::Slots>,
    >>::execute_pass(exec, input, output, with_init)
}

fn finish_fixed_reduce<R, Item, Op>(
    exec: &Executor<R>,
    mut current: FixedReduceStorage<R, Item>,
    mut current_len: usize,
    init: FixedReduceStorage<R, Item>,
) -> Result<FixedReduceStorage<R, Item>, Error>
where
    R: Runtime,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            FixedReduceRead<R, Item>,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            crate::core::read::KernelReadSlots<
                <FixedReduceRead<R, Item> as LowerReadExpression>::Slots,
            >,
            crate::core::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::core::output::LowerOutputExpression>::Slots,
            >,
        >,
{
    loop {
        let next_len = pass_block_count(current_len);
        let final_pass = next_len == 1;
        let mut next = if final_pass {
            init.clone()
        } else {
            Item::alloc_scratch(exec, next_len)
        };
        if !final_pass {
            let extent = RowStorage::logical_extent(&current);
            RowStorage::set_logical_extent(&mut next, extent.ceil_div(exec, TILE_SIZE, next_len)?);
        }
        let input = FixedReduceRead::<R, Item>::new(current.read());
        reduce_pass::<R, _, _, Item, Op>(exec, &input, &next.write(), final_pass)?;
        if final_pass {
            return Ok(init);
        }
        current = next;
        current_len = next_len;
    }
}

impl<R, Input, Item, Op, Slots> ReduceDispatch<R, Input, Item, Op, Slots> for Dispatch<A13, S12>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            Input,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            Slots,
            crate::core::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::core::output::LowerOutputExpression>::Slots,
            >,
        > + ReducePassDispatch<
            R,
            FixedReduceRead<R, Item>,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            crate::core::read::KernelReadSlots<
                <FixedReduceRead<R, Item> as LowerReadExpression>::Slots,
            >,
            crate::core::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::core::output::LowerOutputExpression>::Slots,
            >,
        >,
{
    type Storage = FixedReduceStorage<R, Item>;

    fn execute(
        exec: &Executor<R>,
        input: &Input,
        init: Self::Storage,
    ) -> Result<Self::Storage, Error> {
        let len = input.physical_len()?;
        if len == 0 {
            return Ok(init);
        }
        let extent = input.logical_extent()?;
        let blocks = pass_block_count(len);
        let final_pass = blocks == 1;
        let mut partials = if final_pass {
            init.clone()
        } else {
            Item::alloc_scratch(exec, blocks)
        };
        if !final_pass {
            RowStorage::set_logical_extent(
                &mut partials,
                extent.ceil_div(exec, TILE_SIZE, blocks)?,
            );
        }
        let output = partials.write();
        <Dispatch<A13, S12> as ReducePassDispatch<
            R,
            Input,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            Slots,
            crate::core::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::core::output::LowerOutputExpression>::Slots,
            >,
        >>::execute_pass(exec, input, &output, final_pass)?;
        if final_pass {
            return Ok(init);
        }
        finish_fixed_reduce::<R, Item, Op>(exec, partials, blocks, init)
    }
}

/// Reduces all input items, starting from `init`.
pub(crate) fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::core::read::KernelReadSlots<Input::Slots>,
    >>::Storage,
    _op: Op,
) -> Result<
    <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::core::read::KernelReadSlots<Input::Slots>,
    >>::Storage,
    Error,
>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
    Dispatch<A13, S12>:
        ReduceDispatch<R, Input, Input::Item, Op, crate::core::read::KernelReadSlots<Input::Slots>>,
{
    <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::core::read::KernelReadSlots<Input::Slots>,
    >>::execute(exec, &input, init)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use crate::core::op::UnaryOp;
    use crate::core::read::{Column, Counting, Permute, Transform};
    use crate::op::Identity;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct Double;

    #[cubecl::cube]
    impl UnaryOp<u32> for Double {
        type Output = u32;

        fn apply(input: u32) -> u32 {
            input * 2
        }
    }

    struct AddPair;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for AddPair {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    struct AddTriple;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32, u32)> for AddTriple {
        type Output = u32;

        fn apply(input: (u32, u32, u32)) -> u32 {
            input.0 + input.1 + input.2
        }
    }

    struct AddFour;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32, u32, u32)> for AddFour {
        type Output = u32;

        fn apply(input: (u32, u32, u32, u32)) -> u32 {
            input.0 + input.1 + input.2 + input.3
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct AddSeven;

    #[cubecl::cube]
    impl ReductionOp<Seven> for AddSeven {
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

    fn executor() -> Executor<WgpuRuntime> {
        Executor::new(WgpuDevice::DefaultDevice)
    }

    #[test]
    fn generated_reduce_pass_fits_the_binding_budget() {
        type ScalarLeaves = <u32 as StorageLayout>::StorageLeaves;
        type ScalarExpr = <Column<u32> as LowerReadExpression>::DeviceExpr;
        type ScalarLayout = <u32 as StorageLayout>::DeviceLayout;
        type Kernel = padded_reduce_a13::PaddedReduceA13<
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
        >;

        let exec = executor();
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
            arg,
            true,
            crate::core::launch::plane_count_bound(&exec, BLOCK_SIZE)
        );
        crate::core::launch::assert_binding_budget("reduce pass", &kernel);
    }

    #[test]
    fn one_read_slot_storage1_fuses_column_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let values = exec.to_device(&vec![1_u32; len]);
        let input = Transform::new(values.column(), Double);

        let output = reduce(&exec, input, exec.to_device(&[7]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 7 + 2 * len as u32);
    }

    #[test]
    fn two_read_slots_storage1_fuse_binary_zip_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);

        let output = reduce(&exec, input, exec.to_device(&[0]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 3 * len as u32);
    }

    #[test]
    fn three_read_slots_storage1_use_flat_zip_semantics() {
        let exec = executor();
        let len = 4097;
        let first = exec.to_device(&vec![1_u32; len]);
        let second = exec.to_device(&vec![2_u32; len]);
        let third = exec.to_device(&vec![3_u32; len]);
        let input = Transform::new(
            Zip::new(Zip::new(first.column(), second.column()), third.column()),
            AddTriple,
        );

        let output = reduce(&exec, input, exec.to_device(&[0]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 6 * len as u32);
    }

    #[test]
    fn empty_reduce_returns_init_without_launching_or_staging() {
        let exec = executor();
        let input = Transform::new(Column::<u32>::new(), Identity);
        let output = reduce(&exec, input, exec.to_device(&[42]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![42]);
    }

    #[test]
    fn zip_length_mismatch_is_rejected_before_launch() {
        let exec = executor();
        let left = exec.to_device(&[1_u32, 2]);
        let right = exec.to_device(&[3_u32]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);
        assert!(matches!(
            reduce(&exec, input, exec.to_device(&[0]), Sum),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        ));
    }

    #[test]
    fn four_read_slots_storage1_use_the_fixed_evaluator_path() {
        let exec = executor();
        let columns: Vec<_> = (1_u32..=4)
            .map(|value| exec.to_device(&vec![value; 513]))
            .collect();
        let input = Transform::new(
            Zip::new(
                Zip::new(
                    Zip::new(columns[0].column(), columns[1].column()),
                    columns[2].column(),
                ),
                columns[3].column(),
            ),
            AddFour,
        );
        let output = reduce(&exec, input, exec.to_device(&[5]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 5 + 10 * 513);
    }

    #[test]
    fn eight_read_slots_storage7_reduce_with_physical_leaf_partials() {
        let exec = executor();
        let len = TILE_SIZE * 2 + 17;
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; len]))
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
        let input = Permute::new(seven, Counting::new(0, len));
        let init: Seven = (10, 20, 30, 40, 50, 60, 70);

        let output = reduce(
            &exec,
            input,
            crate::api::value::into_scratch::<WgpuRuntime, Seven>(
                crate::api::value::store(&exec, init).unwrap(),
            ),
            AddSeven,
        )
        .unwrap();
        let output = crate::api::value::read::<WgpuRuntime, Seven>(&exec, &output).unwrap();
        assert_eq!(
            output,
            (
                10 + len as u32,
                20 + 2 * len as u32,
                30 + 3 * len as u32,
                40 + 4 * len as u32,
                50 + 5 * len as u32,
                60 + 6 * len as u32,
                70 + 7 * len as u32,
            )
        );
    }
}
