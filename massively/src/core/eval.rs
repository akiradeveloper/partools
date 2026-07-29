//! Arity-indexed device evaluators.

use core::marker::PhantomData;
use cubecl::prelude::*;
use std::rc::Rc;

use crate::core::op::ReductionOp;
use crate::core::storage::{
    Concat, ConcatExpand, Decompose, JoinedReadRow, ReadFlatLeaves, ReadLayout, ReadRow, Recompose,
    SelectLeaves,
};
use crate::op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp};
use crate::seg::{Segment, SegmentExpand, SegmentReader};

/// A type-level device expression producing `Item`.
#[doc(hidden)]
pub trait DeviceExpr<Item: CubeType>: 'static + Send + Sync {}

/// How a staged leaf slot is interpreted.
#[doc(hidden)]
#[cubecl::cube]
pub trait ReadMode<T: CubePrimitive + cubecl::frontend::Scalar>: 'static + Send + Sync {
    fn read(slot: &[T], offset: u32, index: usize) -> T;
}

/// Reads one element at `offset + index`.
#[doc(hidden)]
pub struct Direct;

/// Reads the single staged value in a constant slot.
#[doc(hidden)]
pub struct Broadcast;

/// Reads a staged start and adds the logical index.
#[doc(hidden)]
pub struct Count;

/// Reads a staged start and advances by a staged stride.
#[doc(hidden)]
pub struct StrideCount;

/// Reads `(start + index) / divisor % modulus`.
#[doc(hidden)]
pub struct DivModCount;

/// Reads a staged final index and subtracts the logical index.
#[doc(hidden)]
pub struct ReverseCount;

/// Device expression for one row of an offset-delimited value stream.
#[doc(hidden)]
pub struct SegmentIteratorExpr<ValuesExpr, OffsetsExpr> {
    _marker: PhantomData<fn() -> (ValuesExpr, OffsetsExpr)>,
}

impl<ValuesExpr, OffsetsExpr, Item> DeviceExpr<Segment<Item>>
    for SegmentIteratorExpr<ValuesExpr, OffsetsExpr>
where
    Item: CubeType + 'static,
    ValuesExpr: DeviceExpr<Item>,
    OffsetsExpr: DeviceExpr<u32>,
{
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> ReadMode<T> for Direct {
    fn read(slot: &[T], offset: u32, index: usize) -> T {
        slot[offset as usize + index]
    }
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> ReadMode<T> for Broadcast {
    fn read(slot: &[T], _offset: u32, _index: usize) -> T {
        slot[0]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for Count {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] + offset + index as u32
    }
}

#[cubecl::cube]
impl ReadMode<u32> for StrideCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] + (offset + index as u32) * slot[1]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for DivModCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        (slot[0] + offset + index as u32) / slot[1] % slot[2]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for ReverseCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] - offset - index as u32
    }
}

macro_rules! define_slots {
    ($($slot:ident),+ $(,)?) => {
        $(
            #[doc(hidden)]
            pub struct $slot<T, Mode> {
                _marker: PhantomData<fn() -> (T, Mode)>,
            }

            impl<T, Mode> DeviceExpr<T> for $slot<T, Mode>
            where
                T: CubePrimitive + cubecl::frontend::Scalar + 'static,
                Mode: ReadMode<T>,
            {
            }
        )+
    };
}

define_slots!(
    Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7, Slot8, Slot9, Slot10, Slot11, Slot12
);

/// Device expression for a binary zip.
#[doc(hidden)]
pub struct ZipExpr<Left, Right, LeftItem, RightItem> {
    _marker: PhantomData<fn() -> (Left, Right, LeftItem, RightItem)>,
}

impl<Left, Right, LeftItem, RightItem> DeviceExpr<JoinedReadRow<LeftItem, RightItem>>
    for ZipExpr<Left, Right, LeftItem, RightItem>
where
    LeftItem: ReadRow + 'static,
    RightItem: ReadRow + 'static,
    LeftItem::ReadLeaves: ReadFlatLeaves<Item = LeftItem> + Concat<RightItem::ReadLeaves>,
    RightItem::ReadLeaves: ReadFlatLeaves<Item = RightItem>,
    <LeftItem::ReadLeaves as Concat<RightItem::ReadLeaves>>::Output: ReadFlatLeaves,
    Left: DeviceExpr<LeftItem>,
    Right: DeviceExpr<RightItem>,
{
}

/// Device expression for a unary transform.
#[doc(hidden)]
pub struct TransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware unary transform.
#[doc(hidden)]
pub struct IndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware adjacent transform.
#[doc(hidden)]
pub struct AdjacentIndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedBinaryOp<InputItem>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for IndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedUnaryOp<InputItem>,
{
}

/// Device expression for adjacent reduction, preserving the first item.
#[doc(hidden)]
pub struct AdjacentExpr<InputExpr, Item, Op, Layout, Leaves> {
    _marker: PhantomData<fn() -> (InputExpr, Item, Op, Layout, Leaves)>,
}

impl<InputExpr, Item, Op, Layout, Leaves> DeviceExpr<Item>
    for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
where
    Item: CubeType + 'static,
    InputExpr: DeviceExpr<Item>,
    Op: ReductionOp<Item>,
    Leaves: CubeType + SelectLeaves + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output> for TransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: UnaryOp<InputItem>,
{
}

/// Device expression for `values[indices[index]]`.
#[doc(hidden)]
pub struct PermuteExpr<ValuesExpr, IndicesExpr> {
    _marker: PhantomData<fn() -> (ValuesExpr, IndicesExpr)>,
}

impl<ValuesExpr, IndicesExpr, Item> DeviceExpr<Item> for PermuteExpr<ValuesExpr, IndicesExpr>
where
    Item: CubeType + 'static,
    ValuesExpr: DeviceExpr<Item>,
    IndicesExpr: DeviceExpr<crate::MIndex>,
{
}

/// Device expression that repeats the first row of its input.
#[doc(hidden)]
pub struct RepeatExpr<InputExpr> {
    _marker: PhantomData<fn() -> InputExpr>,
}

impl<InputExpr, Item> DeviceExpr<Item> for RepeatExpr<InputExpr>
where
    Item: CubeType + 'static,
    InputExpr: DeviceExpr<Item>,
{
}

macro_rules! define_eval {
    ($trait_name:ident, $method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        #[doc = concat!("Evaluates a device expression using `", stringify!($trait_name), "` staged leaves.")]
        #[cubecl::cube]
        pub trait $trait_name<Item: CubeType, $( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+>: DeviceExpr<Item> {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item;
        }

        #[cubecl::cube]
        impl<LeftItem, RightItem, LeftExpr, RightExpr, $( $leaf ),+>
            $trait_name<JoinedReadRow<LeftItem, RightItem>, $( $leaf ),+>
            for ZipExpr<LeftExpr, RightExpr, LeftItem, RightItem>
        where
            LeftItem: ReadRow + 'static,
            RightItem: ReadRow + 'static,
            LeftItem::ReadLeaves:
                ReadFlatLeaves<Item = LeftItem> + Concat<RightItem::ReadLeaves>,
            RightItem::ReadLeaves: ReadFlatLeaves<Item = RightItem>,
            <LeftItem::ReadLeaves as Concat<RightItem::ReadLeaves>>::Output: ReadFlatLeaves,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            LeftExpr: $trait_name<LeftItem, $( $leaf ),+>,
            RightExpr: $trait_name<RightItem, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> JoinedReadRow<LeftItem, RightItem> {
                let left = LeftItem::ReadDeviceLayout::decompose(
                    LeftExpr::$method($( $slot, )+ slot_offsets, index),
                );
                let right = RightItem::ReadDeviceLayout::decompose(
                    RightExpr::$method($( $slot, )+ slot_offsets, index),
                );
                <<JoinedReadRow<LeftItem, RightItem> as ReadLayout>::ReadDeviceLayout as Recompose<
                    JoinedReadRow<LeftItem, RightItem>,
                >>::recompose(left.concat(right))
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+> for TransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: UnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedBinaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let previous = InputExpr::$method(
                    $( $slot, )+
                    slot_offsets,
                    previous_index,
                );
                let current = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(previous, current, index as u32)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for IndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedUnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input, index as u32)
            }
        }

        #[cubecl::cube]
        impl<Item, InputExpr, Op, Layout, Leaves, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+>
            for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            InputExpr: $trait_name<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
            Leaves: CubeType + SelectLeaves + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let first = Layout::decompose(
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                );
                let adjacent = Layout::decompose(Op::apply(
                    InputExpr::$method($( $slot, )+ slot_offsets, previous_index),
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                ));
                Layout::recompose(Leaves::select(index == 0usize, first, adjacent))
            }
        }

        #[cubecl::cube]
        impl<Item, ValuesExpr, IndicesExpr, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+> for PermuteExpr<ValuesExpr, IndicesExpr>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            ValuesExpr: $trait_name<Item, $( $leaf ),+>,
            IndicesExpr: $trait_name<crate::MIndex, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let gathered = IndicesExpr::$method($( $slot, )+ slot_offsets, index);
                ValuesExpr::$method($( $slot, )+ slot_offsets, gathered as usize)
            }
        }

        #[cubecl::cube]
        impl<Item, InputExpr, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+> for RepeatExpr<InputExpr>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar, )+
            InputExpr: $trait_name<Item, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                _index: usize,
            ) -> Item {
                InputExpr::$method($( $slot, )+ slot_offsets, 0usize)
            }
        }

    };
}

define_eval!(Eval13, eval13; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11, L12: slot12);

/// Evaluates a fixed thirteen-slot expression from a subrange of a shared
/// offsets buffer. Two independent expressions can therefore share one
/// binding without changing their slot types or forming an arity cross-product.
#[doc(hidden)]
#[cubecl::cube]
pub trait Eval13At<
    Item: CubeType,
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
>: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        index: usize,
    ) -> Item;
}

#[cubecl::cube]
impl<
    LeftItem,
    RightItem,
    LeftExpr,
    RightExpr,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
>
    Eval13At<
        JoinedReadRow<LeftItem, RightItem>,
        L0,
        L1,
        L2,
        L3,
        L4,
        L5,
        L6,
        L7,
        L8,
        L9,
        L10,
        L11,
        L12,
    > for ZipExpr<LeftExpr, RightExpr, LeftItem, RightItem>
where
    LeftItem: ReadRow + 'static,
    RightItem: ReadRow + 'static,
    LeftItem::ReadLeaves: ReadFlatLeaves<Item = LeftItem> + Concat<RightItem::ReadLeaves>,
    RightItem::ReadLeaves: ReadFlatLeaves<Item = RightItem>,
    <LeftItem::ReadLeaves as Concat<RightItem::ReadLeaves>>::Output: ReadFlatLeaves,
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
    LeftExpr: Eval13At<LeftItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    RightExpr: Eval13At<RightItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        index: usize,
    ) -> JoinedReadRow<LeftItem, RightItem> {
        let left = LeftItem::ReadDeviceLayout::decompose(LeftExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            index,
        ));
        let right = RightItem::ReadDeviceLayout::decompose(RightExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            index,
        ));
        <<JoinedReadRow<LeftItem, RightItem> as ReadLayout>::ReadDeviceLayout as Recompose<
            JoinedReadRow<LeftItem, RightItem>,
        >>::recompose(left.concat(right))
    }
}

macro_rules! impl_unary_eval13_at {
    ($expr:ident, $op:ident, $input:ident, $index:ident; $body:expr) => {
        #[cubecl::cube]
        impl<
            InputItem,
            OutputItem,
            InputExpr,
            Op,
            L0,
            L1,
            L2,
            L3,
            L4,
            L5,
            L6,
            L7,
            L8,
            L9,
            L10,
            L11,
            L12,
        > Eval13At<OutputItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
            for $expr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
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
            InputExpr: Eval13At<InputItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
            Op: $op<InputItem, Output = OutputItem>,
        {
            fn eval13_at(
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
                slot_offsets: &[u32],
                offset_base: usize,
                $index: usize,
            ) -> OutputItem {
                let $input = InputExpr::eval13_at(
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
                    slot_offsets,
                    offset_base,
                    $index,
                );
                $body
            }
        }
    };
}

impl_unary_eval13_at!(TransformExpr, UnaryOp, input, index; Op::apply(input));
impl_unary_eval13_at!(IndexedTransformExpr, IndexedUnaryOp, input, index; Op::apply(input, index as u32));

#[cubecl::cube]
impl<InputItem, OutputItem, InputExpr, Op, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    Eval13At<OutputItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    OutputItem: CubeType + 'static,
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
    InputExpr: Eval13At<InputItem, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Op: IndexedBinaryOp<InputItem, Output = OutputItem>,
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        index: usize,
    ) -> OutputItem {
        let previous_index = if index == 0usize {
            0usize
        } else {
            index - 1usize
        };
        let previous = InputExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            previous_index,
        );
        let current = InputExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            index,
        );
        Op::apply(previous, current, index as u32)
    }
}

#[cubecl::cube]
impl<Item, InputExpr, Op, Layout, Leaves, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
where
    Item: CubeType + 'static,
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
    InputExpr: Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Op: ReductionOp<Item>,
    Leaves: CubeType + SelectLeaves + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        index: usize,
    ) -> Item {
        let previous_index = if index == 0usize {
            0usize
        } else {
            index - 1usize
        };
        let first = Layout::decompose(InputExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            index,
        ));
        let adjacent = Layout::decompose(Op::apply(
            InputExpr::eval13_at(
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
                slot_offsets,
                offset_base,
                previous_index,
            ),
            InputExpr::eval13_at(
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
                slot_offsets,
                offset_base,
                index,
            ),
        ));
        Layout::recompose(Leaves::select(index == 0usize, first, adjacent))
    }
}

#[cubecl::cube]
impl<Item, ValuesExpr, IndicesExpr, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    for PermuteExpr<ValuesExpr, IndicesExpr>
where
    Item: CubeType + 'static,
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
    ValuesExpr: Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    IndicesExpr: Eval13At<crate::MIndex, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        index: usize,
    ) -> Item {
        let gathered = IndicesExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            index,
        );
        ValuesExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            gathered as usize,
        )
    }
}

#[cubecl::cube]
impl<Item, InputExpr, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12> for RepeatExpr<InputExpr>
where
    Item: CubeType + 'static,
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
    InputExpr: Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
{
    fn eval13_at(
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
        slot_offsets: &[u32],
        offset_base: usize,
        _index: usize,
    ) -> Item {
        InputExpr::eval13_at(
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
            slot_offsets,
            offset_base,
            0usize,
        )
    }
}

/// Binds a segmented row to the same staged leaves used by its backing value
/// expression.  There is one implementation per total read arity, not per
/// values-arity x offsets-arity combination.
macro_rules! impl_segment_iterator_eval {
    ($trait_name:ident, $method:ident, $expand_method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        impl<Item, ValuesExpr, OffsetsExpr, $( $leaf ),+>
            $trait_name<Segment<Item>, $( $leaf ),+>
            for SegmentIteratorExpr<ValuesExpr, OffsetsExpr>
        where
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + cubecl::frontend::Scalar + 'static, )+
            ValuesExpr: $trait_name<Item, $( $leaf ),+>,
            OffsetsExpr: $trait_name<u32, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                _slot_offsets: &[u32],
                _index: usize,
            ) -> Segment<Item> {
                let _ = ($( $slot, )+);
                unreachable!("segments are constructed while CubeCL expands a kernel")
            }

            fn $expand_method(
                scope: &Scope,
                $( $slot: &<[$leaf] as CubeType>::ExpandType, )+
                slot_offsets: &<[u32] as CubeType>::ExpandType,
                index: <usize as CubeType>::ExpandType,
            ) -> <Segment<Item> as CubeType>::ExpandType {
                let next_index = ExpandTypeClone::clone_unchecked(&index).__expand_add_method(
                    scope,
                    NativeExpand::from_lit(scope, 1usize),
                );
                let start = OffsetsExpr::$expand_method(
                    scope,
                    $( $slot, )+
                    slot_offsets,
                    ExpandTypeClone::clone_unchecked(&index),
                );
                let end = OffsetsExpr::$expand_method(
                    scope,
                    $( $slot, )+
                    slot_offsets,
                    next_index,
                );

                $( let $slot = ExpandTypeClone::clone_unchecked($slot); )+
                let slot_offsets = ExpandTypeClone::clone_unchecked(slot_offsets);
                let reader: SegmentReader<Item> = Rc::new(move |scope, absolute| {
                    let absolute = <usize as Cast>::__expand_cast_from(scope, absolute);
                    ValuesExpr::$expand_method(
                        scope,
                        $( &$slot, )+
                        &slot_offsets,
                        absolute,
                    )
                });

                SegmentExpand::from_bounds(scope, reader, start, end)
            }
        }
    };
}

impl_segment_iterator_eval!(Eval13, eval13, __expand_eval13; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11, L12: slot12);

impl<Item, ValuesExpr, OffsetsExpr, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    Eval13At<Segment<Item>, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
    for SegmentIteratorExpr<ValuesExpr, OffsetsExpr>
where
    Item: CubeType + Send + Sync + 'static,
    L0: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L1: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L2: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L3: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L4: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L5: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L6: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L7: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L8: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L9: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L10: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L11: CubePrimitive + cubecl::frontend::Scalar + 'static,
    L12: CubePrimitive + cubecl::frontend::Scalar + 'static,
    ValuesExpr: Eval13At<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    OffsetsExpr: Eval13At<u32, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
{
    fn eval13_at(
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
        _slot_offsets: &[u32],
        _offset_base: usize,
        _index: usize,
    ) -> Segment<Item> {
        let _ = (
            slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11,
            slot12,
        );
        unreachable!("segments are constructed while CubeCL expands a kernel")
    }

    fn __expand_eval13_at(
        scope: &Scope,
        slot0: &<[L0] as CubeType>::ExpandType,
        slot1: &<[L1] as CubeType>::ExpandType,
        slot2: &<[L2] as CubeType>::ExpandType,
        slot3: &<[L3] as CubeType>::ExpandType,
        slot4: &<[L4] as CubeType>::ExpandType,
        slot5: &<[L5] as CubeType>::ExpandType,
        slot6: &<[L6] as CubeType>::ExpandType,
        slot7: &<[L7] as CubeType>::ExpandType,
        slot8: &<[L8] as CubeType>::ExpandType,
        slot9: &<[L9] as CubeType>::ExpandType,
        slot10: &<[L10] as CubeType>::ExpandType,
        slot11: &<[L11] as CubeType>::ExpandType,
        slot12: &<[L12] as CubeType>::ExpandType,
        slot_offsets: &<[u32] as CubeType>::ExpandType,
        offset_base: <usize as CubeType>::ExpandType,
        index: <usize as CubeType>::ExpandType,
    ) -> <Segment<Item> as CubeType>::ExpandType {
        let next_index = ExpandTypeClone::clone_unchecked(&index)
            .__expand_add_method(scope, NativeExpand::from_lit(scope, 1usize));
        let start = OffsetsExpr::__expand_eval13_at(
            scope,
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
            slot_offsets,
            ExpandTypeClone::clone_unchecked(&offset_base),
            ExpandTypeClone::clone_unchecked(&index),
        );
        let end = OffsetsExpr::__expand_eval13_at(
            scope,
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
            slot_offsets,
            ExpandTypeClone::clone_unchecked(&offset_base),
            next_index,
        );

        let slot0 = ExpandTypeClone::clone_unchecked(slot0);
        let slot1 = ExpandTypeClone::clone_unchecked(slot1);
        let slot2 = ExpandTypeClone::clone_unchecked(slot2);
        let slot3 = ExpandTypeClone::clone_unchecked(slot3);
        let slot4 = ExpandTypeClone::clone_unchecked(slot4);
        let slot5 = ExpandTypeClone::clone_unchecked(slot5);
        let slot6 = ExpandTypeClone::clone_unchecked(slot6);
        let slot7 = ExpandTypeClone::clone_unchecked(slot7);
        let slot8 = ExpandTypeClone::clone_unchecked(slot8);
        let slot9 = ExpandTypeClone::clone_unchecked(slot9);
        let slot10 = ExpandTypeClone::clone_unchecked(slot10);
        let slot11 = ExpandTypeClone::clone_unchecked(slot11);
        let slot12 = ExpandTypeClone::clone_unchecked(slot12);
        let slot_offsets = ExpandTypeClone::clone_unchecked(slot_offsets);
        let offset_base = ExpandTypeClone::clone_unchecked(&offset_base);
        let reader: SegmentReader<Item> = Rc::new(move |scope, absolute| {
            let absolute = <usize as Cast>::__expand_cast_from(scope, absolute);
            ValuesExpr::__expand_eval13_at(
                scope,
                &slot0,
                &slot1,
                &slot2,
                &slot3,
                &slot4,
                &slot5,
                &slot6,
                &slot7,
                &slot8,
                &slot9,
                &slot10,
                &slot11,
                &slot12,
                &slot_offsets,
                ExpandTypeClone::clone_unchecked(&offset_base),
                absolute,
            )
        });
        SegmentExpand::from_bounds(scope, reader, start, end)
    }
}

macro_rules! impl_slot_eval {
    (
        $trait_name:ident, $method:ident, $slot_expr:ident, $offset_index:literal;
        <$( $generic:ident ),+>;
        [$( $leaf_ty:ty ),+];
        [$( $slot:ident ),+];
        $selected:ident
    ) => {
        #[cubecl::cube]
        impl<Mode, $( $generic ),+> $trait_name<T, $( $leaf_ty ),+> for $slot_expr<T, Mode>
        where
            $( $generic: CubePrimitive + cubecl::frontend::Scalar, )+
            Mode: ReadMode<T>,
        {
            fn $method(
                $( $slot: &[$leaf_ty], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> T {
                let _ = ($( $slot, )+);
                Mode::read($selected, slot_offsets[$offset_index], index)
            }
        }
    };
}

impl_slot_eval!(Eval13, eval13, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot0);
impl_slot_eval!(Eval13, eval13, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot1);
impl_slot_eval!(Eval13, eval13, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot2);
impl_slot_eval!(Eval13, eval13, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot3);
impl_slot_eval!(Eval13, eval13, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot4);
impl_slot_eval!(Eval13, eval13, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot5);
impl_slot_eval!(Eval13, eval13, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot6);
impl_slot_eval!(Eval13, eval13, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot7);
impl_slot_eval!(Eval13, eval13, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot8);
impl_slot_eval!(Eval13, eval13, Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot9);
impl_slot_eval!(Eval13, eval13, Slot10, 10; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot10);
impl_slot_eval!(Eval13, eval13, Slot11, 11; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot11);
impl_slot_eval!(Eval13, eval13, Slot12, 12; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot12);

macro_rules! impl_slot_eval13_at {
    (
        $slot_expr:ident, $offset_index:literal;
        <$( $generic:ident ),+>;
        [$( $leaf_ty:ty ),+];
        [$( $slot:ident ),+];
        $selected:ident
    ) => {
        #[cubecl::cube]
        impl<Mode, $( $generic ),+> Eval13At<T, $( $leaf_ty ),+> for $slot_expr<T, Mode>
        where
            $( $generic: CubePrimitive + cubecl::frontend::Scalar, )+
            Mode: ReadMode<T>,
        {
            fn eval13_at(
                $( $slot: &[$leaf_ty], )+
                slot_offsets: &[u32],
                offset_base: usize,
                index: usize,
            ) -> T {
                let _ = ($( $slot, )+);
                Mode::read($selected, slot_offsets[offset_base + $offset_index], index)
            }
        }
    };
}

impl_slot_eval13_at!(Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot0);
impl_slot_eval13_at!(Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot1);
impl_slot_eval13_at!(Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot2);
impl_slot_eval13_at!(Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot3);
impl_slot_eval13_at!(Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot4);
impl_slot_eval13_at!(Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot5);
impl_slot_eval13_at!(Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot6);
impl_slot_eval13_at!(Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot7);
impl_slot_eval13_at!(Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot8);
impl_slot_eval13_at!(Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot9);
impl_slot_eval13_at!(Slot10, 10; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot10);
impl_slot_eval13_at!(Slot11, 11; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot11);
impl_slot_eval13_at!(Slot12, 12; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot12);
