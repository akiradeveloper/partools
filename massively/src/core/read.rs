//! Read-expression trees and deterministic slot binding.

#![allow(private_interfaces)]

mod staging;
pub use staging::StageRead;

use core::marker::PhantomData;
use cubecl::prelude::*;
use std::ops::{Bound, RangeBounds};

use crate::MIndex;
use crate::core::arity::{
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, AddArity, ReadArity,
};
use crate::core::eval::{
    AdjacentExpr, AdjacentIndexedTransformExpr, Broadcast, Count, DeviceExpr, Direct, DivModCount,
    Eval13, Eval13At, IndexedTransformExpr, PermuteExpr, RepeatExpr, ReverseCount,
    SegmentIteratorExpr, Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7, Slot8, Slot9,
    Slot10, Slot11, Slot12, StrideCount, TransformExpr, ZipExpr,
};
use crate::core::extent::LogicalExtent;
use crate::core::iter::Zip;
use crate::core::op::ReductionOp;
use crate::core::storage::{
    Concat, JoinedReadRow, ReadFlatLeaves, ReadLayout, ReadRow, StorageLayout,
};
use crate::core::value::MStorageElement;
use crate::op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp};
use crate::seg::Segment;

/// Internal runtime-erased contiguous device read leaf.
#[doc(hidden)]
#[derive(Clone, Debug, Default)]
pub struct Column<T> {
    pub(crate) handle: Option<cubecl::server::Handle>,
    pub(crate) len: usize,
    pub(crate) buffer_len: usize,
    pub(crate) offset: u32,
    pub(crate) owner: Option<u64>,
    pub(crate) extent: LogicalExtent,
    _item: PhantomData<fn() -> T>,
}

impl<T> Column<T> {
    #[cfg(test)]
    pub(crate) const fn new() -> Self {
        Self {
            handle: None,
            len: 0,
            buffer_len: 0,
            offset: 0,
            owner: None,
            extent: LogicalExtent::fixed(0),
            _item: PhantomData,
        }
    }

    pub(crate) fn from_handle(
        handle: cubecl::server::Handle,
        len: usize,
        offset: u32,
        owner: u64,
        buffer_len: usize,
    ) -> Self {
        Self::from_handle_with_extent(
            handle,
            len,
            offset,
            owner,
            buffer_len,
            LogicalExtent::fixed(len),
        )
    }

    pub(crate) fn from_handle_with_extent(
        handle: cubecl::server::Handle,
        len: usize,
        offset: u32,
        owner: u64,
        buffer_len: usize,
        extent: LogicalExtent,
    ) -> Self {
        Self {
            handle: Some(handle),
            len,
            buffer_len,
            offset,
            owner: Some(owner),
            extent,
            _item: PhantomData,
        }
    }

    pub(crate) fn with_logical_extent(mut self, extent: LogicalExtent) -> Self {
        self.extent = extent;
        self
    }

    pub(crate) fn slice_usize<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<usize>,
    {
        let (relative, len) = resolve_slice_range(self.len, range);
        Self {
            handle: self.handle.clone(),
            len,
            buffer_len: self.buffer_len,
            offset: self.offset + relative as u32,
            owner: self.owner,
            extent: self.extent.slice(relative, len),
            _item: PhantomData,
        }
    }
}

pub(crate) fn resolve_mindex_slice_range<Range>(len: usize, range: Range) -> (usize, usize)
where
    Range: RangeBounds<MIndex>,
{
    let logical_len = MIndex::try_from(len).expect("slice length does not fit in MIndex");
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => logical_len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= logical_len,
        "slice end ({end}) is out of bounds for length {logical_len}"
    );
    (start as usize, (end - start) as usize)
}

pub(crate) fn resolve_slice_range<Range>(len: usize, range: Range) -> (usize, usize)
where
    Range: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= len,
        "slice end ({end}) is out of bounds for length {len}"
    );
    (start, end - start)
}

/// A scalar leaf broadcast to every logical index.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    pub value: T,
    pub len: usize,
}

/// One scalar read leaf whose bytes may originate on either host or device.
///
/// Both variants use the same direct one-slot device expression. The host
/// variant creates its one-element binding only when a consumer stages the
/// expression.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum Value<T> {
    Host(T),
    Device {
        handle: cubecl::server::Handle,
        owner: u64,
    },
}

impl<T> Constant<T> {
    pub const fn new(value: T, len: usize) -> Self {
        Self { value, len }
    }
}

/// A `u32` counting leaf.
#[derive(Clone, Copy, Debug)]
pub struct Counting {
    pub start: u32,
    pub len: usize,
}

/// A `u32` arithmetic progression leaf.
#[derive(Clone, Copy, Debug)]
pub struct Stride {
    pub start: u32,
    pub step: u32,
    pub len: usize,
}

/// A finite index leaf yielding `(start + index) / divisor % modulus`.
#[derive(Clone, Copy, Debug)]
pub struct DivModCounting {
    pub start: u32,
    pub divisor: u32,
    pub modulus: u32,
    pub len: usize,
}

/// A descending `u32` index leaf yielding `len - 1 - index`.
#[derive(Clone, Copy, Debug)]
pub struct ReverseCounting {
    pub start: usize,
    pub len: usize,
}

impl ReverseCounting {
    pub const fn new(len: usize) -> Self {
        Self {
            start: len.saturating_sub(1),
            len,
        }
    }
}

/// A lazy expression that reads its input in reverse order.
#[derive(Clone, Copy, Debug)]
pub struct Reverse<Values> {
    pub(crate) values: Values,
    pub(crate) offset: usize,
    pub(crate) len: Option<usize>,
}

impl<Values> Reverse<Values> {
    pub const fn new(values: Values) -> Self {
        Self {
            values,
            offset: 0,
            len: None,
        }
    }
}

impl Counting {
    pub const fn new(start: u32, len: usize) -> Self {
        Self { start, len }
    }
}

impl Stride {
    pub const fn new(start: u32, step: u32, len: usize) -> Self {
        Self { start, step, len }
    }
}

impl DivModCounting {
    pub const fn new(start: u32, divisor: u32, modulus: u32, len: usize) -> Self {
        Self {
            start,
            divisor,
            modulus,
            len,
        }
    }
}

/// A finite view of an otherwise unbounded lazy source.
#[derive(Clone, Copy, Debug)]
pub struct Taken<Source> {
    pub(crate) source: Source,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl<Source> Taken<Source> {
    pub(crate) const fn new(source: Source, len: usize) -> Self {
        Self {
            source,
            offset: 0,
            len,
        }
    }

    pub(crate) fn lower(&self) -> Source::Read
    where
        Source: TakenSource,
    {
        self.source.lower(self.offset, self.len)
    }
}

/// Internal lowering contract for unbounded lazy sources.
#[doc(hidden)]
pub trait TakenSource {
    type Read: ReadExpression;

    fn lower(&self, offset: usize, len: usize) -> Self::Read;
}

/// A lazy unary transform expression.
#[derive(Debug)]
pub struct Transform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy transform that also receives the logical input index.
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedTransform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy adjacent transform that also receives the current logical index.
#[doc(hidden)]
#[derive(Debug)]
pub struct AdjacentIndexedTransform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy adjacent reduction expression.
#[doc(hidden)]
#[derive(Debug)]
pub struct Adjacent<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

impl<Input: Clone, Op> Clone for Transform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Transform<Input, Op> {}

impl<Input: Clone, Op> Clone for IndexedTransform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for IndexedTransform<Input, Op> {}

impl<Input: Clone, Op> Clone for AdjacentIndexedTransform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for AdjacentIndexedTransform<Input, Op> {}

impl<Input: Clone, Op> Clone for Adjacent<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Adjacent<Input, Op> {}

impl<Input, Op> Adjacent<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self::from_input(input)
    }

    pub(crate) fn from_input(input: Input) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> Transform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self::from_input(input)
    }

    pub(crate) fn from_input(input: Input) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> IndexedTransform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> AdjacentIndexedTransform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

/// A lazy permutation expression evaluating `values[indices[index]]`.
#[derive(Clone, Copy, Debug)]
pub struct Permute<Values, Indices> {
    pub values: Values,
    pub indices: Indices,
}

/// A finite expression that repeats the first row of its input.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Repeat<Input> {
    pub(crate) input: Input,
    pub(crate) len: usize,
}

impl<Input> Repeat<Input> {
    pub const fn new(input: Input, len: usize) -> Self {
        Self { input, len }
    }
}

impl<Values, Indices> Permute<Values, Indices> {
    pub const fn new(values: Values, indices: Indices) -> Self {
        Self { values, indices }
    }
}

/// A zero-copy logical subrange of a read expression.
///
/// The contained expression has already been sliced according to its own
/// semantics. Keeping a distinct adapter type makes slicing part of iterator
/// composition without adding a staged read leaf or changing read arity.
#[derive(Clone, Copy, Debug)]
pub struct Slice<Runtime, Input> {
    pub input: Input,
    _runtime: PhantomData<fn() -> Runtime>,
}

impl<Runtime, Input> Slice<Runtime, Input> {
    pub const fn new(input: Input) -> Self {
        Self {
            input,
            _runtime: PhantomData,
        }
    }
}

/// Semantic item and physical read arity of a read-expression tree.
pub trait ReadExpression {
    type Item: CubeType + Send + Sync + 'static;
    type ReadArity: ReadArity;
}

impl<T> ReadExpression for Column<T>
where
    T: MStorageElement,
{
    type Item = T;
    type ReadArity = A1;
}

impl<T> ReadExpression for Constant<T>
where
    T: MStorageElement,
{
    type Item = T;
    type ReadArity = A1;
}

impl<T> ReadExpression for Value<T>
where
    T: MStorageElement,
{
    type Item = T;
    type ReadArity = A1;
}

impl ReadExpression for Counting {
    type Item = crate::MIndex;
    type ReadArity = A1;
}

impl ReadExpression for Stride {
    type Item = crate::MIndex;
    type ReadArity = A1;
}

impl ReadExpression for DivModCounting {
    type Item = crate::MIndex;
    type ReadArity = A1;
}

impl ReadExpression for ReverseCounting {
    type Item = crate::MIndex;
    type ReadArity = A1;
}

impl<Source> ReadExpression for Taken<Source>
where
    Source: TakenSource,
{
    type Item = <Source::Read as ReadExpression>::Item;
    type ReadArity = <Source::Read as ReadExpression>::ReadArity;
}

impl<Left, Right> ReadExpression for Zip<Left, Right>
where
    Left: ReadExpression,
    Right: ReadExpression,
    Left::ReadArity: AddArity<Right::ReadArity>,
    Left::Item: ReadRow,
    Right::Item: ReadRow,
    <Left::Item as ReadLayout>::ReadLeaves:
        ReadFlatLeaves<Item = Left::Item> + Concat<<Right::Item as ReadLayout>::ReadLeaves>,
    <Right::Item as ReadLayout>::ReadLeaves: ReadFlatLeaves<Item = Right::Item>,
    <<Left::Item as ReadLayout>::ReadLeaves as Concat<
        <Right::Item as ReadLayout>::ReadLeaves,
    >>::Output: ReadFlatLeaves,
{
    type Item = JoinedReadRow<Left::Item, Right::Item>;
    type ReadArity = <Left::ReadArity as AddArity<Right::ReadArity>>::Output;
}

impl<Input, Op> ReadExpression for Transform<Input, Op>
where
    Input: ReadExpression,
    Op: UnaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for IndexedTransform<Input, Op>
where
    Input: ReadExpression,
    Op: IndexedUnaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression,
    Op: IndexedBinaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for Adjacent<Input, Op>
where
    Input: ReadExpression,
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
{
    type Item = Input::Item;
    type ReadArity = Input::ReadArity;
}

impl<Values, Indices> ReadExpression for Permute<Values, Indices>
where
    Values: ReadExpression,
    Indices: ReadExpression<Item = crate::MIndex>,
    Values::ReadArity: AddArity<Indices::ReadArity>,
{
    type Item = Values::Item;
    type ReadArity = <Values::ReadArity as AddArity<Indices::ReadArity>>::Output;
}

impl<Input> ReadExpression for Repeat<Input>
where
    Input: ReadExpression,
{
    type Item = Input::Item;
    type ReadArity = Input::ReadArity;
}

impl<Values> ReadExpression for Reverse<Values>
where
    Values: ReadExpression,
    Values::ReadArity: AddArity<A1>,
{
    type Item = Values::Item;
    type ReadArity = <Values::ReadArity as AddArity<A1>>::Output;
}

impl<Runtime, Input> ReadExpression for Slice<Runtime, Input>
where
    Input: ReadExpression,
{
    type Item = Input::Item;
    type ReadArity = Input::ReadArity;
}

impl<Values, Offsets> ReadExpression for crate::seg::SegmentRead<Values, Offsets>
where
    Values: ReadExpression,
    Offsets: ReadExpression<Item = u32>,
    Values::ReadArity: AddArity<Offsets::ReadArity>,
{
    type Item = Segment<Values::Item>;
    type ReadArity = <Values::ReadArity as AddArity<Offsets::ReadArity>>::Output;
}

/// Produces a same-arity expression for a logical subrange.
///
/// This operation is defined per expression node so that index-sensitive lazy
/// expressions such as [`Permute`] slice their logical input, rather than
/// blindly shifting every physical leaf.
#[doc(hidden)]
pub trait SliceExpression: ReadExpression + Sized {
    fn slice_expression(&self, start: usize, len: usize) -> Self;
}

impl<T> SliceExpression for Column<T>
where
    T: MStorageElement,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        self.slice_usize(start..start + len)
    }
}

impl<T> SliceExpression for Constant<T>
where
    T: MStorageElement,
{
    fn slice_expression(&self, _start: usize, len: usize) -> Self {
        Self::new(self.value, len)
    }
}

impl<T> SliceExpression for Value<T>
where
    T: MStorageElement,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        assert!(
            start == 0 && len <= 1,
            "value iterator slice is out of bounds"
        );
        self.clone()
    }
}

impl SliceExpression for Counting {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let start = u32::try_from(start).expect("slice start exceeds u32");
        Self::new(
            self.start
                .checked_add(start)
                .expect("counting slice start overflow"),
            len,
        )
    }
}

impl SliceExpression for Stride {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let start = u32::try_from(start).expect("stride slice start exceeds u32");
        Self::new(
            self.start
                .checked_add(
                    start
                        .checked_mul(self.step)
                        .expect("stride slice offset overflow"),
                )
                .expect("stride slice start overflow"),
            self.step,
            len,
        )
    }
}

impl SliceExpression for DivModCounting {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let start = u32::try_from(start).expect("div-mod slice start exceeds u32");
        Self::new(
            self.start
                .checked_add(start)
                .expect("div-mod slice start overflow"),
            self.divisor,
            self.modulus,
            len,
        )
    }
}

impl SliceExpression for ReverseCounting {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            start: self.start.saturating_sub(start),
            len,
        }
    }
}

impl<Source> SliceExpression for Taken<Source>
where
    Source: TakenSource + Clone,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            source: self.source.clone(),
            offset: self
                .offset
                .checked_add(start)
                .expect("taken slice offset overflow"),
            len,
        }
    }
}

impl<Left, Right> SliceExpression for Zip<Left, Right>
where
    Left: SliceExpression,
    Right: SliceExpression,
    Zip<Left, Right>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Zip::new(
            self.0.slice_expression(start, len),
            self.1.slice_expression(start, len),
        )
    }
}

impl<Input, Op> SliceExpression for Transform<Input, Op>
where
    Input: SliceExpression,
    Op: UnaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Transform {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Input, Op> SliceExpression for IndexedTransform<Input, Op>
where
    Input: ReadExpression + SliceExpression,
    Op: IndexedUnaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Input, Op> SliceExpression for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression + SliceExpression,
    Op: IndexedBinaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Values, Indices> SliceExpression for Permute<Values, Indices>
where
    Values: ReadExpression + Clone,
    Indices: SliceExpression<Item = crate::MIndex>,
    Permute<Values, Indices>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Permute::new(
            self.values.clone(),
            self.indices.slice_expression(start, len),
        )
    }
}

impl<Values> SliceExpression for Reverse<Values>
where
    Values: ReadExpression + Clone,
    Reverse<Values>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            values: self.values.clone(),
            offset: self
                .offset
                .checked_add(start)
                .expect("reverse slice offset overflow"),
            len: Some(len),
        }
    }
}

impl<Runtime, Input> SliceExpression for Slice<Runtime, Input>
where
    Input: SliceExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self::new(self.input.slice_expression(start, len))
    }
}

impl<Values, Offsets> SliceExpression for crate::seg::SegmentRead<Values, Offsets>
where
    Values: ReadExpression + Clone,
    Offsets: SliceExpression<Item = u32>,
    crate::seg::SegmentRead<Values, Offsets>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let offset_len = len.checked_add(1).expect("segmented slice length overflow");
        crate::seg::SegmentRead::new(
            self.values().clone(),
            self.offsets().slice_expression(start, offset_len),
        )
    }
}

/// Empty slot environment used as the root binding state.
#[doc(hidden)]
pub struct Env0;
#[doc(hidden)]
pub struct Env1<L0>(PhantomData<fn() -> L0>);
#[doc(hidden)]
pub struct Env2<L0, L1>(PhantomData<fn() -> (L0, L1)>);
#[doc(hidden)]
pub struct Env3<L0, L1, L2>(PhantomData<fn() -> (L0, L1, L2)>);
#[doc(hidden)]
pub struct Env4<L0, L1, L2, L3>(PhantomData<fn() -> (L0, L1, L2, L3)>);
#[doc(hidden)]
pub struct Env5<L0, L1, L2, L3, L4>(PhantomData<fn() -> (L0, L1, L2, L3, L4)>);
#[doc(hidden)]
pub struct Env6<L0, L1, L2, L3, L4, L5>(PhantomData<fn() -> (L0, L1, L2, L3, L4, L5)>);
#[doc(hidden)]
pub struct Env7<L0, L1, L2, L3, L4, L5, L6>(PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6)>);
#[doc(hidden)]
pub struct Env8<L0, L1, L2, L3, L4, L5, L6, L7>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7)>,
);
pub struct Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7, L8)>,
);
pub struct Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9)>,
);
pub struct Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10)>,
);
pub struct Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11)>,
);
pub struct Env13<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12)>,
);

/// Binds an expression's leaves to consecutive slots, starting at `Env`.
///
/// Binary nodes always bind the left subtree first, then pass its resulting
/// environment to the right subtree.
#[doc(hidden)]
pub trait BindSlots<Env> {
    type Expr;
    type NextEnv;
}

macro_rules! impl_leaf_binding {
    (impl <$( $env_ty:ident ),*> $env:ty => $column_next:ty, $counting_next:ty, $slot:ident) => {
        impl<T, $( $env_ty ),*> BindSlots<$env> for Column<T>
        where
            T: MStorageElement,
        {
            type Expr = $slot<T, Direct>;
            type NextEnv = $column_next;
        }

        impl<T, $( $env_ty ),*> BindSlots<$env> for Constant<T>
        where
            T: MStorageElement,
        {
            type Expr = $slot<T, Broadcast>;
            type NextEnv = $column_next;
        }

        impl<T, $( $env_ty ),*> BindSlots<$env> for Value<T>
        where
            T: MStorageElement,
        {
            type Expr = $slot<T, Direct>;
            type NextEnv = $column_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for Counting {
            type Expr = $slot<u32, Count>;
            type NextEnv = $counting_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for Stride {
            type Expr = $slot<u32, StrideCount>;
            type NextEnv = $counting_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for DivModCounting {
            type Expr = $slot<u32, DivModCount>;
            type NextEnv = $counting_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for ReverseCounting {
            type Expr = $slot<u32, ReverseCount>;
            type NextEnv = $counting_next;
        }
    };
}

impl_leaf_binding!(impl <> Env0 => Env1<T>, Env1<u32>, Slot0);
impl_leaf_binding!(impl <L0> Env1<L0> => Env2<L0, T>, Env2<L0, u32>, Slot1);
impl_leaf_binding!(impl <L0, L1> Env2<L0, L1> => Env3<L0, L1, T>, Env3<L0, L1, u32>, Slot2);
impl_leaf_binding!(impl <L0, L1, L2> Env3<L0, L1, L2> => Env4<L0, L1, L2, T>, Env4<L0, L1, L2, u32>, Slot3);
impl_leaf_binding!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3> => Env5<L0, L1, L2, L3, T>, Env5<L0, L1, L2, L3, u32>, Slot4);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4> => Env6<L0, L1, L2, L3, L4, T>, Env6<L0, L1, L2, L3, L4, u32>, Slot5);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5> => Env7<L0, L1, L2, L3, L4, L5, T>, Env7<L0, L1, L2, L3, L4, L5, u32>, Slot6);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6> Env7<L0, L1, L2, L3, L4, L5, L6> => Env8<L0, L1, L2, L3, L4, L5, L6, T>, Env8<L0, L1, L2, L3, L4, L5, L6, u32>, Slot7);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7> Env8<L0, L1, L2, L3, L4, L5, L6, L7> => Env9<L0, L1, L2, L3, L4, L5, L6, L7, T>, Env9<L0, L1, L2, L3, L4, L5, L6, L7, u32>, Slot8);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8> Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8> => Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, T>, Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, u32>, Slot9);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> => Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T>, Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, u32>, Slot10);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> => Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T>, Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, u32>, Slot11);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11> Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11> => Env13<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T>, Env13<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, u32>, Slot12);

impl<Source, Env> BindSlots<Env> for Taken<Source>
where
    Source: TakenSource,
    Source::Read: BindSlots<Env>,
{
    type Expr = <Source::Read as BindSlots<Env>>::Expr;
    type NextEnv = <Source::Read as BindSlots<Env>>::NextEnv;
}

impl<Left, Right, Env> BindSlots<Env> for Zip<Left, Right>
where
    Left: ReadExpression + BindSlots<Env>,
    Right: ReadExpression + BindSlots<Left::NextEnv>,
{
    type Expr = ZipExpr<Left::Expr, Right::Expr, Left::Item, Right::Item>;
    type NextEnv = Right::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for Transform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: UnaryOp<Input::Item>,
{
    type Expr = TransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for IndexedTransform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: IndexedUnaryOp<Input::Item>,
{
    type Expr = IndexedTransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: IndexedBinaryOp<Input::Item>,
{
    type Expr = AdjacentIndexedTransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for Adjacent<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Input::Item: StorageLayout,
    <Input::Item as StorageLayout>::StorageLeaves: crate::core::storage::SelectLeaves,
    Op: ReductionOp<Input::Item>,
{
    type Expr = AdjacentExpr<
        Input::Expr,
        Input::Item,
        Op,
        <Input::Item as StorageLayout>::DeviceLayout,
        <Input::Item as StorageLayout>::StorageLeaves,
    >;
    type NextEnv = Input::NextEnv;
}

impl<Values, Indices, Env> BindSlots<Env> for Permute<Values, Indices>
where
    Values: BindSlots<Env>,
    Indices: ReadExpression<Item = crate::MIndex> + BindSlots<Values::NextEnv>,
{
    type Expr = PermuteExpr<Values::Expr, Indices::Expr>;
    type NextEnv = Indices::NextEnv;
}

impl<Input, Env> BindSlots<Env> for Repeat<Input>
where
    Input: BindSlots<Env>,
{
    type Expr = RepeatExpr<Input::Expr>;
    type NextEnv = Input::NextEnv;
}

impl<Values, Env> BindSlots<Env> for Reverse<Values>
where
    Values: BindSlots<Env>,
    ReverseCounting: BindSlots<Values::NextEnv>,
{
    type Expr = PermuteExpr<Values::Expr, <ReverseCounting as BindSlots<Values::NextEnv>>::Expr>;
    type NextEnv = <ReverseCounting as BindSlots<Values::NextEnv>>::NextEnv;
}

impl<Runtime, Input, Env> BindSlots<Env> for Slice<Runtime, Input>
where
    Input: BindSlots<Env>,
{
    type Expr = Input::Expr;
    type NextEnv = Input::NextEnv;
}

impl<Values, Offsets, Env> BindSlots<Env> for crate::seg::SegmentRead<Values, Offsets>
where
    Values: BindSlots<Env>,
    Offsets: BindSlots<Values::NextEnv>,
{
    type Expr = SegmentIteratorExpr<Values::Expr, Offsets::Expr>;
    type NextEnv = Offsets::NextEnv;
}

/// A non-empty final slot environment and its read arity.
#[doc(hidden)]
pub trait SlotEnvironment {
    type Arity: ReadArity;
}

impl<L0> SlotEnvironment for Env1<L0> {
    type Arity = A1;
}
impl<L0, L1> SlotEnvironment for Env2<L0, L1> {
    type Arity = A2;
}
impl<L0, L1, L2> SlotEnvironment for Env3<L0, L1, L2> {
    type Arity = A3;
}
impl<L0, L1, L2, L3> SlotEnvironment for Env4<L0, L1, L2, L3> {
    type Arity = A4;
}
impl<L0, L1, L2, L3, L4> SlotEnvironment for Env5<L0, L1, L2, L3, L4> {
    type Arity = A5;
}
impl<L0, L1, L2, L3, L4, L5> SlotEnvironment for Env6<L0, L1, L2, L3, L4, L5> {
    type Arity = A6;
}
impl<L0, L1, L2, L3, L4, L5, L6> SlotEnvironment for Env7<L0, L1, L2, L3, L4, L5, L6> {
    type Arity = A7;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7> SlotEnvironment for Env8<L0, L1, L2, L3, L4, L5, L6, L7> {
    type Arity = A8;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8> SlotEnvironment
    for Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8>
{
    type Arity = A9;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> SlotEnvironment
    for Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>
{
    type Arity = A10;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> SlotEnvironment
    for Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>
{
    type Arity = A11;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11> SlotEnvironment
    for Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>
{
    type Arity = A12;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12> SlotEnvironment
    for Env13<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>
{
    type Arity = A13;
}

/// Pads an actual read environment to the fixed thirteen-buffer kernel ABI.
/// Unused slots are typed as `u32`; device expressions never reference them.
#[doc(hidden)]
pub trait PaddedReadSlots: SlotEnvironment {
    type L0: MStorageElement;
    type L1: MStorageElement;
    type L2: MStorageElement;
    type L3: MStorageElement;
    type L4: MStorageElement;
    type L5: MStorageElement;
    type L6: MStorageElement;
    type L7: MStorageElement;
    type L8: MStorageElement;
    type L9: MStorageElement;
    type L10: MStorageElement;
    type L11: MStorageElement;
    type L12: MStorageElement;
}

#[doc(hidden)]
pub type KernelReadSlots<Slots> = Env13<
    <Slots as PaddedReadSlots>::L0,
    <Slots as PaddedReadSlots>::L1,
    <Slots as PaddedReadSlots>::L2,
    <Slots as PaddedReadSlots>::L3,
    <Slots as PaddedReadSlots>::L4,
    <Slots as PaddedReadSlots>::L5,
    <Slots as PaddedReadSlots>::L6,
    <Slots as PaddedReadSlots>::L7,
    <Slots as PaddedReadSlots>::L8,
    <Slots as PaddedReadSlots>::L9,
    <Slots as PaddedReadSlots>::L10,
    <Slots as PaddedReadSlots>::L11,
    <Slots as PaddedReadSlots>::L12,
>;

/// Erases an expression's physical read arity behind the fixed thirteen-slot ABI.
/// The wrapped device expression still evaluates only its real leaves.
#[doc(hidden)]
#[derive(Clone)]
pub struct FixedRead<Input> {
    input: Input,
    extent: Option<LogicalExtent>,
}

impl<Input> FixedRead<Input> {
    pub fn new(input: Input) -> Self {
        Self {
            input,
            extent: None,
        }
    }

    /// Narrows a consumer to an already validated common logical extent.
    /// This changes host scheduling metadata, not the device expression or ABI.
    pub(crate) fn with_logical_extent(mut self, extent: LogicalExtent) -> Self {
        self.extent = Some(extent);
        self
    }
}

impl<Input> ReadExpression for FixedRead<Input>
where
    Input: LowerReadExpression,
{
    type Item = Input::Item;
    type ReadArity = A13;
}

impl<Input> BindSlots<Env0> for FixedRead<Input>
where
    Input: LowerReadExpression,
    Input::Slots: PaddedReadSlots,
{
    type Expr = Input::DeviceExpr;
    type NextEnv = KernelReadSlots<Input::Slots>;
}

macro_rules! impl_padded_read_slots {
    ($env:ty; [$($actual:ident),+]; [$($padded:ty),+]) => {
        impl<$($actual: MStorageElement),+> PaddedReadSlots for $env {
            type L0 = impl_padded_read_slots!(@at 0; $($padded),+);
            type L1 = impl_padded_read_slots!(@at 1; $($padded),+);
            type L2 = impl_padded_read_slots!(@at 2; $($padded),+);
            type L3 = impl_padded_read_slots!(@at 3; $($padded),+);
            type L4 = impl_padded_read_slots!(@at 4; $($padded),+);
            type L5 = impl_padded_read_slots!(@at 5; $($padded),+);
            type L6 = impl_padded_read_slots!(@at 6; $($padded),+);
            type L7 = impl_padded_read_slots!(@at 7; $($padded),+);
            type L8 = impl_padded_read_slots!(@at 8; $($padded),+);
            type L9 = impl_padded_read_slots!(@at 9; $($padded),+);
            type L10 = impl_padded_read_slots!(@at 10; $($padded),+);
            type L11 = impl_padded_read_slots!(@at 11; $($padded),+);
            type L12 = impl_padded_read_slots!(@at 12; $($padded),+);
        }
    };
    (@at 0; $head:ty $(, $tail:ty)*) => { $head };
    (@at 1; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 0; $($tail),+) };
    (@at 2; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 1; $($tail),+) };
    (@at 3; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 2; $($tail),+) };
    (@at 4; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 3; $($tail),+) };
    (@at 5; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 4; $($tail),+) };
    (@at 6; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 5; $($tail),+) };
    (@at 7; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 6; $($tail),+) };
    (@at 8; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 7; $($tail),+) };
    (@at 9; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 8; $($tail),+) };
    (@at 10; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 9; $($tail),+) };
    (@at 11; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 10; $($tail),+) };
    (@at 12; $head:ty, $($tail:ty),+) => { impl_padded_read_slots!(@at 11; $($tail),+) };
}

impl_padded_read_slots!(Env1<L0>; [L0]; [L0,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env2<L0,L1>; [L0,L1]; [L0,L1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env3<L0,L1,L2>; [L0,L1,L2]; [L0,L1,L2,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env4<L0,L1,L2,L3>; [L0,L1,L2,L3]; [L0,L1,L2,L3,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env5<L0,L1,L2,L3,L4>; [L0,L1,L2,L3,L4]; [L0,L1,L2,L3,L4,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env6<L0,L1,L2,L3,L4,L5>; [L0,L1,L2,L3,L4,L5]; [L0,L1,L2,L3,L4,L5,u32,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env7<L0,L1,L2,L3,L4,L5,L6>; [L0,L1,L2,L3,L4,L5,L6]; [L0,L1,L2,L3,L4,L5,L6,u32,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [L0,L1,L2,L3,L4,L5,L6,L7]; [L0,L1,L2,L3,L4,L5,L6,L7,u32,u32,u32,u32,u32]);
impl_padded_read_slots!(Env9<L0,L1,L2,L3,L4,L5,L6,L7,L8>; [L0,L1,L2,L3,L4,L5,L6,L7,L8]; [L0,L1,L2,L3,L4,L5,L6,L7,L8,u32,u32,u32,u32]);
impl_padded_read_slots!(Env10<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9]; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,u32,u32,u32]);
impl_padded_read_slots!(Env11<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10]; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,u32,u32]);
impl_padded_read_slots!(Env12<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11]; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,u32]);
impl_padded_read_slots!(Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12]; [L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12]);

/// Fully bound form of a read expression.
///
/// `Slots` retains the recursively computed [`ReadArity`]. The fixed-evaluator
/// requirements only say that this exact expression can be evaluated by the
/// padded ABI (including from a shared offset table); they do not widen the
/// expression. The actual arity is erased only when [`FixedRead`] is
/// constructed by a consumer.
#[doc(hidden)]
pub trait LowerReadExpression:
    ReadExpression + BindSlots<Env0, NextEnv = Self::Slots, Expr = Self::DeviceExpr>
{
    type Slots: SlotEnvironment<Arity = Self::ReadArity> + PaddedReadSlots;
    type DeviceExpr: DeviceExpr<Self::Item>
        + Eval13<
            Self::Item,
            <Self::Slots as PaddedReadSlots>::L0,
            <Self::Slots as PaddedReadSlots>::L1,
            <Self::Slots as PaddedReadSlots>::L2,
            <Self::Slots as PaddedReadSlots>::L3,
            <Self::Slots as PaddedReadSlots>::L4,
            <Self::Slots as PaddedReadSlots>::L5,
            <Self::Slots as PaddedReadSlots>::L6,
            <Self::Slots as PaddedReadSlots>::L7,
            <Self::Slots as PaddedReadSlots>::L8,
            <Self::Slots as PaddedReadSlots>::L9,
            <Self::Slots as PaddedReadSlots>::L10,
            <Self::Slots as PaddedReadSlots>::L11,
            <Self::Slots as PaddedReadSlots>::L12,
        > + Eval13At<
            Self::Item,
            <Self::Slots as PaddedReadSlots>::L0,
            <Self::Slots as PaddedReadSlots>::L1,
            <Self::Slots as PaddedReadSlots>::L2,
            <Self::Slots as PaddedReadSlots>::L3,
            <Self::Slots as PaddedReadSlots>::L4,
            <Self::Slots as PaddedReadSlots>::L5,
            <Self::Slots as PaddedReadSlots>::L6,
            <Self::Slots as PaddedReadSlots>::L7,
            <Self::Slots as PaddedReadSlots>::L8,
            <Self::Slots as PaddedReadSlots>::L9,
            <Self::Slots as PaddedReadSlots>::L10,
            <Self::Slots as PaddedReadSlots>::L11,
            <Self::Slots as PaddedReadSlots>::L12,
        >;
}

impl<Input> LowerReadExpression for Input
where
    Input: ReadExpression + BindSlots<Env0>,
    Input::NextEnv: SlotEnvironment<Arity = <Input as ReadExpression>::ReadArity> + PaddedReadSlots,
    Input::Expr: DeviceExpr<<Input as ReadExpression>::Item>
        + Eval13<
            <Input as ReadExpression>::Item,
            <Input::NextEnv as PaddedReadSlots>::L0,
            <Input::NextEnv as PaddedReadSlots>::L1,
            <Input::NextEnv as PaddedReadSlots>::L2,
            <Input::NextEnv as PaddedReadSlots>::L3,
            <Input::NextEnv as PaddedReadSlots>::L4,
            <Input::NextEnv as PaddedReadSlots>::L5,
            <Input::NextEnv as PaddedReadSlots>::L6,
            <Input::NextEnv as PaddedReadSlots>::L7,
            <Input::NextEnv as PaddedReadSlots>::L8,
            <Input::NextEnv as PaddedReadSlots>::L9,
            <Input::NextEnv as PaddedReadSlots>::L10,
            <Input::NextEnv as PaddedReadSlots>::L11,
            <Input::NextEnv as PaddedReadSlots>::L12,
        > + Eval13At<
            <Input as ReadExpression>::Item,
            <Input::NextEnv as PaddedReadSlots>::L0,
            <Input::NextEnv as PaddedReadSlots>::L1,
            <Input::NextEnv as PaddedReadSlots>::L2,
            <Input::NextEnv as PaddedReadSlots>::L3,
            <Input::NextEnv as PaddedReadSlots>::L4,
            <Input::NextEnv as PaddedReadSlots>::L5,
            <Input::NextEnv as PaddedReadSlots>::L6,
            <Input::NextEnv as PaddedReadSlots>::L7,
            <Input::NextEnv as PaddedReadSlots>::L8,
            <Input::NextEnv as PaddedReadSlots>::L9,
            <Input::NextEnv as PaddedReadSlots>::L10,
            <Input::NextEnv as PaddedReadSlots>::L11,
            <Input::NextEnv as PaddedReadSlots>::L12,
        >,
{
    type DeviceExpr = Input::Expr;
    type Slots = Input::NextEnv;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::eval::*;
    use crate::op::{Identity, UnaryOp};
    use static_assertions::assert_impl_all;

    type One = Column<u8>;
    type Two = Zip<One, Column<u16>>;
    type Three = Zip<Two, Column<u32>>;
    type Four = Zip<Three, Column<u64>>;
    type Five = Zip<Four, Column<i8>>;
    type Six = Zip<Five, Column<i16>>;
    type Seven = Zip<Six, Column<i32>>;
    type SevenItem = (u8, u16, u32, u64, i8, i16, i32);
    type Indices = Counting;
    type Lazified = Transform<Permute<Seven, Indices>, Identity>;
    type Nine = Transform<Permute<Lazified, Indices>, Identity>;
    type Ten = Transform<Permute<Nine, Indices>, Identity>;
    type Eleven = Transform<Permute<Ten, Indices>, Identity>;
    type Twelve = Transform<Permute<Eleven, Indices>, Identity>;
    type Thirteen = Transform<Permute<Twelve, Indices>, Identity>;
    type Device2 = ZipExpr<Slot0<u8, Direct>, Slot1<u16, Direct>, u8, u16>;
    type Device3 = ZipExpr<Device2, Slot2<u32, Direct>, (u8, u16), u32>;
    type Device4 = ZipExpr<Device3, Slot3<u64, Direct>, (u8, u16, u32), u64>;
    type Device5 = ZipExpr<Device4, Slot4<i8, Direct>, (u8, u16, u32, u64), i8>;
    type Device6 = ZipExpr<Device5, Slot5<i16, Direct>, (u8, u16, u32, u64, i8), i16>;
    type SevenDevice = ZipExpr<Device6, Slot6<i32, Direct>, (u8, u16, u32, u64, i8, i16), i32>;
    type LazifiedDevice =
        TransformExpr<PermuteExpr<SevenDevice, Slot7<u32, Count>>, SevenItem, Identity>;
    type LazifiedSlots = Env8<u8, u16, u32, u64, i8, i16, i32, u32>;

    type RuntimeSevenItem = (u32, u32, u32, u32, u32, u32, u32);
    type RuntimeDevice2 = ZipExpr<Slot0<u32, Direct>, Slot1<u32, Direct>, u32, u32>;
    type RuntimeDevice3 = ZipExpr<RuntimeDevice2, Slot2<u32, Direct>, (u32, u32), u32>;
    type RuntimeDevice4 = ZipExpr<RuntimeDevice3, Slot3<u32, Direct>, (u32, u32, u32), u32>;
    type RuntimeDevice5 = ZipExpr<RuntimeDevice4, Slot4<u32, Direct>, (u32, u32, u32, u32), u32>;
    type RuntimeDevice6 =
        ZipExpr<RuntimeDevice5, Slot5<u32, Direct>, (u32, u32, u32, u32, u32), u32>;
    type RuntimeSevenDevice =
        ZipExpr<RuntimeDevice6, Slot6<u32, Direct>, (u32, u32, u32, u32, u32, u32), u32>;
    type RuntimeLazifiedDevice = TransformExpr<
        PermuteExpr<RuntimeSevenDevice, Slot7<u32, Count>>,
        RuntimeSevenItem,
        Identity,
    >;
    type RuntimeMixedDevice = TransformExpr<
        ZipExpr<Slot0<u32, Broadcast>, Slot1<u32, Count>, u32, u32>,
        (u32, u32),
        AddPair,
    >;

    struct AddPair;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for AddPair {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    #[test]
    fn read_arity_and_semantic_item_are_independent() {
        fn assert_shape<E, Item, Arity>()
        where
            E: ReadExpression<Item = Item, ReadArity = Arity>,
            Item: CubeType + 'static,
            Arity: ReadArity,
        {
        }

        fn assert_arity<E, Arity>()
        where
            E: ReadExpression<ReadArity = Arity>,
            Arity: ReadArity,
        {
        }

        type Pair = Zip<Column<u32>, Column<u32>>;
        type Segments = crate::seg::SegmentRead<Column<u32>, Column<u32>>;
        type ByteSegments = crate::seg::SegmentRead<Column<u8>, Column<u32>>;
        assert_arity::<One, A1>();
        assert_arity::<Two, A2>();
        assert_arity::<Three, A3>();
        assert_arity::<Four, A4>();
        assert_arity::<Five, A5>();
        assert_arity::<Six, A6>();
        assert_shape::<Seven, SevenItem, A7>();
        assert_shape::<Lazified, SevenItem, A8>();
        assert_shape::<Transform<Pair, AddPair>, u32, A2>();
        assert_shape::<Segments, Segment<u32>, A2>();
        assert_shape::<ByteSegments, Segment<u8>, A2>();
    }

    #[test]
    fn binding_is_left_to_right_before_fixed_abi_padding() {
        fn assert_binding<E, Expr, Slots>()
        where
            E: BindSlots<Env0, Expr = Expr, NextEnv = Slots>
                + LowerReadExpression<DeviceExpr = Expr, Slots = Slots>,
            E: ReadExpression,
            Expr: DeviceExpr<E::Item>,
            Slots: SlotEnvironment<Arity = E::ReadArity>,
        {
        }

        assert_binding::<Lazified, LazifiedDevice, LazifiedSlots>();

        type Mixed = Zip<Constant<u16>, Counting>;
        type MixedDevice = ZipExpr<Slot0<u16, Broadcast>, Slot1<u32, Count>, u16, u32>;
        assert_binding::<Mixed, MixedDevice, Env2<u16, u32>>();

        type Segments = crate::seg::SegmentRead<Column<u32>, Column<u32>>;
        type SegmentsDevice = SegmentIteratorExpr<Slot0<u32, Direct>, Slot1<u32, Direct>>;
        assert_binding::<Segments, SegmentsDevice, Env2<u32, u32>>();

        type ByteSegments = crate::seg::SegmentRead<Column<u8>, Column<u32>>;
        type ByteSegmentsDevice = SegmentIteratorExpr<Slot0<u8, Direct>, Slot1<u32, Direct>>;
        assert_binding::<ByteSegments, ByteSegmentsDevice, Env2<u8, u32>>();
    }

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_compiles_fixed_evaluator(
        slot0: &[u8],
        slot1: &[u16],
        slot2: &[u32],
        slot3: &[u64],
        slot4: &[i8],
        slot5: &[i16],
        slot6: &[i32],
        slot7: &[u32],
        slot8: &[u32],
        slot9: &[u32],
        slot10: &[u32],
        slot11: &[u32],
        slot12: &[u32],
        offsets: &[u32],
        index: usize,
    ) -> SevenItem {
        LazifiedDevice::eval13(
            slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11,
            slot12, offsets, index,
        )
    }

    #[cubecl::cube(launch_unchecked)]
    fn fixed_eval_runtime_kernel(
        slot0: &[u32],
        slot1: &[u32],
        slot2: &[u32],
        slot3: &[u32],
        slot4: &[u32],
        slot5: &[u32],
        slot6: &[u32],
        slot7: &[u32],
        slot8: &[u32],
        slot9: &[u32],
        slot10: &[u32],
        slot11: &[u32],
        slot12: &[u32],
        offsets: &[u32],
        output: &mut [u32],
    ) {
        let index = ABSOLUTE_POS as usize;
        if index < output.len() {
            let value = RuntimeLazifiedDevice::eval13(
                slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10,
                slot11, slot12, offsets, index,
            );
            output[index] = value.0 + value.1 + value.2 + value.3 + value.4 + value.5 + value.6;
        }
    }

    #[cubecl::cube(launch_unchecked)]
    fn padded_modes_runtime_kernel(
        constant: &[u32],
        counting: &[u32],
        slot2: &[u32],
        slot3: &[u32],
        slot4: &[u32],
        slot5: &[u32],
        slot6: &[u32],
        slot7: &[u32],
        slot8: &[u32],
        slot9: &[u32],
        slot10: &[u32],
        slot11: &[u32],
        slot12: &[u32],
        offsets: &[u32],
        output: &mut [u32],
    ) {
        let index = ABSOLUTE_POS as usize;
        if index < output.len() {
            output[index] = RuntimeMixedDevice::eval13(
                constant, counting, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10,
                slot11, slot12, offsets, index,
            );
        }
    }

    #[test]
    fn fixed_evaluator_executes_padded_expressions_on_cubecl() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        let columns = [
            [10_u32, 11, 12, 13, 14],
            [20_u32, 21, 22, 23, 24],
            [30_u32, 31, 32, 33, 34],
            [40_u32, 41, 42, 43, 44],
            [50_u32, 51, 52, 53, 54],
            [60_u32, 61, 62, 63, 64],
            [70_u32, 71, 72, 73, 74],
        ];
        let handles = columns.map(|column| client.create_from_slice(u32::as_bytes(&column)));
        let count = client.create_from_slice(u32::as_bytes(&[1]));
        let dummy = client.empty(size_of::<u32>());
        let offsets = client.create_from_slice(u32::as_bytes(&[0, 1, 2, 0, 1, 2, 0, 0]));
        let output = client.empty(2 * size_of::<u32>());

        unsafe {
            fixed_eval_runtime_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                BufferArg::from_raw_parts(handles[0].clone(), columns[0].len()),
                BufferArg::from_raw_parts(handles[1].clone(), columns[1].len()),
                BufferArg::from_raw_parts(handles[2].clone(), columns[2].len()),
                BufferArg::from_raw_parts(handles[3].clone(), columns[3].len()),
                BufferArg::from_raw_parts(handles[4].clone(), columns[4].len()),
                BufferArg::from_raw_parts(handles[5].clone(), columns[5].len()),
                BufferArg::from_raw_parts(handles[6].clone(), columns[6].len()),
                BufferArg::from_raw_parts(count, 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(offsets, 8),
                BufferArg::from_raw_parts(output.clone(), 2),
            );
        }

        let bytes = client.read_one_unchecked(output);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[293, 300]);

        let constant = client.create_from_slice(u32::as_bytes(&[100]));
        let counting = client.create_from_slice(u32::as_bytes(&[5]));
        let offsets = client.create_from_slice(u32::as_bytes(&[99, 3]));
        let output = client.empty(2 * size_of::<u32>());
        unsafe {
            padded_modes_runtime_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                BufferArg::from_raw_parts(constant, 1),
                BufferArg::from_raw_parts(counting, 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy.clone(), 1),
                BufferArg::from_raw_parts(dummy, 1),
                BufferArg::from_raw_parts(offsets, 2),
                BufferArg::from_raw_parts(output.clone(), 2),
            );
        }
        let bytes = client.read_one_unchecked(output);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[108, 109]);
    }

    assert_impl_all!(One: ReadExpression, LowerReadExpression);
    assert_impl_all!(Two: ReadExpression, LowerReadExpression);
    assert_impl_all!(Three: ReadExpression, LowerReadExpression);
    assert_impl_all!(Four: ReadExpression, LowerReadExpression);
    assert_impl_all!(Five: ReadExpression, LowerReadExpression);
    assert_impl_all!(Six: ReadExpression, LowerReadExpression);
    assert_impl_all!(Seven: ReadExpression, LowerReadExpression);
    assert_impl_all!(Lazified: ReadExpression, LowerReadExpression);
    assert_impl_all!(Nine: ReadExpression, LowerReadExpression);
    assert_impl_all!(Ten: ReadExpression, LowerReadExpression);
    assert_impl_all!(Eleven: ReadExpression, LowerReadExpression);
    assert_impl_all!(Twelve: ReadExpression, LowerReadExpression);
    assert_impl_all!(Thirteen: ReadExpression, LowerReadExpression);
    assert_impl_all!(Transform<Zip<Column<u32>, Column<u32>>, AddPair>: ReadExpression, LowerReadExpression);
}
