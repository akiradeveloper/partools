//! Lazy iterator sources and adapters.
//!
//! Lazy expressions allocate no result buffer by themselves. They are evaluated by the algorithm
//! that consumes them.

#![allow(private_interfaces)]

use crate::core::{extent, read};

use core::marker::PhantomData;
use std::ops::RangeBounds;

use cubecl::prelude::Runtime;

use crate::api::iter::MIterExtent;
use crate::core::facade as private;
use crate::core::read::SliceExpression;
use crate::core::value::MStorageElement;
use crate::op::{ReductionOp, UnaryOp};
use crate::{Error, MIndex, MIter};

pub use crate::core::read::Taken;

/// Logical adapter appending one shared table view to every context row.
#[derive(Clone, Copy, Debug)]
pub struct WithTable<Contexts, Table> {
    contexts: Contexts,
    table: Table,
}

impl<Contexts, Table> WithTable<Contexts, Table> {
    pub const fn new(contexts: Contexts, table: Table) -> Self {
        Self { contexts, table }
    }

    /// Returns the iterator that determines the output rows.
    pub const fn contexts(&self) -> &Contexts {
        &self.contexts
    }

    /// Returns the iterator exposed as a shared table view.
    pub const fn table(&self) -> &Table {
        &self.table
    }

    /// Decomposes this adapter into its two inputs.
    pub fn into_parts(self) -> (Contexts, Table) {
        (self.contexts, self.table)
    }
}

type SingleTableSegment<Table> = crate::seg::SegmentIterator<Table, Taken<Stride>>;
type RepeatedTableSegment<Table> = Permute<SingleTableSegment<Table>, Taken<Constant<MIndex>>>;
type WithTableComposition<Contexts, Table> =
    crate::api::iter::Zipped<Contexts, RepeatedTableSegment<Table>>;

impl<Contexts, Table> WithTable<Contexts, Table> {
    fn compose<R>(self) -> WithTableComposition<Contexts, Table>
    where
        R: Runtime,
        Contexts: MIter<R>,
        Table: MIter<R>,
    {
        let context_len = self
            .contexts
            .capacity()
            .expect("with_table contexts must have a valid capacity");
        let table_len = self
            .table
            .capacity()
            .expect("with_table table must have a valid capacity");
        let table = crate::seg::SegmentIterator::new(self.table, stride(0, table_len).take(2));
        let repeated = permute(table, constant(0).take(context_len));
        crate::zip2(self.contexts, repeated)
    }
}

#[doc(hidden)]
impl<R, Contexts, Table> MIter<R> for WithTable<Contexts, Table>
where
    R: Runtime,
    Contexts: MIter<R>,
    Table: MIter<R>,
    WithTableComposition<Contexts, Table>: MIter<R>,
{
    type Item = <WithTableComposition<Contexts, Table> as MIter<R>>::Item;
    type Read = <WithTableComposition<Contexts, Table> as MIter<R>>::Read;
    type Slice = <WithTableComposition<Contexts, Table> as MIter<R>>::Slice;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        self.clone().compose::<R>().slice(range)
    }

    fn lower_read(self) -> Self::Read {
        self.compose::<R>().lower_read()
    }
}

impl<R, Contexts, Table> MIterExtent<R> for WithTable<Contexts, Table>
where
    R: Runtime,
    Contexts: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.contexts.capacity()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.contexts.logical_extent()
    }
}

/// Logical lazy permutation lowered only when an algorithm consumes it.
#[derive(Clone, Copy, Debug)]
pub struct Permute<Values, Indices> {
    values: Values,
    indices: Indices,
}

impl<Values, Indices> Permute<Values, Indices> {
    pub const fn new(values: Values, indices: Indices) -> Self {
        Self { values, indices }
    }
}

#[doc(hidden)]
impl<R, Values, Indices> MIter<R> for Permute<Values, Indices>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    read::Permute<Values::Read, Indices::Read>:
        private::KernelInput<R, Item = Values::Item> + SliceExpression,
{
    type Item = Values::Item;
    type Read = read::Permute<Values::Read, Indices::Read>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice a lazy permutation with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        read::Permute::new(self.values.lower_read(), self.indices.lower_read())
    }
}

impl<R, Values, Indices> MIterExtent<R> for Permute<Values, Indices>
where
    R: Runtime,
    Indices: MIter<R, Item = MIndex>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.indices.capacity()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.indices.logical_extent()
    }
}

/// Logical lazy reverse view lowered only when an algorithm consumes it.
#[derive(Clone, Copy, Debug)]
pub struct Reverse<Values> {
    values: Values,
}

impl<Values> Reverse<Values> {
    pub const fn new(values: Values) -> Self {
        Self { values }
    }
}

#[doc(hidden)]
impl<R, Values> MIter<R> for Reverse<Values>
where
    R: Runtime,
    Values: MIter<R>,
    read::Reverse<Values::Read>: private::KernelInput<R, Item = Values::Item> + SliceExpression,
{
    type Item = Values::Item;
    type Read = read::Reverse<Values::Read>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice a lazy reverse view with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        read::Reverse::new(self.values.lower_read())
    }
}

impl<R, Values> MIterExtent<R> for Reverse<Values>
where
    R: Runtime,
    Values: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.values.capacity()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.values.logical_extent()
    }
}

/// Repeats each input item a fixed number of times without materializing it.
#[derive(Clone, Copy, Debug)]
pub struct RepeatEach<Values> {
    values: Values,
    repeats: MIndex,
}

impl<Values> RepeatEach<Values> {
    pub const fn new(values: Values, repeats: MIndex) -> Self {
        Self { values, repeats }
    }

    /// Returns the repeated input.
    pub const fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the repetition count per input item.
    pub const fn repeats(&self) -> MIndex {
        self.repeats
    }

    /// Decomposes this adapter into its input and repetition count.
    pub fn into_parts(self) -> (Values, MIndex) {
        (self.values, self.repeats)
    }

    fn output_len<R>(&self) -> Result<MIndex, Error>
    where
        R: Runtime,
        Values: MIter<R>,
    {
        repeated_len(self.values.capacity()?, self.repeats)
    }
}

#[doc(hidden)]
impl<R, Values> MIter<R> for RepeatEach<Values>
where
    R: Runtime,
    Values: MIter<R>,
    read::Permute<Values::Read, read::DivModCounting>:
        private::KernelInput<R, Item = Values::Item> + SliceExpression,
{
    type Item = Values::Item;
    type Read = read::Permute<Values::Read, read::DivModCounting>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice lazy repeat_each with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        let input_len = self
            .values
            .capacity()
            .expect("lazy repeat_each input must have a valid capacity");
        let output_len =
            repeated_len(input_len, self.repeats).expect("lazy repeat_each length overflow");
        let indices = read::DivModCounting::new(
            0,
            self.repeats.max(1),
            input_len.max(1),
            output_len as usize,
        );
        read::Permute::new(self.values.lower_read(), indices)
    }
}

impl<R, Values> MIterExtent<R> for RepeatEach<Values>
where
    R: Runtime,
    Values: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.output_len::<R>()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(extent::LogicalExtent::fixed(
            self.output_len::<R>()? as usize
        ))
    }
}

/// Repeats the entire input a fixed number of times without materializing it.
#[derive(Clone, Copy, Debug)]
pub struct Tile<Values> {
    values: Values,
    repeats: MIndex,
}

impl<Values> Tile<Values> {
    pub const fn new(values: Values, repeats: MIndex) -> Self {
        Self { values, repeats }
    }

    /// Returns the tiled input.
    pub const fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the number of complete input repetitions.
    pub const fn repeats(&self) -> MIndex {
        self.repeats
    }

    /// Decomposes this adapter into its input and repetition count.
    pub fn into_parts(self) -> (Values, MIndex) {
        (self.values, self.repeats)
    }

    fn output_len<R>(&self) -> Result<MIndex, Error>
    where
        R: Runtime,
        Values: MIter<R>,
    {
        repeated_len(self.values.capacity()?, self.repeats)
    }
}

#[doc(hidden)]
impl<R, Values> MIter<R> for Tile<Values>
where
    R: Runtime,
    Values: MIter<R>,
    read::Permute<Values::Read, read::DivModCounting>:
        private::KernelInput<R, Item = Values::Item> + SliceExpression,
{
    type Item = Values::Item;
    type Read = read::Permute<Values::Read, read::DivModCounting>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice lazy tile with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        let input_len = self
            .values
            .capacity()
            .expect("lazy tile input must have a valid capacity");
        let output_len = repeated_len(input_len, self.repeats).expect("lazy tile length overflow");
        let indices = read::DivModCounting::new(0, 1, input_len.max(1), output_len as usize);
        read::Permute::new(self.values.lower_read(), indices)
    }
}

impl<R, Values> MIterExtent<R> for Tile<Values>
where
    R: Runtime,
    Values: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.output_len::<R>()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(extent::LogicalExtent::fixed(
            self.output_len::<R>()? as usize
        ))
    }
}

fn repeated_len(input_len: MIndex, repeats: MIndex) -> Result<MIndex, Error> {
    let len = (input_len as usize)
        .checked_mul(repeats as usize)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
}

/// Lazily combines each item with its predecessor, preserving the first item.
#[derive(Debug)]
pub struct AdjacentDifference<Input, Op> {
    input: Input,
    _op: PhantomData<fn() -> Op>,
}

impl<Input: Clone, Op> Clone for AdjacentDifference<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for AdjacentDifference<Input, Op> {}

impl<Input, Op> AdjacentDifference<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }

    /// Returns the input iterator.
    pub const fn input(&self) -> &Input {
        &self.input
    }

    /// Returns the input iterator.
    pub fn into_inner(self) -> Input {
        self.input
    }
}

#[doc(hidden)]
impl<R, Input, Op> MIter<R> for AdjacentDifference<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ReductionOp<Input::Item>,
    read::Permute<read::Adjacent<Input::Read, Op>, read::Counting>:
        private::KernelInput<R, Item = Input::Item> + SliceExpression,
{
    type Item = Input::Item;
    type Read = read::Permute<read::Adjacent<Input::Read, Op>, read::Counting>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice lazy adjacent_difference with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        let len = self
            .input
            .capacity()
            .expect("lazy adjacent_difference input must have a valid capacity");
        read::Permute::new(
            read::Adjacent::from_input(self.input.lower_read()),
            read::Counting::new(0, len as usize),
        )
    }
}

impl<R, Input, Op> MIterExtent<R> for AdjacentDifference<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.input.capacity()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }
}

/// Logical lazy map lowered only when an algorithm consumes it.
#[derive(Debug)]
pub struct Map<Input, Op> {
    input: Input,
    _op: PhantomData<fn() -> Op>,
}

impl<Input: Clone, Op> Clone for Map<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Map<Input, Op> {}

impl<Input, Op> Map<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

#[doc(hidden)]
impl<R, Input, Op> MIter<R> for Map<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item>,
    read::Transform<Input::Read, Op>: private::KernelInput<R, Item = Op::Output> + SliceExpression,
{
    type Item = Op::Output;
    type Read = read::Transform<Input::Read, Op>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input)
            .expect("cannot slice a lazy map with an invalid length");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        read::Transform::from_input(self.input.lower_read())
    }
}

impl<R, Input, Op> MIterExtent<R> for Map<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        self.input.capacity()
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }
}

/// An unbounded stream that repeats one value.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len as usize)
    }
}

/// An unbounded stream of consecutive [`MIndex`] values.
#[derive(Clone, Copy, Debug)]
pub struct Counting {
    start: MIndex,
}

impl Counting {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len as usize)
    }
}

/// An unbounded arithmetic progression of [`MIndex`] values.
#[derive(Clone, Copy, Debug)]
pub struct Stride {
    start: MIndex,
    step: MIndex,
}

impl Stride {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len as usize)
    }
}

impl<T> read::TakenSource for Constant<T>
where
    T: MStorageElement,
{
    type Read = read::Constant<T>;

    fn lower(&self, _offset: usize, len: usize) -> Self::Read {
        read::Constant::new(self.value, len)
    }
}

impl read::TakenSource for Counting {
    type Read = read::Counting;

    fn lower(&self, offset: usize, len: usize) -> Self::Read {
        let offset = u32::try_from(offset).expect("counting offset exceeds u32");
        read::Counting::new(
            self.start
                .checked_add(offset)
                .expect("counting start overflow"),
            len,
        )
    }
}

impl read::TakenSource for Stride {
    type Read = read::Stride;

    fn lower(&self, offset: usize, len: usize) -> Self::Read {
        let offset = u32::try_from(offset).expect("stride offset exceeds u32");
        read::Stride::new(
            self.start
                .checked_add(
                    offset
                        .checked_mul(self.step)
                        .expect("stride offset overflow"),
                )
                .expect("stride start overflow"),
            self.step,
            len,
        )
    }
}

/// Creates an unbounded stream that repeats `value`.
///
/// Call [`.take(len)`](Constant::take) before passing it to an algorithm.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let repeated = lazy::constant(7_u32).take(3);
/// let output = map(&exec, repeated, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7]);
/// ```
pub fn constant<T>(value: T) -> Constant<T> {
    Constant { value }
}

/// Creates an unbounded stream of consecutive indices beginning at `start`.
///
/// Call [`.take(len)`](Counting::take) before passing it to an algorithm.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let indices = lazy::counting(1).take(3);
/// let output = gather(&exec, values.slice(..), indices).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 40]);
/// ```
pub fn counting(start: MIndex) -> Counting {
    Counting { start }
}

/// Creates an unbounded arithmetic progression.
///
/// Item `i` is `start + i * step`. Call
/// [`.take(len)`](Stride::take) before passing it to an algorithm.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = map(&exec, lazy::stride(2, 3).take(4), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&values).unwrap(), vec![2, 5, 8, 11]);
/// ```
pub fn stride(start: MIndex, step: MIndex) -> Stride {
    Stride { start, step }
}

/// Appends a shared view of the entire `table` to every `contexts` item.
///
/// The result has the same length as `contexts`. The table is not copied per
/// item: the consuming kernel receives a [`crate::seg::Segment`] backed
/// directly by the table expression. Flat context tuples stay flat, so calling
/// this twice produces `(T, Segment<A>, Segment<B>)`.
///
/// This is lowered as the ordinary composition of a one-segment
/// [`crate::seg::SegmentIterator`], [`permute`], and [`crate::zip2`].
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, seg::Segment, vector::map};
///
/// struct Lookup;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<(u32, Segment<u32>)> for Lookup {
///     type Output = u32;
///
///     fn apply(input: (u32, Segment<u32>)) -> u32 {
///         input.1.at(input.0)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let table = exec.to_device(&[10_u32, 20, 30]);
/// let input = lazy::with_table(indices.slice(..), table.slice(..));
/// let output = map(&exec, input, Lookup).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10, 20]);
/// ```
pub fn with_table<Contexts, Table>(contexts: Contexts, table: Table) -> WithTable<Contexts, Table> {
    WithTable::new(contexts, table)
}

/// Lazily applies `op` whenever an algorithm reads an item.
///
/// This does not allocate an intermediate device buffer.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// struct Double;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<u32> for Double {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value * 2
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let doubled = lazy::map(input.slice(..), Double);
/// let output = map(&exec, doubled, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6]);
/// ```
pub fn map<Input, Op>(input: Input, op: Op) -> Map<Input, Op> {
    Map::new(input, op)
}

/// Lazily reads `values[indices[i]]`.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0]);
/// let permuted = lazy::permute(values.slice(..), indices.slice(..));
/// let output = map(&exec, permuted, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10]);
/// ```
pub fn permute<Values, Indices>(values: Values, indices: Indices) -> Permute<Values, Indices> {
    Permute::new(values, indices)
}

/// Lazily reads an input in reverse order.
///
/// This generates reverse indices as part of the consuming kernel and does
/// not allocate an intermediate index or value buffer.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30]);
/// let output = map(&exec, lazy::reverse(input.slice(..)), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20, 10]);
/// ```
pub fn reverse<Values>(values: Values) -> Reverse<Values> {
    Reverse::new(values)
}

/// Lazily repeats every input item `repeats` times.
///
/// `repeat_each([a, b], 3)` produces `[a, a, a, b, b, b]`. A zero
/// repetition count or an empty input produces an empty iterator.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20]);
/// let output = map(
///     &exec,
///     lazy::repeat_each(input.slice(..), 3),
///     op::Identity,
/// ).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 10, 10, 20, 20, 20]);
/// ```
pub fn repeat_each<Values>(values: Values, repeats: MIndex) -> RepeatEach<Values> {
    RepeatEach::new(values, repeats)
}

/// Lazily repeats the complete input `repeats` times.
///
/// `tile([a, b], 3)` produces `[a, b, a, b, a, b]`. A zero repetition
/// count or an empty input produces an empty iterator.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20]);
/// let output = map(&exec, lazy::tile(input.slice(..), 3), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 10, 20, 10, 20]);
/// ```
pub fn tile<Values>(values: Values, repeats: MIndex) -> Tile<Values> {
    Tile::new(values, repeats)
}

/// Lazily combines adjacent input items with `op`.
///
/// The first item is preserved. Each later item is `op(previous, current)`.
/// No intermediate value buffer is allocated.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// struct Difference;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Difference {
///     fn apply(previous: u32, current: u32) -> u32 {
///         current - previous
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[3_u32, 8, 10]);
/// let output = map(
///     &exec,
///     lazy::adjacent_difference(input.slice(..), Difference),
///     op::Identity,
/// ).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![3, 5, 2]);
/// ```
pub fn adjacent_difference<Input, Op>(input: Input, op: Op) -> AdjacentDifference<Input, Op> {
    AdjacentDifference::new(input, op)
}

/// Wraps an input in a lazy identity map.
///
/// This is useful in tests and when an explicit lazy map node is required.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = map(&exec, lazy::identity(input.slice(..)), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn identity<Input>(input: Input) -> Map<Input, crate::op::Identity> {
    Map::new(input, crate::op::Identity)
}
