#![allow(private_interfaces)]

use crate::core::{allocation, extent, indexed, output, read, reduce, storage, value};

use core::marker::PhantomData;
use cubecl::prelude::{CubeType, Runtime};
use std::ops::RangeBounds;

use crate::core::facade as private;
use crate::core::iter::Zip;
use crate::core::output::{ReadOutput, SliceOutput};
use crate::core::read::SliceExpression;
use crate::{Error, Executor, MIndex};

pub(crate) trait MStorageExtent<R: Runtime> {
    fn capacity(&self) -> Result<MIndex, Error>;
    fn logical_extent(&self) -> extent::LogicalExtent;
    fn set_logical_extent(&mut self, extent: extent::LogicalExtent);
}

/// Owned device storage for one flat logical row type.
///
/// Length-changing algorithms may keep an initialized logical prefix in an
/// upper-bound allocation. That extent is propagated internally and is not
/// part of the public storage or iterator API.
#[allow(private_bounds, private_interfaces)]
pub trait MStorage<R: Runtime>: MStorageExtent<R> + Sized {
    type Item: CubeType + Send + Sync + 'static;

    /// Owned physical columns in the same flat order as [`Self::Item`].
    ///
    /// A scalar row returns one [`crate::DeviceVec`]. A tuple row returns a
    /// native tuple of device vectors, regardless of the internal storage tree.
    type Columns;

    #[doc(hidden)]
    type Slice<'a>: MIter<R, Item = Self::Item>
    where
        Self: 'a;

    #[doc(hidden)]
    type SliceMut<'a>: MIterMut<R, Item = Self::Item>
    where
        Self: 'a;

    /// Allocates uninitialized storage for `len` logical items.
    ///
    /// The storage must be completely written before it is read.
    #[doc(hidden)]
    fn allocate(exec: &Executor<R>, len: MIndex) -> Self;

    /// Consumes this storage and returns its columns as a flat native tuple.
    fn into_columns(self) -> Self::Columns;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<MIndex>;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<MIndex>;
}

/// Internal algorithm dispatch carried by every canonically allocatable row.
#[doc(hidden)]
pub(crate) trait ItemDispatch<R: Runtime> {
    type Item: CubeType + Send + Sync + Sized + 'static + allocation::ScratchStorage<R>;
    type Storage: MStorage<R, Item = Self::Item>;

    fn store_value(exec: &Executor<R>, value: Self::Item) -> Result<Self::Storage, Error>;

    fn read_value(exec: &Executor<R>, storage: &Self::Storage) -> Result<Self::Item, Error>;

    fn into_scratch(
        storage: Self::Storage,
    ) -> <Self::Item as allocation::ScratchStorage<R>>::Storage;

    fn reduce<Input, Op>(
        exec: &Executor<R>,
        input: Input,
        init: Self::Storage,
        op: Op,
    ) -> Result<Self::Storage, Error>
    where
        Input: MIter<R, Item = Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>;

    fn sort_owned<Input, Less>(
        exec: &Executor<R>,
        input: Input,
        less: Less,
    ) -> Result<Self::Storage, Error>
    where
        Input: MIter<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;
}

impl<R, Item> ItemDispatch<R> for Item
where
    R: Runtime,
    Item: KernelRow + allocation::RowAlloc<R> + allocation::ScratchStorage<R>,
    <Item as allocation::RowAlloc<R>>::RowStorage: MStorage<R, Item = Item>,
    Item: allocation::ScratchStorage<R, Storage = <Item as allocation::RowAlloc<R>>::RowStorage>,
{
    type Item = Item;
    type Storage = <Item as allocation::RowAlloc<R>>::RowStorage;

    fn store_value(exec: &Executor<R>, value: Item) -> Result<Self::Storage, Error> {
        Ok(allocation::RowStorage::from_item(exec, value))
    }

    fn read_value(exec: &Executor<R>, storage: &Self::Storage) -> Result<Item, Error> {
        allocation::RowStorage::read_first(storage, exec)
    }

    fn into_scratch(storage: Self::Storage) -> <Item as allocation::ScratchStorage<R>>::Storage {
        storage
    }

    fn reduce<Input, Op>(
        exec: &Executor<R>,
        input: Input,
        init: Self::Storage,
        op: Op,
    ) -> Result<Self::Storage, Error>
    where
        Input: MIter<R, Item = Item>,
        Op: crate::op::ReductionOp<Item>,
    {
        reduce::reduce(exec, lower_fixed::<R, _>(input), init, op)
    }

    fn sort_owned<Input, Less>(
        exec: &Executor<R>,
        input: Input,
        less: Less,
    ) -> Result<Self::Storage, Error>
    where
        Input: MIter<R, Item = Item>,
        Less: crate::op::BinaryPredicateOp<Item>,
    {
        let len = input.capacity()?;
        let extent = input.logical_extent()?;
        let mut output = <Self::Storage as MStorage<R>>::allocate(exec, len);
        MStorageExtent::set_logical_extent(&mut output, extent);
        crate::vector::sort_values_by_key_into(
            exec,
            input.clone(),
            input,
            less,
            MStorage::slice_mut(&output, ..),
        )?;
        Ok(output)
    }
}

pub(crate) fn checked_len(len: usize) -> Result<MIndex, Error> {
    MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
}

/// Lowers a logical iterator while preserving its actual physical read arity.
pub(crate) fn lower<R, Input>(input: Input) -> Input::Read
where
    R: Runtime,
    Input: MIter<R>,
{
    input.lower_read()
}

/// Lowers a logical iterator and selects the current fixed thirteen-slot ABI.
///
/// Keeping this conversion explicit at consumer call sites leaves room for an
/// exact-arity launch policy without changing [`MIter`] or read expressions.
pub(crate) fn lower_fixed<R, Input>(input: Input) -> read::FixedRead<Input::Read>
where
    R: Runtime,
    Input: MIter<R>,
{
    private::KernelInput::into_fixed(lower::<R, _>(input))
}

/// Internal marker for values supported by the physical storage ABI.
///
/// This is deliberately not required by [`MIter`]; read-only semantic values
/// may have no storage layout. Implementing this marker does not imply that
/// new owned storage can be allocated.
#[doc(hidden)]
pub(crate) trait KernelRow:
    storage::StorageLayout<
        StorageLeaves: private::KernelValue<
            StorageArity = <Self as storage::StorageLayout>::StorageArity,
        > + private::KernelOutputLeaves,
    >
{
}

impl<Item> KernelRow for Item
where
    Item: storage::StorageLayout,
    Item::StorageLeaves:
        private::KernelValue<StorageArity = Item::StorageArity> + private::KernelOutputLeaves,
{
}

/// Item capability for allocating canonical owned device storage.
///
/// This is the capability required by [`Executor::alloc`] and algorithms that
/// return [`MVec`]. [`Owned`](Self::Owned) defines the canonical [`MStorage`]
/// representation for the logical row type. Temporary storage and algorithm
/// dispatch are internal implementation details, not separate public item
/// capabilities.
#[allow(private_bounds)]
pub trait MAlloc<R: Runtime>: CubeType + Send + Sync + Sized + 'static {
    /// The canonical owned SoA storage for this flat logical row type.
    ///
    /// [`MVec<R, Self>`](MVec) is an alias for this associated type.
    type Owned: MStorage<R, Item = Self> + Clone + Send + Sync + 'static;

    #[doc(hidden)]
    type Dispatch: ItemDispatch<R, Item = Self, Storage = Self::Owned>;
}

#[doc(hidden)]
impl<R, Item> MAlloc<R> for Item
where
    R: Runtime,
    Item: allocation::RowAlloc<R>
        + allocation::ScratchStorage<R, Storage = <Item as allocation::RowAlloc<R>>::RowStorage>
        + ItemDispatch<R, Item = Item, Storage = <Item as allocation::RowAlloc<R>>::RowStorage>,
    <Item as allocation::RowAlloc<R>>::RowStorage: MStorage<R, Item = Item>,
{
    type Owned = <Item as allocation::RowAlloc<R>>::RowStorage;
    type Dispatch = Item;
}

/// Backward-compatible name for [`MAlloc`].
#[doc(hidden)]
pub use MAlloc as MItem;

/// Owned device storage for a flat row type.
///
/// Length-changing algorithms may return an upper-bound allocation carrying a
/// smaller device-resident logical length.
pub type MVec<R, Item> = <Item as MAlloc<R>>::Owned;

trait RadixArity {}

impl RadixArity for storage::S1 {}
impl RadixArity for storage::S2 {}
impl RadixArity for storage::S3 {}

mod radix_private {
    use crate::core::storage;
    pub trait Sealed {}

    impl<Item> Sealed for Item
    where
        Item: storage::StorageLayout,
        Item::StorageArity: super::RadixArity,
    {
    }
}

/// A flat numeric value with an order-preserving radix representation.
///
/// Primitive leaves may be `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`,
/// `f32`, or `f64`, provided the runtime supports that scalar type. Integers use
/// their natural ascending numeric order. Floating-point leaves use the same
/// total order as [`f32::total_cmp`] and [`f64::total_cmp`]. Values may contain
/// up to three columns; compound values use lexicographic leaf order from left
/// to right.
#[allow(private_bounds)]
pub trait MRadix<R: Runtime>: MAlloc<R> + radix_private::Sealed {
    #[doc(hidden)]
    fn radix_permutation(
        exec: &Executor<R>,
        keys: &MVec<R, Self>,
        len: usize,
    ) -> Result<crate::DeviceVec<R, u32>, Error>;
}

#[doc(hidden)]
#[allow(private_bounds)]
impl<R, Item> MRadix<R> for Item
where
    R: Runtime,
    Item: MAlloc<R> + radix_private::Sealed,
    MVec<R, Item>: crate::core::radix::RadixStorage<R>,
{
    fn radix_permutation(
        exec: &Executor<R>,
        keys: &MVec<R, Self>,
        len: usize,
    ) -> Result<crate::DeviceVec<R, u32>, Error> {
        crate::core::radix::permutation(exec, keys, len, keys.logical_extent())
    }
}

/// A lowered destination whose concrete ABI is known inside the crate.
#[doc(hidden)]
pub(crate) trait ConcreteOutput<R, Item>:
    output::OutputExpression<Item = Item>
    + output::LowerOutputExpression
    + output::ReadOutput
    + output::StageOutput<R, read::Env0>
    + private::KernelOutput<R>
    + crate::core::selection::FillOutput<R>
    + output::SliceOutput
where
    R: Runtime,
    Item: KernelRow + allocation::ScratchStorage<R>,
    Self::Slots:
        output::PaddedOutputSlots<Leaves = <Item as storage::StorageLayout>::StorageLeaves>,
{
}

impl<R, Item, Output> ConcreteOutput<R, Item> for Output
where
    R: Runtime,
    Item: KernelRow + allocation::ScratchStorage<R>,
    Output: output::OutputExpression<Item = Item>
        + output::LowerOutputExpression
        + output::ReadOutput
        + output::StageOutput<R, read::Env0>
        + private::KernelOutput<R>
        + crate::core::selection::FillOutput<R>
        + output::SliceOutput,
    Output::Slots:
        output::PaddedOutputSlots<Leaves = <Item as storage::StorageLayout>::StorageLeaves>,
{
}

/// An operation that is type-checked only after a concrete output ABI is known.
#[doc(hidden)]
pub(crate) trait OutputOperation<R: Runtime, Item: CubeType + Send + Sync + 'static> {
    type Result;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: KernelRow + allocation::ScratchStorage<R>,
        Output: ConcreteOutput<R, Item>;
}

pub(crate) struct FillValueOperation<'a, R: Runtime, Value> {
    pub(crate) exec: &'a Executor<R>,
    pub(crate) value: Value,
}

impl<R, Item, Value> OutputOperation<R, Item> for FillValueOperation<'_, R, Value>
where
    R: Runtime,
    Item: MAlloc<R>,
    Value: MIter<R, Item = Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: KernelRow + allocation::ScratchStorage<R>,
        Output: ConcreteOutput<R, Item>,
    {
        let len = output::OutputExpression::physical_len(&output)?;
        indexed::gather_direct(
            self.exec,
            lower::<R, _>(self.value),
            read::Constant::new(0u32, len),
            output,
        )
    }
}

pub(crate) trait MIterExtent<R: Runtime> {
    fn capacity(&self) -> Result<MIndex, Error>;

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(extent::LogicalExtent::fixed(self.capacity()? as usize))
    }
}

/// Public read-only logical row stream.
///
/// Device slices, lazy expressions, and values returned by the `zipN` helpers
/// implement this trait. Every tuple item is a native flat tuple, independent
/// of how calls to [`zip2`] are grouped.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, MIter, lazy, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
/// let indices = lazy::counting(0).take(5);
/// let middle = indices.slice(1..4);
/// let output = gather(&exec, values.slice(..), middle).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 40]);
/// ```
#[allow(private_bounds, private_interfaces)]
pub trait MIter<R: Runtime>: MIterExtent<R> + Clone + Sized {
    /// Semantic value produced by one indexed read.
    ///
    /// Reading does not imply that the value has a storage layout, can be
    /// allocated, or can cross a write boundary.
    type Item: CubeType + Send + Sync + 'static;

    /// Exact-arity device read plan for this iterator.
    #[doc(hidden)]
    type Read: private::KernelInput<R, Item = Self::Item> + read::SliceExpression;

    #[doc(hidden)]
    type Slice;

    /// Returns a zero-copy logical subrange of this iterator.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>;

    #[doc(hidden)]
    fn lower_read(self) -> Self::Read;
}

pub(crate) trait MIterMutExtent<R: Runtime> {
    fn capacity(&self) -> Result<MIndex, Error>;
}

/// Public preallocated output stream.
///
/// Device mutable slices and values returned by the `zipN` helpers implement
/// this trait. Their logical item is always a native flat tuple.
#[allow(private_bounds, private_interfaces)]
pub trait MIterMut<R: Runtime>: MIterMutExtent<R> + Sized {
    /// Semantic value stored by one output row.
    ///
    /// Writing to a preallocated destination does not imply that the value can
    /// be allocated as new owned storage.
    type Item: CubeType + Send + Sync + 'static;

    #[doc(hidden)]
    type Slice;

    #[doc(hidden)]
    type SliceMut;

    /// Returns a read-only zero-copy subrange of this output.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>;

    /// Returns a mutable zero-copy subrange of this output.
    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>;

    /// Internal fixed-ABI output tree. This is a structural lowering contract
    /// and contains no algorithm operations.
    #[doc(hidden)]
    type OutputSlots;

    #[doc(hidden)]
    type LoweredOutput: output::OutputExpression<Item = Self::Item>
        + output::LowerOutputExpression<Slots = Self::OutputSlots>
        + output::StageOutput<R, read::Env0>
        + crate::core::selection::FillOutput<R>
        + output::SliceOutput;

    #[doc(hidden)]
    fn lower_output(self) -> Self::LoweredOutput;

    #[doc(hidden)]
    #[allow(private_bounds, private_interfaces)]
    #[allow(private_bounds)]
    fn run_output_operation<Operation>(self, operation: Operation) -> Operation::Result
    where
        Operation: OutputOperation<R, Self::Item>;
}

mod iter_private {
    use crate::core::output;
    pub trait InternalOutput {}

    impl<T> InternalOutput for crate::core::runtime::ColumnMut<T> {}

    impl<Left, Right> InternalOutput for crate::core::iter::Zip<Left, Right>
    where
        Left: InternalOutput,
        Right: InternalOutput,
    {
    }

    impl<R, Output> InternalOutput for output::Slice<R, Output> where Output: InternalOutput {}
}

#[doc(hidden)]
impl<R, T> MIter<R> for crate::DeviceSlice<R, T>
where
    R: Runtime,
    T: value::MStorageElement,
    read::Column<T>: private::KernelInput<R, Item = T> + read::SliceExpression,
{
    type Item = T;
    type Read = read::Column<T>;
    type Slice = Self;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        crate::DeviceSlice::slice(self, range)
    }

    fn lower_read(self) -> Self::Read {
        self.into_column()
    }
}

impl<R, T> MIterExtent<R> for crate::DeviceSlice<R, T>
where
    R: Runtime,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(self.column.len)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        Ok(self.column.extent.clone())
    }
}

#[doc(hidden)]
#[allow(private_bounds)]
impl<R, T> MIterMut<R> for crate::DeviceSliceMut<R, T>
where
    R: Runtime,
    T: KernelRow + allocation::ScratchStorage<R>,
    crate::core::runtime::ColumnMut<T>: output::OutputExpression<Item = T>
        + output::LowerOutputExpression
        + output::ReadOutput
        + output::StageOutput<R, read::Env0>
        + crate::core::selection::FillOutput<R>
        + output::SliceOutput,
    <crate::core::runtime::ColumnMut<T> as output::LowerOutputExpression>::Slots:
        output::PaddedOutputSlots<Leaves = <T as storage::StorageLayout>::StorageLeaves>
            + output::OutputSlotEnvironment<
                StorageArity = <T as storage::StorageLayout>::StorageArity,
            >,
{
    type Item = T;
    type Slice = crate::DeviceSlice<R, T>;
    type SliceMut = Self;
    type OutputSlots = <crate::core::runtime::ColumnMut<T> as output::LowerOutputExpression>::Slots;
    type LoweredOutput = crate::core::runtime::ColumnMut<T>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        crate::DeviceSliceMut::slice(self, range)
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>,
    {
        crate::DeviceSliceMut::slice_mut(self, range)
    }

    fn lower_output(self) -> Self::LoweredOutput {
        self.into_output()
    }

    fn run_output_operation<Operation>(self, operation: Operation) -> Operation::Result
    where
        Operation: OutputOperation<R, Self::Item>,
    {
        operation.run(self.into_output())
    }
}

impl<R: Runtime, T> MIterMutExtent<R> for crate::DeviceSliceMut<R, T> {
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(self.output.len)
    }
}

#[doc(hidden)]
impl<R, Input, Item> MIter<R> for Input
where
    R: Runtime,
    Input: Clone
        + private::KernelInput<R, Item = Item>
        + read::SliceExpression
        + read::LowerReadExpression,
    Item: CubeType + Send + Sync + 'static,
{
    type Item = Item;
    type Read = Input;
    type Slice = read::Slice<R, Input>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = private::physical_len::<R, _>(self)
            .expect("cannot slice an iterator with an invalid length");
        let (start, len) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(self.slice_expression(start, len))
    }

    fn lower_read(self) -> Self::Read {
        self
    }
}

impl<R, Input, Item> MIterExtent<R> for Input
where
    R: Runtime,
    Input: Clone
        + private::KernelInput<R, Item = Item>
        + read::SliceExpression
        + read::LowerReadExpression,
    Item: CubeType + Send + Sync + 'static,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(private::physical_len::<R, _>(self)?)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        private::logical_extent::<R, _>(self)
    }
}

#[doc(hidden)]
impl<R, Output> MIterMut<R> for Output
where
    R: Runtime,
    Output: output::OutputExpression
        + output::LowerOutputExpression
        + output::ReadOutput
        + output::StageOutput<R, read::Env0>
        + crate::core::selection::FillOutput<R>
        + output::SliceOutput
        + iter_private::InternalOutput,
    Output::Item: KernelRow + allocation::ScratchStorage<R>,
    Output::Slots: output::PaddedOutputSlots<Leaves = <Output::Item as storage::StorageLayout>::StorageLeaves>
        + output::OutputSlotEnvironment<
            StorageArity = <Output::Item as storage::StorageLayout>::StorageArity,
        >,
{
    type Item = <Output as output::OutputExpression>::Item;
    type Slice = read::Slice<R, Output::Read>;
    type SliceMut = output::Slice<R, Output>;
    type OutputSlots = Output::Slots;
    type LoweredOutput = Output;
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = output::OutputExpression::physical_len(self)
            .expect("cannot slice an output with an invalid length");
        let (start, len) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(self.slice_read(start..start + len))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = output::OutputExpression::physical_len(self)
            .expect("cannot slice an output with an invalid length");
        let (start, len) = read::resolve_mindex_slice_range(len, range);
        output::Slice::new(self.slice_output(start..start + len))
    }

    fn lower_output(self) -> Self::LoweredOutput {
        self
    }

    #[allow(private_bounds, private_interfaces)]
    fn run_output_operation<Operation>(self, operation: Operation) -> Operation::Result
    where
        Operation: OutputOperation<R, Self::Item>,
    {
        operation.run(self)
    }
}

impl<R, Output> MIterMutExtent<R> for Output
where
    R: Runtime,
    Output: output::OutputExpression + iter_private::InternalOutput,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(output::OutputExpression::physical_len(self)?)
    }
}

/// Logical read view over opaque owned row storage.
#[doc(hidden)]
pub struct StorageSlice<'a, R, Storage> {
    storage: &'a Storage,
    start: usize,
    len: usize,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Storage> Copy for StorageSlice<'_, R, Storage> {}

impl<R, Storage> Clone for StorageSlice<'_, R, Storage> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R, Storage> StorageSlice<'a, R, Storage> {
    pub(crate) const fn new(storage: &'a Storage, start: usize, len: usize) -> Self {
        Self {
            storage,
            start,
            len,
            _runtime: PhantomData,
        }
    }
}

impl<R, Storage> MIter<R> for StorageSlice<'_, R, Storage>
where
    R: Runtime,
    Storage: allocation::RowStorage<R>,
    Storage::Read: private::KernelInput<R, Item = Storage::Item> + read::SliceExpression,
{
    type Item = Storage::Item;
    type Read = Storage::Read;
    type Slice = Self;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let (start, len) = read::resolve_mindex_slice_range(self.len, range);
        Self::new(self.storage, self.start + start, len)
    }

    fn lower_read(self) -> Self::Read {
        allocation::RowStorage::slice(self.storage, self.start..self.start + self.len)
    }
}

impl<R: Runtime, Storage> MIterExtent<R> for StorageSlice<'_, R, Storage>
where
    Storage: allocation::RowStorage<R>,
    Storage::Read: private::KernelInput<R, Item = Storage::Item> + read::SliceExpression,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(self.len)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        private::logical_extent::<R, _>(&(*self).lower_read())
    }
}

/// Logical mutable view over opaque owned row storage.
#[doc(hidden)]
pub struct StorageSliceMut<'a, R, Storage> {
    storage: &'a Storage,
    start: usize,
    len: usize,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Storage> Copy for StorageSliceMut<'_, R, Storage> {}

impl<R, Storage> Clone for StorageSliceMut<'_, R, Storage> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R, Storage> StorageSliceMut<'a, R, Storage> {
    pub(crate) const fn new(storage: &'a Storage, start: usize, len: usize) -> Self {
        Self {
            storage,
            start,
            len,
            _runtime: PhantomData,
        }
    }
}

impl<'a, R, Storage> MIterMut<R> for StorageSliceMut<'a, R, Storage>
where
    R: Runtime,
    Storage: allocation::RowStorage<R>,
    Storage::Item: KernelRow + allocation::ScratchStorage<R>,
    Storage::Read: private::KernelInput<R, Item = Storage::Item> + read::SliceExpression,
    Storage::Write:
        ReadOutput + private::KernelOutput<R> + crate::core::selection::FillOutput<R> + SliceOutput,
    Storage::WriteSlots: output::OutputSlotEnvironment<
            StorageArity = <Storage::Item as storage::StorageLayout>::StorageArity,
        >,
{
    type Item = Storage::Item;
    type Slice = StorageSlice<'a, R, Storage>;
    type SliceMut = Self;
    type OutputSlots = Storage::WriteSlots;
    type LoweredOutput = Storage::Write;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let (start, len) = read::resolve_mindex_slice_range(self.len, range);
        StorageSlice::new(self.storage, self.start + start, len)
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>,
    {
        let (start, len) = read::resolve_mindex_slice_range(self.len, range);
        Self::new(self.storage, self.start + start, len)
    }

    fn lower_output(self) -> Self::LoweredOutput {
        allocation::RowStorage::slice_mut(self.storage, self.start..self.start + self.len)
    }

    #[allow(private_bounds, private_interfaces)]
    fn run_output_operation<Operation>(self, operation: Operation) -> Operation::Result
    where
        Operation: OutputOperation<R, Self::Item>,
    {
        operation.run(self.lower_output())
    }
}

impl<R: Runtime, Storage> MIterMutExtent<R> for StorageSliceMut<'_, R, Storage> {
    fn capacity(&self) -> Result<MIndex, Error> {
        checked_len(self.len)
    }
}

/// Logical composition of two iterator schemas.
///
/// Its operands are lowered into the private physical `Zip` tree only when an
/// algorithm consumes it. The wrapper itself carries no public tree-shape
/// semantics: its item is the flat concatenation of both operand items.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Zipped<Left, Right>(Left, Right);

impl<Left, Right> Zipped<Left, Right> {
    pub(crate) const fn new(left: Left, right: Right) -> Self {
        Self(left, right)
    }
}

impl<R, Left, Right> MIter<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R>,
    Zip<Left::Read, Right::Read>: private::KernelInput<R> + read::SliceExpression,
{
    type Item = <Zip<Left::Read, Right::Read> as read::ReadExpression>::Item;
    type Read = Zip<Left::Read, Right::Read>;
    type Slice = read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let input = self.clone().lower_read();
        let len = private::physical_len::<R, _>(&input).expect("zip operands have equal lengths");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(input.slice_expression(start, count))
    }

    fn lower_read(self) -> Self::Read {
        Zip::new(self.0.lower_read(), self.1.lower_read())
    }
}

impl<R, Left, Right> MIterExtent<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        let left = self.0.capacity()?;
        let right = self.1.capacity()?;
        if left != right {
            return Err(Error::LengthMismatch {
                left: left as usize,
                right: right as usize,
            });
        }
        Ok(left)
    }

    fn logical_extent(&self) -> Result<extent::LogicalExtent, Error> {
        self.0.logical_extent()?.zipped(&self.1.logical_extent()?)
    }
}

impl<R, Left, Right> MIterMut<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIterMut<R> + Clone,
    Right: MIterMut<R> + Clone,
    Zip<Left::LoweredOutput, Right::LoweredOutput>: output::OutputExpression
        + output::LowerOutputExpression
        + output::ReadOutput
        + output::StageOutput<R, read::Env0>
        + crate::core::selection::FillOutput<R>
        + output::SliceOutput
        + private::KernelOutput<R>,
    <Zip<Left::LoweredOutput, Right::LoweredOutput> as output::OutputExpression>::Item:
        KernelRow + allocation::ScratchStorage<R>,
    <Zip<Left::LoweredOutput, Right::LoweredOutput> as output::LowerOutputExpression>::Slots:
        output::PaddedOutputSlots<
            Leaves = <<Zip<Left::LoweredOutput, Right::LoweredOutput> as output::OutputExpression>::Item as storage::StorageLayout>::StorageLeaves,
        > + output::OutputSlotEnvironment<
            StorageArity = <<Zip<Left::LoweredOutput, Right::LoweredOutput> as output::OutputExpression>::Item as storage::StorageLayout>::StorageArity,
        >,
{
    type Item =
        <Zip<Left::LoweredOutput, Right::LoweredOutput> as output::OutputExpression>::Item;
    type Slice = read::Slice<
        R,
        <Zip<Left::LoweredOutput, Right::LoweredOutput> as output::ReadOutput>::Read,
    >;
    type SliceMut = output::Slice<R, Zip<Left::LoweredOutput, Right::LoweredOutput>>;
    type OutputSlots = <Zip<Left::LoweredOutput, Right::LoweredOutput> as output::LowerOutputExpression>::Slots;
    type LoweredOutput = Zip<Left::LoweredOutput, Right::LoweredOutput>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let output = Zip::new(
            self.0.clone().lower_output(),
            self.1.clone().lower_output(),
        );
        let len = output::OutputExpression::physical_len(&output)
            .expect("zip outputs have equal lengths");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        read::Slice::new(output.slice_read(start..start + count))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>,
    {
        let output = Zip::new(
            self.0.clone().lower_output(),
            self.1.clone().lower_output(),
        );
        let len = output::OutputExpression::physical_len(&output)
            .expect("zip outputs have equal lengths");
        let (start, count) = read::resolve_mindex_slice_range(len, range);
        output::Slice::new(output.slice_output(start..start + count))
    }

    fn lower_output(self) -> Self::LoweredOutput {
        Zip::new(self.0.lower_output(), self.1.lower_output())
    }

    #[allow(private_bounds, private_interfaces)]
    fn run_output_operation<Operation>(self, operation: Operation) -> Operation::Result
    where
        Operation: OutputOperation<R, Self::Item>,
    {
        operation.run(self.lower_output())
    }
}

impl<R, Left, Right> MIterMutExtent<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIterMut<R>,
    Right: MIterMut<R>,
{
    fn capacity(&self) -> Result<MIndex, Error> {
        let left = MIterMutExtent::capacity(&self.0)?;
        let right = MIterMutExtent::capacity(&self.1)?;
        if left != right {
            return Err(Error::LengthMismatch {
                left: left as usize,
                right: right as usize,
            });
        }
        Ok(left)
    }
}

/// Combines two iterators into one iterator of paired items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::map, zip2};
///
/// struct AddPair;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<(u32, u32)> for AddPair {
///     type Output = u32;
///
///     fn apply(value: (u32, u32)) -> u32 {
///         value.0 + value.1
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[10_u32, 20, 30]);
/// let input = zip2(left.slice(..), right.slice(..));
/// let output = map(&exec, input, AddPair).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![11, 22, 33]);
/// ```
pub fn zip2<A, B>(a: A, b: B) -> Zipped<A, B> {
    Zipped::new(a, b)
}

/// Combines three iterators into an iterator whose item is `(A, B, C)`.
///
/// See [`zip2`] for a complete example. Grouping `zip2` calls differently does
/// not change the flat logical item type.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> Zipped<Zipped<A, B>, C> {
    Zipped::new(Zipped::new(a, b), c)
}

/// Combines four iterators into an iterator whose item is `(A, B, C, D)`.
///
/// See [`zip2`] for a complete example.
pub fn zip4<A, B, C, D>(a: A, b: B, c: C, d: D) -> Zipped<Zipped<Zipped<A, B>, C>, D> {
    Zipped::new(zip3(a, b, c), d)
}

/// Combines five iterators into a flat five-element tuple iterator.
///
/// See [`zip2`] for a complete example.
pub fn zip5<A, B, C, D, E>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
) -> Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E> {
    Zipped::new(zip4(a, b, c, d), e)
}

/// Combines six iterators into a flat six-element tuple iterator.
///
/// See [`zip2`] for a complete example.
pub fn zip6<A, B, C, D, E, F>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
) -> Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F> {
    Zipped::new(zip5(a, b, c, d, e), f)
}

/// Combines seven iterators into a flat seven-element tuple iterator.
///
/// See [`zip2`] for a complete example.
pub fn zip7<A, B, C, D, E, F, G>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G> {
    Zipped::new(zip6(a, b, c, d, e, f), g)
}

/// Combines eight iterators into a flat eight-element tuple iterator.
#[allow(clippy::too_many_arguments)]
pub fn zip8<A, B, C, D, E, F, G, H>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H> {
    Zipped::new(zip7(a, b, c, d, e, f, g), h)
}

/// Combines nine iterators into a flat nine-element tuple iterator.
#[allow(clippy::too_many_arguments)]
pub fn zip9<A, B, C, D, E, F, G, H, I>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I> {
    Zipped::new(zip8(a, b, c, d, e, f, g, h), i)
}

/// Combines ten iterators into a flat ten-element tuple iterator.
#[allow(clippy::too_many_arguments)]
pub fn zip10<A, B, C, D, E, F, G, H, I, J>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
) -> Zipped<
    Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I>,
    J,
> {
    Zipped::new(zip9(a, b, c, d, e, f, g, h, i), j)
}

/// Combines eleven iterators into a flat eleven-element tuple iterator.
#[allow(clippy::too_many_arguments)]
pub fn zip11<A, B, C, D, E, F, G, H, I, J, K>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
    k: K,
) -> Zipped<
    Zipped<
        Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I>,
        J,
    >,
    K,
> {
    Zipped::new(zip10(a, b, c, d, e, f, g, h, i, j), k)
}

/// Combines twelve iterators into a flat twelve-element tuple iterator.
#[allow(clippy::too_many_arguments)]
pub fn zip12<A, B, C, D, E, F, G, H, I, J, K, L>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
    k: K,
    l: L,
) -> Zipped<
    Zipped<
        Zipped<
            Zipped<
                Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>,
                I,
            >,
            J,
        >,
        K,
    >,
    L,
> {
    Zipped::new(zip11(a, b, c, d, e, f, g, h, i, j, k), l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::arity::A2;
    use crate::core::arity::{A1, A13};
    use crate::core::read::{FixedRead, ReadExpression};
    use crate::core::storage::StorageLayout;
    use cubecl::prelude::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use static_assertions::{assert_impl_all, assert_not_impl_any, assert_type_eq_all};

    #[derive(CubeType, Clone, Copy)]
    struct ReadOnlyValue {
        value: u32,
    }

    struct MakeReadOnly;

    struct MakeReadOnlyFromU64;

    struct ReadOnlyEqual;

    struct ReadOnlyLess;

    #[cubecl::cube]
    impl crate::op::UnaryOp<u32> for MakeReadOnly {
        type Output = ReadOnlyValue;

        fn apply(value: u32) -> ReadOnlyValue {
            ReadOnlyValue { value }
        }
    }

    #[cubecl::cube]
    impl crate::op::UnaryOp<u64> for MakeReadOnlyFromU64 {
        type Output = ReadOnlyValue;

        fn apply(value: u64) -> ReadOnlyValue {
            ReadOnlyValue {
                value: value as u32,
            }
        }
    }

    #[cubecl::cube]
    impl crate::op::BinaryPredicateOp<ReadOnlyValue> for ReadOnlyEqual {
        fn apply(lhs: ReadOnlyValue, rhs: ReadOnlyValue) -> crate::MFlag {
            crate::flag::from_bool(lhs.value == rhs.value)
        }
    }

    #[cubecl::cube]
    impl crate::op::BinaryPredicateOp<ReadOnlyValue> for ReadOnlyLess {
        fn apply(lhs: ReadOnlyValue, rhs: ReadOnlyValue) -> crate::MFlag {
            crate::flag::from_bool(lhs.value < rhs.value)
        }
    }

    type ReadOnlyIter = read::Transform<read::Counting, MakeReadOnly>;
    type ExactRead = <ReadOnlyIter as MIter<WgpuRuntime>>::Read;
    type TwoColumnRead = <Zipped<read::Counting, read::Counting> as MIter<WgpuRuntime>>::Read;
    type Fixed = FixedRead<ExactRead>;

    #[test]
    fn readable_item_does_not_require_a_storage_layout() {
        assert_not_impl_any!(ReadOnlyValue: StorageLayout);
        assert_impl_all!(ReadOnlyIter: MIter<WgpuRuntime>);
    }

    #[test]
    fn logical_lowering_retains_exact_arity_until_fixed_adapter() {
        assert_type_eq_all!(<ExactRead as ReadExpression>::ReadArity, A1);
        assert_type_eq_all!(<TwoColumnRead as ReadExpression>::ReadArity, A2);
        assert_type_eq_all!(<Fixed as ReadExpression>::ReadArity, A13);
    }

    #[test]
    fn exact_prefix_reallocates_every_physical_column() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let storage = exec.alloc::<(u32, u32)>(8);
        crate::api::algorithm::fill(&exec, (11_u32, 22_u32), MStorage::slice_mut(&storage, ..))
            .unwrap();
        let exact = crate::api::materialize::into_exact_prefix::<WgpuRuntime, (u32, u32)>(
            &exec, storage, 3,
        )
        .unwrap();

        let (left, right) = exact.into_columns();
        let left_bytes = exec.client().read_one(left.handle.clone()).unwrap();
        let right_bytes = exec.client().read_one(right.handle.clone()).unwrap();
        assert_eq!(left_bytes.len(), 3 * core::mem::size_of::<u32>());
        assert_eq!(right_bytes.len(), 3 * core::mem::size_of::<u32>());
        assert_eq!(exec.to_host(&left).unwrap(), vec![11; 3]);
        assert_eq!(exec.to_host(&right).unwrap(), vec![22; 3]);
    }

    #[test]
    fn non_storage_keys_support_comparison_without_materialization() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let keys = read::Transform::new(read::Counting::new(7, 3), MakeReadOnly);

        assert_eq!(
            crate::vector::is_sorted(&exec, keys, ReadOnlyLess).unwrap(),
            crate::flag::from_bool(true)
        );
    }

    #[test]
    fn non_storage_keys_can_build_a_sort_permutation() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let backing = exec.to_device(&[9_u32, 7, 8]);
        let keys = read::Transform::new(backing.column(), MakeReadOnly);

        let permutation = crate::core::ordering::sort_control_with(
            &exec,
            lower_fixed::<WgpuRuntime, _>(keys),
            ReadOnlyLess,
        )
        .unwrap();

        assert_eq!(exec.to_host(&permutation).unwrap(), vec![1, 2, 0]);
    }

    #[test]
    fn non_storage_sort_by_key_crosses_merge_tiles_stably() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 10_000usize;
        let host_keys = (0..len)
            .map(|index| ((index * 37) % 97) as u32)
            .collect::<Vec<_>>();
        let host_values = (0..len as u32).collect::<Vec<_>>();
        let mut expected = host_values.clone();
        expected.sort_by_key(|index| host_keys[*index as usize]);

        let key_storage = exec.to_device(&host_keys);
        let keys = read::Transform::new(key_storage.column(), MakeReadOnly);
        let value_storage = exec.to_device(&host_values);
        let sorted =
            crate::vector::sort_by_key(&exec, keys, value_storage.slice(..), ReadOnlyLess).unwrap();

        assert_eq!(exec.to_host(&sorted).unwrap(), expected);
    }

    #[test]
    fn value_only_by_key_algorithms_accept_read_only_keys() {
        assert_not_impl_any!(ReadOnlyValue: MAlloc<WgpuRuntime>);

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

        let sort_key_storage = exec.to_device(&[3_u32, 1, 2]);
        let sort_keys = read::Transform::new(sort_key_storage.column(), MakeReadOnly);
        let sort_values = exec.to_device(&[30_u32, 10, 20]);
        let sorted =
            crate::vector::sort_by_key(&exec, sort_keys, sort_values.slice(..), ReadOnlyLess)
                .unwrap();
        assert_eq!(exec.to_host(&sorted).unwrap(), vec![10, 20, 30]);

        let unique_key_storage = exec.to_device(&[1_u32, 1, 2, 2]);
        let unique_keys = read::Transform::new(unique_key_storage.column(), MakeReadOnly);
        let unique_values = exec.to_device(&[10_u32, 11, 20, 21]);
        let unique = crate::vector::unique_by_key(
            &exec,
            unique_keys,
            unique_values.slice(..),
            ReadOnlyEqual,
        )
        .unwrap();
        assert_eq!(exec.to_host(&unique).unwrap(), vec![10, 20]);

        let left_key_storage = exec.to_device(&[1_u32, 3]);
        let right_key_storage = exec.to_device(&[2_u64, 4]);
        let left_keys = read::Transform::new(left_key_storage.column(), MakeReadOnly);
        let right_keys = read::Transform::new(right_key_storage.column(), MakeReadOnlyFromU64);
        let left_values = exec.to_device(&[10_u32, 30]);
        let right_values = exec.to_device(&[20_u32, 40]);
        let merged = crate::vector::merge_by_key(
            &exec,
            left_keys,
            left_values.slice(..),
            right_keys,
            right_values.slice(..),
            ReadOnlyLess,
        )
        .unwrap();
        assert_eq!(exec.to_host(&merged).unwrap(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn two_input_comparison_accepts_independent_physical_slot_types() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = read::Transform::new(read::Counting::new(7, 3), MakeReadOnly);
        let right_values = exec.to_device(&[7_u64, 8, 9]);
        let right = read::Transform::new(right_values.column(), MakeReadOnlyFromU64);

        assert_eq!(
            crate::vector::equal(&exec, left, right, ReadOnlyEqual).unwrap(),
            crate::flag::from_bool(true)
        );
    }
}
