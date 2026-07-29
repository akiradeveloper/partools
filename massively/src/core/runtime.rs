//! Runtime executor and contiguous device storage.

#![allow(private_interfaces)]

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use cubecl::prelude::*;
use std::ops::RangeBounds;

use crate::Error;
use crate::core::extent::LogicalExtent;
use crate::core::read::Column;
use crate::core::value::MStorageElement;

static NEXT_EXECUTOR_ID: AtomicU64 = AtomicU64::new(1);

fn assert_indexable_len(len: usize) {
    assert!(
        crate::MIndex::try_from(len).is_ok(),
        "device vector length does not fit in MIndex"
    );
}

/// Execution context for one CubeCL runtime.
#[derive(Clone)]
pub struct Executor<R: Runtime> {
    client: ComputeClient<R>,
    id: u64,
}

impl<R: Runtime> Executor<R> {
    pub(crate) fn from_client(client: &ComputeClient<R>, id: u64) -> Self {
        Self {
            client: client.clone(),
            id,
        }
    }

    /// Creates an executor for one CubeCL device.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3]);
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![1, 2, 3]);
    /// ```
    pub fn new(device: R::Device) -> Self {
        Self {
            client: R::client(&device),
            id: NEXT_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    #[doc(hidden)]
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    #[doc(hidden)]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Explicitly copies a host slice to device memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[10_u32, 20, 30]);
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30]);
    /// ```
    pub fn to_device<T>(&self, input: &[T]) -> DeviceVec<R, T>
    where
        T: MStorageElement,
    {
        assert_indexable_len(input.len());
        let handle = if input.is_empty() {
            self.client.empty(size_of::<T>().max(1))
        } else {
            self.client.create_from_slice(T::as_bytes(input))
        };
        DeviceVec {
            handle,
            len: input.len(),
            owner: self.id,
            extent: LogicalExtent::fixed(input.len()),
            _runtime: PhantomData,
        }
    }

    /// Allocates device storage that must be fully written before it is read.
    pub(crate) fn alloc_column<T>(&self, len: usize) -> DeviceVec<R, T>
    where
        T: MStorageElement,
    {
        assert_indexable_len(len);
        DeviceVec {
            handle: self.client.empty(len.max(1) * size_of::<T>()),
            len,
            owner: self.id,
            extent: LogicalExtent::fixed(len),
            _runtime: PhantomData,
        }
    }

    pub(crate) fn column_from_handle<T>(
        &self,
        handle: cubecl::server::Handle,
        len: usize,
    ) -> DeviceVec<R, T>
    where
        T: MStorageElement,
    {
        assert_indexable_len(len);
        DeviceVec {
            handle,
            len,
            owner: self.id,
            extent: LogicalExtent::fixed(len),
            _runtime: PhantomData,
        }
    }

    /// Explicitly copies a device vector to the host.
    ///
    /// `input` may be a [`DeviceVec`], [`DeviceSlice`], or [`DeviceSliceMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[10_u32, 20, 30, 40]);
    ///
    /// assert_eq!(exec.to_host(&values.slice(1..3)).unwrap(), vec![20, 30]);
    /// ```
    pub fn to_host<Input>(&self, input: &Input) -> Result<Vec<Input::HostElement>, Error>
    where
        Input: DeviceRange,
    {
        if input.owner() != self.id {
            return Err(Error::ForeignExecutor);
        }
        let logical_len = input.extent().read(self)?;
        if logical_len == 0 {
            return Ok(Vec::new());
        }
        if logical_len > input.capacity() {
            return Err(Error::LengthMismatch {
                left: logical_len,
                right: input.capacity(),
            });
        }
        let bytes = self
            .client
            .read_one(input.handle())
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        let values = Input::Element::from_bytes(&bytes);
        let start = input.offset();
        let end = start + logical_len;
        Ok(values[start..end]
            .iter()
            .copied()
            .map(Input::to_host_element)
            .collect())
    }

    /// Waits for all work submitted through this executor to complete.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, vector::fill};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let output = exec.alloc::<u32>(4);
    /// fill(&exec, 7_u32, output.slice_mut(..)).unwrap();
    /// exec.sync().unwrap();
    /// ```
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.client.sync()).map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })
    }
}

/// Owned single-column device storage.
pub struct DeviceVec<R: Runtime, T> {
    pub(crate) handle: cubecl::server::Handle,
    len: usize,
    pub(crate) owner: u64,
    extent: LogicalExtent,
    _runtime: PhantomData<fn() -> (R, T)>,
}

/// Read-only contiguous device view tied to its originating runtime.
#[derive(Clone, Debug)]
pub struct DeviceSlice<R: Runtime, T> {
    pub(crate) column: Column<T>,
    _runtime: PhantomData<fn() -> R>,
}

impl<R: Runtime, T> DeviceSlice<R, T> {
    pub(crate) const fn from_column(column: Column<T>) -> Self {
        Self {
            column,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn into_column(self) -> Column<T> {
        self.column
    }

    /// Returns a read-only subview without copying device data.
    pub fn slice<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<crate::MIndex>,
    {
        let (offset, len) = crate::core::read::resolve_mindex_slice_range(self.column.len, range);
        Self::from_column(self.column.slice_usize(offset..offset + len))
    }
}

/// Mutable contiguous device view tied to its originating runtime.
#[derive(Clone, Debug)]
pub struct DeviceSliceMut<R: Runtime, T> {
    pub(crate) output: ColumnMut<T>,
    _runtime: PhantomData<fn() -> R>,
}

impl<R: Runtime, T> DeviceSliceMut<R, T> {
    pub(crate) const fn from_output(output: ColumnMut<T>) -> Self {
        Self {
            output,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn into_output(self) -> ColumnMut<T> {
        self.output
    }

    /// Returns a read-only subview of this mutable view.
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<R, T>
    where
        Range: RangeBounds<crate::MIndex>,
    {
        let (offset, len) = crate::core::read::resolve_mindex_slice_range(self.output.len, range);
        DeviceSlice::from_column(self.output.slice_usize(offset..offset + len))
    }

    /// Returns a mutable subview of this mutable view.
    pub fn slice_mut<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<crate::MIndex>,
    {
        let (offset, len) = crate::core::read::resolve_mindex_slice_range(self.output.len, range);
        Self::from_output(self.output.slice_mut_usize(offset..offset + len))
    }
}

impl<R: Runtime, T> Clone for DeviceVec<R, T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            len: self.len,
            owner: self.owner,
            extent: self.extent.clone(),
            _runtime: PhantomData,
        }
    }
}

impl<R: Runtime, T> DeviceVec<R, T> {
    pub(crate) fn len(&self) -> crate::MIndex {
        crate::MIndex::try_from(self.len).expect("device vector length does not fit in MIndex")
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the physical allocation bound without synchronizing.
    pub(crate) fn capacity(&self) -> usize {
        self.len
    }

    pub(crate) fn logical_extent(&self) -> LogicalExtent {
        self.extent.clone()
    }

    pub(crate) fn set_logical_extent(&mut self, extent: LogicalExtent) {
        debug_assert!(extent.upper_bound() <= self.len);
        self.extent = extent;
    }

    /// Creates an internal read-expression leaf over the whole allocation.
    pub(crate) fn column(&self) -> Column<T> {
        Column::from_handle_with_extent(
            self.handle.clone(),
            self.len,
            0,
            self.owner,
            self.len,
            self.extent.clone(),
        )
    }

    /// Returns a read-only view into this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4]);
    /// let middle = values.slice(1..3);
    ///
    /// assert_eq!(exec.to_host(&middle).unwrap(), vec![2, 3]);
    /// ```
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<R, T>
    where
        Range: RangeBounds<crate::MIndex>,
    {
        let (offset, len) = crate::core::read::resolve_mindex_slice_range(self.len, range);
        DeviceSlice::from_column(self.column().slice_usize(offset..offset + len))
    }

    pub(crate) fn slice_usize<Range>(&self, range: Range) -> Column<T>
    where
        Range: RangeBounds<usize>,
    {
        self.column().slice_usize(range)
    }

    /// Returns a mutable view into this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector::replace_where};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4]);
    /// let stencil = lazy::constant(1_u32).take(2);
    /// replace_where(&exec, 9_u32, stencil, values.slice_mut(1..3)).unwrap();
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![1, 9, 9, 4]);
    /// ```
    pub fn slice_mut<Range>(&self, range: Range) -> DeviceSliceMut<R, T>
    where
        Range: RangeBounds<crate::MIndex>,
    {
        let (offset, len) = crate::core::read::resolve_mindex_slice_range(self.len, range);
        DeviceSliceMut::from_output(self.slice_mut_usize(offset..offset + len))
    }

    pub(crate) fn slice_mut_usize<Range>(&self, range: Range) -> ColumnMut<T>
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::core::read::resolve_slice_range(self.len, range);
        ColumnMut {
            handle: self.handle.clone(),
            len,
            offset: offset as u32,
            owner: self.owner,
            buffer_len: self.len,
            extent: self.extent.slice(offset, len),
            _item: PhantomData,
        }
    }
}

#[doc(hidden)]
pub trait DeviceRange {
    type Element: MStorageElement;
    type HostElement;
    fn handle(&self) -> cubecl::server::Handle;
    fn capacity(&self) -> usize;
    fn offset(&self) -> usize;
    fn owner(&self) -> u64;
    #[doc(hidden)]
    fn extent(&self) -> LogicalExtent;
    #[doc(hidden)]
    fn to_host_element(value: Self::Element) -> Self::HostElement;
}

impl<R: Runtime, T: MStorageElement> DeviceRange for DeviceVec<R, T> {
    type Element = T;
    type HostElement = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.handle.clone()
    }
    fn capacity(&self) -> usize {
        self.len
    }
    fn offset(&self) -> usize {
        0
    }
    fn owner(&self) -> u64 {
        self.owner
    }
    fn extent(&self) -> LogicalExtent {
        self.extent.clone()
    }
    fn to_host_element(value: T) -> T {
        value
    }
}

impl<T: MStorageElement> DeviceRange for Column<T> {
    type Element = T;
    type HostElement = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.handle.clone().expect("bound device column")
    }
    fn capacity(&self) -> usize {
        self.len
    }
    fn offset(&self) -> usize {
        self.offset as usize
    }
    fn owner(&self) -> u64 {
        self.owner.expect("bound device column")
    }
    fn extent(&self) -> LogicalExtent {
        self.extent.clone()
    }
    fn to_host_element(value: T) -> T {
        value
    }
}

impl<R: Runtime, T: MStorageElement> DeviceRange for DeviceSlice<R, T> {
    type Element = T;
    type HostElement = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.column.handle.clone().expect("bound device slice")
    }
    fn capacity(&self) -> usize {
        self.column.len
    }
    fn offset(&self) -> usize {
        self.column.offset as usize
    }
    fn owner(&self) -> u64 {
        self.column.owner.expect("bound device slice")
    }
    fn extent(&self) -> LogicalExtent {
        self.column.extent.clone()
    }
    fn to_host_element(value: T) -> T {
        value
    }
}

/// Internal runtime-erased mutable output leaf.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ColumnMut<T> {
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) offset: u32,
    pub(crate) owner: u64,
    pub(crate) buffer_len: usize,
    pub(crate) extent: LogicalExtent,
    pub(crate) _item: PhantomData<fn() -> T>,
}

impl<T> ColumnMut<T> {
    pub(crate) fn capacity(&self) -> usize {
        self.len
    }

    pub(crate) fn slice_usize<Range>(&self, range: Range) -> Column<T>
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::core::read::resolve_slice_range(self.len, range);
        Column::from_handle(
            self.handle.clone(),
            len,
            self.offset + offset as u32,
            self.owner,
            self.buffer_len,
        )
        .with_logical_extent(self.extent.slice(offset, len))
    }

    pub(crate) fn slice_mut_usize<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::core::read::resolve_slice_range(self.len, range);
        Self {
            handle: self.handle.clone(),
            len,
            offset: self.offset + offset as u32,
            owner: self.owner,
            buffer_len: self.buffer_len,
            extent: self.extent.slice(offset, len),
            _item: PhantomData,
        }
    }
}

impl<R: Runtime, T: MStorageElement> DeviceRange for DeviceSliceMut<R, T> {
    type Element = T;
    type HostElement = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.output.handle.clone()
    }
    fn capacity(&self) -> usize {
        self.output.len
    }
    fn offset(&self) -> usize {
        self.output.offset as usize
    }
    fn owner(&self) -> u64 {
        self.output.owner
    }
    fn extent(&self) -> LogicalExtent {
        self.output.extent.clone()
    }
    fn to_host_element(value: T) -> T {
        value
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn maximum_mindex_length_is_indexable() {
        super::assert_indexable_len(crate::MIndex::MAX as usize);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "device vector length does not fit in MIndex")]
    fn length_above_mindex_is_rejected_before_allocation() {
        super::assert_indexable_len(crate::MIndex::MAX as usize + 1);
    }
}
