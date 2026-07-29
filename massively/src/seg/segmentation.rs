use cubecl::prelude::*;
use std::ops::RangeBounds;

use crate::api::iter::MIterExtent;
use crate::core::read::SliceExpression;
use crate::{DeviceSlice, DeviceVec, Error, Executor, MIndex, MIter};

use super::{ExclusiveScan, Executable, ForEachSegment, SegmentIterator};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn validate_offsets_kernel(offsets: &[u32], status: &[Atomic<u32>]) {
    let index = ABSOLUTE_POS as usize;
    if index < offsets.len() {
        let offset = offsets[index];
        if (index == 0usize && offset != 0u32)
            || (index > 0usize && offset < offsets[index - 1usize])
        {
            status[0].fetch_max(1u32);
        }
        if index + 1usize == offsets.len() {
            status[1].fetch_max(offset);
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn validate_segment_ids_kernel(ids: &[u32], segment_count: &[u32], invalid: &[Atomic<u32>]) {
    let index = ABSOLUTE_POS as usize;
    if index < ids.len() {
        let id = ids[index];
        if id >= segment_count[0] || (index > 0usize && id < ids[index - 1usize]) {
            invalid[0].fetch_max(1u32);
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn offsets_from_prefix_kernel(prefix: &[u32], offsets: &mut [u32], status: &[Atomic<u32>]) {
    let index = ABSOLUTE_POS as usize;
    if index < prefix.len() {
        let end = prefix[index];
        if index == 0usize {
            offsets[0] = 0u32;
        } else if end < prefix[index - 1usize] {
            status[0].fetch_max(1u32);
        }
        offsets[index + 1usize] = end;
        if index + 1usize == prefix.len() {
            status[1].fetch_max(end);
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn lengths_from_offsets_kernel(offsets: &[u32], lengths: &mut [u32]) {
    let segment = ABSOLUTE_POS as usize;
    if segment < lengths.len() {
        lengths[segment] = offsets[segment + 1usize] - offsets[segment];
    }
}

#[cubecl::cube(launch_unchecked)]
fn mark_zero_based_heads_kernel(offsets: &[u32], heads: &mut [u32]) {
    let segment = ABSOLUTE_POS as usize;
    if segment + 1usize < offsets.len() {
        let start = offsets[segment] as usize;
        let end = offsets[segment + 1usize] as usize;
        if start < end {
            heads[start] = segment as u32;
        }
    }
}

fn checked_offset_count(segment_count: MIndex) -> Result<MIndex, Error> {
    segment_count
        .checked_add(1)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })
}

/// A reusable partition of one flat value range into ordered segments.
///
/// A segmentation has three equivalent representations:
///
/// - segment lengths, such as `[1, 2, 3]`;
/// - one zero-based segment id per flat value, such as
///   `[0, 1, 1, 2, 2, 2]`;
/// - CSR-style offsets, such as `[0, 1, 3, 6]`.
///
/// Offsets are the canonical representation because they preserve empty
/// segments and provide constant-time segment bounds. Constructing a
/// segmentation validates and privately materializes that representation, so
/// later methods can rely on its invariants.
///
/// Constructors may synchronize once to observe validation status and the
/// exact flat value count. Once constructed, [`lengths`](Self::lengths),
/// [`segment_ids`](Self::segment_ids), and
/// [`local_indices`](Self::local_indices) only enqueue fixed-shape GPU work;
/// they do not observe a value on the host.
///
/// Per-segment context needs no dedicated adapter: derive segment IDs, use
/// [`crate::lazy::permute`] to broadcast the context to flat entries, combine
/// it with the values using [`crate::zip2`], and apply this segmentation to the
/// combined rows. A uniform context uses [`crate::lazy::constant`] instead.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, seg::Segmentation};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let lengths = exec.to_device(&[1_u32, 2, 3]);
/// let segmentation =
///     Segmentation::from_lengths(&exec, lengths.slice(..)).unwrap();
///
/// assert_eq!(
///     exec.to_host(&segmentation.offsets()).unwrap(),
///     vec![0, 1, 3, 6],
/// );
/// assert_eq!(
///     exec.to_host(&segmentation.segment_ids(&exec).unwrap()).unwrap(),
///     vec![0, 1, 1, 2, 2, 2],
/// );
/// assert_eq!(
///     exec.to_host(&segmentation.local_indices(&exec).unwrap()).unwrap(),
///     vec![0, 0, 1, 0, 1, 2],
/// );
/// ```
pub struct Segmentation<R: Runtime> {
    offsets: DeviceVec<R, MIndex>,
    value_count: MIndex,
}

impl<R: Runtime> Clone for Segmentation<R> {
    fn clone(&self) -> Self {
        Self {
            offsets: self.offsets.clone(),
            value_count: self.value_count,
        }
    }
}

#[allow(private_interfaces)]
impl<R: Runtime> MIter<R> for Segmentation<R> {
    type Item = MIndex;
    type Read = <DeviceSlice<R, MIndex> as MIter<R>>::Read;
    type Slice = DeviceSlice<R, MIndex>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        self.offsets.slice(range)
    }

    fn lower_read(self) -> Self::Read {
        <DeviceSlice<R, MIndex> as MIter<R>>::lower_read(self.offsets.slice(..))
    }
}

impl<R: Runtime> MIterExtent<R> for Segmentation<R> {
    fn capacity(&self) -> Result<MIndex, Error> {
        Ok(self.offsets.len())
    }

    fn logical_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error> {
        self.offsets.slice(..).logical_extent()
    }
}

impl<R: Runtime> Segmentation<R> {
    pub(crate) fn from_generated_offsets(
        offsets: DeviceVec<R, MIndex>,
        value_count: MIndex,
    ) -> Result<Self, Error> {
        super::segment_count(offsets.len() as usize)?;
        Ok(Self {
            offsets,
            value_count,
        })
    }

    /// Builds a segmentation from CSR-style offsets.
    ///
    /// Offsets must be nonempty, start at zero, and be nondecreasing. The last
    /// offset is the flat value count.
    pub fn from_offsets<Offsets>(exec: &Executor<R>, offsets: Offsets) -> Result<Self, Error>
    where
        Offsets: MIter<R, Item = MIndex>,
    {
        let offsets = crate::api::materialize::exact(exec, offsets)?;
        if offsets.is_empty() {
            return Err(Error::InvalidSegmentation);
        }

        let status = exec.to_device(&[0u32, 0u32]);
        let offset_count = offsets.len();
        unsafe {
            validate_offsets_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::core::launch::cube_count_1d(
                    (offset_count as usize).div_ceil(BLOCK_SIZE as usize),
                )?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
                BufferArg::from_raw_parts(status.handle.clone(), status.capacity()),
            );
        }
        let status = exec.to_host(&status)?;
        if status[0] != 0 {
            return Err(Error::InvalidSegmentation);
        }

        Ok(Self {
            offsets,
            value_count: status[1],
        })
    }

    /// Builds a segmentation from one length per segment.
    ///
    /// The total value count must fit in [`MIndex`]. A segment count equal to
    /// `MIndex::MAX` is rejected because its offsets need one additional item.
    pub fn from_lengths<Lengths>(exec: &Executor<R>, lengths: Lengths) -> Result<Self, Error>
    where
        Lengths: MIter<R, Item = MIndex>,
    {
        let segment_count = crate::api::iter::checked_len(lengths.logical_extent()?.read(exec)?)?;
        let offset_count = checked_offset_count(segment_count)?;
        if segment_count == 0 {
            return Ok(Self {
                offsets: exec.to_device(&[0u32]),
                value_count: 0,
            });
        }

        let lengths = lengths
            .lower_read()
            .slice_expression(0, segment_count as usize);
        let prefix = crate::vector::inclusive_scan(exec, lengths, super::control::SumU32)?;
        let offsets =
            exec.alloc::<u32>(MIndex::try_from(offset_count).expect("offset count was validated"));
        let status = exec.to_device(&[0u32, 0u32]);
        unsafe {
            offsets_from_prefix_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::core::launch::cube_count_1d(
                    (segment_count as usize).div_ceil(BLOCK_SIZE as usize),
                )?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(prefix.handle.clone(), prefix.capacity()),
                BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
                BufferArg::from_raw_parts(status.handle.clone(), status.capacity()),
            );
        }
        let status = exec.to_host(&status)?;
        if status[0] != 0 {
            return Err(Error::LengthTooLarge {
                len: (u32::MAX as usize).saturating_add(1),
            });
        }

        Ok(Self {
            offsets,
            value_count: status[1],
        })
    }

    /// Builds a segmentation from one zero-based segment id per flat value.
    ///
    /// IDs must be nondecreasing and less than `segment_count`. The explicit
    /// segment count preserves trailing and all-empty segments, which cannot be
    /// inferred from the IDs alone. `segment_count == MIndex::MAX` is rejected
    /// because its offsets need one additional item.
    pub fn from_segment_ids<Ids>(
        exec: &Executor<R>,
        ids: Ids,
        segment_count: MIndex,
    ) -> Result<Self, Error>
    where
        Ids: MIter<R, Item = MIndex>,
    {
        let offset_count = checked_offset_count(segment_count)?;
        let ids = crate::api::materialize::exact(exec, ids)?;
        let value_count = ids.len();

        if value_count != 0 {
            let invalid = exec.to_device(&[0u32]);
            let segment_count_value = exec.to_device(&[segment_count]);
            unsafe {
                validate_segment_ids_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(
                        (value_count as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(ids.handle.clone(), ids.capacity()),
                    BufferArg::from_raw_parts(
                        segment_count_value.handle.clone(),
                        segment_count_value.capacity(),
                    ),
                    BufferArg::from_raw_parts(invalid.handle.clone(), invalid.capacity()),
                );
            }
            if exec.to_host(&invalid)?[0] != 0 {
                return Err(Error::InvalidSegmentation);
            }
        }

        let offsets = if value_count == 0 {
            let offsets = exec.alloc::<u32>(offset_count);
            crate::vector::fill(exec, 0u32, offsets.slice_mut(..))?;
            offsets
        } else {
            crate::vector::lower_bound(
                exec,
                ids.slice(..),
                crate::lazy::counting(0).take(offset_count),
                super::control::LessU32,
            )?
        };
        Ok(Self {
            offsets,
            value_count,
        })
    }

    /// Returns the canonical CSR-style offsets as a read-only zero-copy view.
    pub fn offsets(&self) -> DeviceSlice<R, MIndex> {
        self.offsets.slice(..)
    }

    /// Returns the number of segments.
    pub fn segment_count(&self) -> MIndex {
        self.offsets.len() - 1
    }

    /// Returns the number of values in the partitioned flat range.
    pub const fn value_count(&self) -> MIndex {
        self.value_count
    }

    /// Materializes one length per segment.
    pub fn lengths(&self, exec: &Executor<R>) -> Result<crate::MVec<R, MIndex>, Error> {
        if exec.id() != self.offsets.owner {
            return Err(Error::ForeignExecutor);
        }
        let segment_count = self.segment_count();
        let lengths = exec.alloc::<u32>(segment_count);
        if segment_count != 0 {
            unsafe {
                lengths_from_offsets_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(
                        (segment_count as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.capacity()),
                    BufferArg::from_raw_parts(lengths.handle.clone(), lengths.capacity()),
                );
            }
        }
        Ok(lengths)
    }

    /// Materializes one zero-based segment id per flat value.
    ///
    /// These IDs can index per-segment data with [`crate::lazy::permute`].
    pub fn segment_ids(&self, exec: &Executor<R>) -> Result<crate::MVec<R, MIndex>, Error> {
        if exec.id() != self.offsets.owner {
            return Err(Error::ForeignExecutor);
        }
        let value_count = self.value_count;
        let ids = exec.alloc::<u32>(value_count);
        if value_count == 0 {
            return Ok(ids);
        }

        let heads = exec.alloc::<u32>(value_count);
        crate::vector::fill(exec, 0u32, heads.slice_mut(..))?;
        let segment_count = self.segment_count();
        if segment_count != 0 {
            unsafe {
                mark_zero_based_heads_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::core::launch::cube_count_1d(
                        (segment_count as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.capacity()),
                    BufferArg::from_raw_parts(heads.handle.clone(), heads.capacity()),
                );
            }
        }
        crate::vector::inclusive_scan_into(
            exec,
            heads.slice(..),
            super::control::MaxU32,
            ids.slice_mut(..),
        )?;
        Ok(ids)
    }

    /// Materializes each flat item's zero-based index within its segment.
    ///
    /// Empty segments produce no item. For offsets `[0, 2, 2, 5]`, this
    /// returns `[0, 1, 0, 1, 2]`.
    pub fn local_indices(&self, exec: &Executor<R>) -> Result<crate::MVec<R, MIndex>, Error> {
        if exec.id() != self.offsets.owner {
            return Err(Error::ForeignExecutor);
        }
        let ones = crate::lazy::constant(1u32).take(self.value_count);
        let output = ForEachSegment(ExclusiveScan(super::control::SumU32, 0u32))
            .run(exec, self.segments(ones)?)?;
        Ok(output.into_parts().0)
    }

    /// Applies this partition to a flat logical iterator.
    ///
    /// The iterator length must equal [`value_count`](Self::value_count).
    pub fn segments<Values>(&self, values: Values) -> Result<SegmentIterator<Values, Self>, Error>
    where
        Values: MIter<R>,
    {
        let value_len = values.capacity()?;
        if value_len != self.value_count {
            return Err(Error::LengthMismatch {
                left: value_len as usize,
                right: self.value_count as usize,
            });
        }
        Ok(SegmentIterator::new(values, self.clone()))
    }
}
