//! Device-resident active-prefix extents for internal scratch storage.

use std::{fmt, sync::Arc};

use cubecl::prelude::*;

use crate::{DeviceVec, Error, Executor, MIndex};

#[doc(hidden)]
#[derive(Clone)]
pub struct DeviceExtentSource {
    handle: cubecl::server::Handle,
    owner: u64,
    upper_bound: usize,
}

/// Active item count carried by an internal upper-bound allocation.
///
/// `Fixed` is used when the host already knows the exact count. `Device`
/// keeps a one-element `u32` allocation produced by GPU work. `start` and
/// `limit` make ordinary host range slicing composable without reading that
/// scalar back: the represented value is
/// `min(source.saturating_sub(start), limit)`.
///
/// Public owned vectors retain this metadata when a length-changing algorithm
/// returns an initialized prefix inside an upper-bound allocation.
#[doc(hidden)]
#[derive(Clone)]
pub enum LogicalExtent {
    Fixed(usize),
    Device {
        source: Arc<DeviceExtentSource>,
        start: usize,
        limit: usize,
    },
}

impl fmt::Debug for LogicalExtent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(len) => f.debug_tuple("Fixed").field(len).finish(),
            Self::Device {
                source,
                start,
                limit,
            } => f
                .debug_struct("Device")
                .field("owner", &source.owner)
                .field("upper_bound", &source.upper_bound)
                .field("start", start)
                .field("limit", limit)
                .finish_non_exhaustive(),
        }
    }
}

impl Default for LogicalExtent {
    fn default() -> Self {
        Self::Fixed(0)
    }
}

#[cubecl::cube(launch_unchecked)]
fn clamp_extent_kernel(source: &[u32], parameters: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        let start = parameters[0];
        let limit = parameters[1];
        let source_len = source[0];
        let remaining = source_len.saturating_sub(start);
        output[0] = u32::min(remaining, limit);
    }
}

#[cubecl::cube(launch_unchecked)]
fn add_extent_kernel(left: &[u32], right: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = left[0] + right[0];
    }
}

#[cubecl::cube(launch_unchecked)]
fn min_extent_kernel(left: &[u32], right: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = u32::min(left[0], right[0]);
    }
}

#[cubecl::cube(launch_unchecked)]
fn extent_less_kernel(left: &[u32], right: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = crate::flag::from_bool(left[0] < right[0]);
    }
}

#[cubecl::cube(launch_unchecked)]
fn copy_extent_kernel(source: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = source[0];
    }
}

#[cubecl::cube(launch_unchecked)]
fn reverse_start_kernel(source: &[u32], offset: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = if source[0] > offset[0] {
            source[0] - offset[0] - 1u32
        } else {
            0u32
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn ceil_div_extent_kernel(source: &[u32], divisor: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        let value = source[0];
        let divisor = divisor[0];
        output[0] = if value == 0u32 {
            0u32
        } else {
            (value - 1u32) / divisor + 1u32
        };
    }
}

impl LogicalExtent {
    pub(crate) const fn fixed(len: usize) -> Self {
        Self::Fixed(len)
    }

    pub(crate) fn from_device<R: Runtime>(value: &DeviceVec<R, u32>, upper_bound: usize) -> Self {
        Self::Device {
            source: Arc::new(DeviceExtentSource {
                handle: value.handle.clone(),
                owner: value.owner,
                upper_bound,
            }),
            start: 0,
            limit: upper_bound,
        }
    }

    pub(crate) fn host_len(&self) -> Option<usize> {
        match self {
            Self::Fixed(len) => Some(*len),
            Self::Device { .. } => None,
        }
    }

    pub(crate) fn upper_bound(&self) -> usize {
        match self {
            Self::Fixed(len) => *len,
            Self::Device { limit, .. } => *limit,
        }
    }

    pub(crate) fn slice(&self, start: usize, len: usize) -> Self {
        match self {
            Self::Fixed(current) => Self::Fixed(current.saturating_sub(start).min(len)),
            Self::Device {
                source,
                start: current_start,
                limit: current_limit,
            } => Self::Device {
                source: source.clone(),
                start: current_start.saturating_add(start),
                limit: current_limit.saturating_sub(start).min(len),
            },
        }
    }

    /// Chooses the common logical extent for a zip operation.
    ///
    /// A fixed operand can safely be narrowed by a device-resident operand as
    /// long as its physical range covers the latter's upper bound. Two device
    /// extents must originate from the same scalar and represent the same
    /// slice; accepting unrelated values would silently turn zip's equality
    /// contract into `zip_min`.
    pub(crate) fn zipped(&self, rhs: &Self) -> Result<Self, Error> {
        match (self, rhs) {
            (Self::Fixed(left), Self::Fixed(right)) if left == right => Ok(self.clone()),
            (Self::Fixed(left), device @ Self::Device { .. }) if *left >= device.upper_bound() => {
                Ok(device.clone())
            }
            (device @ Self::Device { .. }, Self::Fixed(right))
                if *right >= device.upper_bound() =>
            {
                Ok(device.clone())
            }
            (
                Self::Device {
                    source: left_source,
                    start: left_start,
                    limit: left_limit,
                },
                Self::Device {
                    source: right_source,
                    start: right_start,
                    limit: right_limit,
                },
            ) if Arc::ptr_eq(left_source, right_source)
                && left_start == right_start
                && left_limit == right_limit =>
            {
                Ok(self.clone())
            }
            _ => Err(Error::LengthMismatch {
                left: self.upper_bound(),
                right: rhs.upper_bound(),
            }),
        }
    }

    /// Produces a one-element device buffer containing this logical length.
    pub(crate) fn materialize<R: Runtime>(
        &self,
        exec: &Executor<R>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        match self {
            Self::Fixed(len) => {
                let len =
                    MIndex::try_from(*len).map_err(|_| Error::LengthTooLarge { len: *len })?;
                Ok(exec.to_device(&[len]))
            }
            Self::Device {
                source,
                start,
                limit,
            } => {
                if source.owner != exec.id() {
                    return Err(Error::ForeignExecutor);
                }
                if *start == 0 && *limit >= source.upper_bound {
                    return Ok(exec.column_from_handle(source.handle.clone(), 1));
                }
                let start =
                    MIndex::try_from(*start).map_err(|_| Error::LengthTooLarge { len: *start })?;
                let limit =
                    MIndex::try_from(*limit).map_err(|_| Error::LengthTooLarge { len: *limit })?;
                let parameters = exec.to_device(&[start, limit]);
                let output = exec.alloc_column::<u32>(1);
                unsafe {
                    clamp_extent_kernel::launch_unchecked::<R>(
                        exec.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(1),
                        BufferArg::from_raw_parts(source.handle.clone(), 1),
                        BufferArg::from_raw_parts(parameters.handle.clone(), 2),
                        BufferArg::from_raw_parts(output.handle.clone(), 1),
                    );
                }
                Ok(output)
            }
        }
    }

    pub(crate) fn add<R: Runtime>(
        exec: &Executor<R>,
        left: &Self,
        right: &Self,
        upper_bound: usize,
    ) -> Result<Self, Error> {
        if let (Some(left), Some(right)) = (left.host_len(), right.host_len()) {
            return left
                .checked_add(right)
                .map(Self::Fixed)
                .ok_or(Error::LengthTooLarge { len: usize::MAX });
        }
        let left = left.materialize(exec)?;
        let right = right.materialize(exec)?;
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            add_extent_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(left.handle.clone(), 1),
                BufferArg::from_raw_parts(right.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(Self::from_device(&output, upper_bound))
    }

    pub(crate) fn min<R: Runtime>(
        exec: &Executor<R>,
        left: &Self,
        right: &Self,
    ) -> Result<Self, Error> {
        if let (Some(left), Some(right)) = (left.host_len(), right.host_len()) {
            return Ok(Self::Fixed(left.min(right)));
        }
        let upper_bound = left.upper_bound().min(right.upper_bound());
        let left = left.materialize(exec)?;
        let right = right.materialize(exec)?;
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            min_extent_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(left.handle.clone(), 1),
                BufferArg::from_raw_parts(right.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(Self::from_device(&output, upper_bound))
    }

    pub(crate) fn less_value<R: Runtime>(
        exec: &Executor<R>,
        left: &Self,
        right: &Self,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if let (Some(left), Some(right)) = (left.host_len(), right.host_len()) {
            return Ok(exec.to_device(&[crate::flag::from_bool(left < right)]));
        }
        let left = left.materialize(exec)?;
        let right = right.materialize(exec)?;
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            extent_less_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(left.handle.clone(), 1),
                BufferArg::from_raw_parts(right.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(output)
    }

    pub(crate) fn copy_value<R: Runtime>(
        &self,
        exec: &Executor<R>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        let source = self.materialize(exec)?;
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            copy_extent_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(source.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(output)
    }

    pub(crate) fn reverse_start<R: Runtime>(
        &self,
        exec: &Executor<R>,
        offset: usize,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if let Some(len) = self.host_len() {
            let start = len.saturating_sub(offset).saturating_sub(1);
            let start = u32::try_from(start).map_err(|_| Error::LengthTooLarge {
                len: start.saturating_add(1),
            })?;
            return Ok(exec.to_device(&[start]));
        }
        let source = self.materialize(exec)?;
        let offset = u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
        let offset = exec.to_device(&[offset]);
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            reverse_start_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(source.handle.clone(), 1),
                BufferArg::from_raw_parts(offset.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(output)
    }

    pub(crate) fn ceil_div<R: Runtime>(
        &self,
        exec: &Executor<R>,
        divisor: usize,
        upper_bound: usize,
    ) -> Result<Self, Error> {
        assert!(divisor != 0, "logical extent divisor must be nonzero");
        if let Some(len) = self.host_len() {
            return Ok(Self::Fixed(len.div_ceil(divisor)));
        }
        let source = self.materialize(exec)?;
        let divisor =
            MIndex::try_from(divisor).map_err(|_| Error::LengthTooLarge { len: divisor })?;
        let divisor = exec.to_device(&[divisor]);
        let output = exec.alloc_column::<u32>(1);
        unsafe {
            ceil_div_extent_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(source.handle.clone(), 1),
                BufferArg::from_raw_parts(divisor.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), 1),
            );
        }
        Ok(Self::from_device(&output, upper_bound))
    }

    pub(crate) fn read<R: Runtime>(&self, exec: &Executor<R>) -> Result<usize, Error> {
        if let Some(len) = self.host_len() {
            return Ok(len);
        }
        let value = self.materialize(exec)?;
        Ok(exec.to_host(&value)?[0] as usize)
    }
}
