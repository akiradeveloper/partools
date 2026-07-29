//! Stable least-significant-digit radix ordering controls.

use cubecl::prelude::*;

use crate::core::iter::Zip;
use crate::core::launch::cube_count_1d;
use crate::core::value::MStorageElement;
use crate::{DeviceVec, Error, Executor};

const BLOCK_SIZE: u32 = 256;
const RADIX_BITS: u32 = 8;
const RADIX_BUCKETS: usize = 1usize << RADIX_BITS;
const RADIX_MASK: u32 = RADIX_BUCKETS as u32 - 1u32;

/// Returns one eight-bit digit of the order-preserving radix representation.
#[cubecl::cube]
trait RadixDigitOp<T: CubePrimitive + cubecl::frontend::Scalar>: 'static + Send + Sync {
    fn apply(value: T, shift: u32) -> u32;
}

macro_rules! unsigned_radix_op {
    ($name:ident, $ty:ty, $wide:ty) => {
        struct $name;

        #[cubecl::cube]
        impl RadixDigitOp<$ty> for $name {
            fn apply(value: $ty, shift: u32) -> u32 {
                (((value as $wide) >> (shift as $wide)) & (RADIX_MASK as $wide)) as u32
            }
        }
    };
}

macro_rules! signed_radix_op {
    ($name:ident, $ty:ty, $wide:ty, $mask:expr, $sign:expr) => {
        struct $name;

        #[cubecl::cube]
        impl RadixDigitOp<$ty> for $name {
            fn apply(value: $ty, shift: u32) -> u32 {
                let encoded = (((value as $wide) & $mask) ^ $sign) as $wide;
                ((encoded >> (shift as $wide)) & (RADIX_MASK as $wide)) as u32
            }
        }
    };
}

unsigned_radix_op!(U8RadixZero, u8, u32);
unsigned_radix_op!(U16RadixZero, u16, u32);
unsigned_radix_op!(U32RadixZero, u32, u32);
unsigned_radix_op!(U64RadixZero, u64, u64);
signed_radix_op!(I8RadixZero, i8, u32, 0xffu32, 0x80u32);
signed_radix_op!(I16RadixZero, i16, u32, 0xffffu32, 0x8000u32);
signed_radix_op!(I32RadixZero, i32, u32, 0xffff_ffffu32, 0x8000_0000u32);
signed_radix_op!(
    I64RadixZero,
    i64,
    u64,
    0xffff_ffff_ffff_ffffu64,
    0x8000_0000_0000_0000u64
);

struct F32RadixZero;

#[cubecl::cube]
impl RadixDigitOp<f32> for F32RadixZero {
    fn apply(value: f32, shift: u32) -> u32 {
        let raw = u32::reinterpret(value);
        let encoded = if (raw & 0x8000_0000u32) != 0u32 {
            !raw
        } else {
            raw ^ 0x8000_0000u32
        };
        (encoded >> shift) & RADIX_MASK
    }
}

struct F64RadixZero;

#[cubecl::cube]
impl RadixDigitOp<f64> for F64RadixZero {
    fn apply(value: f64, shift: u32) -> u32 {
        let raw = u64::reinterpret(value);
        let encoded = if (raw & 0x8000_0000_0000_0000u64) != 0u64 {
            !raw
        } else {
            raw ^ 0x8000_0000_0000_0000u64
        };
        ((encoded >> (shift as u64)) & (RADIX_MASK as u64)) as u32
    }
}

trait RadixScalar: MStorageElement {
    const BITS: u32;
    type DigitOp: RadixDigitOp<Self>;
}

macro_rules! impl_radix_scalar {
    ($($ty:ty => ($bits:expr, $op:ty)),+ $(,)?) => {
        $(
            impl RadixScalar for $ty {
                const BITS: u32 = $bits;
                type DigitOp = $op;
            }
        )+
    };
}

impl_radix_scalar!(
    u8 => (8, U8RadixZero),
    u16 => (16, U16RadixZero),
    u32 => (32, U32RadixZero),
    u64 => (64, U64RadixZero),
    i8 => (8, I8RadixZero),
    i16 => (16, I16RadixZero),
    i32 => (32, I32RadixZero),
    i64 => (64, I64RadixZero),
    f32 => (32, F32RadixZero),
    f64 => (64, F64RadixZero),
);

/// Places one key column in the current permutation order.  Subsequent radix
/// passes can then read keys linearly instead of gathering the original key
/// column through an increasingly shuffled permutation.
#[cubecl::cube(launch_unchecked)]
fn align_radix_keys_kernel<T: CubePrimitive + cubecl::frontend::Scalar>(
    keys: &[T],
    permutation: &[u32],
    logical_len: &[u32],
    params: &[u32],
    aligned: &mut [T],
) {
    let index = ABSOLUTE_POS as usize;
    if index < logical_len[0] as usize {
        let source = if params[2] != 0u32 {
            index
        } else {
            permutation[index] as usize
        };
        aligned[index] = keys[source];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn block_histogram_kernel<T: CubePrimitive + cubecl::frontend::Scalar, Op: RadixDigitOp<T>>(
    keys: &[T],
    logical_len: &[u32],
    params: &[u32],
    histograms: &mut [u32],
    #[comptime] items_per_unit: usize,
) {
    let block = CUBE_POS as usize;
    if block >= params[1] as usize {
        terminate!();
    }
    let counts = Shared::<[Atomic<u32>]>::new_slice(RADIX_BUCKETS);
    counts[UNIT_POS as usize].store(0u32);
    sync_cube();
    #[unroll]
    for item in 0usize..items_per_unit {
        let index = block * (BLOCK_SIZE as usize * items_per_unit)
            + item * BLOCK_SIZE as usize
            + UNIT_POS as usize;
        let valid = index < logical_len[0] as usize;
        let digit = if valid {
            Op::apply(keys[index], params[0])
        } else {
            0u32
        };
        let (rank, count, _) = crate::core::collective::plane_digit_rank(digit, valid);
        if valid && rank == 0u32 {
            counts[digit as usize].fetch_add(count);
        }
    }
    sync_cube();
    histograms[UNIT_POS as usize * params[1] as usize + block] = counts[UNIT_POS as usize].load();
}

#[cubecl::cube(launch_unchecked)]
fn histogram_prefix_kernel(
    histograms: &[u32],
    params: &[u32],
    block_prefixes: &mut [u32],
    bucket_totals: &mut [u32],
) {
    crate::core::collective::row_exclusive_sum(
        histograms,
        params[1] as usize,
        block_prefixes,
        bucket_totals,
    );
}

/// Scans one 256-block histogram chunk per workgroup and records its total.
#[cubecl::cube(launch_unchecked)]
fn histogram_chunk_scan_kernel(
    histograms: &[u32],
    params: &[u32],
    block_prefixes: &mut [u32],
    chunk_totals: &mut [u32],
) {
    let blocks = params[1] as usize;
    let chunks = blocks.div_ceil(BLOCK_SIZE as usize);
    let group = CUBE_POS as usize;
    let bucket = group / chunks;
    let chunk = group - bucket * chunks;
    let lane = UNIT_POS as usize;
    let block = chunk * BLOCK_SIZE as usize + lane;
    let value = if block < blocks {
        histograms[bucket * blocks + block]
    } else {
        0u32
    };

    let (prefix, total) = crate::core::collective::block_exclusive_sum(value);
    if block < blocks {
        block_prefixes[bucket * blocks + block] = prefix;
    }
    if lane == 0usize {
        chunk_totals[group] = total;
    }
}

/// Exclusively scans the much smaller list of chunk totals for each bucket.
#[cubecl::cube(launch_unchecked)]
fn histogram_chunk_prefix_kernel(
    chunk_totals: &[u32],
    params: &[u32],
    chunk_prefixes: &mut [u32],
    bucket_totals: &mut [u32],
) {
    let chunks = (params[1] as usize).div_ceil(BLOCK_SIZE as usize);
    crate::core::collective::row_exclusive_sum(chunk_totals, chunks, chunk_prefixes, bucket_totals);
}

/// Adds each chunk's global prefix to the chunk-local block prefixes.  This
/// keeps the hot scatter path to one prefix lookup per item.
#[cubecl::cube(launch_unchecked)]
fn add_histogram_chunk_prefix_kernel(
    chunk_prefixes: &[u32],
    params: &[u32],
    block_prefixes: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    let blocks = params[1] as usize;
    let entry_count = RADIX_BUCKETS * blocks;
    if index < entry_count {
        let chunks = blocks.div_ceil(BLOCK_SIZE as usize);
        let bucket = index / blocks;
        let block = index - bucket * blocks;
        let chunk = block / BLOCK_SIZE as usize;
        block_prefixes[index] += chunk_prefixes[bucket * chunks + chunk];
    }
}

#[cubecl::cube(launch_unchecked)]
fn bucket_offsets_kernel(bucket_totals: &[u32], bucket_offsets: &mut [u32]) {
    let (prefix, _) =
        crate::core::collective::block_exclusive_sum(bucket_totals[UNIT_POS as usize]);
    bucket_offsets[UNIT_POS as usize] = prefix;
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn stable_digit_scatter_kernel<T: CubePrimitive + cubecl::frontend::Scalar, Op: RadixDigitOp<T>>(
    keys: &[T],
    permutation: &[u32],
    block_prefixes: &[u32],
    bucket_offsets: &[u32],
    logical_len: &[u32],
    params: &[u32],
    key_output: &mut [T],
    output: &mut [u32],
    #[comptime] items_per_unit: usize,
) {
    let block = CUBE_POS as usize;
    let start = block * (BLOCK_SIZE as usize * items_per_unit);
    if start >= logical_len[0] as usize {
        terminate!();
    }
    let counts = Shared::<[Atomic<u32>]>::new_slice(RADIX_BUCKETS);
    counts[UNIT_POS as usize].store(0u32);
    let mut values = Array::<T>::new(items_per_unit);
    let mut indices = Array::<u32>::new(items_per_unit);
    let mut digits = Array::<u32>::new(items_per_unit);
    let mut ranks = Array::<u32>::new(items_per_unit);
    let mut matches = Array::<u32>::new(items_per_unit);
    let mut leaders = Array::<u32>::new(items_per_unit);
    #[unroll]
    for item in 0usize..items_per_unit {
        // Each subgroup owns a contiguous run, preserving source order when
        // subgroup prefixes are accumulated below.
        let index = start
            + PLANE_POS as usize * PLANE_DIM as usize * items_per_unit
            + item * PLANE_DIM as usize
            + UNIT_POS_PLANE as usize;
        let valid = index < logical_len[0] as usize;
        let digit = if valid {
            Op::apply(keys[index], params[0])
        } else {
            0u32
        };
        if valid {
            values[item] = keys[index];
            indices[item] = if params[2] != 0u32 {
                index as u32
            } else {
                permutation[index]
            };
        }
        let (rank, count, first) = crate::core::collective::plane_digit_rank(digit, valid);
        digits[item] = digit;
        ranks[item] = rank;
        matches[item] = count;
        leaders[item] = first;
    }
    sync_cube();
    let plane = RuntimeCell::<u32>::new(0u32);
    while plane.read() < CUBE_DIM.div_ceil(PLANE_DIM) {
        if PLANE_POS == plane.read() {
            #[unroll]
            for item in 0usize..items_per_unit {
                let prefix = RuntimeCell::<u32>::new(0u32);
                if ranks[item] == 0u32 && matches[item] != 0u32 {
                    prefix.store(counts[digits[item] as usize].fetch_add(matches[item]));
                }
                ranks[item] += plane_shuffle(prefix.read(), leaders[item]);
            }
        }
        sync_cube();
        plane.store(plane.read() + 1u32);
    }
    let (bucket_start, _) =
        crate::core::collective::block_exclusive_sum(counts[UNIT_POS as usize].load());
    let mut starts = Shared::<[u32]>::new_slice(RADIX_BUCKETS);
    starts[UNIT_POS as usize] = bucket_start;
    sync_cube();
    let mut sorted_keys = Shared::<[T]>::new_slice(BLOCK_SIZE as usize * items_per_unit);
    let mut sorted_indices = Shared::<[u32]>::new_slice(BLOCK_SIZE as usize * items_per_unit);
    #[unroll]
    for item in 0usize..items_per_unit {
        let index = start
            + PLANE_POS as usize * PLANE_DIM as usize * items_per_unit
            + item * PLANE_DIM as usize
            + UNIT_POS_PLANE as usize;
        if index < logical_len[0] as usize {
            let destination = (starts[digits[item] as usize] + ranks[item]) as usize;
            sorted_keys[destination] = values[item];
            sorted_indices[destination] = indices[item];
        }
    }
    sync_cube();
    // Transfer in bucket order so consecutive lanes write consecutive rows.
    // The control still carries only keys and the permutation; arbitrary
    // payload columns are gathered once after all radix passes.
    #[unroll]
    for item in 0usize..items_per_unit {
        let local = item * BLOCK_SIZE as usize + UNIT_POS as usize;
        if start + local < logical_len[0] as usize {
            let key = sorted_keys[local];
            let digit = Op::apply(key, params[0]) as usize;
            let rank = local as u32 - starts[digit];
            let destination =
                bucket_offsets[digit] + block_prefixes[digit * params[1] as usize + block] + rank;
            key_output[destination as usize] = key;
            output[destination as usize] = sorted_indices[local];
        }
    }
}

pub(crate) struct RadixControl<R: Runtime> {
    current: DeviceVec<R, u32>,
    scratch: DeviceVec<R, u32>,
    histograms: DeviceVec<R, u32>,
    block_prefixes: DeviceVec<R, u32>,
    histogram_chunk_totals: DeviceVec<R, u32>,
    histogram_chunk_prefixes: DeviceVec<R, u32>,
    bucket_totals: DeviceVec<R, u32>,
    bucket_offsets: DeviceVec<R, u32>,
    block_count: usize,
    items_per_unit: usize,
    initialized: bool,
}

impl<R: Runtime> RadixControl<R> {
    fn new(
        exec: &Executor<R>,
        len: usize,
        extent: crate::core::extent::LogicalExtent,
        items_per_unit: usize,
    ) -> Result<Self, Error> {
        let block_count = len.div_ceil(BLOCK_SIZE as usize * items_per_unit);
        let histogram_chunk_count = block_count.div_ceil(BLOCK_SIZE as usize);
        let mut current = exec.alloc_row::<u32>(len);
        current.set_logical_extent(extent.clone());
        let mut scratch = exec.alloc_row::<u32>(len);
        scratch.set_logical_extent(extent);
        Ok(Self {
            current,
            scratch,
            histograms: exec.alloc_row::<u32>(RADIX_BUCKETS * block_count),
            block_prefixes: exec.alloc_row::<u32>(RADIX_BUCKETS * block_count),
            histogram_chunk_totals: exec.alloc_row::<u32>(RADIX_BUCKETS * histogram_chunk_count),
            histogram_chunk_prefixes: exec.alloc_row::<u32>(RADIX_BUCKETS * histogram_chunk_count),
            bucket_totals: exec.alloc_row::<u32>(RADIX_BUCKETS),
            bucket_offsets: exec.alloc_row::<u32>(RADIX_BUCKETS),
            block_count,
            items_per_unit,
            initialized: false,
        })
    }

    fn pass<T: RadixScalar>(
        &mut self,
        exec: &Executor<R>,
        keys: &DeviceVec<R, T>,
        key_scratch: &DeviceVec<R, T>,
        shift: u32,
    ) -> Result<(), Error> {
        let len = self.current.capacity();
        if len == 0 {
            return Ok(());
        }
        let logical_len = self.current.logical_extent().materialize(exec)?;
        let params_handle = exec.client().create_from_slice(u32::as_bytes(&[
            shift,
            u32::try_from(self.block_count).map_err(|_| Error::LengthTooLarge {
                len: self.block_count,
            })?,
            u32::from(!self.initialized),
        ]));
        let rank_count = cube_count_1d(self.block_count)?;
        unsafe {
            block_histogram_kernel::launch_unchecked::<T, T::DigitOp, R>(
                exec.client(),
                rank_count.clone(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(keys.handle.clone(), keys.capacity()),
                BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                BufferArg::from_raw_parts(params_handle.clone(), 3),
                BufferArg::from_raw_parts(
                    self.histograms.handle.clone(),
                    self.histograms.capacity(),
                ),
                self.items_per_unit,
            );
            if self.block_count <= BLOCK_SIZE as usize {
                histogram_prefix_kernel::launch_unchecked::<R>(
                    exec.client(),
                    cube_count_1d(RADIX_BUCKETS)?,
                    CubeDim::new_1d(self.block_count.max(1) as u32),
                    BufferArg::from_raw_parts(
                        self.histograms.handle.clone(),
                        self.histograms.capacity(),
                    ),
                    BufferArg::from_raw_parts(params_handle.clone(), 3),
                    BufferArg::from_raw_parts(
                        self.block_prefixes.handle.clone(),
                        self.block_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(
                        self.bucket_totals.handle.clone(),
                        self.bucket_totals.capacity(),
                    ),
                );
            } else {
                let chunk_count = self.block_count.div_ceil(BLOCK_SIZE as usize);
                let chunk_entries = RADIX_BUCKETS * chunk_count;
                histogram_chunk_scan_kernel::launch_unchecked::<R>(
                    exec.client(),
                    cube_count_1d(chunk_entries)?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(
                        self.histograms.handle.clone(),
                        self.histograms.capacity(),
                    ),
                    BufferArg::from_raw_parts(params_handle.clone(), 3),
                    BufferArg::from_raw_parts(
                        self.block_prefixes.handle.clone(),
                        self.block_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(
                        self.histogram_chunk_totals.handle.clone(),
                        self.histogram_chunk_totals.capacity(),
                    ),
                );
                histogram_chunk_prefix_kernel::launch_unchecked::<R>(
                    exec.client(),
                    cube_count_1d(RADIX_BUCKETS)?,
                    CubeDim::new_1d(chunk_count.min(BLOCK_SIZE as usize) as u32),
                    BufferArg::from_raw_parts(
                        self.histogram_chunk_totals.handle.clone(),
                        self.histogram_chunk_totals.capacity(),
                    ),
                    BufferArg::from_raw_parts(params_handle.clone(), 3),
                    BufferArg::from_raw_parts(
                        self.histogram_chunk_prefixes.handle.clone(),
                        self.histogram_chunk_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(
                        self.bucket_totals.handle.clone(),
                        self.bucket_totals.capacity(),
                    ),
                );
                add_histogram_chunk_prefix_kernel::launch_unchecked::<R>(
                    exec.client(),
                    cube_count_1d(self.block_prefixes.capacity().div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(
                        self.histogram_chunk_prefixes.handle.clone(),
                        self.histogram_chunk_prefixes.capacity(),
                    ),
                    BufferArg::from_raw_parts(params_handle.clone(), 3),
                    BufferArg::from_raw_parts(
                        self.block_prefixes.handle.clone(),
                        self.block_prefixes.capacity(),
                    ),
                );
            }
            bucket_offsets_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::new_single(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(
                    self.bucket_totals.handle.clone(),
                    self.bucket_totals.capacity(),
                ),
                BufferArg::from_raw_parts(
                    self.bucket_offsets.handle.clone(),
                    self.bucket_offsets.capacity(),
                ),
            );
            stable_digit_scatter_kernel::launch_unchecked::<T, T::DigitOp, R>(
                exec.client(),
                rank_count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(keys.handle.clone(), len),
                BufferArg::from_raw_parts(self.current.handle.clone(), len),
                BufferArg::from_raw_parts(
                    self.block_prefixes.handle.clone(),
                    self.block_prefixes.capacity(),
                ),
                BufferArg::from_raw_parts(
                    self.bucket_offsets.handle.clone(),
                    self.bucket_offsets.capacity(),
                ),
                BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(key_scratch.handle.clone(), len),
                BufferArg::from_raw_parts(self.scratch.handle.clone(), len),
                self.items_per_unit,
            );
        }
        core::mem::swap(&mut self.current, &mut self.scratch);
        self.initialized = true;
        Ok(())
    }
}

/// Flat-row key storage that can contribute lexicographic radix passes.
///
/// Zip nodes process the right child first because each binary pass is stable;
/// this makes the leftmost physical column the primary key.
pub(crate) trait RadixStorage<R: Runtime> {
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error>;
}

impl<R, T> RadixStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: RadixScalar,
{
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error> {
        let len = control.current.capacity();
        if len == 0 {
            return Ok(());
        }
        let extent = control.current.logical_extent();
        let mut key_a = exec.alloc_row::<T>(len);
        key_a.set_logical_extent(extent.clone());
        let mut key_b = exec.alloc_row::<T>(len);
        key_b.set_logical_extent(extent.clone());
        let mut shifts = (0..T::BITS).step_by(RADIX_BITS as usize);

        // The first physical key column is already in identity order.  Feed it
        // directly into the first pass and avoid an otherwise redundant copy.
        if !control.initialized {
            if let Some(shift) = shifts.next() {
                control.pass(exec, self, &key_a, shift)?;
            }
        } else {
            let logical_len = extent.materialize(exec)?;
            let params = exec.client().create_from_slice(u32::as_bytes(&[
                0u32,
                u32::try_from(control.block_count).map_err(|_| Error::LengthTooLarge {
                    len: control.block_count,
                })?,
                0u32,
            ]));
            unsafe {
                align_radix_keys_kernel::launch_unchecked::<T, R>(
                    exec.client(),
                    cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.handle.clone(), self.capacity()),
                    BufferArg::from_raw_parts(control.current.handle.clone(), len),
                    BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                    BufferArg::from_raw_parts(params, 3),
                    BufferArg::from_raw_parts(key_a.handle.clone(), len),
                );
            }
        }

        for shift in shifts {
            control.pass(exec, &key_a, &key_b, shift)?;
            core::mem::swap(&mut key_a, &mut key_b);
        }
        Ok(())
    }
}

impl<R, Left, Right> RadixStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: RadixStorage<R>,
    Right: RadixStorage<R>,
{
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error> {
        self.1.radix_passes(exec, control)?;
        self.0.radix_passes(exec, control)
    }
}

pub(crate) fn permutation<R, Keys>(
    exec: &Executor<R>,
    keys: &Keys,
    len: usize,
    extent: crate::core::extent::LogicalExtent,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: RadixStorage<R>,
{
    // One launch shape per device, shared by all key widths and row layouts.
    // Eight rows use less than 28 KiB even for 64-bit keys; four fit in 16 KiB.
    let items_per_unit = if exec.client().properties().hardware.max_shared_memory_size >= 28 * 1024
    {
        8
    } else {
        4
    };
    let mut control = RadixControl::new(exec, len, extent, items_per_unit)?;
    keys.radix_passes(exec, &mut control)?;
    Ok(control.current)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RADIX_BLOCK_ITEMS: usize = BLOCK_SIZE as usize * 4;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn radix_tiles_preserve_stability_across_partial_blocks() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        for items_per_unit in [4, 8] {
            if items_per_unit == 8
                && exec.client().properties().hardware.max_shared_memory_size < 28 * 1024
            {
                continue;
            }
            let len = BLOCK_SIZE as usize * items_per_unit * 3 + 19;
            let keys: Vec<u32> = (0..len)
                .map(|i| ((i * 37 % 23) as u32) | (((i % 19) as u32) << 24))
                .collect();
            let mut expected: Vec<u32> = (0..len as u32).collect();
            expected.sort_by_key(|&index| keys[index as usize]);
            let keys = exec.to_device(&keys);
            let mut control = RadixControl::new(
                &exec,
                len,
                crate::core::extent::LogicalExtent::fixed(len),
                items_per_unit,
            )
            .unwrap();
            keys.radix_passes(&exec, &mut control).unwrap();
            assert_eq!(exec.to_host(&control.current).unwrap(), expected);
        }
    }

    #[test]
    fn histogram_prefix_and_bucket_offsets_are_independent_controls() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let blocks = 2usize;
        let logical_len = RADIX_BLOCK_ITEMS + 4usize;
        let mut input_histograms = vec![0u32; RADIX_BUCKETS * blocks];
        input_histograms[0] = 100;
        input_histograms[1] = 2;
        input_histograms[2 * blocks + 1] = 2;
        input_histograms[3 * blocks] = (RADIX_BLOCK_ITEMS - 100usize) as u32;

        let histograms = exec.to_device(&input_histograms);
        let params = exec.to_device(&[0u32, blocks as u32, 0u32]);
        let prefixes = exec.alloc_column::<u32>(RADIX_BUCKETS * blocks);
        let totals = exec.alloc_column::<u32>(RADIX_BUCKETS);
        let offsets = exec.alloc_column::<u32>(RADIX_BUCKETS);

        unsafe {
            histogram_prefix_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                crate::core::launch::cube_count_1d(RADIX_BUCKETS).unwrap(),
                CubeDim::new_1d(blocks as u32),
                BufferArg::from_raw_parts(histograms.handle.clone(), histograms.capacity()),
                BufferArg::from_raw_parts(params.handle.clone(), 3),
                BufferArg::from_raw_parts(prefixes.handle.clone(), prefixes.capacity()),
                BufferArg::from_raw_parts(totals.handle.clone(), totals.capacity()),
            );
            bucket_offsets_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                CubeCount::new_single(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(totals.handle.clone(), totals.capacity()),
                BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
            );
        }

        assert_eq!(exec.to_host(&histograms).unwrap(), input_histograms);

        let mut expected_prefixes = vec![0u32; RADIX_BUCKETS * blocks];
        expected_prefixes[1] = 100;
        expected_prefixes[3 * blocks + 1] = (RADIX_BLOCK_ITEMS - 100usize) as u32;
        assert_eq!(exec.to_host(&prefixes).unwrap(), expected_prefixes);

        let mut expected_totals = vec![0u32; RADIX_BUCKETS];
        expected_totals[0] = 102;
        expected_totals[2] = 2;
        expected_totals[3] = (RADIX_BLOCK_ITEMS - 100usize) as u32;
        assert_eq!(exec.to_host(&totals).unwrap(), expected_totals);

        let mut expected_offsets = vec![logical_len as u32; RADIX_BUCKETS];
        expected_offsets[0] = 0;
        expected_offsets[1] = 102;
        expected_offsets[2] = 102;
        expected_offsets[3] = 104;
        assert_eq!(exec.to_host(&offsets).unwrap(), expected_offsets);
    }

    #[test]
    fn hierarchical_histogram_prefix_matches_serial_exclusive_scan() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let blocks = BLOCK_SIZE as usize + 17usize;
        let chunks = blocks.div_ceil(BLOCK_SIZE as usize);
        let histograms: Vec<u32> = (0..RADIX_BUCKETS)
            .flat_map(|bucket| {
                (0..blocks).map(move |block| ((bucket * 3usize + block * 5usize) % 7usize) as u32)
            })
            .collect();
        let histogram_device = exec.to_device(&histograms);
        let params = exec.to_device(&[0u32, blocks as u32, 0u32]);
        let prefixes = exec.alloc_column::<u32>(histograms.len());
        let chunk_totals = exec.alloc_column::<u32>(RADIX_BUCKETS * chunks);
        let chunk_prefixes = exec.alloc_column::<u32>(RADIX_BUCKETS * chunks);
        let totals = exec.alloc_column::<u32>(RADIX_BUCKETS);

        unsafe {
            histogram_chunk_scan_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                crate::core::launch::cube_count_1d(RADIX_BUCKETS * chunks).unwrap(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(
                    histogram_device.handle.clone(),
                    histogram_device.capacity(),
                ),
                BufferArg::from_raw_parts(params.handle.clone(), 3),
                BufferArg::from_raw_parts(prefixes.handle.clone(), prefixes.capacity()),
                BufferArg::from_raw_parts(chunk_totals.handle.clone(), chunk_totals.capacity()),
            );
            histogram_chunk_prefix_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                crate::core::launch::cube_count_1d(RADIX_BUCKETS).unwrap(),
                CubeDim::new_1d(chunks as u32),
                BufferArg::from_raw_parts(chunk_totals.handle.clone(), chunk_totals.capacity()),
                BufferArg::from_raw_parts(params.handle.clone(), 3),
                BufferArg::from_raw_parts(chunk_prefixes.handle.clone(), chunk_prefixes.capacity()),
                BufferArg::from_raw_parts(totals.handle.clone(), totals.capacity()),
            );
            add_histogram_chunk_prefix_kernel::launch_unchecked::<WgpuRuntime>(
                exec.client(),
                crate::core::launch::cube_count_1d(
                    prefixes.capacity().div_ceil(BLOCK_SIZE as usize),
                )
                .unwrap(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(chunk_prefixes.handle.clone(), chunk_prefixes.capacity()),
                BufferArg::from_raw_parts(params.handle.clone(), 3),
                BufferArg::from_raw_parts(prefixes.handle.clone(), prefixes.capacity()),
            );
        }

        let actual = exec.to_host(&prefixes).unwrap();
        let actual_totals = exec.to_host(&totals).unwrap();
        for bucket in 0..RADIX_BUCKETS {
            let mut expected = 0u32;
            for block in 0..blocks {
                let index = bucket * blocks + block;
                assert_eq!(actual[index], expected, "bucket {bucket}, block {block}");
                expected += histograms[index];
            }
            assert_eq!(actual_totals[bucket], expected, "bucket {bucket}");
        }
    }
}
