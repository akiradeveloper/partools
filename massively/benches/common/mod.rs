#![allow(dead_code)]

use std::time::Duration;

use criterion::Criterion;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;

pub const SIZES: &[usize] = &[1_024, 16 * 1_024, 256 * 1_024, 1_024 * 1_024];
pub const SORT_SIZES: &[usize] = SIZES;
pub const SORT_PATTERN_SIZE: usize = 256 * 1_024;

pub fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

pub fn dense_f32(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 251) as f32).collect()
}

pub fn shuffled_u32(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % len.max(1)) as u32)
        .collect()
}

pub fn reverse_indices(len: usize) -> Vec<u32> {
    (0..len).rev().map(|index| index as u32).collect()
}

pub fn reverse_u32(len: usize) -> Vec<u32> {
    reverse_indices(len)
}

pub fn flags(len: usize, selected_per_100: usize) -> Vec<u32> {
    (0..len)
        .map(|index| u32::from(index % 100 < selected_per_100))
        .collect()
}

pub fn run_keys(len: usize, run_len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| (index / run_len.max(1)) as u32)
        .collect()
}

pub fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250))
}

/// Keeps the result alive through the resident GPU completion boundary.
pub fn completed<T>(exec: &Executor<WgpuRuntime>, result: T) -> T {
    exec.sync().unwrap();
    criterion::black_box(result)
}
