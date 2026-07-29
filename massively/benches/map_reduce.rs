use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, lazy, op::ReductionOp, op::UnaryOp, vector::reduce, zip2, zip3};

const SIZES: &[usize] = &[256 * 1024, 1024 * 1024, 16 * 1024 * 1024];

struct Sum;

#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

struct AddPair;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for AddPair {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 + input.1
    }
}

struct AddTriple;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32)> for AddTriple {
    type Output = f32;

    fn apply(input: (f32, f32, f32)) -> f32 {
        input.0 + input.1 + input.2
    }
}

fn dense_f32(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 251) as f32).collect()
}

fn bench_map_reduce(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let mut column_group = c.benchmark_group("map_reduce");
    for &len in SIZES {
        let values = exec.to_device(&dense_f32(len));
        let init = 0.0_f32;
        exec.sync().unwrap();
        column_group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let input = lazy::map(values.slice(..), MulTwo);
                black_box(reduce(&exec, input, init, Sum).unwrap())
            })
        });
    }
    column_group.finish();

    let mut zip2_group = c.benchmark_group("map_reduce_zip2");
    for &len in SIZES {
        let left = exec.to_device(&dense_f32(len));
        let right = exec.to_device(&dense_f32(len));
        let init = 0.0_f32;
        exec.sync().unwrap();
        zip2_group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let input = lazy::map(
                    zip2(black_box(left.slice(..)), black_box(right.slice(..))),
                    AddPair,
                );
                black_box(reduce(&exec, input, init, Sum).unwrap())
            })
        });
    }
    zip2_group.finish();

    let mut zip3_group = c.benchmark_group("map_reduce_zip3");
    for &len in SIZES {
        let first = exec.to_device(&dense_f32(len));
        let second = exec.to_device(&dense_f32(len));
        let third = exec.to_device(&dense_f32(len));
        let init = 0.0_f32;
        exec.sync().unwrap();
        zip3_group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let input = lazy::map(
                    zip3(
                        black_box(first.slice(..)),
                        black_box(second.slice(..)),
                        black_box(third.slice(..)),
                    ),
                    AddTriple,
                );
                black_box(reduce(&exec, input, init, Sum).unwrap())
            })
        });
    }
    zip3_group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250));
    targets = bench_map_reduce
}
criterion_main!(benches);
