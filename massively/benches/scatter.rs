mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{op::ReductionOp, vector::scatter, vector::scatter_reduce};

struct Add;

#[cubecl::cube]
impl ReductionOp<f32> for Add {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

fn bench_scatter(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("scatter");
    for &len in common::SIZES {
        let input = exec.to_device(&common::dense_f32(len));
        let indices = exec.to_device(&common::reverse_indices(len));
        let output = exec.alloc::<f32>(len.try_into().unwrap());
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("reverse", len), |b| {
            b.iter(|| {
                scatter(
                    &exec,
                    input.slice(..),
                    indices.slice(..),
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });

        let collision_indices: Vec<u32> = (0..len)
            .map(|index| (index % (len / 4).max(1)) as u32)
            .collect();
        let collision_indices = exec.to_device(&collision_indices);
        let init = 0.0_f32;
        group.bench_function(BenchmarkId::new("reduce_4_to_1", len), |b| {
            b.iter(|| {
                scatter_reduce(
                    &exec,
                    input.slice(..),
                    collision_indices.slice(..),
                    init,
                    Add,
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_scatter }
criterion_main!(benches);
