mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use massively::{vector::gather_where, vector::scatter_where};

fn bench_where_indexed(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("where_indexed");
    for &len in common::SIZES {
        let input = exec.to_device(&common::dense_f32(len));
        let indices = exec.to_device(&common::reverse_indices(len));
        let flags = exec.to_device(&common::flags(len, 50));
        let output = exec.alloc::<f32>(len.try_into().unwrap());
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("gather_where", len), |b| {
            b.iter(|| {
                gather_where(
                    &exec,
                    input.slice(..),
                    indices.slice(..),
                    flags.slice(..),
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("scatter_where", len), |b| {
            b.iter(|| {
                scatter_where(
                    &exec,
                    input.slice(..),
                    indices.slice(..),
                    flags.slice(..),
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_where_indexed }
criterion_main!(benches);
