mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use massively::vector::gather;

fn bench_gather(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("gather");
    for &len in common::SIZES {
        let input = exec.to_device(&common::dense_f32(len));
        let indices = exec.to_device(&common::reverse_indices(len));
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("reverse", len), |b| {
            b.iter(|| {
                std::hint::black_box(gather(&exec, input.slice(..), indices.slice(..)).unwrap());
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_gather }
criterion_main!(benches);
