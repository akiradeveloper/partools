mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::BinaryPredicateOp, op::ReductionOp, vector::exclusive_scan, vector::exclusive_scan_by_key,
    vector::inclusive_scan, vector::inclusive_scan_by_key,
};

struct Sum;
struct Equal;
#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}
#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

fn bench_scan(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("scan");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let keys = exec.to_device(&common::run_keys(len, 8));
        let init = 0.0_f32;
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("inclusive", len), |b| {
            b.iter(|| {
                std::hint::black_box(inclusive_scan(&exec, values.slice(..), Sum).unwrap());
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("exclusive", len), |b| {
            b.iter(|| {
                std::hint::black_box(exclusive_scan(&exec, values.slice(..), init, Sum).unwrap());
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("inclusive_by_key", len), |b| {
            b.iter(|| {
                std::hint::black_box(
                    inclusive_scan_by_key(&exec, keys.slice(..), values.slice(..), Equal, Sum)
                        .unwrap(),
                );
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("exclusive_by_key", len), |b| {
            b.iter(|| {
                std::hint::black_box(
                    exclusive_scan_by_key(
                        &exec,
                        keys.slice(..),
                        values.slice(..),
                        Equal,
                        init,
                        Sum,
                    )
                    .unwrap(),
                );
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_scan }
criterion_main!(benches);
