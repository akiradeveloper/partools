mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::BinaryPredicateOp, op::ReductionOp, vector::exclusive_scan_by_key,
    vector::inclusive_scan_by_key, vector::reduce_by_key, vector::unique_by_key,
};

struct Equal;
struct Sum;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

fn key_patterns(len: usize) -> [(String, Vec<u32>); 4] {
    [
        ("run1".into(), common::run_keys(len, 1)),
        ("run8".into(), common::run_keys(len, 8)),
        ("run128".into(), common::run_keys(len, 128)),
        ("all".into(), common::run_keys(len, len)),
    ]
}

fn bench_by_key_patterns(c: &mut Criterion) {
    let exec = common::exec();
    let mut inclusive = c.benchmark_group("inclusive_scan_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            inclusive.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = inclusive_scan_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        Sum,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    inclusive.finish();

    let mut exclusive = c.benchmark_group("exclusive_scan_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let init = 0.0_f32;
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            exclusive.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = exclusive_scan_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        init,
                        Sum,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    exclusive.finish();

    let mut reduce = c.benchmark_group("reduce_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let init = 0.0_f32;
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            reduce.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = reduce_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        init,
                        Sum,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    reduce.finish();

    let mut unique = c.benchmark_group("unique_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&(0..len as u32).collect::<Vec<_>>());
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            unique.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = unique_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    unique.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_by_key_patterns
}
criterion_main!(benches);
