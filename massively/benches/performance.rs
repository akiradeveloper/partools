mod common;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    lazy,
    op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp},
    vector::{
        all_of, any_of, copy_where, count_if, exclusive_scan, exclusive_scan_by_key, find_if,
        gather, gather_where, inclusive_scan, inclusive_scan_by_key, is_partitioned, lower_bound,
        map, max_element, merge, merge_by_key, min_element, minmax_element, none_of, partition,
        radix_sort_by_key, reduce, reduce_by_key, remove_where, scatter, scatter_reduce,
        scatter_where, set_difference, set_intersection, set_union, sort, sort_by_key, unique,
        unique_by_key, upper_bound,
    },
};

const N: usize = 10_000_000;

struct EqualU32;
struct EvenIndex;
struct LessU32;
struct LessIndex;
struct MulTwo;
struct PositiveF32;
struct SumF32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for EvenIndex {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2u32 == 0u32)
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessIndex {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

#[cubecl::cube]
impl PredicateOp<f32> for PositiveF32 {
    fn apply(input: f32) -> massively::MFlag {
        massively::flag::from_bool(input > 0.0)
    }
}

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

fn ascending(len: usize) -> Vec<u32> {
    (0..len).map(|index| index as u32).collect()
}

fn shifted(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index + len / 2) as u32).collect()
}

fn search_queries(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % (len.max(1) * 2)) as u32)
        .collect()
}

macro_rules! benchmark {
    ($group:ident, $exec:ident, $name:literal, $expression:expr) => {
        $group.bench_function($name, |b| {
            b.iter(|| {
                let result = black_box($expression);
                $exec.sync().unwrap();
                result
            })
        });
    };
}

macro_rules! benchmark_synchronous {
    ($group:ident, $name:literal, $expression:expr) => {
        $group.bench_function($name, |b| b.iter(|| black_box($expression)));
    };
}

fn bench_performance(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("performance");

    {
        let values = exec.to_device(&common::dense_f32(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "map",
            map(&exec, black_box(values.slice(..)), MulTwo).unwrap()
        );
    }

    {
        let values = exec.to_device(&common::dense_f32(N));
        let keys = exec.to_device(&common::run_keys(N, 8));
        let reduce_init = 0.0_f32;
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "exclusive_scan",
            exclusive_scan(&exec, black_box(values.slice(..)), reduce_init, SumF32,).unwrap()
        );
        benchmark!(
            group,
            exec,
            "exclusive_scan_by_key",
            exclusive_scan_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
                EqualU32,
                reduce_init,
                SumF32,
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "inclusive_scan",
            inclusive_scan(&exec, black_box(values.slice(..)), SumF32).unwrap()
        );
        benchmark!(
            group,
            exec,
            "inclusive_scan_by_key",
            inclusive_scan_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
                EqualU32,
                SumF32,
            )
            .unwrap()
        );
        benchmark_synchronous!(
            group,
            "reduce",
            reduce(&exec, black_box(values.slice(..)), reduce_init, SumF32,).unwrap()
        );
        benchmark!(
            group,
            exec,
            "reduce_by_key",
            reduce_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
                EqualU32,
                reduce_init,
                SumF32,
            )
            .unwrap()
        );
    }

    {
        let values = exec.to_device(&common::dense_f32(N));
        let flags = exec.to_device(&common::flags(N, 50));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "copy_where",
            copy_where(
                &exec,
                black_box(values.slice(..)),
                black_box(flags.slice(..)),
            )
            .unwrap()
        );
        benchmark_synchronous!(group, "partition", {
            let (output, boundary) =
                partition(&exec, black_box(values.slice(..)), PositiveF32).unwrap();
            black_box(boundary);
            output
        });
        benchmark!(
            group,
            exec,
            "remove_where",
            remove_where(
                &exec,
                black_box(values.slice(..)),
                black_box(flags.slice(..)),
            )
            .unwrap()
        );
    }

    {
        let values = exec.to_device(&common::dense_f32(N));
        let reverse_indices = exec.to_device(&common::reverse_indices(N));
        let collision_indices = exec.to_device(
            &(0..N)
                .map(|index| (index % (N / 4)) as u32)
                .collect::<Vec<_>>(),
        );
        let flags = exec.to_device(&common::flags(N, 50));
        let reduce_init = 0.0_f32;
        let output = exec.alloc::<f32>(N.try_into().unwrap());
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "gather",
            gather(
                &exec,
                black_box(values.slice(..)),
                black_box(reverse_indices.slice(..)),
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "gather_where",
            gather_where(
                &exec,
                black_box(values.slice(..)),
                black_box(reverse_indices.slice(..)),
                black_box(flags.slice(..)),
                output.slice_mut(..),
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "scatter",
            scatter(
                &exec,
                black_box(values.slice(..)),
                black_box(reverse_indices.slice(..)),
                output.slice_mut(..),
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "scatter_reduce",
            scatter_reduce(
                &exec,
                black_box(values.slice(..)),
                black_box(collision_indices.slice(..)),
                reduce_init,
                SumF32,
                output.slice_mut(..),
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "scatter_where",
            scatter_where(
                &exec,
                black_box(values.slice(..)),
                black_box(reverse_indices.slice(..)),
                black_box(flags.slice(..)),
                output.slice_mut(..),
            )
            .unwrap()
        );
    }

    {
        let keys = exec.to_device(&common::shuffled_u32(N));
        let values = exec.to_device(&common::dense_f32(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "radix_sort_by_key",
            radix_sort_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "sort_by_key",
            sort_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
                LessU32,
            )
            .unwrap()
        );
    }

    {
        let keys = exec.to_device(&common::run_keys(N, 8));
        let values = exec.to_device(&ascending(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "unique",
            unique(&exec, black_box(keys.slice(..)), EqualU32).unwrap()
        );
        benchmark!(
            group,
            exec,
            "unique_by_key",
            unique_by_key(
                &exec,
                black_box(keys.slice(..)),
                black_box(values.slice(..)),
                EqualU32,
            )
            .unwrap()
        );
    }

    {
        let values = exec.to_device(&common::shuffled_u32(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "sort",
            sort(&exec, black_box(values.slice(..)), LessU32).unwrap()
        );
    }

    {
        let left = exec.to_device(&(0..N).map(|index| (index * 2) as u32).collect::<Vec<_>>());
        let right = exec.to_device(
            &(0..N)
                .map(|index| (index * 2 + 1) as u32)
                .collect::<Vec<_>>(),
        );
        let left_values = exec.to_device(&common::dense_f32(N));
        let right_values = exec.to_device(&common::dense_f32(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "merge",
            merge(
                &exec,
                black_box(left.slice(..)),
                black_box(right.slice(..)),
                LessU32,
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "merge_by_key",
            merge_by_key(
                &exec,
                black_box(left.slice(..)),
                black_box(left_values.slice(..)),
                black_box(right.slice(..)),
                black_box(right_values.slice(..)),
                LessU32,
            )
            .unwrap()
        );
    }

    {
        let left = exec.to_device(&ascending(N));
        let right = exec.to_device(&shifted(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "set_difference",
            set_difference(
                &exec,
                black_box(left.slice(..)),
                black_box(right.slice(..)),
                LessU32,
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "set_intersection",
            set_intersection(
                &exec,
                black_box(left.slice(..)),
                black_box(right.slice(..)),
                LessU32,
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "set_union",
            set_union(
                &exec,
                black_box(left.slice(..)),
                black_box(right.slice(..)),
                LessU32,
            )
            .unwrap()
        );
    }

    {
        let source = exec.to_device(&ascending(N));
        let queries = exec.to_device(&search_queries(N));
        exec.sync().unwrap();
        benchmark!(
            group,
            exec,
            "lower_bound",
            lower_bound(
                &exec,
                black_box(source.slice(..)),
                black_box(queries.slice(..)),
                LessU32,
            )
            .unwrap()
        );
        benchmark!(
            group,
            exec,
            "upper_bound",
            upper_bound(
                &exec,
                black_box(source.slice(..)),
                black_box(queries.slice(..)),
                LessU32,
            )
            .unwrap()
        );
    }

    benchmark_synchronous!(
        group,
        "all_of",
        all_of(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "any_of",
        any_of(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "count_if",
        count_if(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "find_if",
        find_if(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "is_partitioned",
        is_partitioned(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "none_of",
        none_of(&exec, lazy::counting(0).take(N as u32), EvenIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "max_element",
        max_element(&exec, lazy::counting(0).take(N as u32), LessIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "min_element",
        min_element(&exec, lazy::counting(0).take(N as u32), LessIndex).unwrap()
    );
    benchmark_synchronous!(
        group,
        "minmax_element",
        minmax_element(&exec, lazy::counting(0).take(N as u32), LessIndex).unwrap()
    );

    group.finish();
}

criterion_group! { name = benches; config = common::criterion(); targets = bench_performance }
criterion_main!(benches);
