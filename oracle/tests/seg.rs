#[path = "vector/common.rs"]
#[allow(dead_code)]
mod common;

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use massively::seg::{
    AdjacentDifference, AdjacentFind, AllOf, AnyOf, CountIf, ExclusiveScan, Executable, Filter,
    FindIf, ForEachSegment, InclusiveScan, IsSorted, IsSortedUntil, Map, NoneOf, Reduce, Reverse,
    SegmentIterator, Sort, Take, Unique,
};
use massively::{
    Executor, MStorage, op::BinaryPredicateOp, op::PredicateOp, op::ReductionOp, op::UnaryOp, zip2,
};
use oracle::{op, seg as reference};
use proptest::prelude::*;

use common::*;

const SEGMENT_LENGTHS: [usize; 12] = [0, 1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257];

fn oracle_segments() -> impl Strategy<Value = Vec<Vec<u32>>> {
    prop::collection::vec(
        prop::sample::select(&SEGMENT_LENGTHS)
            .prop_flat_map(|len| prop::collection::vec(0_u32..100, len)),
        0..6,
    )
}

fn flatten<T: Clone>(segments: &[Vec<T>]) -> (Vec<T>, Vec<u32>) {
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(segments.len() + 1);
    offsets.push(0);
    for segment in segments {
        values.extend_from_slice(segment);
        offsets.push(values.len() as u32);
    }
    (values, offsets)
}

macro_rules! length_preserving_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment(($algorithm)(&exec)).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                let output_offsets = massively::vector::map(
                    &exec,
                    output.offsets().clone(),
                    massively::op::Identity,
                ).unwrap();
                prop_assert_eq!(exec.to_host(output.values()).unwrap(), expected);
                prop_assert_eq!(exec.to_host(&output_offsets).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! compacting_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();
                let (expected_values, expected_offsets) = flatten(&$oracle(&segments));
                prop_assert_eq!(exec.to_host(output.values()).unwrap(), expected_values);
                prop_assert_eq!(
                    exec.to_host(&output.offsets().offsets()).unwrap(),
                    expected_offsets,
                );
            }
        }
    };
}

macro_rules! summarizing_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment(($algorithm)(&exec)).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();
                prop_assert_eq!(exec.to_host(&output).unwrap(), $oracle(&segments));
            }
        }
    };
}

length_preserving_case!(seg_map, |_| Map(AddOne), |segments| {
    reference::map(segments, AddOne)
});
length_preserving_case!(seg_sort, |_| Sort(Less), |segments| reference::sort(
    segments, Less
));
length_preserving_case!(seg_reverse, |_| Reverse, reference::reverse);
length_preserving_case!(seg_inclusive_scan, |_| InclusiveScan(Sum), |segments| {
    reference::inclusive_scan(segments, Sum)
});
length_preserving_case!(
    seg_exclusive_scan,
    |_exec: &Executor<WgpuRuntime>| ExclusiveScan(Sum, 7),
    |segments| { reference::exclusive_scan(segments, Sum, 7) }
);
length_preserving_case!(
    seg_adjacent_difference,
    |_| AdjacentDifference(Sum),
    |segments| reference::adjacent_difference(segments, Sum)
);

compacting_case!(seg_unique, Unique(Equal), |segments| reference::unique(
    segments, Equal
));
compacting_case!(seg_filter, Filter(Even), |segments| reference::filter(
    segments, Even
));
compacting_case!(seg_take, Take(37), |segments| reference::take(segments, 37));

summarizing_case!(
    seg_reduce,
    |_exec: &Executor<WgpuRuntime>| Reduce(Sum, 7),
    |segments| reference::reduce(segments, Sum, 7)
);
summarizing_case!(seg_count_if, |_| CountIf(Even), |segments| {
    reference::count_if(segments, Even)
});
summarizing_case!(seg_find_if, |_| FindIf(Even), |segments| {
    reference::find_if(segments, Even)
});
summarizing_case!(seg_adjacent_find, |_| AdjacentFind(Equal), |segments| {
    reference::adjacent_find(segments, Equal)
});
summarizing_case!(seg_all_of, |_| AllOf(Even), |segments| expected_flags(
    reference::all_of(segments, Even)
));
summarizing_case!(seg_any_of, |_| AnyOf(Even), |segments| expected_flags(
    reference::any_of(segments, Even)
));
summarizing_case!(seg_none_of, |_| NoneOf(Even), |segments| expected_flags(
    reference::none_of(segments, Even)
));
summarizing_case!(seg_is_sorted, |_| IsSorted(Less), |segments| {
    expected_flags(reference::is_sorted(segments, Less))
});
summarizing_case!(seg_is_sorted_until, |_| IsSortedUntil(Less), |segments| {
    reference::is_sorted_until(segments, Less)
});

type Pair = (u32, u32);

struct PairAddOne;
struct PairSum;
struct PairEven;
struct PairEqual;
struct PairLess;

#[cubecl::cube]
impl UnaryOp<Pair> for PairAddOne {
    type Output = Pair;

    fn apply(input: Pair) -> Pair {
        (input.0 + 1u32, input.1 + 1u32)
    }
}

impl op::UnaryOp<Pair> for PairAddOne {
    type Output = Pair;

    fn apply(input: Pair) -> Pair {
        (input.0 + 1, input.1 + 1)
    }
}

#[cubecl::cube]
impl ReductionOp<Pair> for PairSum {
    fn apply(lhs: Pair, rhs: Pair) -> Pair {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

impl op::ReductionOp<Pair> for PairSum {
    fn apply(lhs: Pair, rhs: Pair) -> Pair {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

#[cubecl::cube]
impl PredicateOp<Pair> for PairEven {
    fn apply(input: Pair) -> massively::MFlag {
        massively::flag::from_bool(input.0 % 2u32 == 0u32)
    }
}

impl op::PredicateOp<Pair> for PairEven {
    fn apply(input: Pair) -> bool {
        input.0 % 2 == 0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Pair> for PairEqual {
    fn apply(lhs: Pair, rhs: Pair) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 == rhs.0 && lhs.1 == rhs.1)
    }
}

impl op::BinaryPredicateOp<Pair> for PairEqual {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Pair> for PairLess {
    fn apply(lhs: Pair, rhs: Pair) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

impl op::BinaryPredicateOp<Pair> for PairLess {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs.0 < rhs.0
    }
}

fn oracle_pair_segments() -> impl Strategy<Value = Vec<Vec<Pair>>> {
    oracle_segments().prop_map(|segments| {
        segments
            .into_iter()
            .enumerate()
            .map(|(segment, values)| {
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        (
                            value,
                            value.wrapping_add(segment as u32 * 17 + index as u32),
                        )
                    })
                    .collect()
            })
            .collect()
    })
}

fn pair_columns(rows: &[Pair]) -> (Vec<u32>, Vec<u32>) {
    rows.iter().copied().unzip()
}

fn pair_rows(first: Vec<u32>, second: Vec<u32>) -> Vec<Pair> {
    first.into_iter().zip(second).collect()
}

macro_rules! pair_length_preserving_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment(($algorithm)(&exec)).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                let (output_first, output_second) =
                    MStorage::into_columns(output.values().clone());
                prop_assert_eq!(
                    pair_rows(
                        exec.to_host(&output_first).unwrap(),
                        exec.to_host(&output_second).unwrap(),
                    ),
                    expected,
                );
                let output_offsets = massively::vector::map(
                    &exec,
                    output.offsets().clone(),
                    massively::op::Identity,
                ).unwrap();
                prop_assert_eq!(exec.to_host(&output_offsets).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! pair_compacting_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();
                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                let (output_first, output_second) =
                    MStorage::into_columns(output.values().clone());
                let output_first = exec.to_host(&output_first).unwrap();
                let output_second = exec.to_host(&output_second).unwrap();
                prop_assert_eq!(
                    pair_rows(output_first, output_second),
                    expected,
                );
                prop_assert_eq!(
                    exec.to_host(&output.offsets().offsets()).unwrap(),
                    expected_offsets,
                );
            }
        }
    };
}

macro_rules! pair_item_reduce_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment(($algorithm)(&exec)).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (output_first, output_second) = MStorage::into_columns(output);
                prop_assert_eq!(
                    pair_rows(
                        exec.to_host(&output_first).unwrap(),
                        exec.to_host(&output_second).unwrap(),
                    ),
                    $oracle(&segments),
                );
            }
        }
    };
}

macro_rules! pair_flag_reduce_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                prop_assert_eq!(exec.to_host(&output).unwrap(), $oracle(&segments));
            }
        }
    };
}

pair_length_preserving_case!(seg_pair_map, |_| Map(PairAddOne), |segments| {
    reference::map(segments, PairAddOne)
});
pair_length_preserving_case!(seg_pair_sort, |_| Sort(PairLess), |segments| {
    reference::sort(segments, PairLess)
});
pair_length_preserving_case!(seg_pair_reverse, |_| Reverse, reference::reverse);
pair_length_preserving_case!(
    seg_pair_inclusive_scan,
    |_| InclusiveScan(PairSum),
    |segments| reference::inclusive_scan(segments, PairSum)
);
pair_length_preserving_case!(
    seg_pair_exclusive_scan,
    |_exec: &Executor<WgpuRuntime>| ExclusiveScan(PairSum, (7, 11)),
    |segments| reference::exclusive_scan(segments, PairSum, (7, 11))
);
pair_length_preserving_case!(
    seg_pair_adjacent_difference,
    |_| AdjacentDifference(PairSum),
    |segments| reference::adjacent_difference(segments, PairSum)
);

pair_compacting_case!(seg_pair_unique, Unique(PairEqual), |segments| {
    reference::unique(segments, PairEqual)
});
pair_compacting_case!(seg_pair_filter, Filter(PairEven), |segments| {
    reference::filter(segments, PairEven)
});
pair_compacting_case!(seg_pair_take, Take(37), |segments| reference::take(
    segments, 37
));

pair_item_reduce_case!(
    seg_pair_reduce,
    |_exec: &Executor<WgpuRuntime>| Reduce(PairSum, (7, 11)),
    |segments| { reference::reduce(segments, PairSum, (7, 11)) }
);
pair_flag_reduce_case!(seg_pair_count_if, CountIf(PairEven), |segments| {
    reference::count_if(segments, PairEven)
});
pair_flag_reduce_case!(seg_pair_find_if, FindIf(PairEven), |segments| {
    reference::find_if(segments, PairEven)
});
pair_flag_reduce_case!(
    seg_pair_adjacent_find,
    AdjacentFind(PairEqual),
    |segments| { reference::adjacent_find(segments, PairEqual) }
);
pair_flag_reduce_case!(seg_pair_all_of, AllOf(PairEven), |segments| {
    expected_flags(reference::all_of(segments, PairEven))
});
pair_flag_reduce_case!(seg_pair_any_of, AnyOf(PairEven), |segments| {
    expected_flags(reference::any_of(segments, PairEven))
});
pair_flag_reduce_case!(seg_pair_none_of, NoneOf(PairEven), |segments| {
    expected_flags(reference::none_of(segments, PairEven))
});
pair_flag_reduce_case!(seg_pair_is_sorted, IsSorted(PairLess), |segments| {
    expected_flags(reference::is_sorted(segments, PairLess))
});
pair_flag_reduce_case!(
    seg_pair_is_sorted_until,
    IsSortedUntil(PairLess),
    |segments| reference::is_sorted_until(segments, PairLess)
);
