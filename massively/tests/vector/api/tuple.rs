use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MFlag, MStorage,
    op::{BinaryPredicateOp, UnaryOp},
    vector::{map, replace_where, sort},
};

type Twelve = (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);

type Wide = (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

struct WideColumns;
struct WideSum;

#[cubecl::cube]
impl UnaryOp<u32> for WideColumns {
    type Output = Wide;
    fn apply(input: u32) -> Wide {
        let value = input as u64;
        (
            value * 1u64,
            value * 2u64,
            value * 3u64,
            value * 4u64,
            value * 5u64,
            value * 6u64,
            value * 7u64,
            value * 8u64,
            value * 9u64,
            value * 10u64,
            value * 11u64,
            value * 12u64,
        )
    }
}

#[cubecl::cube]
impl massively::op::ReductionOp<Wide> for WideSum {
    fn apply(lhs: Wide, rhs: Wide) -> Wide {
        (
            lhs.0 + rhs.0,
            lhs.1 + rhs.1,
            lhs.2 + rhs.2,
            lhs.3 + rhs.3,
            lhs.4 + rhs.4,
            lhs.5 + rhs.5,
            lhs.6 + rhs.6,
            lhs.7 + rhs.7,
            lhs.8 + rhs.8,
            lhs.9 + rhs.9,
            lhs.10 + rhs.10,
            lhs.11 + rhs.11,
        )
    }
}

#[test]
fn scans_and_reduction_preserve_twelve_wide_columns_across_blocks() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let len = 17_003u32;
    let input = massively::lazy::map(massively::lazy::counting(1u32).take(len), WideColumns);
    let initial = (
        1u64, 2u64, 3u64, 4u64, 5u64, 6u64, 7u64, 8u64, 9u64, 10u64, 11u64, 12u64,
    );
    let reduced = massively::vector::reduce(&exec, input.clone(), initial, WideSum).unwrap();
    for (column, actual) in [
        reduced.0, reduced.1, reduced.2, reduced.3, reduced.4, reduced.5, reduced.6, reduced.7,
        reduced.8, reduced.9, reduced.10, reduced.11,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            actual,
            (column as u64 + 1) * (1 + u64::from(len) * (u64::from(len) + 1) / 2)
        );
    }
    for exclusive in [false, true] {
        let output = if exclusive {
            massively::vector::exclusive_scan(&exec, input.clone(), initial, WideSum).unwrap()
        } else {
            massively::vector::inclusive_scan(&exec, input.clone(), WideSum).unwrap()
        };
        let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) = MStorage::into_columns(output);
        for (column, values) in [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
            .iter()
            .enumerate()
        {
            let actual = exec.to_host(values).unwrap();
            let expected: Vec<_> = (0..u64::from(len))
                .map(|index| {
                    let terms = if exclusive { index } else { index + 1 };
                    (column as u64 + 1) * (u64::from(exclusive) + terms * (terms + 1) / 2)
                })
                .collect();
            assert_eq!(actual, expected, "column {column}, exclusive {exclusive}");
        }
    }
}

struct AscendingTuple;

#[cubecl::cube]
impl UnaryOp<u32> for AscendingTuple {
    type Output = Twelve;

    fn apply(input: u32) -> Self::Output {
        (
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        )
    }
}

struct ReversedTuple;

struct LessTwelve;

#[cubecl::cube]
impl BinaryPredicateOp<Twelve> for LessTwelve {
    fn apply(lhs: Twelve, rhs: Twelve) -> MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for ReversedTuple {
    type Output = Twelve;

    fn apply(input: u32) -> Self::Output {
        let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = (
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        );
        (a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0)
    }
}

#[test]
fn flat_tuples_can_be_destructured_directly_inside_cube_ops() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = map(&exec, input.slice(..), ReversedTuple).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];

    for (column, offset) in outputs
        .into_iter()
        .zip([11_u32, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}

#[test]
fn tuple_outputs_expose_flat_owned_columns() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = map(&exec, input.slice(..), AscendingTuple).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];

    for (column, offset) in outputs.into_iter().zip(0_u32..) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}

#[test]
fn sort_composes_ordering_and_gather_for_twelve_columns() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[30_u32, 10, 20]);
    let rows = map(&exec, input.slice(..), AscendingTuple).unwrap();
    let output = sort(&exec, rows.slice(..), LessTwelve).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(output);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];

    for (column, offset) in outputs.into_iter().zip(0_u32..) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset, 30 + offset]
        );
    }
}

#[test]
fn replace_where_repeats_one_twelve_column_row_without_materializing_it() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]);
    let rows = map(&exec, input.slice(..), AscendingTuple).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0]);
    let replacement = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111);

    replace_where(&exec, replacement, stencil.slice(..), rows.slice_mut(..)).unwrap();

    let (a, b, c, d, e, f, g, h, i, j, k, l) = MStorage::into_columns(rows);
    let outputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l];
    for (column, offset) in outputs.into_iter().zip(0_u32..) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![1 + offset, 100 + offset, 3 + offset]
        );
    }
}
