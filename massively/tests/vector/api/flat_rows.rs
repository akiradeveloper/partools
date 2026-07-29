use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::*, vector::*, *};

type Triple = (u32, u32, u32);

fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

macro_rules! nested_input {
    ($a:expr, $b:expr, $c:expr) => {
        zip2($a.slice(..), zip2($b.slice(..), $c.slice(..)))
    };
}

struct SumTriple;

#[cubecl::cube]
impl ReductionOp<Triple> for SumTriple {
    fn apply(lhs: Triple, rhs: Triple) -> Triple {
        (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    }
}

struct LessTriple;

#[cubecl::cube]
impl BinaryPredicateOp<Triple> for LessTriple {
    fn apply(lhs: Triple, rhs: Triple) -> MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

struct AddOne;

#[cubecl::cube]
impl UnaryOp<Triple> for AddOne {
    type Output = Triple;

    fn apply(input: Triple) -> Triple {
        (input.0 + 1, input.1 + 1, input.2 + 1)
    }
}

fn stencil<Input>(input: Input) -> Input {
    input
}

#[test]
fn inclusive_scan_treats_nested_zip_calls_as_flat_rows() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let output = inclusive_scan(&exec, nested_input!(a, b, c), SumTriple).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 3, 6]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 30, 60]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 300, 600]);
}

#[test]
fn sort_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[3_u32, 1, 2]);
    let b = exec.to_device(&[30_u32, 10, 20]);
    let c = exec.to_device(&[300_u32, 100, 200]);
    let output = sort(&exec, nested_input!(a, b, c), LessTriple).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300]);
}

#[test]
fn copy_where_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);
    let flags = exec.to_device(&[0_u32, 1, 1, 0]);
    let output = copy_where(&exec, nested_input!(a, b, c), stencil(flags.slice(..))).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![2, 3]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![200, 300]);
}

#[test]
fn gather_returns_flat_owned_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let indices = exec.to_device(&[2_u32, 0]);
    let output = gather(&exec, nested_input!(a, b, c), indices.slice(..)).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![3, 1]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 10]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300, 100]);
}

#[test]
fn merge_returns_flat_owned_columns() {
    let exec = exec();
    let la = exec.to_device(&[1_u32, 3]);
    let lb = exec.to_device(&[10_u32, 30]);
    let lc = exec.to_device(&[100_u32, 300]);
    let ra = exec.to_device(&[2_u32, 4]);
    let rb = exec.to_device(&[20_u32, 40]);
    let rc = exec.to_device(&[200_u32, 400]);
    let output = merge(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300, 400]);
}

#[test]
fn set_operations_preserve_flat_row_payloads() {
    let exec = exec();
    let la = exec.to_device(&[1_u32, 2, 2, 4]);
    let lb = exec.to_device(&[10_u32, 20, 21, 40]);
    let lc = exec.to_device(&[100_u32, 200, 210, 400]);
    let ra = exec.to_device(&[2_u32, 2, 2, 3, 4, 4]);
    let rb = exec.to_device(&[200_u32, 201, 202, 300, 400, 401]);
    let rc = exec.to_device(&[2_000_u32, 2_010, 2_020, 3_000, 4_000, 4_010]);

    let union = set_union(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(union);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 2, 2, 2, 3, 4, 4]);
    assert_eq!(
        exec.to_host(&b).unwrap(),
        vec![10, 20, 21, 202, 300, 40, 401]
    );
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![100, 200, 210, 2_020, 3_000, 400, 4_010]
    );

    let intersection = set_intersection(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(intersection);
    assert_eq!(exec.to_host(&a).unwrap(), vec![2, 2, 4]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![20, 21, 40]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![200, 210, 400]);

    let difference = set_difference(
        &exec,
        nested_input!(la, lb, lc),
        nested_input!(ra, rb, rc),
        LessTriple,
    )
    .unwrap();
    let (a, b, c) = MStorage::into_columns(difference);
    assert_eq!(exec.to_host(&a).unwrap(), vec![1]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100]);
}

#[test]
fn sort_by_key_returns_flat_value_columns() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]);
    let a = exec.to_device(&[30_u32, 10, 20]);
    let b = exec.to_device(&[300_u32, 100, 200]);
    let c = exec.to_device(&[3_000_u32, 1_000, 2_000]);
    let output = sort_by_key(&exec, keys.slice(..), nested_input!(a, b, c), LessU32).unwrap();
    let (a, b, c) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![100, 200, 300]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![1_000, 2_000, 3_000]);
}

#[test]
fn transform_where_writes_a_flat_operation_result() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let flags = exec.to_device(&[1_u32, 0, 1]);
    let oa = exec.to_device(&[90_u32, 90, 90]);
    let ob = exec.to_device(&[80_u32, 80, 80]);
    let oc = exec.to_device(&[70_u32, 70, 70]);
    transform_where(
        &exec,
        nested_input!(a, b, c),
        AddOne,
        stencil(flags.slice(..)),
        zip3(oa.slice_mut(..), ob.slice_mut(..), oc.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&oa).unwrap(), vec![2, 90, 4]);
    assert_eq!(exec.to_host(&ob).unwrap(), vec![11, 80, 31]);
    assert_eq!(exec.to_host(&oc).unwrap(), vec![101, 70, 301]);
}

#[test]
fn controls_compose_over_device_resident_empty_and_partial_extents() {
    let exec = exec();
    let len = 1_025usize;
    let a = exec.to_device(&(0..len as u32).rev().collect::<Vec<_>>());
    let b = exec.to_device(&vec![2u32; len]);
    let c = exec.to_device(&vec![3u32; len]);
    for selected_len in [0usize, 1, 31, 32, 33, 255, 256, 257, 1_023, 1_025] {
        let flags = exec.to_device(
            &(0..len)
                .map(|i| if i < selected_len { 7u32 } else { 0u32 })
                .collect::<Vec<_>>(),
        );
        let selected = copy_where(&exec, nested_input!(a, b, c), flags.slice(..)).unwrap();
        let n = selected_len as u32;
        let first_sum = n * (2 * len as u32 - n - 1) / 2;
        assert_eq!(
            reduce(&exec, selected.slice(..), (7, 11, 13), SumTriple).unwrap(),
            (7 + first_sum, 11 + 2 * n, 13 + 3 * n),
        );
        let inclusive = inclusive_scan(&exec, selected.slice(..), SumTriple).unwrap();
        let exclusive = exclusive_scan(&exec, selected.slice(..), (7, 11, 13), SumTriple).unwrap();
        let (ia, ib, ic) = MStorage::into_columns(inclusive);
        let (ea, eb, ec) = MStorage::into_columns(exclusive);
        let mut running = (0u32, 0u32, 0u32);
        let mut expected_inclusive = [Vec::new(), Vec::new(), Vec::new()];
        let mut expected_exclusive = [Vec::new(), Vec::new(), Vec::new()];
        for index in 0..n {
            expected_exclusive[0].push(7 + running.0);
            expected_exclusive[1].push(11 + running.1);
            expected_exclusive[2].push(13 + running.2);
            running = (
                running.0 + len as u32 - index - 1,
                running.1 + 2,
                running.2 + 3,
            );
            expected_inclusive[0].push(running.0);
            expected_inclusive[1].push(running.1);
            expected_inclusive[2].push(running.2);
        }
        for (column, expected) in [ia, ib, ic].iter().zip(expected_inclusive) {
            assert_eq!(exec.to_host(column).unwrap(), expected);
        }
        for (column, expected) in [ea, eb, ec].iter().zip(expected_exclusive) {
            assert_eq!(exec.to_host(column).unwrap(), expected);
        }
        let (keys, _, _) = MStorage::into_columns(selected.clone());
        let sorted = radix_sort_by_key(&exec, keys.slice(..), selected.slice(..)).unwrap();
        let (sa, sb, sc) = MStorage::into_columns(sorted);
        assert_eq!(
            exec.to_host(&sa).unwrap(),
            ((len as u32 - n)..len as u32).collect::<Vec<_>>()
        );
        assert_eq!(exec.to_host(&sb).unwrap(), vec![2u32; selected_len]);
        assert_eq!(exec.to_host(&sc).unwrap(), vec![3u32; selected_len]);
    }
}

struct EqualU32;
struct EvenTriple;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<Triple> for EvenTriple {
    fn apply(value: Triple) -> MFlag {
        massively::flag::from_bool(value.0 % 2u32 == 0u32)
    }
}

fn host_triples(exec: &Executor<WgpuRuntime>, rows: MVec<WgpuRuntime, Triple>) -> Vec<Triple> {
    let (a, b, c) = MStorage::into_columns(rows);
    exec.to_host(&a)
        .unwrap()
        .into_iter()
        .zip(exec.to_host(&b).unwrap())
        .zip(exec.to_host(&c).unwrap())
        .map(|((a, b), c)| (a, b, c))
        .collect()
}

#[test]
fn owned_algorithms_propagate_shared_and_narrowed_device_extents() {
    let exec = exec();
    let a = exec.to_device(&[4u32, 3, 2, 1, 0]);
    let b = exec.to_device(&[40u32, 30, 20, 10, 0]);
    let c = exec.to_device(&[400u32, 300, 200, 100, 0]);
    let keys = exec.to_device(&[1u32, 1, 1, 2, 3]);
    let indices = exec.to_device(&[2u32, 0, 1, 3, 4]);
    let ascending_keys = exec.to_device(&[2u32, 3, 4, 8, 9]);
    for count in [0usize, 3] {
        let flags = exec.to_device(&(0..5).map(|i| u32::from(i < count)).collect::<Vec<_>>());
        let selected = copy_where(&exec, nested_input!(a, b, c), flags.slice(..)).unwrap();
        let selected_indices = copy_where(&exec, indices.slice(..), flags.slice(..)).unwrap();
        assert_eq!(
            massively::vector::max_element(&exec, selected_indices.slice(..), LessU32).unwrap(),
            if count == 0 { None } else { Some(0) },
        );
        let expected_sorted: Vec<_> = [(2, 20, 200), (3, 30, 300), (4, 40, 400)]
            .into_iter()
            .take(count)
            .collect();
        assert_eq!(
            host_triples(&exec, reverse(&exec, selected.slice(..)).unwrap()),
            expected_sorted
        );
        assert_eq!(
            host_triples(
                &exec,
                gather(&exec, nested_input!(a, b, c), selected_indices.slice(..)).unwrap()
            ),
            [(2, 20, 200), (4, 40, 400), (3, 30, 300)]
                .into_iter()
                .take(count)
                .collect::<Vec<_>>()
        );
        for sorted in [
            sort_by_key(&exec, a.slice(..), selected.slice(..), LessU32).unwrap(),
            radix_sort_by_key(&exec, a.slice(..), selected.slice(..)).unwrap(),
        ] {
            assert_eq!(host_triples(&exec, sorted), expected_sorted);
        }
        let sorted = sort(&exec, selected.slice(..), LessTriple).unwrap();
        let expected_merged: Vec<_> = expected_sorted.iter().flat_map(|&row| [row, row]).collect();
        assert_eq!(
            host_triples(
                &exec,
                merge(&exec, sorted.slice(..), sorted.slice(..), LessTriple).unwrap()
            ),
            expected_merged
        );
        assert_eq!(
            host_triples(
                &exec,
                merge_by_key(
                    &exec,
                    ascending_keys.slice(..),
                    sorted.slice(..),
                    ascending_keys.slice(..),
                    sorted.slice(..),
                    LessU32
                )
                .unwrap()
            ),
            expected_merged
        );
        let (partitioned, boundary) = partition(&exec, selected.slice(..), EvenTriple).unwrap();
        assert_eq!(boundary, if count == 0 { 0 } else { 2 });
        assert_eq!(
            host_triples(&exec, partitioned),
            [(4, 40, 400), (2, 20, 200), (3, 30, 300)]
                .into_iter()
                .take(count)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            host_triples(
                &exec,
                inclusive_scan_by_key(
                    &exec,
                    keys.slice(..),
                    selected.slice(..),
                    EqualU32,
                    SumTriple
                )
                .unwrap()
            ),
            [(4, 40, 400), (7, 70, 700), (9, 90, 900)]
                .into_iter()
                .take(count)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            host_triples(
                &exec,
                exclusive_scan_by_key(
                    &exec,
                    keys.slice(..),
                    selected.slice(..),
                    EqualU32,
                    (7, 11, 13),
                    SumTriple
                )
                .unwrap()
            ),
            [(7, 11, 13), (11, 51, 413), (14, 81, 713)]
                .into_iter()
                .take(count)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            host_triples(
                &exec,
                unique_by_key(&exec, keys.slice(..), selected.slice(..), EqualU32).unwrap()
            ),
            [(4, 40, 400)]
                .into_iter()
                .take(count.min(1))
                .collect::<Vec<_>>()
        );
        let (reduced_keys, reduced) = reduce_by_key(
            &exec,
            keys.slice(..),
            selected.slice(..),
            EqualU32,
            (7, 11, 13),
            SumTriple,
        )
        .unwrap();
        assert_eq!(
            exec.to_host(&reduced_keys).unwrap(),
            vec![1u32; count.min(1)]
        );
        assert_eq!(
            host_triples(&exec, reduced),
            [(16, 101, 913)]
                .into_iter()
                .take(count.min(1))
                .collect::<Vec<_>>()
        );
    }
}
