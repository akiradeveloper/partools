use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MFlag, MIter, MIterMut, MStorage, lazy, op::Identity, op::UnaryOp, vector::gather,
    vector::map, zip2, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
};

fn transform_where_into<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), massively::Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = MFlag>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item, Output = Output::Item>,
{
    massively::vector::transform_where(exec, input, op, stencil, output)
}

struct AddThree;
struct IdentityTriple;
struct SumFour;
struct AddPair;
struct EncodeFlagIndex;

#[test]
fn custom_preallocated_functions_do_not_need_allocation_bound() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]);
    let output = exec.to_device(&[0_u32; 3]);

    transform_where_into(
        &exec,
        input.slice(..),
        Identity,
        lazy::constant(1_u32).take(3),
        output.slice_mut(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for AddThree {
    type Output = u32;

    fn apply(input: (u32, u32, u32)) -> u32 {
        input.0 + input.1 + input.2
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32)> for IdentityTriple {
    type Output = (u32, u32, u32);

    fn apply(input: (u32, u32, u32)) -> (u32, u32, u32) {
        input
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32, u32, u32)> for SumFour {
    type Output = u32;

    fn apply(input: (u32, u32, u32, u32)) -> u32 {
        input.0 + input.1 + input.2 + input.3
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

#[cubecl::cube]
impl UnaryOp<(MFlag, massively::MIndex)> for EncodeFlagIndex {
    type Output = u32;

    fn apply(input: (MFlag, massively::MIndex)) -> u32 {
        if massively::flag::is_set(input.0) {
            input.1
        } else {
            0u32
        }
    }
}

#[test]
fn zip_flattens_flag_scalars() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = zip2(lazy::constant(1_u32).take(3), lazy::counting(4).take(3));
    let output = map(&exec, input, EncodeFlagIndex).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![4, 5, 6]);
}

#[test]
fn zip_helpers_expose_flat_public_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let columns: Vec<_> = (0_u32..12)
        .map(|base| exec.to_device(&[base + 1, base + 2]))
        .collect();

    fn assert_iter<R: Runtime, Input: MIter<R>>(_input: &Input) {}
    macro_rules! assert_zip {
        ($zip:ident; $($index:expr),+ $(,)?) => {{
            let input = $zip($(columns[$index].slice(..)),+);
            assert_iter::<WgpuRuntime, _>(&input);
        }};
    }

    assert_zip!(zip2; 0, 1);
    assert_zip!(zip3; 0, 1, 2);
    assert_zip!(zip4; 0, 1, 2, 3);
    assert_zip!(zip5; 0, 1, 2, 3, 4);
    assert_zip!(zip6; 0, 1, 2, 3, 4, 5);
    assert_zip!(zip7; 0, 1, 2, 3, 4, 5, 6);
    assert_zip!(zip8; 0, 1, 2, 3, 4, 5, 6, 7);
    assert_zip!(zip9; 0, 1, 2, 3, 4, 5, 6, 7, 8);
    assert_zip!(zip10; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    assert_zip!(zip11; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    assert_zip!(zip12; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

    let output = map(
        &exec,
        zip3(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
        ),
        AddThree,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![6, 9]);
}

#[test]
fn zip_grouping_does_not_change_the_logical_row() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2]);
    let b = exec.to_device(&[3_u32, 4]);
    let c = exec.to_device(&[5_u32, 6]);

    let left_grouped = map(
        &exec,
        zip2(zip2(a.slice(..), b.slice(..)), c.slice(..)),
        IdentityTriple,
    )
    .unwrap();
    let right_grouped = map(
        &exec,
        zip2(a.slice(..), zip2(b.slice(..), c.slice(..))),
        IdentityTriple,
    )
    .unwrap();

    let (left_a, left_b, left_c) = MStorage::into_columns(left_grouped);
    let (right_a, right_b, right_c) = MStorage::into_columns(right_grouped);
    assert_eq!(exec.to_host(&left_a).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&left_b).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&left_c).unwrap(), vec![5, 6]);
    assert_eq!(exec.to_host(&right_a).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&right_b).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&right_c).unwrap(), vec![5, 6]);
}

#[test]
fn read_slice_adapters_compose_on_binary_zip_trees() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);

    let sliced = zip3(a.slice(..), b.slice(..), c.slice(..))
        .slice(1..4)
        .slice(1..2);
    let output = map(&exec, sliced, AddThree).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![333]);
}

#[test]
fn mutable_slice_adapters_compose_and_can_be_read_back() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let out_a = exec.to_device(&[0_u32; 5]);
    let out_b = exec.to_device(&[0_u32; 5]);
    let output = zip2(out_a.slice_mut(..), out_b.slice_mut(..));

    massively::vector::replace_where(
        &exec,
        (7_u32, 9_u32),
        lazy::constant(1_u32).take(1),
        output.slice_mut(1..4).slice_mut(1..2),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0, 0, 7, 0, 0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 0, 9, 0, 0]);

    let read = output.slice(1..4).slice(1..2);
    let copy = map(&exec, read, Identity).unwrap();
    let (first, second) = MStorage::into_columns(copy);
    assert_eq!(exec.to_host(&first).unwrap(), vec![7]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![9]);
}

#[test]
fn gather_keeps_an_eight_slot_value_expression_lazy() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let columns: Vec<_> = (0_u32..8)
        .map(|column| exec.to_device(&[column, column + 10, column + 20]))
        .collect();
    let indices = exec.to_device(&[2_u32, 0]);
    let left = lazy::map(
        zip4(
            columns[0].slice(..),
            columns[1].slice(..),
            columns[2].slice(..),
            columns[3].slice(..),
        ),
        SumFour,
    );
    let right = lazy::map(
        zip4(
            columns[4].slice(..),
            columns[5].slice(..),
            columns[6].slice(..),
            columns[7].slice(..),
        ),
        SumFour,
    );
    let values = lazy::map(zip2(left, right), AddPair);
    let output = gather(&exec, values, indices.slice(..)).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![188, 28]);
}
