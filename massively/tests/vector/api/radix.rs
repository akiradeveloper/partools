use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Error, Executor, MRadix, MStorage, lazy, op::UnaryOp, vector, zip3, zip7, zip12};
use static_assertions::{assert_impl_all, assert_not_impl_any};

struct CompoundKey;

#[test]
fn radix_keys_are_limited_to_three_columns() {
    assert_impl_all!(u32: MRadix<WgpuRuntime>);
    assert_impl_all!((u32, u32): MRadix<WgpuRuntime>);
    assert_impl_all!((u32, u32, u32): MRadix<WgpuRuntime>);
    assert_not_impl_any!((u32, u32, u32, u32): MRadix<WgpuRuntime>);
}

#[cubecl::cube]
impl UnaryOp<u32> for CompoundKey {
    type Output = (u32, u32, u32);

    fn apply(value: u32) -> Self::Output {
        (value % 3u32, (value / 3u32) % 2u32, value)
    }
}

#[test]
fn radix_sort_by_key_orders_wgpu_integer_scalars_stably() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    macro_rules! check_integer {
        ($keys:expr) => {{
            let keys = $keys;
            let values: Vec<u32> = (0..keys.len() as u32).collect();
            let key_device = exec.to_device(&keys);
            let value_device = exec.to_device(&values);
            let output =
                vector::radix_sort_by_key(&exec, key_device.slice(..), value_device.slice(..))
                    .unwrap();
            let mut order: Vec<_> = (0..keys.len()).collect();
            order.sort_by_key(|&index| keys[index]);
            let expected: Vec<_> = order.into_iter().map(|index| values[index]).collect();
            assert_eq!(exec.to_host(&output).unwrap(), expected);
        }};
    }

    check_integer!([u32::MAX, 0_u32, 1, 1 << 31, 0, u32::MAX]);
    check_integer!([u64::MAX, 0_u64, 1, 1 << 63, 0, u64::MAX]);
    check_integer!([i32::MAX, i32::MIN, -1_i32, 0, i32::MIN, i32::MAX]);
    check_integer!([i64::MAX, i64::MIN, -1_i64, 0, i64::MIN, i64::MAX]);
}

#[test]
fn lazy_flat_compound_keys_preserve_lexicographic_order() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let source = [8_u32, 1, 7, 0, 6, 5, 2, 4, 3];
    let values: Vec<u32> = (0..source.len() as u32).collect();
    let source_device = exec.to_device(&source);
    let value_device = exec.to_device(&values);
    let output = vector::radix_sort_by_key(
        &exec,
        lazy::map(source_device.slice(..), CompoundKey),
        value_device.slice(..),
    )
    .unwrap();
    let mut order: Vec<_> = (0..source.len()).collect();
    order.sort_by_key(|&index| {
        let value = source[index];
        (value % 3, (value / 3) % 2, value)
    });
    let expected: Vec<_> = order.into_iter().map(|index| values[index]).collect();
    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn radix_sort_by_key_uses_float_total_order() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    macro_rules! check_float {
        ($keys:expr) => {{
            let keys = $keys;
            let values: Vec<u32> = (0..keys.len() as u32).collect();
            let key_device = exec.to_device(&keys);
            let value_device = exec.to_device(&values);
            let output =
                vector::radix_sort_by_key(&exec, key_device.slice(..), value_device.slice(..))
                    .unwrap();
            let mut order: Vec<_> = (0..keys.len()).collect();
            order.sort_by(|&lhs, &rhs| keys[lhs].total_cmp(&keys[rhs]));
            let expected: Vec<_> = order.into_iter().map(|index| values[index]).collect();
            assert_eq!(exec.to_host(&output).unwrap(), expected);
        }};
    }

    check_float!([
        f32::from_bits(0x7fc0_0001),
        f32::INFINITY,
        -0.0_f32,
        0.0_f32,
        f32::NEG_INFINITY,
        f32::from_bits(0xffc0_0001),
        1.5_f32,
        -1.5_f32,
        -0.0_f32,
    ]);
    check_float!([
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::INFINITY,
        -0.0_f64,
        0.0_f64,
        f64::NEG_INFINITY,
        f64::from_bits(0xfff8_0000_0000_0001),
        1.5_f64,
        -1.5_f64,
        -0.0_f64,
    ]);
}

#[test]
fn compound_radix_keys_reorder_seven_value_columns_once() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = [1_i32, -1, 1, -1, 1, -1, 1, -1, 1];
    let second = [2_u32, 2, 1, 2, 1, 1, 1, 1, 1];
    let third = [0_u32, 0, 2, 0, 1, 2, 1, 1, 1];
    let first_device = exec.to_device(&first);
    let second_device = exec.to_device(&second);
    let third_device = exec.to_device(&third);
    let value_columns: Vec<Vec<u32>> = (0..7_u32)
        .map(|column| {
            (0..first.len())
                .map(|index| column * 100 + index as u32)
                .collect()
        })
        .collect();
    let value_devices: Vec<_> = value_columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();

    let output = vector::radix_sort_by_key(
        &exec,
        zip3(
            first_device.slice(..),
            second_device.slice(..),
            third_device.slice(..),
        ),
        zip7(
            value_devices[0].slice(..),
            value_devices[1].slice(..),
            value_devices[2].slice(..),
            value_devices[3].slice(..),
            value_devices[4].slice(..),
            value_devices[5].slice(..),
            value_devices[6].slice(..),
        ),
    )
    .unwrap();

    let mut order: Vec<_> = (0..first.len()).collect();
    order.sort_by_key(|&index| (first[index], second[index], third[index]));
    let outputs = MStorage::into_columns(output);
    let outputs = [
        outputs.0, outputs.1, outputs.2, outputs.3, outputs.4, outputs.5, outputs.6,
    ];
    for (column, output) in outputs.into_iter().enumerate() {
        let expected: Vec<_> = order
            .iter()
            .map(|&index| value_columns[column][index])
            .collect();
        assert_eq!(exec.to_host(&output).unwrap(), expected);
    }
}

#[test]
fn three_key_columns_reorder_twelve_value_columns_once() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let rows = [
        [0_u32, 0, 0],
        [0_u32, 0, 1],
        [0_u32, 0, 0],
        [1_u32, 0, 0],
        [0_u32, 1, 0],
        [0_u32, 0, 0],
        [0_u32, 1, 1],
    ];
    let key_columns: Vec<Vec<u32>> = (0..3)
        .map(|column| rows.iter().map(|row| row[column]).collect())
        .collect();
    let value_columns: Vec<Vec<u32>> = (0..12_u32)
        .map(|column| {
            (0..rows.len())
                .map(|index| column * 100 + index as u32)
                .collect()
        })
        .collect();
    let key_devices: Vec<_> = key_columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();
    let value_devices: Vec<_> = value_columns
        .iter()
        .map(|column| exec.to_device(column))
        .collect();

    let output = vector::radix_sort_by_key(
        &exec,
        zip3(
            key_devices[0].slice(..),
            key_devices[1].slice(..),
            key_devices[2].slice(..),
        ),
        zip12(
            value_devices[0].slice(..),
            value_devices[1].slice(..),
            value_devices[2].slice(..),
            value_devices[3].slice(..),
            value_devices[4].slice(..),
            value_devices[5].slice(..),
            value_devices[6].slice(..),
            value_devices[7].slice(..),
            value_devices[8].slice(..),
            value_devices[9].slice(..),
            value_devices[10].slice(..),
            value_devices[11].slice(..),
        ),
    )
    .unwrap();

    let mut order: Vec<_> = (0..rows.len()).collect();
    order.sort_by_key(|&index| rows[index]);
    let outputs = MStorage::into_columns(output);
    let outputs = [
        outputs.0, outputs.1, outputs.2, outputs.3, outputs.4, outputs.5, outputs.6, outputs.7,
        outputs.8, outputs.9, outputs.10, outputs.11,
    ];
    for (column, output) in outputs.into_iter().enumerate() {
        let expected: Vec<_> = order
            .iter()
            .map(|&index| value_columns[column][index])
            .collect();
        assert_eq!(exec.to_host(&output).unwrap(), expected);
    }
}

#[test]
fn radix_sort_by_key_handles_empty_and_rejects_length_mismatch() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty_keys = exec.to_device(&Vec::<u32>::new());
    let empty_values = exec.to_device(&Vec::<u32>::new());
    let output =
        vector::radix_sort_by_key(&exec, empty_keys.slice(..), empty_values.slice(..)).unwrap();
    assert!(exec.to_host(&output).unwrap().is_empty());

    let keys = exec.to_device(&[1_u32, 2]);
    let values = exec.to_device(&[10_u32]);
    let error = match vector::radix_sort_by_key(&exec, keys.slice(..), values.slice(..)) {
        Ok(_) => panic!("mismatched key and value lengths must fail"),
        Err(error) => error,
    };
    assert_eq!(error, Error::LengthMismatch { left: 2, right: 1 },);
}

#[test]
fn radix_sort_by_key_crosses_macro_block_boundaries_stably() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let len = 6_145;
    let keys: Vec<u32> = (0..len).map(|index| ((index * 37) % 11) as u32).collect();
    let values: Vec<u32> = (0..len as u32).collect();
    let key_device = exec.to_device(&keys);
    let value_device = exec.to_device(&values);
    let output =
        vector::radix_sort_by_key(&exec, key_device.slice(..), value_device.slice(..)).unwrap();
    let mut order: Vec<_> = (0..len).collect();
    order.sort_by_key(|&index| keys[index]);
    let expected: Vec<_> = order.into_iter().map(|index| values[index]).collect();
    assert_eq!(exec.to_host(&output).unwrap(), expected);
}
