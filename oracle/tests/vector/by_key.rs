use oracle::vector as reference;
use proptest::prelude::*;

use super::common::*;

macro_rules! by_key_case {
    ($name:ident, |$exec:ident, $keys:ident, $values:ident, $key_gpu:ident, $value_gpu:ident| $body:block) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(
                pairs in oracle_vec((0_u32..12, 0_u32..50)),
            ) {
                let (raw_keys, raw_values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
                let $keys = raw_keys.as_slice();
                let $values = raw_values.as_slice();
                let $exec = exec();
                let $key_gpu = $exec.to_device($keys);
                let $value_gpu = $exec.to_device($values);
                $body
            }
        }
    };
}

by_key_case!(
    inclusive_scan_by_key,
    |exec, keys, values, key_gpu, value_gpu| {
        let output = massively::vector::inclusive_scan_by_key(
            &exec,
            lazify(key_gpu.slice(..)),
            lazify(value_gpu.slice(..)),
            Equal,
            Sum,
        )
        .unwrap();
        prop_assert_eq!(
            exec.to_host(&output).unwrap(),
            reference::inclusive_scan_by_key(keys, values, Equal, Sum),
        );
    }
);

by_key_case!(
    exclusive_scan_by_key,
    |exec, keys, values, key_gpu, value_gpu| {
        let output = massively::vector::exclusive_scan_by_key(
            &exec,
            lazify(key_gpu.slice(..)),
            lazify(value_gpu.slice(..)),
            Equal,
            7,
            Sum,
        )
        .unwrap();
        prop_assert_eq!(
            exec.to_host(&output).unwrap(),
            reference::exclusive_scan_by_key(keys, values, Equal, 7, Sum),
        );
    }
);

by_key_case!(reduce_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let (out_keys, out_values) = massively::vector::reduce_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Equal,
        7,
        Sum,
    )
    .unwrap();
    let (expected_keys, expected_values) = reference::reduce_by_key(keys, values, Equal, 7, Sum);
    prop_assert_eq!(exec.to_host(&out_keys).unwrap(), expected_keys);
    prop_assert_eq!(exec.to_host(&out_values).unwrap(), expected_values);
});

by_key_case!(unique_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let output = massively::vector::unique_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Equal,
    )
    .unwrap();
    let (_, expected_values) = reference::unique_by_key(keys, values, Equal);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});

by_key_case!(sort_by_key, |exec, keys, values, key_gpu, value_gpu| {
    let output = massively::vector::sort_by_key(
        &exec,
        lazify(key_gpu.slice(..)),
        lazify(value_gpu.slice(..)),
        Less,
    )
    .unwrap();
    let (_, expected_values) = reference::sort_by_key(keys, values, Less);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});

by_key_case!(
    radix_sort_by_key,
    |exec, keys, values, key_gpu, value_gpu| {
        let output = massively::vector::radix_sort_by_key(
            &exec,
            lazify(key_gpu.slice(..)),
            lazify(value_gpu.slice(..)),
        )
        .unwrap();
        let mut order: Vec<_> = (0..keys.len()).collect();
        order.sort_by_key(|&index| keys[index]);
        let expected_values: Vec<_> = order.into_iter().map(|index| values[index]).collect();
        prop_assert_eq!(exec.to_host(&output).unwrap(), expected_values);
    }
);

by_key_case!(merge_by_key, |exec, keys, values, _key_gpu, _value_gpu| {
    let split = keys.len() / 2;
    let (left_keys, left_values) = reference::sort_by_key(&keys[..split], &values[..split], Less);
    let (right_keys, right_values) = reference::sort_by_key(&keys[split..], &values[split..], Less);
    let left_keys_gpu = exec.to_device(&left_keys);
    let left_values_gpu = exec.to_device(&left_values);
    let right_keys_gpu = exec.to_device(&right_keys);
    let right_values_gpu = exec.to_device(&right_values);
    let output = massively::vector::merge_by_key(
        &exec,
        lazify(left_keys_gpu.slice(..)),
        lazify(left_values_gpu.slice(..)),
        lazify(right_keys_gpu.slice(..)),
        lazify(right_values_gpu.slice(..)),
        Less,
    )
    .unwrap();
    let (_, expected_values) =
        reference::merge_by_key(&left_keys, &left_values, &right_keys, &right_values, Less);
    prop_assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});
