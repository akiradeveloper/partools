use oracle::vector as reference;

use super::common::*;

macro_rules! scale_test {
    ($name:ident, $body:block) => {
        #[test]
        fn $name() $body
    };
}

#[test]
fn scale_prime_block_dispatch_guard() {
    const LEN: usize = 65_537 * 256;
    let input: Vec<_> = (0..LEN as u32).collect();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::reduce(&exec, lazify(gpu.slice(..)), 0, MaxU32,).unwrap(),
        ),
        (LEN - 1) as u32
    );
    assert_eq!(
        read_optional_index_pair(
            &exec,
            massively::vector::minmax_element(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ),
        Some((0, LEN - 1))
    );
    let output = massively::vector::map(&exec, lazify(gpu.slice(..)), IdentityU32).unwrap();
    let actual = exec.to_host(&output).unwrap();
    assert_eq!(actual[0], 0);
    assert_eq!(actual[LEN - 1], (LEN - 1) as u32);
    let output = massively::vector::inclusive_scan(&exec, lazify(gpu.slice(..)), MaxU32).unwrap();
    let actual = exec.to_host(&output).unwrap();
    assert_eq!(actual[0], 0);
    assert_eq!(actual[LEN - 1], (LEN - 1) as u32);
}

scale_test!(scale_map, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::map(&exec, lazify(gpu.slice(..)), IdentityU32).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), input);
});

scale_test!(scale_map_matches_reference, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::map(&exec, lazify(gpu.slice(..)), IdentityU32).unwrap();
    let expected = reference::map(&input, IdentityU32);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

scale_test!(scale_transform_where, {
    let input = scale_input();
    let flags = flags_for(&input);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&vec![0u32; input.len()]);
    let mut expected = vec![0; input.len()];
    massively::vector::transform_where(
        &exec,
        lazify(gpu.slice(..)),
        IdentityU32,
        as_stencil(lazify(flags_gpu.slice(..))),
        output.slice_mut(..),
    )
    .unwrap();
    reference::transform_where(&input, IdentityU32, &flags, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

scale_test!(scale_reduce, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::reduce(&exec, lazify(gpu.slice(..)), 0, MaxU32,).unwrap(),
        ),
        reference::reduce(&input, 0, MaxU32)
    );
});

scale_test!(scale_inclusive_scan, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::inclusive_scan(&exec, lazify(gpu.slice(..)), MaxU32).unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::inclusive_scan(&input, MaxU32)
    );
});

scale_test!(scale_exclusive_scan, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output =
        massively::vector::exclusive_scan(&exec, lazify(gpu.slice(..)), 0, MaxU32).unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::exclusive_scan(&input, 0, MaxU32)
    );
});

scale_test!(scale_adjacent_difference, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output =
        massively::vector::adjacent_difference(&exec, lazify(gpu.slice(..)), MaxU32).unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::adjacent_difference(&input, MaxU32)
    );
});

scale_test!(scale_copy_where, {
    let input = scale_input();
    let flags = flags_for(&input);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = massively::vector::copy_where(
        &exec,
        lazify(gpu.slice(..)),
        as_stencil(lazify(flags_gpu.slice(..))),
    )
    .unwrap();
    let expected = reference::copy_where(&input, &flags);
    assert_eq!(logical_len(&exec, &output), expected.len());
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

scale_test!(scale_remove_where, {
    let input = scale_input();
    let flags = flags_for(&input);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let flags_gpu = exec.to_device(&flags);
    let output = massively::vector::remove_where(
        &exec,
        lazify(gpu.slice(..)),
        as_stencil(lazify(flags_gpu.slice(..))),
    )
    .unwrap();
    let expected = reference::remove_where(&input, &flags);
    assert_eq!(logical_len(&exec, &output), expected.len());
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

scale_test!(scale_partition, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let (output, boundary) =
        massively::vector::partition(&exec, lazify(gpu.slice(..)), NonZero).unwrap();
    let boundary = read_value(&exec, boundary) as usize;
    let (mut selected, rejected) = reference::partition(&input, NonZero);
    assert_eq!(boundary, selected.len());
    selected.extend(rejected);
    assert_eq!(exec.to_host(&output).unwrap(), selected);
});

scale_test!(scale_count_if, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::count_if(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ) as usize,
        reference::count_if(&input, NonZero)
    );
});
scale_test!(scale_all_of, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::all_of(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ),
        expected_flag(reference::all_of(&input, NonZero))
    );
});
scale_test!(scale_any_of, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::any_of(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ),
        expected_flag(reference::any_of(&input, NonZero))
    );
});
scale_test!(scale_none_of, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::none_of(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ),
        expected_flag(reference::none_of(&input, NonZero))
    );
});
scale_test!(scale_find_if, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::find_if(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ),
        reference::find_if(&input, NonZero)
    );
});
scale_test!(scale_is_partitioned, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::is_partitioned(&exec, lazify(gpu.slice(..)), NonZero).unwrap(),
        ),
        expected_flag(reference::is_partitioned(&input, NonZero))
    );
});

scale_test!(scale_gather, {
    let input = scale_input();
    let indices = indices_for(input.len());
    let exec = exec();
    let gpu = exec.to_device(&input);
    let indices_gpu = exec.to_device(&indices);
    let mut expected = vec![0; input.len()];
    let output = massively::vector::gather(
        &exec,
        lazify(gpu.slice(..)),
        as_indices(lazify(indices_gpu.slice(..))),
    )
    .unwrap();
    reference::gather(&input, &indices, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});
scale_test!(scale_gather_where, {
    let input = scale_input();
    let indices = indices_for(input.len());
    let flags = flags_for(&input);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let indices_gpu = exec.to_device(&indices);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    let mut expected = input.clone();
    massively::vector::gather_where(
        &exec,
        lazify(gpu.slice(..)),
        as_indices(lazify(indices_gpu.slice(..))),
        as_stencil(lazify(flags_gpu.slice(..))),
        output.slice_mut(..),
    )
    .unwrap();
    reference::gather_where(&input, &indices, &flags, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});
scale_test!(scale_scatter, {
    let input = scale_input();
    let indices = indices_for(input.len());
    let exec = exec();
    let gpu = exec.to_device(&input);
    let indices_gpu = exec.to_device(&indices);
    let output = exec.to_device(&vec![0u32; input.len()]);
    let mut expected = vec![0; input.len()];
    massively::vector::scatter(
        &exec,
        lazify(gpu.slice(..)),
        as_indices(lazify(indices_gpu.slice(..))),
        output.slice_mut(..),
    )
    .unwrap();
    reference::scatter(&input, &indices, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});
scale_test!(scale_scatter_where, {
    let input = scale_input();
    let indices = indices_for(input.len());
    let flags = flags_for(&input);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let indices_gpu = exec.to_device(&indices);
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    let mut expected = input.clone();
    massively::vector::scatter_where(
        &exec,
        lazify(gpu.slice(..)),
        as_indices(lazify(indices_gpu.slice(..))),
        as_stencil(lazify(flags_gpu.slice(..))),
        output.slice_mut(..),
    )
    .unwrap();
    reference::scatter_where(&input, &indices, &flags, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

scale_test!(scale_equal, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::equal(
                &exec,
                lazify(gpu.slice(..)),
                lazify(gpu.slice(..)),
                EqualU32
            )
            .unwrap(),
        ),
        expected_flag(reference::equal(&input, &input, EqualU32))
    );
});
scale_test!(scale_mismatch, {
    let input = scale_input();
    let other = scale_other();
    let exec = exec();
    let left = exec.to_device(&input);
    let right = exec.to_device(&other);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::mismatch(
                &exec,
                lazify(left.slice(..)),
                lazify(right.slice(..)),
                EqualU32
            )
            .unwrap(),
        ),
        reference::mismatch(&input, &other, EqualU32)
    );
});
scale_test!(scale_adjacent_find, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::adjacent_find(&exec, lazify(gpu.slice(..)), EqualU32).unwrap(),
        ),
        reference::adjacent_find(&input, EqualU32)
    );
});
scale_test!(scale_find_first_of, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::find_first_of(
                &exec,
                lazify(gpu.slice(..)),
                lazify(gpu.slice(..)),
                EqualU32
            )
            .unwrap(),
        ),
        reference::find_first_of(&input, &input, EqualU32)
    );
});
scale_test!(scale_fill, {
    let input = scale_input();
    let exec = exec();
    let mut expected = input.clone();
    let output = exec.alloc::<u32>(input.len().try_into().unwrap());
    let value = 123_u32;
    massively::vector::fill(&exec, value, output.slice_mut(..)).unwrap();
    reference::fill(123, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});
scale_test!(scale_replace_where, {
    let input = scale_input();
    let flags = flags_for(&input);
    let exec = exec();
    let flags_gpu = exec.to_device(&flags);
    let output = exec.to_device(&input);
    let mut expected = input.clone();
    let value = 123_u32;
    massively::vector::replace_where(
        &exec,
        value,
        as_stencil(lazify(flags_gpu.slice(..))),
        output.slice_mut(..),
    )
    .unwrap();
    reference::replace_where(123, &flags, &mut expected);
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});
scale_test!(scale_reverse, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::reverse(&exec, lazify(gpu.slice(..))).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), reference::reverse(&input));
});

scale_test!(scale_sort, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::sort(&exec, lazify(gpu.slice(..)), LessU32).unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::sort(&input, LessU32)
    );
});
scale_test!(scale_merge, {
    let left = reference::sort(&scale_input(), LessU32);
    let right = reference::sort(&scale_other(), LessU32);
    let exec = exec();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    let output = massively::vector::merge(
        &exec,
        lazify(left_gpu.slice(..)),
        lazify(right_gpu.slice(..)),
        LessU32,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::merge(&left, &right, LessU32)
    );
});
scale_test!(scale_is_sorted, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::is_sorted(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ),
        expected_flag(reference::is_sorted(&input, LessU32))
    );
});
scale_test!(scale_is_sorted_until, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::is_sorted_until(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ) as usize,
        reference::is_sorted_until(&input, LessU32)
    );
});
scale_test!(scale_lexicographical_compare, {
    let left = scale_input();
    let right = scale_other();
    let exec = exec();
    let left_gpu = exec.to_device(&left);
    let right_gpu = exec.to_device(&right);
    assert_eq!(
        read_value(
            &exec,
            massively::vector::lexicographical_compare(
                &exec,
                lazify(left_gpu.slice(..)),
                lazify(right_gpu.slice(..)),
                LessU32
            )
            .unwrap(),
        ),
        expected_flag(reference::lexicographical_compare(&left, &right, LessU32))
    );
});
scale_test!(scale_min_element, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::min_element(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ),
        reference::min_element(&input, LessU32)
    );
});
scale_test!(scale_max_element, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index(
            &exec,
            massively::vector::max_element(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ),
        reference::max_element(&input, LessU32)
    );
});
scale_test!(scale_minmax_element, {
    let input = scale_input();
    let exec = exec();
    let gpu = exec.to_device(&input);
    assert_eq!(
        read_optional_index_pair(
            &exec,
            massively::vector::minmax_element(&exec, lazify(gpu.slice(..)), LessU32).unwrap(),
        ),
        reference::minmax_element(&input, LessU32)
    );
});
scale_test!(scale_lower_bound, {
    let source = reference::sort(&scale_input(), LessU32);
    let values = scale_other();
    let exec = exec();
    let source_gpu = exec.to_device(&source);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::lower_bound(
        &exec,
        lazify(source_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        LessU32,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::lower_bound(&source, &values, LessU32)
    );
});
scale_test!(scale_upper_bound, {
    let source = reference::sort(&scale_input(), LessU32);
    let values = scale_other();
    let exec = exec();
    let source_gpu = exec.to_device(&source);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::upper_bound(
        &exec,
        lazify(source_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        LessU32,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::upper_bound(&source, &values, LessU32)
    );
});
scale_test!(scale_unique, {
    let input = reference::sort(&scale_input(), LessU32);
    let exec = exec();
    let gpu = exec.to_device(&input);
    let output = massively::vector::unique(&exec, lazify(gpu.slice(..)), EqualU32).unwrap();
    let expected = reference::unique(&input, EqualU32);
    assert_eq!(logical_len(&exec, &output), expected.len());
    assert_eq!(exec.to_host(&output).unwrap(), expected);
});

macro_rules! scale_set_test {
    ($name:ident, $algorithm:ident, $oracle:ident) => {
        scale_test!($name, {
            let left = reference::sort(&scale_input(), LessU32);
            let right = reference::sort(&scale_other(), LessU32);
            let exec = exec();
            let left_gpu = exec.to_device(&left);
            let right_gpu = exec.to_device(&right);
            let output = massively::vector::$algorithm(
                &exec,
                lazify(left_gpu.slice(..)),
                lazify(right_gpu.slice(..)),
                LessU32,
            )
            .unwrap();
            let expected = reference::$oracle(&left, &right, LessU32);
            assert_eq!(logical_len(&exec, &output), expected.len());
            assert_eq!(exec.to_host(&output).unwrap(), expected);
        });
    };
}
scale_set_test!(scale_set_union, set_union, set_union);
scale_set_test!(scale_set_intersection, set_intersection, set_intersection);
scale_set_test!(scale_set_difference, set_difference, set_difference);

scale_test!(scale_sort_by_key, {
    let keys = scale_input();
    let values = scale_other();
    let exec = exec();
    let keys_gpu = exec.to_device(&keys);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::sort_by_key(
        &exec,
        lazify(keys_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        LessU32,
    )
    .unwrap();
    let (_, expected_values) = reference::sort_by_key(&keys, &values, LessU32);
    assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});
scale_test!(scale_inclusive_scan_by_key, {
    let keys: Vec<_> = scale_input().into_iter().map(|v| v % 1024).collect();
    let values = scale_other();
    let exec = exec();
    let keys_gpu = exec.to_device(&keys);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::inclusive_scan_by_key(
        &exec,
        lazify(keys_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        EqualU32,
        MaxU32,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::inclusive_scan_by_key(&keys, &values, EqualU32, MaxU32)
    );
});
scale_test!(scale_exclusive_scan_by_key, {
    let keys: Vec<_> = scale_input().into_iter().map(|v| v % 1024).collect();
    let values = scale_other();
    let exec = exec();
    let keys_gpu = exec.to_device(&keys);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::exclusive_scan_by_key(
        &exec,
        lazify(keys_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        EqualU32,
        0,
        MaxU32,
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&output).unwrap(),
        reference::exclusive_scan_by_key(&keys, &values, EqualU32, 0, MaxU32)
    );
});
scale_test!(scale_reduce_by_key, {
    let keys: Vec<_> = scale_input().into_iter().map(|v| v % 1024).collect();
    let values = scale_other();
    let exec = exec();
    let keys_gpu = exec.to_device(&keys);
    let values_gpu = exec.to_device(&values);
    let (out_keys, out_values) = massively::vector::reduce_by_key(
        &exec,
        lazify(keys_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        EqualU32,
        0,
        MaxU32,
    )
    .unwrap();
    let (expected_keys, expected_values) =
        reference::reduce_by_key(&keys, &values, EqualU32, 0, MaxU32);
    assert_eq!(logical_len(&exec, &out_keys), expected_keys.len());
    assert_eq!(exec.to_host(&out_keys).unwrap(), expected_keys);
    assert_eq!(exec.to_host(&out_values).unwrap(), expected_values);
});
scale_test!(scale_unique_by_key, {
    let keys: Vec<_> = scale_input().into_iter().map(|v| v % 1024).collect();
    let values = scale_other();
    let exec = exec();
    let keys_gpu = exec.to_device(&keys);
    let values_gpu = exec.to_device(&values);
    let output = massively::vector::unique_by_key(
        &exec,
        lazify(keys_gpu.slice(..)),
        lazify(values_gpu.slice(..)),
        EqualU32,
    )
    .unwrap();
    let (_, expected_values) = reference::unique_by_key(&keys, &values, EqualU32);
    assert_eq!(logical_len(&exec, &output), expected_values.len());
    assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});
scale_test!(scale_merge_by_key, {
    let keys = scale_input();
    let values = scale_other();
    let split = keys.len() / 2;
    let (left_keys, left_values) =
        reference::sort_by_key(&keys[..split], &values[..split], LessU32);
    let (right_keys, right_values) =
        reference::sort_by_key(&keys[split..], &values[split..], LessU32);
    let exec = exec();
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
        LessU32,
    )
    .unwrap();
    let (_, expected_values) = reference::merge_by_key(
        &left_keys,
        &left_values,
        &right_keys,
        &right_values,
        LessU32,
    );
    assert_eq!(exec.to_host(&output).unwrap(), expected_values);
});
