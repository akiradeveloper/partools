//! CPU references for vector algorithms.
//!
//! These implementations use AoS values (`Vec<(...)>`) while `massively` uses
//! Zip device columns.

use std::cmp::Ordering;

use crate::op;

pub fn map<T, Op>(source: &[T], _op: Op) -> Vec<Op::Output>
where
    T: Copy,
    Op: op::UnaryOp<T>,
{
    let mut output = Vec::with_capacity(source.len());
    output.extend(source.iter().copied().map(Op::apply));
    output
}

pub fn transform_where<T, Op>(source: &[T], _op: Op, stencil: &[u32], output: &mut [Op::Output])
where
    T: Copy,
    Op: op::UnaryOp<T>,
    Op::Output: Copy,
{
    for i in 0..source.len() {
        if stencil[i] != 0 {
            output[i] = Op::apply(source[i]);
        }
    }
}

pub fn reduce<T, Op>(source: &[T], init: T, _op: Op) -> T
where
    T: Copy,
    Op: op::ReductionOp<T>,
{
    source
        .iter()
        .copied()
        .fold(init, |acc, value| Op::apply(acc, value))
}

pub fn inclusive_scan<T, Op>(source: &[T], _op: Op) -> Vec<T>
where
    T: Copy,
    Op: op::ReductionOp<T>,
{
    let mut out = Vec::with_capacity(source.len());
    let mut acc = None;
    for value in source.iter().copied() {
        let next = match acc {
            Some(prev) => Op::apply(prev, value),
            None => value,
        };
        out.push(next);
        acc = Some(next);
    }
    out
}

pub fn exclusive_scan<T, Op>(source: &[T], init: T, _op: Op) -> Vec<T>
where
    T: Copy,
    Op: op::ReductionOp<T>,
{
    let mut out = Vec::with_capacity(source.len());
    let mut acc = init;
    for value in source.iter().copied() {
        out.push(acc);
        acc = Op::apply(acc, value);
    }
    out
}

pub fn adjacent_difference<T, Op>(source: &[T], _op: Op) -> Vec<T>
where
    T: Copy,
    Op: op::ReductionOp<T>,
{
    if source.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(source.len());
    out.push(source[0]);
    for i in 1..source.len() {
        out.push(Op::apply(source[i - 1], source[i]));
    }
    out
}

pub fn inclusive_scan_by_key<K, V, KeyEq, Op>(
    keys: &[K],
    values: &[V],
    _key_eq: KeyEq,
    _op: Op,
) -> Vec<V>
where
    K: Copy,
    V: Copy,
    KeyEq: op::BinaryPredicateOp<K>,
    Op: op::ReductionOp<V>,
{
    let mut out = Vec::with_capacity(values.len());
    let mut acc = None;
    for i in 0..values.len() {
        let starts_segment = i == 0 || !KeyEq::apply(keys[i - 1], keys[i]);
        let next = if starts_segment {
            values[i]
        } else {
            Op::apply(acc.expect("segment accumulator"), values[i])
        };
        out.push(next);
        acc = Some(next);
    }
    out
}

pub fn exclusive_scan_by_key<K, V, KeyEq, Op>(
    keys: &[K],
    values: &[V],
    _key_eq: KeyEq,
    init: V,
    _op: Op,
) -> Vec<V>
where
    K: Copy,
    V: Copy,
    KeyEq: op::BinaryPredicateOp<K>,
    Op: op::ReductionOp<V>,
{
    let mut out = Vec::with_capacity(values.len());
    let mut acc = init;
    for i in 0..values.len() {
        if i == 0 || !KeyEq::apply(keys[i - 1], keys[i]) {
            acc = init;
        }
        out.push(acc);
        acc = Op::apply(acc, values[i]);
    }
    out
}

pub fn reduce_by_key<K, V, KeyEq, Op>(
    keys: &[K],
    values: &[V],
    _key_eq: KeyEq,
    init: V,
    _op: Op,
) -> (Vec<K>, Vec<V>)
where
    K: Copy,
    V: Copy,
    KeyEq: op::BinaryPredicateOp<K>,
    Op: op::ReductionOp<V>,
{
    let mut out_keys = Vec::new();
    let mut out_values = Vec::new();
    let mut i = 0;
    while i < values.len() {
        let key = keys[i];
        let mut acc = init;
        while i < values.len() && (i == 0 || KeyEq::apply(key, keys[i])) {
            acc = Op::apply(acc, values[i]);
            i += 1;
            if i < values.len() && !KeyEq::apply(key, keys[i]) {
                break;
            }
        }
        out_keys.push(key);
        out_values.push(acc);
    }
    (out_keys, out_values)
}

pub fn count_if<T, Pred>(source: &[T], _pred: Pred) -> usize
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    source
        .iter()
        .copied()
        .filter(|value| Pred::apply(*value))
        .count()
}

pub fn all_of<T, Pred>(source: &[T], _pred: Pred) -> bool
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    source.iter().copied().all(Pred::apply)
}

pub fn any_of<T, Pred>(source: &[T], _pred: Pred) -> bool
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    source.iter().copied().any(Pred::apply)
}

pub fn none_of<T, Pred>(source: &[T], _pred: Pred) -> bool
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    !any_of(source, _pred)
}

pub fn find_if<T, Pred>(source: &[T], _pred: Pred) -> Option<usize>
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    source.iter().copied().position(Pred::apply)
}

pub fn partition<T, Pred>(source: &[T], _pred: Pred) -> (Vec<T>, Vec<T>)
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    let mut matching = Vec::new();
    let mut failing = Vec::new();
    for value in source.iter().copied() {
        if Pred::apply(value) {
            matching.push(value);
        } else {
            failing.push(value);
        }
    }
    (matching, failing)
}

pub fn is_partitioned<T, Pred>(source: &[T], _pred: Pred) -> bool
where
    T: Copy,
    Pred: op::PredicateOp<T>,
{
    let mut seen_false = false;
    for value in source.iter().copied() {
        if Pred::apply(value) {
            if seen_false {
                return false;
            }
        } else {
            seen_false = true;
        }
    }
    true
}

pub fn copy_where<T: Copy>(source: &[T], stencil: &[u32]) -> Vec<T> {
    source
        .iter()
        .copied()
        .zip(stencil.iter().copied())
        .filter_map(|(value, flag)| (flag != 0).then_some(value))
        .collect()
}

pub fn remove_where<T: Copy>(source: &[T], stencil: &[u32]) -> Vec<T> {
    source
        .iter()
        .copied()
        .zip(stencil.iter().copied())
        .filter_map(|(value, flag)| (flag == 0).then_some(value))
        .collect()
}

pub fn fill<T: Copy>(value: T, output: &mut [T]) {
    output.fill(value);
}

pub fn replace_where<T: Copy>(value: T, stencil: &[u32], output: &mut [T]) {
    for i in 0..output.len() {
        if stencil[i] != 0 {
            output[i] = value;
        }
    }
}

pub fn gather<T: Copy>(source: &[T], indices: &[u32], output: &mut [T]) {
    for i in 0..indices.len() {
        output[i] = source[indices[i] as usize];
    }
}

pub fn gather_where<T: Copy>(source: &[T], indices: &[u32], stencil: &[u32], output: &mut [T]) {
    for i in 0..indices.len() {
        if stencil[i] != 0 {
            output[i] = source[indices[i] as usize];
        }
    }
}

pub fn scatter<T: Copy>(source: &[T], indices: &[u32], output: &mut [T]) {
    for i in 0..source.len() {
        output[indices[i] as usize] = source[i];
    }
}

pub fn scatter_where<T: Copy>(source: &[T], indices: &[u32], stencil: &[u32], output: &mut [T]) {
    for i in 0..source.len() {
        if stencil[i] != 0 {
            output[indices[i] as usize] = source[i];
        }
    }
}

pub fn reverse<T: Copy>(source: &[T]) -> Vec<T> {
    source.iter().copied().rev().collect()
}

pub fn equal<T, Eq>(left: &[T], right: &[T], _eq: Eq) -> bool
where
    T: Copy,
    Eq: op::BinaryPredicateOp<T>,
{
    left.len() == right.len()
        && left
            .iter()
            .copied()
            .zip(right.iter().copied())
            .all(|(lhs, rhs)| Eq::apply(lhs, rhs))
}

pub fn mismatch<T, Eq>(left: &[T], right: &[T], _eq: Eq) -> Option<usize>
where
    T: Copy,
    Eq: op::BinaryPredicateOp<T>,
{
    let len = left.len().min(right.len());
    for i in 0..len {
        if !Eq::apply(left[i], right[i]) {
            return Some(i);
        }
    }
    (left.len() != right.len()).then_some(len)
}

pub fn adjacent_find<T, Eq>(source: &[T], _eq: Eq) -> Option<usize>
where
    T: Copy,
    Eq: op::BinaryPredicateOp<T>,
{
    source
        .windows(2)
        .position(|pair| Eq::apply(pair[0], pair[1]))
}

pub fn find_first_of<T, Eq>(source: &[T], needles: &[T], _eq: Eq) -> Option<usize>
where
    T: Copy,
    Eq: op::BinaryPredicateOp<T>,
{
    source.iter().copied().position(|value| {
        needles
            .iter()
            .copied()
            .any(|needle| Eq::apply(value, needle))
    })
}

pub fn is_sorted_until<T, Less>(source: &[T], _less: Less) -> usize
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    for i in 1..source.len() {
        if Less::apply(source[i], source[i - 1]) {
            return i;
        }
    }
    source.len()
}

pub fn is_sorted<T, Less>(source: &[T], less: Less) -> bool
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    is_sorted_until(source, less) == source.len()
}

pub fn lexicographical_compare<T, Less>(left: &[T], right: &[T], _less: Less) -> bool
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    for i in 0..left.len().min(right.len()) {
        if Less::apply(left[i], right[i]) {
            return true;
        }
        if Less::apply(right[i], left[i]) {
            return false;
        }
    }
    left.len() < right.len()
}

pub fn lower_bound<T, Less>(source: &[T], values: &[T], _less: Less) -> Vec<u32>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    values
        .iter()
        .copied()
        .map(|value| source.partition_point(|candidate| Less::apply(*candidate, value)) as u32)
        .collect()
}

pub fn upper_bound<T, Less>(source: &[T], values: &[T], _less: Less) -> Vec<u32>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    values
        .iter()
        .copied()
        .map(|value| source.partition_point(|candidate| !Less::apply(value, *candidate)) as u32)
        .collect()
}

pub fn min_element<T, Less>(source: &[T], _less: Less) -> Option<usize>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    if source.is_empty() {
        return None;
    }
    let mut best = 0;
    for i in 1..source.len() {
        if Less::apply(source[i], source[best]) {
            best = i;
        }
    }
    Some(best)
}

pub fn max_element<T, Less>(source: &[T], _less: Less) -> Option<usize>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    if source.is_empty() {
        return None;
    }
    let mut best = 0;
    for i in 1..source.len() {
        if Less::apply(source[best], source[i]) {
            best = i;
        }
    }
    Some(best)
}

pub fn minmax_element<T, Less>(source: &[T], _less: Less) -> Option<(usize, usize)>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    if source.is_empty() {
        return None;
    }
    let mut min = 0;
    let mut max = 0;
    for i in 1..source.len() {
        if !Less::apply(source[min], source[i]) {
            min = i;
        }
        if Less::apply(source[max], source[i]) {
            max = i;
        }
    }
    Some((min, max))
}

pub fn sort<T, Less>(source: &[T], _less: Less) -> Vec<T>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    let mut out = source.to_vec();
    sort_by_less(&mut out, |lhs, rhs| Less::apply(lhs, rhs));
    out
}

pub fn merge<T, Less>(left: &[T], right: &[T], _less: Less) -> Vec<T>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut l = 0;
    let mut r = 0;
    while l < left.len() && r < right.len() {
        if Less::apply(right[r], left[l]) {
            out.push(right[r]);
            r += 1;
        } else {
            out.push(left[l]);
            l += 1;
        }
    }
    out.extend_from_slice(&left[l..]);
    out.extend_from_slice(&right[r..]);
    out
}

pub fn sort_by_key<K, V, Less>(keys: &[K], values: &[V], _less: Less) -> (Vec<K>, Vec<V>)
where
    K: Copy,
    V: Copy,
    Less: op::BinaryPredicateOp<K>,
{
    let mut pairs: Vec<_> = keys.iter().copied().zip(values.iter().copied()).collect();
    sort_by_less(&mut pairs, |lhs, rhs| Less::apply(lhs.0, rhs.0));
    pairs.into_iter().unzip()
}

pub fn merge_by_key<K, V, Less>(
    left_keys: &[K],
    left_values: &[V],
    right_keys: &[K],
    right_values: &[V],
    _less: Less,
) -> (Vec<K>, Vec<V>)
where
    K: Copy,
    V: Copy,
    Less: op::BinaryPredicateOp<K>,
{
    let mut out_keys = Vec::with_capacity(left_keys.len() + right_keys.len());
    let mut out_values = Vec::with_capacity(left_values.len() + right_values.len());
    let mut l = 0;
    let mut r = 0;
    while l < left_keys.len() && r < right_keys.len() {
        if Less::apply(right_keys[r], left_keys[l]) {
            out_keys.push(right_keys[r]);
            out_values.push(right_values[r]);
            r += 1;
        } else {
            out_keys.push(left_keys[l]);
            out_values.push(left_values[l]);
            l += 1;
        }
    }
    out_keys.extend_from_slice(&left_keys[l..]);
    out_values.extend_from_slice(&left_values[l..]);
    out_keys.extend_from_slice(&right_keys[r..]);
    out_values.extend_from_slice(&right_values[r..]);
    (out_keys, out_values)
}

pub fn unique<T, Eq>(source: &[T], _eq: Eq) -> Vec<T>
where
    T: Copy,
    Eq: op::BinaryPredicateOp<T>,
{
    let mut out = Vec::new();
    for value in source.iter().copied() {
        if out
            .last()
            .copied()
            .is_none_or(|prev| !Eq::apply(prev, value))
        {
            out.push(value);
        }
    }
    out
}

pub fn unique_by_key<K, V, Eq>(keys: &[K], values: &[V], _eq: Eq) -> (Vec<K>, Vec<V>)
where
    K: Copy,
    V: Copy,
    Eq: op::BinaryPredicateOp<K>,
{
    let mut out_keys = Vec::new();
    let mut out_values = Vec::new();
    for i in 0..keys.len() {
        if i == 0 || !Eq::apply(keys[i - 1], keys[i]) {
            out_keys.push(keys[i]);
            out_values.push(values[i]);
        }
    }
    (out_keys, out_values)
}

pub fn set_union<T, Less>(left: &[T], right: &[T], _less: Less) -> Vec<T>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    let mut out = Vec::new();
    let mut l = 0;
    let mut r = 0;
    while l < left.len() && r < right.len() {
        if Less::apply(left[l], right[r]) {
            out.push(left[l]);
            l += 1;
        } else if Less::apply(right[r], left[l]) {
            out.push(right[r]);
            r += 1;
        } else {
            out.push(left[l]);
            l += 1;
            r += 1;
        }
    }
    out.extend_from_slice(&left[l..]);
    out.extend_from_slice(&right[r..]);
    out
}

pub fn set_intersection<T, Less>(left: &[T], right: &[T], _less: Less) -> Vec<T>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    let mut out = Vec::new();
    let mut l = 0;
    let mut r = 0;
    while l < left.len() && r < right.len() {
        if Less::apply(left[l], right[r]) {
            l += 1;
        } else if Less::apply(right[r], left[l]) {
            r += 1;
        } else {
            out.push(left[l]);
            l += 1;
            r += 1;
        }
    }
    out
}

pub fn set_difference<T, Less>(left: &[T], right: &[T], _less: Less) -> Vec<T>
where
    T: Copy,
    Less: op::BinaryPredicateOp<T>,
{
    let mut out = Vec::new();
    let mut l = 0;
    let mut r = 0;
    while l < left.len() && r < right.len() {
        if Less::apply(left[l], right[r]) {
            out.push(left[l]);
            l += 1;
        } else if Less::apply(right[r], left[l]) {
            r += 1;
        } else {
            l += 1;
            r += 1;
        }
    }
    out.extend_from_slice(&left[l..]);
    out
}

fn sort_by_less<T, F>(values: &mut [T], less: F)
where
    F: Fn(T, T) -> bool,
    T: Copy,
{
    values.sort_by(|lhs, rhs| {
        if less(*lhs, *rhs) {
            Ordering::Less
        } else if less(*rhs, *lhs) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
}
