//! CPU references for segment algorithms applied independently to nested vectors.

use std::cmp::Ordering;

use crate::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};

pub fn map<T, Op>(segments: &[Vec<T>], _op: Op) -> Vec<Vec<Op::Output>>
where
    T: Copy,
    Op: UnaryOp<T>,
{
    segments
        .iter()
        .map(|segment| segment.iter().copied().map(Op::apply).collect())
        .collect()
}

pub fn sort<T, Less>(segments: &[Vec<T>], _less: Less) -> Vec<Vec<T>>
where
    T: Copy,
    Less: BinaryPredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            let mut output = segment.clone();
            output.sort_by(|lhs, rhs| {
                if Less::apply(*lhs, *rhs) {
                    Ordering::Less
                } else if Less::apply(*rhs, *lhs) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            output
        })
        .collect()
}

pub fn reverse<T: Copy>(segments: &[Vec<T>]) -> Vec<Vec<T>> {
    segments
        .iter()
        .map(|segment| segment.iter().copied().rev().collect())
        .collect()
}

pub fn inclusive_scan<T, Op>(segments: &[Vec<T>], _op: Op) -> Vec<Vec<T>>
where
    T: Copy,
    Op: ReductionOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            let mut output = Vec::with_capacity(segment.len());
            let mut accumulator = None;
            for value in segment.iter().copied() {
                let next = match accumulator {
                    Some(previous) => Op::apply(previous, value),
                    None => value,
                };
                output.push(next);
                accumulator = Some(next);
            }
            output
        })
        .collect()
}

pub fn exclusive_scan<T, Op>(segments: &[Vec<T>], _op: Op, init: T) -> Vec<Vec<T>>
where
    T: Copy,
    Op: ReductionOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            let mut output = Vec::with_capacity(segment.len());
            let mut accumulator = init;
            for value in segment.iter().copied() {
                output.push(accumulator);
                accumulator = Op::apply(accumulator, value);
            }
            output
        })
        .collect()
}

pub fn adjacent_difference<T, Op>(segments: &[Vec<T>], _op: Op) -> Vec<Vec<T>>
where
    T: Copy,
    Op: ReductionOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            let Some(first) = segment.first().copied() else {
                return Vec::new();
            };
            let mut output = Vec::with_capacity(segment.len());
            output.push(first);
            output.extend(segment.windows(2).map(|pair| Op::apply(pair[0], pair[1])));
            output
        })
        .collect()
}

pub fn unique<T, Equal>(segments: &[Vec<T>], _equal: Equal) -> Vec<Vec<T>>
where
    T: Copy,
    Equal: BinaryPredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            let mut output = Vec::with_capacity(segment.len());
            for value in segment.iter().copied() {
                if output
                    .last()
                    .is_none_or(|previous| !Equal::apply(*previous, value))
                {
                    output.push(value);
                }
            }
            output
        })
        .collect()
}

pub fn filter<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<Vec<T>>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .iter()
                .copied()
                .filter(|value| Pred::apply(*value))
                .collect()
        })
        .collect()
}

pub fn take<T: Copy>(segments: &[Vec<T>], count: usize) -> Vec<Vec<T>> {
    segments
        .iter()
        .map(|segment| segment.iter().copied().take(count).collect())
        .collect()
}

pub fn reduce<T, Op>(segments: &[Vec<T>], _op: Op, init: T) -> Vec<T>
where
    T: Copy,
    Op: ReductionOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .iter()
                .copied()
                .fold(init, |acc, value| Op::apply(acc, value))
        })
        .collect()
}

pub fn count_if<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<u32>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .iter()
                .copied()
                .filter(|value| Pred::apply(*value))
                .count() as u32
        })
        .collect()
}

pub fn find_if<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<u32>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .iter()
                .copied()
                .position(Pred::apply)
                .unwrap_or(segment.len()) as u32
        })
        .collect()
}

pub fn adjacent_find<T, Equal>(segments: &[Vec<T>], _equal: Equal) -> Vec<u32>
where
    T: Copy,
    Equal: BinaryPredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .windows(2)
                .position(|pair| Equal::apply(pair[0], pair[1]))
                .unwrap_or(segment.len()) as u32
        })
        .collect()
}

pub fn all_of<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<bool>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| segment.iter().copied().all(Pred::apply))
        .collect()
}

pub fn any_of<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<bool>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| segment.iter().copied().any(Pred::apply))
        .collect()
}

pub fn none_of<T, Pred>(segments: &[Vec<T>], _pred: Pred) -> Vec<bool>
where
    T: Copy,
    Pred: PredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| !segment.iter().copied().any(Pred::apply))
        .collect()
}

pub fn is_sorted<T, Less>(segments: &[Vec<T>], _less: Less) -> Vec<bool>
where
    T: Copy,
    Less: BinaryPredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .windows(2)
                .all(|pair| !Less::apply(pair[1], pair[0]))
        })
        .collect()
}

pub fn is_sorted_until<T, Less>(segments: &[Vec<T>], _less: Less) -> Vec<u32>
where
    T: Copy,
    Less: BinaryPredicateOp<T>,
{
    segments
        .iter()
        .map(|segment| {
            segment
                .windows(2)
                .position(|pair| Less::apply(pair[1], pair[0]))
                .map_or(segment.len(), |index| index + 1) as u32
        })
        .collect()
}
