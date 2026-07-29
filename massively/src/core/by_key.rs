//! By-key algorithms split into key-control and value-apply phases.

use cubecl::prelude::*;

use crate::core::allocation::ScratchStorage;
use crate::core::op::BinaryPredicateOp;
use crate::core::ordering::{AdjacentFlagInput, UniqueHead, unique_head_flags};
use crate::core::read::ReadExpression;
use crate::core::selection::{CopySelected, SelectionControl};
use crate::{DeviceVec, Error, Executor};

/// Key-only phase producing segment head flags.
#[doc(hidden)]
pub trait SegmentKeyInput<R: Runtime, Equal>: ReadExpression + Sized {
    fn segment_heads(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Keys, Equal> SegmentKeyInput<R, Equal> for Keys
where
    R: Runtime,
    Keys: ReadExpression + AdjacentFlagInput<R, UniqueHead<Equal>>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    fn segment_heads(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        unique_head_flags::<R, _, Equal>(exec, self)
    }
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn inclusive_scan_by_key_lowered<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    _op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output: crate::core::facade::KernelOutput<R>
        + crate::core::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.segment_heads(exec)?;
    crate::core::segmented::segmented_inclusive::<R, _, _, Values::Item, Op>(
        exec, &values, &heads, &output,
    )
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn exclusive_scan_by_key_lowered<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    _op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output: crate::core::facade::KernelOutput<R>
        + crate::core::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.segment_heads(exec)?;
    crate::core::segmented::segmented_exclusive::<R, _, _, Values::Item, Op>(
        exec, &values, &heads, init, &output,
    )
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn reduce_by_key_lowered<R, Keys, Values, Equal, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    op: Op,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: crate::core::facade::KernelInput<R, Item = KeyOutput::Item>
        + SegmentKeyInput<R, Equal>
        + CopySelected<R, KeyOutput>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    KeyOutput: crate::core::facade::KernelOutput<R>,
    ValueOutput: crate::core::facade::KernelOutput<R>
        + crate::core::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.clone().segment_heads(exec)?;
    let head_control = SelectionControl::from_flags(exec, heads.clone())?;
    let head_positions = head_control.materialize_positions(exec)?;
    crate::core::segmented::segmented_reduce::<R, _, _, Values::Item, Op>(
        exec,
        &values,
        &heads,
        &head_positions,
        head_control.count(),
        init,
        op,
        &value_output,
    )?;
    keys.copy_selected(exec, &head_control, key_output)?;
    Ok(head_control.count().clone())
}

/// Reduces values using an already-computed segment-head control.
pub(crate) fn reduce_values_by_heads_lowered<R, Values, Op, Output>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    head_control: &SelectionControl<R>,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output: crate::core::facade::KernelOutput<R>
        + crate::core::output::OutputExpression<Item = Values::Item>,
{
    let head_positions = head_control.materialize_positions(exec)?;
    crate::core::segmented::segmented_reduce::<R, _, _, Values::Item, Op>(
        exec,
        &values,
        heads,
        &head_positions,
        head_control.count(),
        init,
        op,
        &output,
    )
}

/// Key phase for stable adjacent-key deduplication.
///
/// The resulting selection control is applied only to values, avoiding both
/// key materialization and key-arity × value-arity dispatch.
#[doc(hidden)]
pub trait UniqueByKeyKeys<R: Runtime, Equal>: ReadExpression + Sized {
    fn unique_key_len(&self) -> Result<usize, Error>;
    fn unique_key_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error>;
    fn unique_key_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
}

impl<R, Keys, Equal> UniqueByKeyKeys<R, Equal> for Keys
where
    R: Runtime,
    Keys: ReadExpression
        + AdjacentFlagInput<R, UniqueHead<Equal>>
        + crate::core::read::StageRead<R, crate::core::read::Env0>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    fn unique_key_len(&self) -> Result<usize, Error> {
        crate::core::read::StageRead::physical_len(self)
    }

    fn unique_key_extent(&self) -> Result<crate::core::extent::LogicalExtent, Error> {
        crate::core::read::StageRead::logical_extent(self)
    }

    fn unique_key_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let flags = unique_head_flags::<R, _, Equal>(exec, self)?;
        SelectionControl::from_flags(exec, flags)
    }
}

/// Keeps the first value of every adjacent equal-key run.
///
/// `value_output` is preallocated and may be larger than the returned logical
/// length.
pub(crate) fn unique_by_key<R, Keys, Values, Equal, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    value_output: ValueOutput,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: UniqueByKeyKeys<R, Equal>,
    Values: CopySelected<R, ValueOutput>,
{
    let key_len = keys.unique_key_len()?;
    let value_len = values.source_len()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len,
            right: value_len,
        });
    }
    keys.unique_key_extent()?.zipped(&values.source_extent()?)?;
    let control = keys.unique_key_control(exec)?;
    values.copy_selected(exec, &control, value_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::allocation::RowStorage;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Three = (u32, u32, u32);
    struct EqualThree;

    #[cubecl::cube]
    impl BinaryPredicateOp<Three> for EqualThree {
        fn apply(lhs: Three, rhs: Three) -> crate::MFlag {
            crate::flag::from_bool(lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2)
        }
    }

    struct EqualU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for EqualU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs == rhs)
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct SumSeven;

    #[cubecl::cube]
    impl crate::op::ReductionOp<Seven> for SumSeven {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                lhs.1 + rhs.1,
                lhs.2 + rhs.2,
                lhs.3 + rhs.3,
                lhs.4 + rhs.4,
                lhs.5 + rhs.5,
                lhs.6 + rhs.6,
            )
        }
    }

    #[test]
    fn unique_by_key_separates_three_key_and_seven_value_arities() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let k0 = exec.to_device(&[1_u32, 1, 1, 2, 2]);
        let k1 = exec.to_device(&[10_u32, 10, 11, 10, 10]);
        let k2 = exec.to_device(&[100_u32, 100, 100, 200, 200]);
        let values: Vec<_> = (0_u32..7)
            .map(|column| {
                exec.to_device(&[
                    column * 100 + 10,
                    column * 100 + 11,
                    column * 100 + 12,
                    column * 100 + 20,
                    column * 100 + 21,
                ])
            })
            .collect();
        let out_values: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 5])).collect();

        let keys = Zip::new(k0.column(), Zip::new(k1.column(), k2.column()));
        let value_input = Zip::new(
            values[0].column(),
            Zip::new(
                values[1].column(),
                Zip::new(
                    values[2].column(),
                    Zip::new(
                        values[3].column(),
                        Zip::new(
                            values[4].column(),
                            Zip::new(values[5].column(), values[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let value_output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(out_values[0].slice_mut(..), out_values[1].slice_mut(..)),
                            out_values[2].slice_mut(..),
                        ),
                        out_values[3].slice_mut(..),
                    ),
                    out_values[4].slice_mut(..),
                ),
                out_values[5].slice_mut(..),
            ),
            out_values[6].slice_mut(..),
        );

        let count = unique_by_key(&exec, keys, value_input, EqualThree, value_output).unwrap();
        let count = exec.to_host(&count).unwrap()[0] as usize;
        assert_eq!(count, 3);
        for (column, output) in out_values.iter().enumerate() {
            let base = column as u32 * 100;
            assert_eq!(
                exec.to_host(output).unwrap()[..3],
                [base + 10, base + 12, base + 20]
            );
        }
    }

    #[test]
    fn unique_by_key_rejects_key_value_length_mismatch() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let keys = exec.to_device(&[1_u32, 1]);
        let values = exec.to_device(&[10_u32]);
        let out_values = exec.to_device(&[0_u32; 2]);
        assert!(matches!(
            unique_by_key(
                &exec,
                keys.column(),
                values.column(),
                EqualU32,
                out_values.slice_mut(..),
            ),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        ));
    }

    #[test]
    fn segmented_by_key_algorithms_separate_eight_slot_keys_from_storage7_values() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 600usize;
        let keys_host: Vec<u32> = (0..len)
            .map(|index| if index < 300 { 1 } else { 2 })
            .collect();
        let keys = exec.to_device(&keys_host);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; len]))
            .collect();
        let values = || {
            let seven = Zip::new(
                columns[0].column(),
                Zip::new(
                    columns[1].column(),
                    Zip::new(
                        columns[2].column(),
                        Zip::new(
                            columns[3].column(),
                            Zip::new(
                                columns[4].column(),
                                Zip::new(columns[5].column(), columns[6].column()),
                            ),
                        ),
                    ),
                ),
            );
            Permute::new(seven, Counting::new(0, len))
        };

        let inclusive = exec.alloc_row::<Seven>(len);
        crate::api::algorithm::inclusive_scan_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            SumSeven,
            inclusive.write(),
        )
        .unwrap();
        let (first, _, _, _, _, _, seventh) = crate::MStorage::into_columns(inclusive);
        let first = exec.to_host(&first).unwrap();
        let seventh = exec.to_host(&seventh).unwrap();
        assert_eq!((first[299], first[300], first[599]), (300, 1, 300));
        assert_eq!((seventh[299], seventh[300], seventh[599]), (2100, 7, 2100));

        let init: Seven = (10, 20, 30, 40, 50, 60, 70);
        let init = crate::api::value::store(&exec, init).unwrap();
        let exclusive = exec.alloc_row::<Seven>(len);
        crate::api::algorithm::exclusive_scan_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            init.clone(),
            SumSeven,
            exclusive.write(),
        )
        .unwrap();
        let (first, _, _, _, _, _, _) = crate::MStorage::into_columns(exclusive);
        let first = exec.to_host(&first).unwrap();
        assert_eq!(
            (first[0], first[1], first[300], first[301]),
            (10, 11, 10, 11)
        );

        // A caller may know the maximum number of segments independently of
        // the input length.  Compact outputs must not require one slot per
        // input item.
        let key_output = exec.to_device(&[0_u32; 2]);
        let value_output = exec.alloc_row::<Seven>(2);
        let count = crate::api::algorithm::reduce_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            init,
            SumSeven,
            key_output.slice_mut(..),
            value_output.write(),
        )
        .unwrap();
        let count = crate::api::value::read::<WgpuRuntime, u32>(&exec, &count).unwrap();
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&key_output.slice(..2)).unwrap(), vec![1, 2]);
        let (first, _, _, _, _, _, seventh) = crate::MStorage::into_columns(value_output);
        assert_eq!(exec.to_host(&first.slice(..2)).unwrap(), vec![310, 310]);
        assert_eq!(exec.to_host(&seventh.slice(..2)).unwrap(), vec![2170, 2170]);
    }
}
