//! Predicate algorithms composed from materialization and reduction.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::core::arity::Dispatch;
use crate::core::op::{PredicateOp, ReductionOp};
use crate::core::read::{
    AdjacentIndexedTransform, Env0, IndexedTransform, LowerReadExpression, ReadExpression,
    StageRead, Transform,
};
use crate::core::reduce::{ReduceDispatch, reduce};
use crate::op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp};
use crate::{DeviceVec, Error, Executor, MFlag, MIndex, MVec};

#[cubecl::cube]
pub(crate) fn predicate<Input, Pred>(input: Input) -> bool
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    crate::flag::is_set(Pred::apply(input))
}

#[doc(hidden)]
pub struct PredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> UnaryOp<Input> for PredicateMap<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = MFlag;

    fn apply(input: Input) -> MFlag {
        crate::flag::from_bool(crate::flag::is_set(Pred::apply(input)))
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

struct FirstMatchingIndex<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> IndexedUnaryOp<Input> for FirstMatchingIndex<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(input: Input, index: u32) -> u32 {
        if crate::core::predicate::predicate::<Input, Pred>(input) {
            index
        } else {
            4_294_967_295u32
        }
    }
}

struct FirstPartitionViolation<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> IndexedBinaryOp<Input> for FirstPartitionViolation<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(previous: Input, current: Input, index: u32) -> u32 {
        if index != 0u32
            && !crate::core::predicate::predicate::<Input, Pred>(previous)
            && crate::core::predicate::predicate::<Input, Pred>(current)
        {
            index
        } else {
            4_294_967_295u32
        }
    }
}

/// Internal capability proving that the input has a supported predicate kernel.
#[doc(hidden)]
pub trait PredicateInput<R: Runtime, Pred>: ReadExpression + Sized {
    fn predicate_count(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_first(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_control(
        self,
        exec: &Executor<R>,
    ) -> Result<crate::core::selection::SelectionControl<R>, Error>;
}

impl<R, Input, Pred> PredicateInput<R, Pred> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Pred: PredicateOp<Input::Item>,
    Transform<Input, PredicateMap<Pred>>:
        ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
    IndexedTransform<Input, FirstMatchingIndex<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::core::arity::A13, crate::core::storage::S12>:
        ReduceDispatch<
                R,
                IndexedTransform<Input, FirstMatchingIndex<Pred>>,
                u32,
                MinU32,
                crate::core::read::KernelReadSlots<
                    <IndexedTransform<Input, FirstMatchingIndex<Pred>> as LowerReadExpression>::Slots,
                >,
                Storage = DeviceVec<R, u32>,
            >,
    AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::core::arity::A13, crate::core::storage::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>>,
            u32,
            MinU32,
            crate::core::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>> as LowerReadExpression>::Slots,
            >,
            Storage = DeviceVec<R, u32>,
        >,
    Dispatch<crate::core::arity::A13, crate::core::storage::S12>:
        ReduceDispatch<
                R,
                Transform<Input, PredicateMap<Pred>>,
                u32,
                SumU32,
                crate::core::read::KernelReadSlots<
                    <Transform<Input, PredicateMap<Pred>> as LowerReadExpression>::Slots,
                >,
                Storage = DeviceVec<R, u32>,
            >,
    Transform<Input, PredicateMap<Pred>>: crate::core::selection::FlagInput<R>,
{
    fn predicate_count(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            Transform::new(self, PredicateMap::<Pred>(PhantomData)),
            exec.to_device(&[0u32]),
            SumU32,
        )
    }

    fn predicate_first(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            IndexedTransform::new(self, FirstMatchingIndex::<Pred>(PhantomData)),
            exec.to_device(&[u32::MAX]),
            MinU32,
        )
    }

    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            AdjacentIndexedTransform::new(
                self,
                FirstPartitionViolation::<Pred>(PhantomData),
            ),
            exec.to_device(&[u32::MAX]),
            MinU32,
        )
    }

    fn predicate_control(
        self,
        exec: &Executor<R>,
    ) -> Result<crate::core::selection::SelectionControl<R>, Error> {
        crate::core::selection::FlagInput::selected_control(
            Transform::new(self, PredicateMap::<Pred>(PhantomData)),
            exec,
        )
    }

}

/// Counts elements satisfying `pred`.
pub(crate) fn count_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_count(exec)
}

/// Returns the first matching index, or `u32::MAX` when none matches.
pub(crate) fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_first(exec)
}

/// Returns the first partition violation, or `u32::MAX` when there is none.
pub(crate) fn is_partitioned<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVec<R, u32>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_is_partitioned(exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::iter::Zip;
    use crate::core::read::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn read(exec: &Executor<WgpuRuntime>, value: MVec<WgpuRuntime, u32>) -> u32 {
        crate::api::value::read::<WgpuRuntime, u32>(exec, &value).unwrap()
    }

    struct MatchTriple;

    #[cubecl::cube]
    impl PredicateOp<(u32, u32, u32)> for MatchTriple {
        fn apply(input: (u32, u32, u32)) -> MFlag {
            crate::flag::from_bool(input.0 + input.1 == input.2)
        }
    }

    #[test]
    fn predicate_queries_receive_flat_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_u32, 20, 30, 40]);
        let c = exec.to_device(&[11_u32, 0, 33, 0]);

        let input = || Zip::new(a.column(), Zip::new(b.column(), c.column()));
        assert_eq!(
            read(&exec, count_if(&exec, input(), MatchTriple).unwrap()),
            2
        );
        assert_eq!(
            read(&exec, count_if(&exec, input(), MatchTriple).unwrap()),
            2
        );
        assert_eq!(
            read(&exec, count_if(&exec, input(), MatchTriple).unwrap()),
            2
        );
        assert_eq!(
            read(&exec, count_if(&exec, input(), MatchTriple).unwrap()),
            2
        );
        assert_eq!(
            read(&exec, find_if(&exec, input(), MatchTriple).unwrap()),
            0
        );
        assert_eq!(
            read(&exec, is_partitioned(&exec, input(), MatchTriple).unwrap(),),
            2
        );
    }

    struct LastLeafIsThree;

    struct IsEven;

    #[cubecl::cube]
    impl PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> MFlag {
            crate::flag::from_bool(input % 2u32 == 0u32)
        }
    }

    struct RawFlag;

    #[cubecl::cube]
    impl PredicateOp<u32> for RawFlag {
        fn apply(input: u32) -> MFlag {
            input
        }
    }

    #[test]
    fn predicate_consumers_normalize_nonzero_flags() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[7_u32, 0, 3]);
        assert_eq!(
            read(&exec, count_if(&exec, input.column(), RawFlag).unwrap()),
            2
        );
    }

    #[test]
    fn partitioned_accepts_a_two_item_all_passing_range() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[20_u32, 0]);
        assert_eq!(
            read(
                &exec,
                is_partitioned(&exec, input.column(), IsEven).unwrap(),
            ),
            u32::MAX
        );
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    #[cubecl::cube]
    impl PredicateOp<Seven> for LastLeafIsThree {
        fn apply(input: Seven) -> MFlag {
            crate::flag::from_bool(input.6 == 3)
        }
    }

    #[test]
    fn predicate_materialization_uses_padded_fixed_evaluator() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0_u32..7)
            .map(|_| exec.to_device(&[1_u32, 2, 3, 4]))
            .collect();
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
        let input = Permute::new(seven, Counting::new(0, 4));
        assert_eq!(
            read(&exec, count_if(&exec, input, LastLeafIsThree).unwrap()),
            1
        );
    }

    struct Never;

    #[cubecl::cube]
    impl PredicateOp<u32> for Never {
        fn apply(_input: u32) -> MFlag {
            crate::flag::from_bool(false)
        }
    }

    #[test]
    fn find_if_returns_none_without_a_numeric_sentinel() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        assert_eq!(
            read(&exec, find_if(&exec, input.column(), Never).unwrap()),
            u32::MAX
        );
    }
}
