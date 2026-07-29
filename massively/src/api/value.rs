use cubecl::prelude::Runtime;

use crate::{Error, Executor, MAlloc, MIndex, MVec};

pub(crate) fn store<R, T>(exec: &Executor<R>, value: T) -> Result<MVec<R, T>, Error>
where
    R: Runtime,
    T: MAlloc<R>,
{
    <T::Dispatch as crate::api::iter::ItemDispatch<R>>::store_value(exec, value)
}

pub(crate) fn read<R, T>(exec: &Executor<R>, storage: &MVec<R, T>) -> Result<T, Error>
where
    R: Runtime,
    T: MAlloc<R>,
{
    <T::Dispatch as crate::api::iter::ItemDispatch<R>>::read_value(exec, storage)
}

pub(crate) fn into_scratch<R, T>(
    storage: MVec<R, T>,
) -> <T as crate::core::allocation::ScratchStorage<R>>::Storage
where
    R: Runtime,
    T: MAlloc<R> + crate::core::allocation::ScratchStorage<R>,
{
    <T::Dispatch as crate::api::iter::ItemDispatch<R>>::into_scratch(storage)
}

pub(crate) fn read_optional_index<R>(
    exec: &Executor<R>,
    storage: &MVec<R, MIndex>,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    MIndex: MAlloc<R>,
{
    let value: MIndex = read(exec, storage)?;
    Ok((value != MIndex::MAX).then_some(value))
}

pub(crate) fn read_optional_index_pair<R>(
    exec: &Executor<R>,
    storage: &MVec<R, (MIndex, MIndex)>,
) -> Result<Option<(MIndex, MIndex)>, Error>
where
    R: Runtime,
    (MIndex, MIndex): MAlloc<R>,
{
    let value: (MIndex, MIndex) = read(exec, storage)?;
    Ok((value.0 != MIndex::MAX).then_some(value))
}
