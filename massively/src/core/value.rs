//! Semantic and physical value traits.

use cubecl::prelude::{CubeElement, CubePrimitive};

/// A scalar that can occupy one physical storage column.
pub trait MStorageElement:
    CubePrimitive
    + cubecl::frontend::Scalar
    + CubeElement
    + crate::core::storage::StorageLayout<
        StorageArity = crate::core::storage::S1,
        StorageLeaves = crate::core::storage::Last<Self>,
    > + crate::core::storage::ReadElement
    + Copy
    + Send
    + Sync
    + 'static
{
}

macro_rules! impl_storage_element {
    ($($ty:ty),+ $(,)?) => {
        $(impl MStorageElement for $ty {})+
    };
}

impl_storage_element!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;

    use super::MStorageElement;

    assert_not_impl_any!(bool: MStorageElement);
    assert_not_impl_any!(usize: MStorageElement);
}
