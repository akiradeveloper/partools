//! Storage arity and write-boundary normalization.

use cubecl::prelude::*;

/// A physical storage arity supported by public storage values.
pub trait StorageArity: private::Sealed + 'static {}

macro_rules! define_arities {
    ($($arity:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Storage arity marker `", stringify!($arity), "`.")]
            #[derive(Clone, Copy, Debug, Eq, PartialEq)]
            pub struct $arity;

            impl StorageArity for $arity {}
            impl private::Sealed for $arity {}
        )+
    };
}

define_arities!(S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12);

/// Type-level addition for storage arities whose sum is at most twelve.
#[doc(hidden)]
pub trait AddStorageArity<Rhs: StorageArity>: StorageArity {
    type Output: StorageArity;
}

macro_rules! impl_add_arity {
    ($lhs:ty, $rhs:ty => $output:ty) => {
        impl AddStorageArity<$rhs> for $lhs {
            type Output = $output;
        }
    };
}

impl_add_arity!(S1, S1 => S2);
impl_add_arity!(S1, S2 => S3);
impl_add_arity!(S1, S3 => S4);
impl_add_arity!(S1, S4 => S5);
impl_add_arity!(S1, S5 => S6);
impl_add_arity!(S1, S6 => S7);
impl_add_arity!(S1, S7 => S8);
impl_add_arity!(S1, S8 => S9);
impl_add_arity!(S1, S9 => S10);
impl_add_arity!(S1, S10 => S11);
impl_add_arity!(S1, S11 => S12);
impl_add_arity!(S2, S1 => S3);
impl_add_arity!(S2, S2 => S4);
impl_add_arity!(S2, S3 => S5);
impl_add_arity!(S2, S4 => S6);
impl_add_arity!(S2, S5 => S7);
impl_add_arity!(S2, S6 => S8);
impl_add_arity!(S2, S7 => S9);
impl_add_arity!(S2, S8 => S10);
impl_add_arity!(S2, S9 => S11);
impl_add_arity!(S2, S10 => S12);
impl_add_arity!(S3, S1 => S4);
impl_add_arity!(S3, S2 => S5);
impl_add_arity!(S3, S3 => S6);
impl_add_arity!(S3, S4 => S7);
impl_add_arity!(S3, S5 => S8);
impl_add_arity!(S3, S6 => S9);
impl_add_arity!(S3, S7 => S10);
impl_add_arity!(S3, S8 => S11);
impl_add_arity!(S3, S9 => S12);
impl_add_arity!(S4, S1 => S5);
impl_add_arity!(S4, S2 => S6);
impl_add_arity!(S4, S3 => S7);
impl_add_arity!(S4, S4 => S8);
impl_add_arity!(S4, S5 => S9);
impl_add_arity!(S4, S6 => S10);
impl_add_arity!(S4, S7 => S11);
impl_add_arity!(S4, S8 => S12);
impl_add_arity!(S5, S1 => S6);
impl_add_arity!(S5, S2 => S7);
impl_add_arity!(S5, S3 => S8);
impl_add_arity!(S5, S4 => S9);
impl_add_arity!(S5, S5 => S10);
impl_add_arity!(S5, S6 => S11);
impl_add_arity!(S5, S7 => S12);
impl_add_arity!(S6, S1 => S7);
impl_add_arity!(S6, S2 => S8);
impl_add_arity!(S6, S3 => S9);
impl_add_arity!(S6, S4 => S10);
impl_add_arity!(S6, S5 => S11);
impl_add_arity!(S6, S6 => S12);
impl_add_arity!(S7, S1 => S8);
impl_add_arity!(S7, S2 => S9);
impl_add_arity!(S7, S3 => S10);
impl_add_arity!(S7, S4 => S11);
impl_add_arity!(S7, S5 => S12);
impl_add_arity!(S8, S1 => S9);
impl_add_arity!(S8, S2 => S10);
impl_add_arity!(S8, S3 => S11);
impl_add_arity!(S8, S4 => S12);
impl_add_arity!(S9, S1 => S10);
impl_add_arity!(S9, S2 => S11);
impl_add_arity!(S9, S3 => S12);
impl_add_arity!(S10, S1 => S11);
impl_add_arity!(S10, S2 => S12);
impl_add_arity!(S11, S1 => S12);

/// The final leaf in a non-empty, ordered storage-leaf list.
#[doc(hidden)]
#[derive(CubeType, Clone, Copy, Debug, Eq, PartialEq)]
pub struct Last<T: CubeType> {
    pub(crate) value: T,
}

#[cubecl::cube]
impl<T: CubeType> Last<T> {
    #[allow(dead_code)]
    pub(crate) fn new(value: T) -> Self {
        Last::<T> { value }
    }
}

/// A non-final leaf in a non-empty, ordered storage-leaf list.
#[doc(hidden)]
#[derive(CubeType, Clone, Copy, Debug, Eq, PartialEq)]
pub struct More<Head: CubeType, Tail: CubeType> {
    pub(crate) head: Head,
    pub(crate) tail: Tail,
}

#[doc(hidden)]
pub type Leaves2<A, B> = More<A, Last<B>>;
#[doc(hidden)]
pub type Leaves3<A, B, C> = More<A, Leaves2<B, C>>;
#[doc(hidden)]
pub type Leaves4<A, B, C, D> = More<A, Leaves3<B, C, D>>;
#[doc(hidden)]
pub type Leaves5<A, B, C, D, E> = More<A, Leaves4<B, C, D, E>>;
#[doc(hidden)]
pub type Leaves6<A, B, C, D, E, F> = More<A, Leaves5<B, C, D, E, F>>;
#[doc(hidden)]
pub type Leaves7<A, B, C, D, E, F, G> = More<A, Leaves6<B, C, D, E, F, G>>;
#[doc(hidden)]
pub type Leaves8<A, B, C, D, E, F, G, H> = More<A, Leaves7<B, C, D, E, F, G, H>>;
#[doc(hidden)]
pub type Leaves9<A, B, C, D, E, F, G, H, I> = More<A, Leaves8<B, C, D, E, F, G, H, I>>;
#[doc(hidden)]
pub type Leaves10<A, B, C, D, E, F, G, H, I, J> = More<A, Leaves9<B, C, D, E, F, G, H, I, J>>;
#[doc(hidden)]
pub type Leaves11<A, B, C, D, E, F, G, H, I, J, K> =
    More<A, Leaves10<B, C, D, E, F, G, H, I, J, K>>;
#[doc(hidden)]
pub type Leaves12<A, B, C, D, E, F, G, H, I, J, K, L> =
    More<A, Leaves11<B, C, D, E, F, G, H, I, J, K, L>>;

#[cubecl::cube]
impl<Head: CubeType, Tail: CubeType> More<Head, Tail> {
    #[allow(dead_code)]
    fn new(head: Head, tail: Tail) -> Self {
        More::<Head, Tail> { head, tail }
    }
}

/// Concatenates two non-empty leaf lists and can split the result again.
#[doc(hidden)]
#[cubecl::cube]
pub trait Concat<Rhs: CubeType>: CubeType + Sized {
    type Output: CubeType;

    fn concat(self, rhs: Rhs) -> Self::Output;
    fn split(output: Self::Output) -> (Self, Rhs);
}

#[cubecl::cube]
impl<Head: CubeType, Rhs: CubeType> Concat<Rhs> for Last<Head> {
    type Output = More<Head, Rhs>;

    fn concat(self, rhs: Rhs) -> Self::Output {
        More::<Head, Rhs> {
            head: self.value,
            tail: rhs,
        }
    }

    fn split(output: Self::Output) -> (Self, Rhs) {
        (Last::<Head> { value: output.head }, output.tail)
    }
}

#[cubecl::cube]
impl<Head: CubeType, Tail: CubeType, Rhs: CubeType> Concat<Rhs> for More<Head, Tail>
where
    Tail: Concat<Rhs>,
{
    type Output = More<Head, Tail::Output>;

    fn concat(self, rhs: Rhs) -> Self::Output {
        More::<Head, Tail::Output> {
            head: self.head,
            tail: self.tail.concat(rhs),
        }
    }

    fn split(output: Self::Output) -> (Self, Rhs) {
        let (tail, rhs) = Tail::split(output.tail);
        (
            More::<Head, Tail> {
                head: output.head,
                tail,
            },
            rhs,
        )
    }
}

/// Describes the ordered physical leaves of a semantic item shape.
///
/// Tuple nesting is erased only by [`into_storage_leaves`](Self::into_storage_leaves)
/// and restored by [`from_storage_leaves`](Self::from_storage_leaves).  The
/// associated leaf-list type records both leaf order and leaf types.
pub trait StorageLayout: CubeType + Sized + Send + Sync + 'static {
    type StorageArity: StorageArity;
    type StorageLeaves: CubeType + SelectLeaves + Send + Sync + 'static;
    type DeviceLayout: Decompose<Self, Leaves = Self::StorageLeaves>
        + Recompose<Self, Leaves = Self::StorageLeaves>;

    fn into_storage_leaves(self) -> Self::StorageLeaves;
    fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self;
}

/// Describes the ordered semantic fields of a read-only row.
///
/// Unlike [`StorageLayout`], a read layout may contain values such as
/// [`crate::seg::Segment`] that exist only while a kernel is being expanded
/// and therefore cannot own physical storage.
#[doc(hidden)]
pub trait ReadLayout: CubeType + Sized + Send + Sync + 'static {
    type ReadLeaves: CubeType + Send + Sync + 'static;
    type ReadDeviceLayout: Decompose<Self, Leaves = Self::ReadLeaves>
        + Recompose<Self, Leaves = Self::ReadLeaves>;
}

macro_rules! impl_scalar_layout {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl StorageLayout for $ty {
                type StorageArity = S1;
                type StorageLeaves = Last<Self>;
                type DeviceLayout = ScalarLayout<Self>;

                fn into_storage_leaves(self) -> Self::StorageLeaves {
                    Last::<$ty> { value: self }
                }

                fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
                    leaves.value
                }
            }

            impl ReadLayout for $ty {
                type ReadLeaves = Last<Self>;
                type ReadDeviceLayout = ScalarLayout<Self>;
            }
        )+
    };
}

impl_scalar_layout!(bool, usize, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<T> ReadLayout for crate::seg::Segment<T>
where
    T: CubeType + Send + Sync + 'static,
{
    type ReadLeaves = Last<Self>;
    type ReadDeviceLayout = ScalarLayout<Self>;
}

/// One indivisible field in a read-only flat row.
#[doc(hidden)]
pub trait ReadElement:
    ReadLayout<ReadLeaves = Last<Self>, ReadDeviceLayout = ScalarLayout<Self>>
    + Copy
    + Send
    + Sync
    + 'static
{
}

impl<T> ReadElement for T where
    T: ReadLayout<ReadLeaves = Last<Self>, ReadDeviceLayout = ScalarLayout<Self>>
        + Copy
        + Send
        + Sync
        + 'static
{
}

/// A scalar that may participate in a flat logical row.
///
/// Unlike [`crate::core::value::MStorageElement`], this includes read-only semantic values
/// such as `bool` and `usize` that cannot own a device column.
#[doc(hidden)]
pub trait FlatElement:
    CubePrimitive
    + cubecl::frontend::Scalar
    + StorageLayout<StorageArity = S1, StorageLeaves = Last<Self>>
    + Copy
    + Send
    + Sync
    + 'static
{
}

impl<T> FlatElement for T where
    T: CubePrimitive
        + cubecl::frontend::Scalar
        + StorageLayout<StorageArity = S1, StorageLeaves = Last<Self>>
        + Copy
        + Send
        + Sync
        + 'static
{
}

impl<Left, Right> ReadLayout for (Left, Right)
where
    Left: ReadElement,
    Right: ReadElement,
{
    type ReadLeaves = Leaves2<Left, Right>;
    type ReadDeviceLayout = PairLayout<ScalarLayout<Left>, ScalarLayout<Right>>;
}

impl<Left, Right> StorageLayout for (Left, Right)
where
    Left: StorageLayout + 'static,
    Right: StorageLayout + 'static,
    Left::StorageArity: AddStorageArity<Right::StorageArity>,
    Left::StorageLeaves: Concat<Right::StorageLeaves>,
    <Left::StorageLeaves as Concat<Right::StorageLeaves>>::Output:
        SelectLeaves + Send + Sync + 'static,
{
    type StorageArity = <Left::StorageArity as AddStorageArity<Right::StorageArity>>::Output;
    type StorageLeaves = <Left::StorageLeaves as Concat<Right::StorageLeaves>>::Output;
    type DeviceLayout = PairLayout<Left::DeviceLayout, Right::DeviceLayout>;

    fn into_storage_leaves(self) -> Self::StorageLeaves {
        self.0
            .into_storage_leaves()
            .concat(self.1.into_storage_leaves())
    }

    fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
        let (left, right) = <Left::StorageLeaves as Concat<Right::StorageLeaves>>::split(leaves);
        (
            Left::from_storage_leaves(left),
            Right::from_storage_leaves(right),
        )
    }
}

macro_rules! impl_flat_tuple_layout {
    (
        $arity:ty;
        ($( $leaf:ident : $value:ident ),+);
        $leaves:ty;
        $build:expr;
        $source:ident => $rebuild:expr
    ) => {
        impl<$( $leaf ),+> StorageLayout for ($( $leaf, )+)
        where
            $( $leaf: FlatElement, )+
        {
            type StorageArity = $arity;
            type StorageLeaves = $leaves;
            type DeviceLayout = FlatTupleLayout<($( $leaf, )+)>;

            fn into_storage_leaves(self) -> Self::StorageLeaves {
                let ($( $value, )+) = self;
                $build
            }

            fn from_storage_leaves($source: Self::StorageLeaves) -> Self {
                $rebuild
            }
        }

        impl<$( $leaf ),+> ReadLayout for ($( $leaf, )+)
        where
            $( $leaf: ReadElement, )+
        {
            type ReadLeaves = $leaves;
            type ReadDeviceLayout = FlatTupleLayout<($( $leaf, )+)>;
        }

        #[cubecl::cube]
        impl<$( $leaf: CubeType + 'static ),+> Decompose<($( $leaf, )+)>
            for FlatTupleLayout<($( $leaf, )+)>
        {
            type Leaves = $leaves;

            fn decompose(item: ($( $leaf, )+)) -> Self::Leaves {
                let ($( $value, )+) = item;
                $build
            }
        }

        #[cubecl::cube]
        impl<$( $leaf: CubeType + 'static ),+> Recompose<($( $leaf, )+)>
            for FlatTupleLayout<($( $leaf, )+)>
        {
            type Leaves = $leaves;

            fn recompose($source: Self::Leaves) -> ($( $leaf, )+) {
                $rebuild
            }
        }
    };
}

impl_flat_tuple_layout!(
    S3;
    (A:a, B:b, C:c);
    Leaves3<A, B, C>;
    Leaves3::<A, B, C> { head: a, tail: Leaves2::<B, C> { head: b, tail: Last::<C> { value: c } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.value)
);
impl_flat_tuple_layout!(
    S4;
    (A:a, B:b, C:c, D:d);
    Leaves4<A, B, C, D>;
    Leaves4::<A, B, C, D> { head: a, tail: Leaves3::<B, C, D> { head: b, tail: Leaves2::<C, D> { head: c, tail: Last::<D> { value: d } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S5;
    (A:a, B:b, C:c, D:d, E:e);
    Leaves5<A, B, C, D, E>;
    Leaves5::<A, B, C, D, E> { head: a, tail: Leaves4::<B, C, D, E> { head: b, tail: Leaves3::<C, D, E> { head: c, tail: Leaves2::<D, E> { head: d, tail: Last::<E> { value: e } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S6;
    (A:a, B:b, C:c, D:d, E:e, F:f);
    Leaves6<A, B, C, D, E, F>;
    Leaves6::<A, B, C, D, E, F> { head: a, tail: Leaves5::<B, C, D, E, F> { head: b, tail: Leaves4::<C, D, E, F> { head: c, tail: Leaves3::<D, E, F> { head: d, tail: Leaves2::<E, F> { head: e, tail: Last::<F> { value: f } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S7;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g);
    Leaves7<A, B, C, D, E, F, G>;
    Leaves7::<A, B, C, D, E, F, G> { head: a, tail: Leaves6::<B, C, D, E, F, G> { head: b, tail: Leaves5::<C, D, E, F, G> { head: c, tail: Leaves4::<D, E, F, G> { head: d, tail: Leaves3::<E, F, G> { head: e, tail: Leaves2::<F, G> { head: f, tail: Last::<G> { value: g } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S8;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h);
    Leaves8<A, B, C, D, E, F, G, H>;
    Leaves8::<A, B, C, D, E, F, G, H> { head: a, tail: Leaves7::<B, C, D, E, F, G, H> { head: b, tail: Leaves6::<C, D, E, F, G, H> { head: c, tail: Leaves5::<D, E, F, G, H> { head: d, tail: Leaves4::<E, F, G, H> { head: e, tail: Leaves3::<F, G, H> { head: f, tail: Leaves2::<G, H> { head: g, tail: Last::<H> { value: h } } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S9;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h, I:i);
    Leaves9<A, B, C, D, E, F, G, H, I>;
    Leaves9::<A, B, C, D, E, F, G, H, I> { head: a, tail: Leaves8::<B, C, D, E, F, G, H, I> { head: b, tail: Leaves7::<C, D, E, F, G, H, I> { head: c, tail: Leaves6::<D, E, F, G, H, I> { head: d, tail: Leaves5::<E, F, G, H, I> { head: e, tail: Leaves4::<F, G, H, I> { head: f, tail: Leaves3::<G, H, I> { head: g, tail: Leaves2::<H, I> { head: h, tail: Last::<I> { value: i } } } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S10;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h, I:i, J:j);
    Leaves10<A, B, C, D, E, F, G, H, I, J>;
    Leaves10::<A, B, C, D, E, F, G, H, I, J> { head: a, tail: Leaves9::<B, C, D, E, F, G, H, I, J> { head: b, tail: Leaves8::<C, D, E, F, G, H, I, J> { head: c, tail: Leaves7::<D, E, F, G, H, I, J> { head: d, tail: Leaves6::<E, F, G, H, I, J> { head: e, tail: Leaves5::<F, G, H, I, J> { head: f, tail: Leaves4::<G, H, I, J> { head: g, tail: Leaves3::<H, I, J> { head: h, tail: Leaves2::<I, J> { head: i, tail: Last::<J> { value: j } } } } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S11;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h, I:i, J:j, K:k);
    Leaves11<A, B, C, D, E, F, G, H, I, J, K>;
    Leaves11::<A, B, C, D, E, F, G, H, I, J, K> { head: a, tail: Leaves10::<B, C, D, E, F, G, H, I, J, K> { head: b, tail: Leaves9::<C, D, E, F, G, H, I, J, K> { head: c, tail: Leaves8::<D, E, F, G, H, I, J, K> { head: d, tail: Leaves7::<E, F, G, H, I, J, K> { head: e, tail: Leaves6::<F, G, H, I, J, K> { head: f, tail: Leaves5::<G, H, I, J, K> { head: g, tail: Leaves4::<H, I, J, K> { head: h, tail: Leaves3::<I, J, K> { head: i, tail: Leaves2::<J, K> { head: j, tail: Last::<K> { value: k } } } } } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value)
);
impl_flat_tuple_layout!(
    S12;
    (A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h, I:i, J:j, K:k, L:l);
    Leaves12<A, B, C, D, E, F, G, H, I, J, K, L>;
    Leaves12::<A, B, C, D, E, F, G, H, I, J, K, L> { head: a, tail: Leaves11::<B, C, D, E, F, G, H, I, J, K, L> { head: b, tail: Leaves10::<C, D, E, F, G, H, I, J, K, L> { head: c, tail: Leaves9::<D, E, F, G, H, I, J, K, L> { head: d, tail: Leaves8::<E, F, G, H, I, J, K, L> { head: e, tail: Leaves7::<F, G, H, I, J, K, L> { head: f, tail: Leaves6::<G, H, I, J, K, L> { head: g, tail: Leaves5::<H, I, J, K, L> { head: h, tail: Leaves4::<I, J, K, L> { head: i, tail: Leaves3::<J, K, L> { head: j, tail: Leaves2::<K, L> { head: k, tail: Last::<L> { value: l } } } } } } } } } } } };
    leaves => (leaves.head, leaves.tail.head, leaves.tail.tail.head, leaves.tail.tail.tail.head, leaves.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.head, leaves.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value)
);

/// Maps an ordered physical leaf list to its unique flat logical row type.
#[doc(hidden)]
pub trait FlatLeaves: CubeType {
    type Item: StorageLayout<StorageLeaves = Self>;
}

/// Maps an ordered semantic leaf list to its unique flat read-only row type.
#[doc(hidden)]
pub trait ReadFlatLeaves: CubeType {
    type Item: ReadLayout<ReadLeaves = Self>;
}

impl<T> FlatLeaves for Last<T>
where
    T: FlatElement,
{
    type Item = T;
}

impl<T> ReadFlatLeaves for Last<T>
where
    T: ReadElement,
{
    type Item = T;
}

macro_rules! impl_flat_leaves {
    ($leaves:ty => ($( $leaf:ident ),+)) => {
        impl<$( $leaf ),+> FlatLeaves for $leaves
        where
            $( $leaf: FlatElement, )+
        {
            type Item = ($( $leaf, )+);
        }

        impl<$( $leaf ),+> ReadFlatLeaves for $leaves
        where
            $( $leaf: ReadElement, )+
        {
            type Item = ($( $leaf, )+);
        }
    };
}

impl_flat_leaves!(Leaves2<A, B> => (A, B));
impl_flat_leaves!(Leaves3<A, B, C> => (A, B, C));
impl_flat_leaves!(Leaves4<A, B, C, D> => (A, B, C, D));
impl_flat_leaves!(Leaves5<A, B, C, D, E> => (A, B, C, D, E));
impl_flat_leaves!(Leaves6<A, B, C, D, E, F> => (A, B, C, D, E, F));
impl_flat_leaves!(Leaves7<A, B, C, D, E, F, G> => (A, B, C, D, E, F, G));
impl_flat_leaves!(Leaves8<A, B, C, D, E, F, G, H> => (A, B, C, D, E, F, G, H));
impl_flat_leaves!(Leaves9<A, B, C, D, E, F, G, H, I> => (A, B, C, D, E, F, G, H, I));
impl_flat_leaves!(Leaves10<A, B, C, D, E, F, G, H, I, J> => (A, B, C, D, E, F, G, H, I, J));
impl_flat_leaves!(Leaves11<A, B, C, D, E, F, G, H, I, J, K> => (A, B, C, D, E, F, G, H, I, J, K));
impl_flat_leaves!(Leaves12<A, B, C, D, E, F, G, H, I, J, K, L> => (A, B, C, D, E, F, G, H, I, J, K, L));

/// A scalar-like value or native tuple whose read-only shape already is flat.
#[doc(hidden)]
pub trait ReadRow: ReadLayout<ReadLeaves: ReadFlatLeaves<Item = Self>> {}

impl<Item> ReadRow for Item
where
    Item: ReadLayout,
    Item::ReadLeaves: ReadFlatLeaves<Item = Item>,
{
}

/// A storable scalar or native tuple whose public shape already is flat.
#[doc(hidden)]
pub trait FlatRow: StorageLayout<StorageLeaves: FlatLeaves<Item = Self>> {}

impl<Item> FlatRow for Item
where
    Item: StorageLayout,
    Item::StorageLeaves: FlatLeaves<Item = Item>,
{
}

/// The flat logical row obtained by concatenating two flat row schemas.
#[doc(hidden)]
pub type JoinedRow<Left, Right> = <<<Left as StorageLayout>::StorageLeaves as Concat<
    <Right as StorageLayout>::StorageLeaves,
>>::Output as FlatLeaves>::Item;

/// The flat logical row obtained by concatenating two read-only row schemas.
#[doc(hidden)]
pub type JoinedReadRow<Left, Right> = <<<Left as ReadLayout>::ReadLeaves as Concat<
    <Right as ReadLayout>::ReadLeaves,
>>::Output as ReadFlatLeaves>::Item;

/// Device-side layout marker for one physical leaf.
#[doc(hidden)]
pub struct ScalarLayout<T> {
    _marker: core::marker::PhantomData<fn() -> T>,
}

/// Device-side layout marker for a semantic pair.
#[doc(hidden)]
pub struct PairLayout<Left, Right> {
    _marker: core::marker::PhantomData<fn() -> (Left, Right)>,
}

/// Device-side layout marker for one native, flat tuple.
#[doc(hidden)]
pub struct FlatTupleLayout<Item> {
    _marker: core::marker::PhantomData<fn() -> Item>,
}

/// Decomposes a semantic value into its ordered physical leaves in device code.
#[doc(hidden)]
#[cubecl::cube]
pub trait Decompose<Item: CubeType>: 'static + Send + Sync {
    type Leaves: CubeType;
    fn decompose(item: Item) -> Self::Leaves;
}

#[doc(hidden)]
#[cubecl::cube]
pub trait Recompose<Item: CubeType>: 'static + Send + Sync {
    type Leaves: CubeType;
    fn recompose(leaves: Self::Leaves) -> Item;
}

/// Selects one of two physical leaf lists without assigning a composite
/// semantic value in a device branch.
#[doc(hidden)]
#[cubecl::cube]
pub trait SelectLeaves: CubeType + Sized {
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self;
}

/// Mutable register cells for an ordered physical leaf list.
#[doc(hidden)]
#[cubecl::cube]
pub trait MutableLeaves: CubeType + Sized {
    type Cells: CubeType;

    fn into_cells(self) -> Self::Cells;
    fn read(cells: &Self::Cells) -> Self;
    fn store(cells: &Self::Cells, value: Self);
}

/// Applies one plane shuffle to every physical leaf.
#[doc(hidden)]
#[cubecl::cube]
pub trait PlaneShuffleLeaves: CubeType + Sized {
    fn shuffle_leaves_down(value: Self, offset: u32) -> Self;
    fn shuffle_leaves_up(value: Self, offset: u32) -> Self;
}

/// Workgroup-local storage with the same heterogeneous leaf shape as a value.
/// Each implementation allocates exactly one shared array per real leaf.
#[derive(CubeType)]
pub struct SharedLast<T: CubePrimitive + cubecl::frontend::Scalar> {
    value: Shared<[T]>,
}

#[derive(CubeType)]
pub struct SharedMore<Head: CubePrimitive + cubecl::frontend::Scalar, Tail: CubeType> {
    head: Shared<[Head]>,
    tail: Tail,
}

#[cubecl::cube]
pub trait SharedLeaves: CubeType + Sized {
    type Shared: CubeType;

    fn new_shared(#[comptime] len: usize) -> Self::Shared;
    fn load_shared(shared: &Self::Shared, index: usize) -> Self;
    fn store_shared(self, shared: &mut Self::Shared, index: usize);
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> SharedLeaves for Last<T> {
    type Shared = SharedLast<T>;

    fn new_shared(#[comptime] len: usize) -> Self::Shared {
        SharedLast::<T> {
            value: Shared::<[T]>::new_slice(len),
        }
    }

    fn load_shared(shared: &Self::Shared, index: usize) -> Self {
        Last::<T> {
            value: shared.value[index],
        }
    }

    fn store_shared(self, shared: &mut Self::Shared, index: usize) {
        shared.value[index] = self.value;
    }
}

#[cubecl::cube]
impl<Head, Tail> SharedLeaves for More<Head, Tail>
where
    Head: CubePrimitive + cubecl::frontend::Scalar,
    Tail: SharedLeaves,
{
    type Shared = SharedMore<Head, Tail::Shared>;

    fn new_shared(#[comptime] len: usize) -> Self::Shared {
        SharedMore::<Head, Tail::Shared> {
            head: Shared::<[Head]>::new_slice(len),
            tail: Tail::new_shared(len),
        }
    }

    fn load_shared(shared: &Self::Shared, index: usize) -> Self {
        More::<Head, Tail> {
            head: shared.head[index],
            tail: Tail::load_shared(&shared.tail, index),
        }
    }

    fn store_shared(self, shared: &mut Self::Shared, index: usize) {
        shared.head[index] = self.head;
        self.tail.store_shared(&mut shared.tail, index);
    }
}

#[cubecl::cube]
fn shuffle_primitive<T: CubePrimitive + cubecl::frontend::Scalar>(value: T, offset: u32) -> T {
    plane_shuffle_down(value, offset)
}

#[cubecl::cube]
fn shuffle_primitive_up<T: CubePrimitive + cubecl::frontend::Scalar>(value: T, offset: u32) -> T {
    plane_shuffle_up(value, offset)
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> SelectLeaves for Last<T> {
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self {
        Last::<T> {
            value: if condition {
                if_true.value
            } else {
                if_false.value
            },
        }
    }
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> MutableLeaves for Last<T> {
    type Cells = Last<RuntimeCell<T>>;

    fn into_cells(self) -> Self::Cells {
        Last::new(RuntimeCell::<T>::new(self.value))
    }

    fn read(cells: &Self::Cells) -> Self {
        Last::<T> {
            value: cells.value.read(),
        }
    }

    fn store(cells: &Self::Cells, value: Self) {
        cells.value.store(value.value);
    }
}

#[cubecl::cube]
impl<T: CubePrimitive + cubecl::frontend::Scalar> PlaneShuffleLeaves for Last<T> {
    fn shuffle_leaves_down(value: Self, offset: u32) -> Self {
        Last::<T> {
            value: shuffle_primitive::<T>(value.value, offset),
        }
    }

    fn shuffle_leaves_up(value: Self, offset: u32) -> Self {
        Last::<T> {
            value: shuffle_primitive_up::<T>(value.value, offset),
        }
    }
}

#[cubecl::cube]
impl<Head, Tail> SelectLeaves for More<Head, Tail>
where
    Head: CubePrimitive + cubecl::frontend::Scalar,
    Tail: SelectLeaves,
{
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self {
        More::<Head, Tail> {
            head: if condition {
                if_true.head
            } else {
                if_false.head
            },
            tail: Tail::select(condition, if_true.tail, if_false.tail),
        }
    }
}

#[cubecl::cube]
impl<Head, Tail> MutableLeaves for More<Head, Tail>
where
    Head: CubePrimitive + cubecl::frontend::Scalar,
    Tail: MutableLeaves,
{
    type Cells = More<RuntimeCell<Head>, Tail::Cells>;

    fn into_cells(self) -> Self::Cells {
        More::new(RuntimeCell::<Head>::new(self.head), self.tail.into_cells())
    }

    fn read(cells: &Self::Cells) -> Self {
        More::<Head, Tail> {
            head: cells.head.read(),
            tail: Tail::read(&cells.tail),
        }
    }

    fn store(cells: &Self::Cells, value: Self) {
        cells.head.store(value.head);
        Tail::store(&cells.tail, value.tail);
    }
}

#[cubecl::cube]
impl<Head, Tail> PlaneShuffleLeaves for More<Head, Tail>
where
    Head: CubePrimitive + cubecl::frontend::Scalar,
    Tail: PlaneShuffleLeaves,
{
    fn shuffle_leaves_down(value: Self, offset: u32) -> Self {
        More::<Head, Tail> {
            head: shuffle_primitive::<Head>(value.head, offset),
            tail: Tail::shuffle_leaves_down(value.tail, offset),
        }
    }

    fn shuffle_leaves_up(value: Self, offset: u32) -> Self {
        More::<Head, Tail> {
            head: shuffle_primitive_up::<Head>(value.head, offset),
            tail: Tail::shuffle_leaves_up(value.tail, offset),
        }
    }
}

#[cubecl::cube]
impl<T: CubeType + 'static> Decompose<T> for ScalarLayout<T> {
    type Leaves = Last<T>;

    fn decompose(item: T) -> Self::Leaves {
        Last::<T> { value: item }
    }
}

#[cubecl::cube]
impl<T: CubeType + 'static> Recompose<T> for ScalarLayout<T> {
    type Leaves = Last<T>;
    fn recompose(leaves: Self::Leaves) -> T {
        leaves.value
    }
}

#[cubecl::cube]
impl<LeftItem, RightItem, Left, Right> Decompose<(LeftItem, RightItem)> for PairLayout<Left, Right>
where
    LeftItem: CubeType + 'static,
    RightItem: CubeType + 'static,
    Left: Decompose<LeftItem>,
    Right: Decompose<RightItem>,
    Left::Leaves: Concat<Right::Leaves>,
{
    type Leaves = <Left::Leaves as Concat<Right::Leaves>>::Output;

    fn decompose(item: (LeftItem, RightItem)) -> Self::Leaves {
        Left::decompose(item.0).concat(Right::decompose(item.1))
    }
}

#[cubecl::cube]
impl<LeftItem, RightItem, Left, Right> Recompose<(LeftItem, RightItem)> for PairLayout<Left, Right>
where
    LeftItem: CubeType + 'static,
    RightItem: CubeType + 'static,
    Left: Recompose<LeftItem>,
    Right: Recompose<RightItem>,
    Left::Leaves: Concat<Right::Leaves>,
{
    type Leaves = <Left::Leaves as Concat<Right::Leaves>>::Output;

    fn recompose(leaves: Self::Leaves) -> (LeftItem, RightItem) {
        let (left, right) = Left::Leaves::split(leaves);
        (Left::recompose(left), Right::recompose(right))
    }
}

macro_rules! define_store_leaves {
    (
        $trait_name:ident;
        $leaves:ty;
        $( $leaf:ident : $out:ident ),+;
        $this:ident, $offsets:ident, $index:ident;
        $body:block
    ) => {
        #[doc(hidden)]
        #[cubecl::cube]
        #[allow(unused_mut)]
        pub trait $trait_name<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+>: CubeType {
            fn store(
                self,
                $( $out: &mut [$leaf], )+
                offsets: &[u32],
                index: usize,
            );
        }

        #[cubecl::cube]
        #[allow(unused_mut)]
        impl<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn store(
                self,
                $( $out: &mut [$leaf], )+
                $offsets: &[u32],
                $index: usize,
            ) {
                let $this = self;
                $body
            }
        }
    };
}

define_store_leaves!(StoreLeaves1; Last<L0>; L0: out0; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.value;
});
define_store_leaves!(StoreLeaves2; More<L0, Last<L1>>; L0: out0, L1: out1; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.value;
});
define_store_leaves!(StoreLeaves3; More<L0, More<L1, Last<L2>>>; L0: out0, L1: out1, L2: out2; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.value;
});
define_store_leaves!(StoreLeaves4; More<L0, More<L1, More<L2, Last<L3>>>>; L0: out0, L1: out1, L2: out2, L3: out3; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves5; More<L0, More<L1, More<L2, More<L3, Last<L4>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves6; More<L0, More<L1, More<L2, More<L3, More<L4, Last<L5>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves7; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, Last<L6>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.value;
});

define_store_leaves!(StoreLeaves8; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, Last<L7>>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6, L7: out7; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.head;
    out7[offsets[7] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves9; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, Last<L8>>>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6, L7: out7, L8: out8; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.head;
    out7[offsets[7] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.head;
    out8[offsets[8] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves10; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, Last<L9>>>>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6, L7: out7, L8: out8, L9: out9; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.head;
    out7[offsets[7] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.head;
    out8[offsets[8] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out9[offsets[9] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves11; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, Last<L10>>>>>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6, L7: out7, L8: out8, L9: out9, L10: out10; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.head;
    out7[offsets[7] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.head;
    out8[offsets[8] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out9[offsets[9] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out10[offsets[10] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves12; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, More<L10, Last<L11>>>>>>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6, L7: out7, L8: out8, L9: out9, L10: out10, L11: out11; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.head;
    out7[offsets[7] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.head;
    out8[offsets[8] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out9[offsets[9] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out10[offsets[10] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.head;
    out11[offsets[11] as usize + index] = this.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.tail.value;
});

/// Stores an actual one-to-twelve-leaf value through a fixed twelve-buffer
/// kernel ABI. Implementations intentionally ignore inactive output buffers.
#[doc(hidden)]
#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
pub trait StorePadded12: CubeType {
    type O0: crate::core::value::MStorageElement;
    type O1: crate::core::value::MStorageElement;
    type O2: crate::core::value::MStorageElement;
    type O3: crate::core::value::MStorageElement;
    type O4: crate::core::value::MStorageElement;
    type O5: crate::core::value::MStorageElement;
    type O6: crate::core::value::MStorageElement;
    type O7: crate::core::value::MStorageElement;
    type O8: crate::core::value::MStorageElement;
    type O9: crate::core::value::MStorageElement;
    type O10: crate::core::value::MStorageElement;
    type O11: crate::core::value::MStorageElement;

    fn store_padded(
        self,
        out0: &mut [Self::O0],
        out1: &mut [Self::O1],
        out2: &mut [Self::O2],
        out3: &mut [Self::O3],
        out4: &mut [Self::O4],
        out5: &mut [Self::O5],
        out6: &mut [Self::O6],
        out7: &mut [Self::O7],
        out8: &mut [Self::O8],
        out9: &mut [Self::O9],
        out10: &mut [Self::O10],
        out11: &mut [Self::O11],
        offsets: &[u32],
        index: usize,
    );
}

macro_rules! impl_store_padded12 {
    (
        $leaves:ty, $store_trait:ident;
        [$out0:ident, $out1:ident, $out2:ident, $out3:ident, $out4:ident, $out5:ident,
         $out6:ident, $out7:ident, $out8:ident, $out9:ident, $out10:ident, $out11:ident];
        [$a0:ty, $a1:ty, $a2:ty, $a3:ty, $a4:ty, $a5:ty,
         $a6:ty, $a7:ty, $a8:ty, $a9:ty, $a10:ty, $a11:ty];
        $($active_ty:ident : $active_out:ident),+ $(,)?
    ) => {
        #[cubecl::cube]
        #[allow(unused_variables, clippy::too_many_arguments)]
        impl<$($active_ty),+> StorePadded12 for $leaves
        where
            $($active_ty: crate::core::value::MStorageElement,)+
        {
            type O0 = $a0;
            type O1 = $a1;
            type O2 = $a2;
            type O3 = $a3;
            type O4 = $a4;
            type O5 = $a5;
            type O6 = $a6;
            type O7 = $a7;
            type O8 = $a8;
            type O9 = $a9;
            type O10 = $a10;
            type O11 = $a11;

            #[allow(unused_variables)]
            fn store_padded(
                self,
                $out0: &mut [$a0],
                $out1: &mut [$a1],
                $out2: &mut [$a2],
                $out3: &mut [$a3],
                $out4: &mut [$a4],
                $out5: &mut [$a5],
                $out6: &mut [$a6],
                $out7: &mut [$a7],
                $out8: &mut [$a8],
                $out9: &mut [$a9],
                $out10: &mut [$a10],
                $out11: &mut [$a11],
                offsets: &[u32],
                index: usize,
            ) {
                self.store($($active_out,)+ offsets, index);
            }
        }
    };
}

mod store_padded_impls {
    use super::*;

    impl_store_padded12!(Last<O0>, StoreLeaves1; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0: out0);
    impl_store_padded12!(More<O0, Last<O1>>, StoreLeaves2; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0: out0, O1: out1);
    impl_store_padded12!(More<O0, More<O1, Last<O2>>>, StoreLeaves3; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2);
    impl_store_padded12!(More<O0, More<O1, More<O2, Last<O3>>>>, StoreLeaves4; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,u32,u32,u32,u32,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, Last<O4>>>>>, StoreLeaves5; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,u32,u32,u32,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, Last<O5>>>>>>, StoreLeaves6; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,u32,u32,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, Last<O6>>>>>>>, StoreLeaves7; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,u32,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, More<O6, Last<O7>>>>>>>>, StoreLeaves8; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,O7,u32,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6, O7: out7);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, More<O6, More<O7, Last<O8>>>>>>>>>, StoreLeaves9; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,u32,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6, O7: out7, O8: out8);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, More<O6, More<O7, More<O8, Last<O9>>>>>>>>>>, StoreLeaves10; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,u32,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6, O7: out7, O8: out8, O9: out9);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, More<O6, More<O7, More<O8, More<O9, Last<O10>>>>>>>>>>>, StoreLeaves11; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,u32]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6, O7: out7, O8: out8, O9: out9, O10: out10);
    impl_store_padded12!(More<O0, More<O1, More<O2, More<O3, More<O4, More<O5, More<O6, More<O7, More<O8, More<O9, More<O10, Last<O11>>>>>>>>>>>>, StoreLeaves12; [out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]; O0: out0, O1: out1, O2: out2, O3: out3, O4: out4, O5: out5, O6: out6, O7: out7, O8: out8, O9: out9, O10: out10, O11: out11);
}

macro_rules! define_load_leaves {
    ($trait_name:ident; $leaves:ty; $( $leaf:ident:$input:ident ),+; $offsets:ident,$index:ident; $body:expr) => {
        #[doc(hidden)]
        #[cubecl::cube]
        pub trait $trait_name<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+>: CubeType {
            fn load($( $input: &[$leaf], )+ offsets: &[u32], index: usize) -> Self;
        }

        #[cubecl::cube]
        impl<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn load($( $input: &[$leaf], )+ $offsets: &[u32], $index: usize) -> Self {
                $body
            }
        }
    };
}

define_load_leaves!(LoadLeaves1; Last<L0>; L0:in0; offsets,index;
    Last::new(in0[offsets[0] as usize + index])
);
define_load_leaves!(LoadLeaves2; More<L0,Last<L1>>; L0:in0,L1:in1; offsets,index;
    More::new(in0[offsets[0] as usize + index], Last::new(in1[offsets[1] as usize + index]))
);
define_load_leaves!(LoadLeaves3; More<L0,More<L1,Last<L2>>>; L0:in0,L1:in1,L2:in2; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], Last::new(in2[offsets[2] as usize + index])))
);
define_load_leaves!(LoadLeaves4; More<L0,More<L1,More<L2,Last<L3>>>>; L0:in0,L1:in1,L2:in2,L3:in3; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], Last::new(in3[offsets[3] as usize + index]))))
);
define_load_leaves!(LoadLeaves5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], Last::new(in4[offsets[4] as usize + index])))))
);
define_load_leaves!(LoadLeaves6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], Last::new(in5[offsets[5] as usize + index]))))))
);
define_load_leaves!(LoadLeaves7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], Last::new(in6[offsets[6] as usize + index])))))))
);

define_load_leaves!(LoadLeaves8; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, Last<L7>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], Last::new(in7[offsets[7] as usize + index]))))))))
);
define_load_leaves!(LoadLeaves9; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, Last<L8>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], Last::new(in8[offsets[8] as usize + index])))))))))
);
define_load_leaves!(LoadLeaves10; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, Last<L9>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], Last::new(in9[offsets[9] as usize + index]))))))))))
);
define_load_leaves!(LoadLeaves11; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, Last<L10>>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9,L10:in10; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], More::new(in9[offsets[9] as usize + index], Last::new(in10[offsets[10] as usize + index])))))))))))
);
define_load_leaves!(LoadLeaves12; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, More<L10, Last<L11>>>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9,L10:in10,L11:in11; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], More::new(in9[offsets[9] as usize + index], More::new(in10[offsets[10] as usize + index], Last::new(in11[offsets[11] as usize + index]))))))))))))
);

macro_rules! define_load_mut_leaves {
    ($trait_name:ident; $leaves:ty; $( $leaf:ident:$input:ident ),+; $offsets:ident,$index:ident; $body:expr) => {
        #[doc(hidden)]
        #[cubecl::cube]
        pub trait $trait_name<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+>: CubeType {
            fn load_mut($( $input: &mut [$leaf], )+ offsets: &[u32], index: usize) -> Self;
        }

        #[cubecl::cube]
        impl<$( $leaf: CubePrimitive + cubecl::frontend::Scalar ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn load_mut($( $input: &mut [$leaf], )+ $offsets: &[u32], $index: usize) -> Self {
                $body
            }
        }
    };
}

define_load_mut_leaves!(LoadMutLeaves1; Last<L0>; L0:in0; offsets,index;
    Last::new(in0[offsets[0] as usize + index])
);
define_load_mut_leaves!(LoadMutLeaves2; More<L0,Last<L1>>; L0:in0,L1:in1; offsets,index;
    More::new(in0[offsets[0] as usize + index], Last::new(in1[offsets[1] as usize + index]))
);
define_load_mut_leaves!(LoadMutLeaves3; More<L0,More<L1,Last<L2>>>; L0:in0,L1:in1,L2:in2; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], Last::new(in2[offsets[2] as usize + index])))
);
define_load_mut_leaves!(LoadMutLeaves4; More<L0,More<L1,More<L2,Last<L3>>>>; L0:in0,L1:in1,L2:in2,L3:in3; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], Last::new(in3[offsets[3] as usize + index]))))
);
define_load_mut_leaves!(LoadMutLeaves5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], Last::new(in4[offsets[4] as usize + index])))))
);
define_load_mut_leaves!(LoadMutLeaves6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], Last::new(in5[offsets[5] as usize + index]))))))
);
define_load_mut_leaves!(LoadMutLeaves7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], Last::new(in6[offsets[6] as usize + index])))))))
);

define_load_mut_leaves!(LoadMutLeaves8; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, Last<L7>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], Last::new(in7[offsets[7] as usize + index]))))))))
);
define_load_mut_leaves!(LoadMutLeaves9; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, Last<L8>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], Last::new(in8[offsets[8] as usize + index])))))))))
);
define_load_mut_leaves!(LoadMutLeaves10; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, Last<L9>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], Last::new(in9[offsets[9] as usize + index]))))))))))
);
define_load_mut_leaves!(LoadMutLeaves11; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, Last<L10>>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9,L10:in10; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], More::new(in9[offsets[9] as usize + index], Last::new(in10[offsets[10] as usize + index])))))))))))
);
define_load_mut_leaves!(LoadMutLeaves12; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, More<L6, More<L7, More<L8, More<L9, More<L10, Last<L11>>>>>>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6,L7:in7,L8:in8,L9:in9,L10:in10,L11:in11; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], More::new(in6[offsets[6] as usize + index], More::new(in7[offsets[7] as usize + index], More::new(in8[offsets[8] as usize + index], More::new(in9[offsets[9] as usize + index], More::new(in10[offsets[10] as usize + index], Last::new(in11[offsets[11] as usize + index]))))))))))))
);

/// Loads an actual one-to-twelve-leaf value through the fixed twelve-buffer
/// kernel ABI. Inactive buffers are never read.
#[doc(hidden)]
#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
pub trait LoadPadded12: StorePadded12 {
    fn load_padded(
        input0: &[Self::O0],
        input1: &[Self::O1],
        input2: &[Self::O2],
        input3: &[Self::O3],
        input4: &[Self::O4],
        input5: &[Self::O5],
        input6: &[Self::O6],
        input7: &[Self::O7],
        input8: &[Self::O8],
        input9: &[Self::O9],
        input10: &[Self::O10],
        input11: &[Self::O11],
        offsets: &[u32],
        index: usize,
    ) -> Self;
}

macro_rules! impl_load_padded12 {
    (
        $leaves:ty, $load_trait:ident;
        [$input0:ident, $input1:ident, $input2:ident, $input3:ident, $input4:ident, $input5:ident,
         $input6:ident, $input7:ident, $input8:ident, $input9:ident, $input10:ident, $input11:ident];
        [$a0:ty, $a1:ty, $a2:ty, $a3:ty, $a4:ty, $a5:ty,
         $a6:ty, $a7:ty, $a8:ty, $a9:ty, $a10:ty, $a11:ty];
        $($active_ty:ident : $active_input:ident),+ $(,)?
    ) => {
        #[cubecl::cube]
        #[allow(unused_variables, clippy::too_many_arguments)]
        impl<$($active_ty),+> LoadPadded12 for $leaves
        where
            $($active_ty: crate::core::value::MStorageElement,)+
        {
            #[allow(unused_variables)]
            fn load_padded(
                $input0: &[$a0],
                $input1: &[$a1],
                $input2: &[$a2],
                $input3: &[$a3],
                $input4: &[$a4],
                $input5: &[$a5],
                $input6: &[$a6],
                $input7: &[$a7],
                $input8: &[$a8],
                $input9: &[$a9],
                $input10: &[$a10],
                $input11: &[$a11],
                offsets: &[u32],
                index: usize,
            ) -> Self {
                <Self as $load_trait<$($active_ty),+>>::load(
                    $($active_input,)+ offsets, index,
                )
            }
        }
    };
}

mod load_padded_impls {
    use super::*;

    impl_load_padded12!(Last<O0>, LoadLeaves1; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0);
    impl_load_padded12!(More<O0,Last<O1>>, LoadLeaves2; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1);
    impl_load_padded12!(More<O0,More<O1,Last<O2>>>, LoadLeaves3; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2);
    impl_load_padded12!(More<O0,More<O1,More<O2,Last<O3>>>>, LoadLeaves4; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>, LoadLeaves5; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>, LoadLeaves6; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>, LoadLeaves7; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,Last<O7>>>>>>>>, LoadLeaves8; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,Last<O8>>>>>>>>>, LoadLeaves9; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,Last<O9>>>>>>>>>>, LoadLeaves10; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,Last<O10>>>>>>>>>>>, LoadLeaves11; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9,O10:in10);
    impl_load_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,More<O10,Last<O11>>>>>>>>>>>>, LoadLeaves12; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9,O10:in10,O11:in11);
}

/// Loads an actual one-to-twelve-leaf value from mutable buffers through the
/// fixed twelve-buffer kernel ABI. Inactive buffers are never read.
#[doc(hidden)]
#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
pub trait LoadMutPadded12: StorePadded12 {
    fn load_mut_padded(
        input0: &mut [Self::O0],
        input1: &mut [Self::O1],
        input2: &mut [Self::O2],
        input3: &mut [Self::O3],
        input4: &mut [Self::O4],
        input5: &mut [Self::O5],
        input6: &mut [Self::O6],
        input7: &mut [Self::O7],
        input8: &mut [Self::O8],
        input9: &mut [Self::O9],
        input10: &mut [Self::O10],
        input11: &mut [Self::O11],
        offsets: &[u32],
        index: usize,
    ) -> Self;
}

macro_rules! impl_load_mut_padded12 {
    (
        $leaves:ty, $load_trait:ident;
        [$input0:ident, $input1:ident, $input2:ident, $input3:ident, $input4:ident, $input5:ident,
         $input6:ident, $input7:ident, $input8:ident, $input9:ident, $input10:ident, $input11:ident];
        [$a0:ty, $a1:ty, $a2:ty, $a3:ty, $a4:ty, $a5:ty,
         $a6:ty, $a7:ty, $a8:ty, $a9:ty, $a10:ty, $a11:ty];
        $($active_ty:ident : $active_input:ident),+ $(,)?
    ) => {
        #[cubecl::cube]
        #[allow(unused_variables, clippy::too_many_arguments)]
        impl<$($active_ty),+> LoadMutPadded12 for $leaves
        where
            $($active_ty: crate::core::value::MStorageElement,)+
        {
            #[allow(unused_variables)]
            fn load_mut_padded(
                $input0: &mut [$a0],
                $input1: &mut [$a1],
                $input2: &mut [$a2],
                $input3: &mut [$a3],
                $input4: &mut [$a4],
                $input5: &mut [$a5],
                $input6: &mut [$a6],
                $input7: &mut [$a7],
                $input8: &mut [$a8],
                $input9: &mut [$a9],
                $input10: &mut [$a10],
                $input11: &mut [$a11],
                offsets: &[u32],
                index: usize,
            ) -> Self {
                <Self as $load_trait<$($active_ty),+>>::load_mut(
                    $($active_input,)+ offsets, index,
                )
            }
        }
    };
}

mod load_mut_padded_impls {
    use super::*;

    impl_load_mut_padded12!(Last<O0>, LoadMutLeaves1; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0);
    impl_load_mut_padded12!(More<O0,Last<O1>>, LoadMutLeaves2; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1);
    impl_load_mut_padded12!(More<O0,More<O1,Last<O2>>>, LoadMutLeaves3; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,u32,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,Last<O3>>>>, LoadMutLeaves4; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,u32,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>, LoadMutLeaves5; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,u32,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>, LoadMutLeaves6; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,u32,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>, LoadMutLeaves7; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,u32,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,Last<O7>>>>>>>>, LoadMutLeaves8; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,u32,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,Last<O8>>>>>>>>>, LoadMutLeaves9; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,u32,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,Last<O9>>>>>>>>>>, LoadMutLeaves10; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,u32,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,Last<O10>>>>>>>>>>>, LoadMutLeaves11; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,u32]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9,O10:in10);
    impl_load_mut_padded12!(More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,More<O6,More<O7,More<O8,More<O9,More<O10,Last<O11>>>>>>>>>>>>, LoadMutLeaves12; [in0,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]; O0:in0,O1:in1,O2:in2,O3:in3,O4:in4,O5:in5,O6:in6,O7:in7,O8:in8,O9:in9,O10:in10,O11:in11);
}

mod private {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    type Flat3 = (u32, f32, i16);
    type Supported12 = (u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_accepts_flat_rows(source: Flat3) -> Flat3 {
        (source.0, source.1, source.2)
    }

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_accepts_recursive_leaf_concat(
        a: u32,
        b: f32,
        c: i16,
    ) -> More<u32, More<f32, Last<i16>>> {
        let left = Last::new(a);
        let right = More::new(b, Last::new(c));
        left.concat(right)
    }

    #[test]
    fn layout_arity_is_derived_from_the_flat_row() {
        fn assert_arity<T, A>()
        where
            T: StorageLayout<StorageArity = A>,
            A: StorageArity,
        {
        }

        assert_arity::<u32, S1>();
        assert_arity::<Flat3, S3>();
        assert_arity::<(u8, u16, u32, f32), S4>();
    }

    assert_impl_all!(Flat3: FlatRow);
    assert_impl_all!(Supported12: FlatRow);
}
