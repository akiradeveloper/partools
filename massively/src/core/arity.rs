//! Type-level read arity and kernel dispatch keys.

/// Dispatch key combining a kernel's read and storage arities.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch<Read, Storage>(core::marker::PhantomData<fn() -> (Read, Storage)>);

/// A supported number of physical leaves read by an expression.
pub trait ReadArity: private::Sealed + 'static {}

macro_rules! define_arities {
    ($($arity:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Read arity marker `", stringify!($arity), "`.")]
            #[derive(Clone, Copy, Debug, Eq, PartialEq)]
            pub struct $arity;

            impl ReadArity for $arity {}
            impl private::Sealed for $arity {}
        )+
    };
}

define_arities!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13);

/// Type-level addition for read arities whose sum is at most thirteen.
///
/// There is deliberately no implementation for sums greater than thirteen.
pub trait AddArity<Rhs: ReadArity>: ReadArity {
    type Output: ReadArity;
}

macro_rules! impl_add_arity {
    ($lhs:ty, $rhs:ty => $output:ty) => {
        impl AddArity<$rhs> for $lhs {
            type Output = $output;
        }
    };
}

impl_add_arity!(A1, A1 => A2);
impl_add_arity!(A1, A2 => A3);
impl_add_arity!(A1, A3 => A4);
impl_add_arity!(A1, A4 => A5);
impl_add_arity!(A1, A5 => A6);
impl_add_arity!(A1, A6 => A7);
impl_add_arity!(A1, A7 => A8);
impl_add_arity!(A1, A8 => A9);
impl_add_arity!(A1, A9 => A10);
impl_add_arity!(A1, A10 => A11);
impl_add_arity!(A1, A11 => A12);
impl_add_arity!(A1, A12 => A13);
impl_add_arity!(A2, A1 => A3);
impl_add_arity!(A2, A2 => A4);
impl_add_arity!(A2, A3 => A5);
impl_add_arity!(A2, A4 => A6);
impl_add_arity!(A2, A5 => A7);
impl_add_arity!(A2, A6 => A8);
impl_add_arity!(A2, A7 => A9);
impl_add_arity!(A2, A8 => A10);
impl_add_arity!(A2, A9 => A11);
impl_add_arity!(A2, A10 => A12);
impl_add_arity!(A2, A11 => A13);
impl_add_arity!(A3, A1 => A4);
impl_add_arity!(A3, A2 => A5);
impl_add_arity!(A3, A3 => A6);
impl_add_arity!(A3, A4 => A7);
impl_add_arity!(A3, A5 => A8);
impl_add_arity!(A3, A6 => A9);
impl_add_arity!(A3, A7 => A10);
impl_add_arity!(A3, A8 => A11);
impl_add_arity!(A3, A9 => A12);
impl_add_arity!(A3, A10 => A13);
impl_add_arity!(A4, A1 => A5);
impl_add_arity!(A4, A2 => A6);
impl_add_arity!(A4, A3 => A7);
impl_add_arity!(A4, A4 => A8);
impl_add_arity!(A4, A5 => A9);
impl_add_arity!(A4, A6 => A10);
impl_add_arity!(A4, A7 => A11);
impl_add_arity!(A4, A8 => A12);
impl_add_arity!(A4, A9 => A13);
impl_add_arity!(A5, A1 => A6);
impl_add_arity!(A5, A2 => A7);
impl_add_arity!(A5, A3 => A8);
impl_add_arity!(A5, A4 => A9);
impl_add_arity!(A5, A5 => A10);
impl_add_arity!(A5, A6 => A11);
impl_add_arity!(A5, A7 => A12);
impl_add_arity!(A5, A8 => A13);
impl_add_arity!(A6, A1 => A7);
impl_add_arity!(A6, A2 => A8);
impl_add_arity!(A6, A3 => A9);
impl_add_arity!(A6, A4 => A10);
impl_add_arity!(A6, A5 => A11);
impl_add_arity!(A6, A6 => A12);
impl_add_arity!(A6, A7 => A13);
impl_add_arity!(A7, A1 => A8);
impl_add_arity!(A7, A2 => A9);
impl_add_arity!(A7, A3 => A10);
impl_add_arity!(A7, A4 => A11);
impl_add_arity!(A7, A5 => A12);
impl_add_arity!(A7, A6 => A13);
impl_add_arity!(A8, A1 => A9);
impl_add_arity!(A8, A2 => A10);
impl_add_arity!(A8, A3 => A11);
impl_add_arity!(A8, A4 => A12);
impl_add_arity!(A8, A5 => A13);
impl_add_arity!(A9, A1 => A10);
impl_add_arity!(A9, A2 => A11);
impl_add_arity!(A9, A3 => A12);
impl_add_arity!(A9, A4 => A13);
impl_add_arity!(A10, A1 => A11);
impl_add_arity!(A10, A2 => A12);
impl_add_arity!(A10, A3 => A13);
impl_add_arity!(A11, A1 => A12);
impl_add_arity!(A11, A2 => A13);
impl_add_arity!(A12, A1 => A13);

mod private {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    assert_impl_all!(A1: AddArity<A12, Output = A13>);
    assert_impl_all!(A6: AddArity<A7, Output = A13>);
    assert_not_impl_any!(A13: AddArity<A1>);
    assert_not_impl_any!(A12: AddArity<A2>);
}
