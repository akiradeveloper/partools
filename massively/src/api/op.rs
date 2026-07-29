//! Compile-time operations embedded in GPU algorithms and lazy expressions.

pub use crate::core::op::{
    BinaryPredicateOp, ExpandOp, Identity, PredicateOp, ReductionOp, UnaryOp,
};
pub(crate) use crate::core::op::{IndexedBinaryOp, IndexedUnaryOp};
