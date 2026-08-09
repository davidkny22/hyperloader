//! Default collation mirrors torch's pinned recursive type contract.

mod container;
mod dispatch;
mod kind;
mod tensor;

pub use dispatch::default_collate;
