//! Writable Python buffers whose lifetime is one native arena slot.

use crate::arena::{ArenaAllocator, DeliveryView, SlotRef};
use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{CString, c_int, c_void};
use std::ptr::{self, NonNull};

const OWNER: u32 = 0;

#[pyclass(name = "_NativeSlot", unsendable)]
pub(super) struct NativeSlot {
    allocator: ArenaAllocator,
    slot: SlotRef,
    pointer: NonNull<u8>,
    capacity: usize,
    delivery: Option<DeliveryView>,
}

impl NativeSlot {
    pub(super) fn reserve(allocator: ArenaAllocator, required: usize) -> PyResult<Self> {
        let slot = allocator.reserve(required, OWNER).map_err(runtime_error)?;
        let result = unsafe { allocator.writing_ptr_len(slot, OWNER) };
        let (pointer, capacity) = match result {
            Ok(value) => value,
            Err(error) => {
                let _ = allocator.cancel(slot, OWNER);
                return Err(runtime_error(error));
            }
        };
        Ok(Self {
            allocator,
            slot,
            pointer,
            capacity,
            delivery: None,
        })
    }
}

#[pymethods]
impl NativeSlot {
    fn publish(&mut self, length: usize) -> PyResult<()> {
        if self.delivery.is_some() {
            return Err(PyRuntimeError::new_err(
                "native arena slot is already published",
            ));
        }
        self.allocator
            .publish(self.slot, OWNER, length)
            .map_err(runtime_error)?;
        self.delivery = self
            .allocator
            .deliver(
                self.slot,
                std::num::NonZeroU32::new(1).expect("one view is nonzero"),
            )
            .map_err(runtime_error)?
            .pop();
        Ok(())
    }

    #[getter]
    const fn capacity(&self) -> usize {
        self.capacity
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("buffer view is null"));
        }
        let owner = slf.into_any();
        let borrowed = owner.cast::<Self>()?.borrow();
        unsafe {
            (*view).obj = owner.into_ptr();
            (*view).buf = borrowed.pointer.as_ptr().cast::<c_void>();
            (*view).len = borrowed.capacity as isize;
            (*view).readonly = 0;
            (*view).itemsize = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                CString::new("B")
                    .expect("static format has no nul")
                    .into_raw()
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                &mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        unsafe {
            if !view.is_null() && !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = ptr::null_mut();
            }
        }
    }
}

impl Drop for NativeSlot {
    fn drop(&mut self) {
        if self.delivery.is_none() {
            let _ = self.allocator.cancel(self.slot, OWNER);
        }
    }
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
