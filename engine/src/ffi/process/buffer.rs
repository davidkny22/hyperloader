//! Python buffer ownership for one delivered arena slot.

use crate::arena::DeliveryView;
use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::ffi::{CString, c_int, c_void};
use std::ptr::{self, NonNull};

/// One writable Python buffer that keeps its arena delivery slot alive.
#[pyclass(name = "_ArenaBuffer", unsendable)]
pub(super) struct ArenaBuffer {
    _view: DeliveryView,
    pointer: NonNull<u8>,
    length: usize,
}

impl ArenaBuffer {
    pub(super) fn new(view: DeliveryView) -> Result<Self, crate::arena::ArenaError> {
        // SAFETY: this object owns the delivery view for its full lifetime, so the returned
        // mapping remains live and its slot cannot be recycled while Python holds the buffer.
        let (pointer, length) = unsafe { view.as_ptr_len()? };
        Ok(Self {
            _view: view,
            pointer,
            length,
        })
    }
}

#[pymethods]
impl ArenaBuffer {
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
        // SAFETY: view is non-null, the buffer owns its live delivery mapping, and every shape
        // pointer below refers to storage inside the caller-owned Py_buffer structure.
        unsafe {
            (*view).obj = owner.into_ptr();
            (*view).buf = borrowed.pointer.as_ptr().cast::<c_void>();
            (*view).len = borrowed.length as isize;
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
        // SAFETY: Python returns the same live view initialized by __getbuffer__.
        unsafe {
            if !view.is_null() && !(*view).format.is_null() {
                drop(CString::from_raw((*view).format));
                (*view).format = ptr::null_mut();
            }
        }
    }
}
