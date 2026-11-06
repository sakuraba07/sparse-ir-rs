//! The raw pointers of the C API, checked in one place
//!
//! A C function turns each pointer argument into a reference or an [`Out`]
//! at its start, with one `unsafe` block whose SAFETY comment names the
//! caller's obligation; the rest of the body is safe Rust. A null pointer
//! becomes `SPIR_INVALID_ARGUMENT`, and [`status_call`] and [`new_handle`]
//! turn the result of the body (or a panic) into the status the C caller
//! sees.

use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT, StatusCode};
use std::marker::PhantomData;
use std::panic::UnwindSafe;
use std::ptr::NonNull;

/// The object behind a handle, or `SPIR_INVALID_ARGUMENT` if it is null
///
/// # Safety
/// A non-null `ptr` must point to a live `T` (a handle returned by this
/// library and not yet released) that nothing mutates during `'a`.
pub(crate) unsafe fn obj<'a, T>(ptr: *const T) -> Result<&'a T, StatusCode> {
    // SAFETY: forwarded from the caller; `as_ref` handles null.
    unsafe { ptr.as_ref() }.ok_or(SPIR_INVALID_ARGUMENT)
}

/// A non-null output pointer of the C caller, written with [`Out::write`]
///
/// It is not a `&mut T`: the caller's memory may be uninitialized, and it is
/// only ever written, never read.
pub(crate) struct Out<'a, T> {
    ptr: NonNull<T>,
    _call: PhantomData<&'a mut T>,
}

impl<'a, T> Out<'a, T> {
    /// The output `ptr`, or `SPIR_INVALID_ARGUMENT` if it is null
    ///
    /// # Safety
    /// A non-null `ptr` must be aligned and valid for writes of a `T` during
    /// `'a`.
    pub(crate) unsafe fn new(ptr: *mut T) -> Result<Self, StatusCode> {
        let ptr = NonNull::new(ptr).ok_or(SPIR_INVALID_ARGUMENT)?;
        Ok(Self {
            ptr,
            _call: PhantomData,
        })
    }

    /// Store `value` (the old value is not dropped)
    pub(crate) fn write(self, value: T) {
        // SAFETY: `new` was given an aligned pointer valid for writes of a T.
        unsafe { self.ptr.as_ptr().write(value) }
    }
}

/// Copy `src` to the `dst` buffer of the C caller
///
/// # Safety
/// `dst` must be non-null, aligned and valid for writes of `src.len()`
/// elements, and must not overlap `src`.
pub(crate) unsafe fn write_slice<T: Copy>(dst: *mut T, src: &[T]) {
    // SAFETY: forwarded from the caller.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) }
}

/// Run the body of a C function that returns a status
///
/// `Ok` is `SPIR_COMPUTATION_SUCCESS`, `Err(code)` is `code`, and a panic is
/// `SPIR_INTERNAL_ERROR`.
pub(crate) fn status_call(f: impl FnOnce() -> Result<(), StatusCode> + UnwindSafe) -> StatusCode {
    match crate::ffi_call(f) {
        Ok(Ok(())) => SPIR_COMPUTATION_SUCCESS,
        Ok(Err(code)) => code,
        Err(_) => SPIR_INTERNAL_ERROR,
    }
}

/// Run the body of a C constructor: the new handle, or null, with the
/// status written to `status`
///
/// A null `status` returns null without running `f`.
///
/// # Safety
/// A non-null `status` must be aligned and valid for a write of a
/// `StatusCode`.
pub(crate) unsafe fn new_handle<T>(
    status: *mut StatusCode,
    f: impl FnOnce() -> Result<T, StatusCode> + UnwindSafe,
) -> *mut T {
    // SAFETY: forwarded from the caller.
    let Ok(status) = (unsafe { Out::new(status) }) else {
        return std::ptr::null_mut();
    };
    let (handle, code) = match crate::ffi_call(f) {
        Ok(Ok(obj)) => (Box::into_raw(Box::new(obj)), SPIR_COMPUTATION_SUCCESS),
        Ok(Err(code)) => (std::ptr::null_mut(), code),
        Err(_) => (std::ptr::null_mut(), SPIR_INTERNAL_ERROR),
    };
    status.write(code);
    handle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_pointers_are_invalid_arguments() {
        // SAFETY: null is always accepted.
        unsafe {
            assert_eq!(
                obj::<f64>(std::ptr::null()).err(),
                Some(SPIR_INVALID_ARGUMENT)
            );
            assert!(Out::<f64>::new(std::ptr::null_mut()).is_err());
            assert!(new_handle(std::ptr::null_mut(), || Ok(1.0)).is_null());
        }
    }

    #[test]
    fn status_of_each_outcome() {
        assert_eq!(status_call(|| Ok(())), SPIR_COMPUTATION_SUCCESS);
        assert_eq!(status_call(|| Err(-3)), -3);
        assert_eq!(status_call(|| panic!("boom")), SPIR_INTERNAL_ERROR);

        for (outcome, expected) in [(Ok(2.5), SPIR_COMPUTATION_SUCCESS), (Err(-3), -3)] {
            let mut status = 99;
            // SAFETY: `status` is a live local.
            let handle = unsafe { new_handle(&mut status, || outcome) };
            assert_eq!(status, expected);
            assert_eq!(handle.is_null(), outcome.is_err());
            if !handle.is_null() {
                // SAFETY: a non-null handle is a fresh `Box` from `new_handle`.
                assert_eq!(*unsafe { Box::from_raw(handle) }, 2.5);
            }
        }
        let mut status = 99;
        // SAFETY: `status` is a live local.
        let handle = unsafe { new_handle::<f64>(&mut status, || panic!("boom")) };
        assert!(handle.is_null());
        assert_eq!(status, SPIR_INTERNAL_ERROR);
    }
}
