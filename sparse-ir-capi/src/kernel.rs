//! Kernel API for C
//!
//! Functions for creating and manipulating kernel objects.

use sparse_ir::kernel::SVEHints;

use crate::ffi::{Out, new_handle, obj, status_call, write_slice};
use crate::status::status_from;
use crate::types::{KernelType, spir_kernel};
use crate::{SPIR_INVALID_ARGUMENT, StatusCode};

/// The SVE hints of `kernel` for the accuracy `epsilon`
fn sve_hints(kernel: &spir_kernel, epsilon: f64) -> Box<dyn SVEHints<f64>> {
    use sparse_ir::kernel::SVEKernel;
    match kernel.inner() {
        KernelType::Logistic(k) => Box::new(k.sve_hints::<f64>(epsilon)),
        KernelType::RegularizedBose(k) => Box::new(k.sve_hints::<f64>(epsilon)),
    }
}

/// `SPIR_INVALID_ARGUMENT` unless `epsilon` is positive and finite
fn check_epsilon(epsilon: f64) -> Result<(), StatusCode> {
    if epsilon > 0.0 && epsilon.is_finite() {
        Ok(())
    } else {
        Err(SPIR_INVALID_ARGUMENT)
    }
}

/// Create a new Logistic kernel
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// * Pointer to the newly created kernel object, or NULL if creation fails
///
/// # Safety
/// The caller must ensure `status` is a valid pointer.
///
/// # Example (C)
/// ```c
/// int status;
/// spir_kernel* kernel = spir_logistic_kernel_new(10.0, &status);
/// if (kernel != NULL) {
///     // Use kernel...
///     spir_kernel_release(kernel);
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn spir_logistic_kernel_new(
    lambda: f64,
    status: *mut StatusCode,
) -> *mut spir_kernel {
    // SAFETY: `status` is null or writable (the C contract above).
    unsafe {
        new_handle(status, || {
            spir_kernel::new_logistic(lambda).map_err(|e| status_from(&e))
        })
    }
}

/// Create a new RegularizedBose kernel
///
/// # Deprecated
/// Use `spir_logistic_kernel_new`, the default kernel for both statistics.
/// `RegularizedBoseKernel` will be removed in a future release
/// (https://github.com/SpM-lab/sparse-ir-rs/issues/273).
///
/// # Arguments
/// * `lambda` - The kernel parameter Λ = β * ωmax (must be > 0)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// * Pointer to the newly created kernel object, or NULL if creation fails
#[unsafe(no_mangle)]
pub extern "C" fn spir_reg_bose_kernel_new(
    lambda: f64,
    status: *mut StatusCode,
) -> *mut spir_kernel {
    // SAFETY: `status` is null or writable (the C contract).
    unsafe {
        new_handle(status, || {
            spir_kernel::new_regularized_bose(lambda).map_err(|e| status_from(&e))
        })
    }
}

/// Get the lambda parameter of a kernel
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `lambda_out` - Pointer to store the lambda value
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if kernel or lambda_out is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_lambda(
    kernel: *const spir_kernel,
    lambda_out: *mut f64,
) -> StatusCode {
    status_call(|| {
        // SAFETY: each pointer is null or, by the C contract, a live kernel
        // handle and a writable double.
        let (kernel, lambda_out) = unsafe { (obj(kernel)?, Out::new(lambda_out)?) };
        lambda_out.write(kernel.lambda());
        Ok(())
    })
}

/// Compute kernel value K(x, y)
///
/// # Arguments
/// * `kernel` - Kernel object
/// * `x` - First argument (typically in [-1, 1])
/// * `y` - Second argument (typically in [-1, 1])
/// * `out` - Pointer to store the result
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if kernel or out is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_compute(
    kernel: *const spir_kernel,
    x: f64,
    y: f64,
    out: *mut f64,
) -> StatusCode {
    status_call(|| {
        // SAFETY: as in `spir_kernel_get_lambda`
        let (kernel, out) = unsafe { (obj(kernel)?, Out::new(out)?) };
        out.write(kernel.compute(x, y));
        Ok(())
    })
}

/// Manual release function (replaces macro-generated one)
///
/// # Safety
/// This function drops the kernel. The inner KernelType data is automatically freed
/// by the Drop implementation when the spir_kernel structure is dropped.
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_release(kernel: *mut spir_kernel) {
    if !kernel.is_null() {
        // SAFETY: a non-null `kernel` is a handle that this library boxed and
        // the caller releases once. Dropping it frees the inner KernelType.
        drop(unsafe { Box::from_raw(kernel) });
    }
}

/// Manual clone function (replaces macro-generated one)
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_clone(src: *const spir_kernel) -> *mut spir_kernel {
    let result = crate::ffi_call(|| {
        // SAFETY: `src` is null or a live kernel handle (the C contract).
        let src = unsafe { obj(src) }.ok()?;
        Some(Box::into_raw(Box::new(src.clone())))
    });
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}

/// Check if the kernel pointer is non-null.
///
/// Note: This only performs a null check. It cannot detect dangling
/// pointers; dereferencing an arbitrary non-null pointer would be
/// undefined behaviour that `catch_unwind` cannot reliably catch.
///
/// # Returns
/// 1 if the pointer is non-null, 0 otherwise
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_is_assigned(obj: *const spir_kernel) -> i32 {
    if obj.is_null() { 0 } else { 1 }
}

/// Get kernel domain boundaries
///
/// # Arguments
/// * `k` - Kernel object
/// * `xmin` - Pointer to store minimum x value
/// * `xmax` - Pointer to store maximum x value
/// * `ymin` - Pointer to store minimum y value
/// * `ymax` - Pointer to store maximum y value
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if any pointer is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_domain(
    k: *const spir_kernel,
    xmin: *mut f64,
    xmax: *mut f64,
    ymin: *mut f64,
    ymax: *mut f64,
) -> StatusCode {
    status_call(|| {
        // SAFETY: each pointer is null or, by the C contract, a live kernel
        // handle and four writable doubles.
        let (kernel, xmin, xmax, ymin, ymax) = unsafe {
            (
                obj(k)?,
                Out::new(xmin)?,
                Out::new(xmax)?,
                Out::new(ymin)?,
                Out::new(ymax)?,
            )
        };
        let (xmin_val, xmax_val, ymin_val, ymax_val) = kernel.domain();
        xmin.write(xmin_val);
        xmax.write(xmax_val);
        ymin.write(ymin_val);
        ymax.write(ymax_val);
        Ok(())
    })
}

/// Body of `spir_kernel_get_sve_hints_segments_{x,y}`: the half [0, 1] of the
/// segments of these centrosymmetric kernels, in x or (`y`) in y
fn sve_hint_segments(
    k: *const spir_kernel,
    epsilon: f64,
    segments: *mut f64,
    n_segments: *mut libc::c_int,
    y: bool,
) -> StatusCode {
    status_call(|| {
        // SAFETY: `k` is null or a live kernel handle, and `n_segments` null or
        // a readable and writable int (the C contract).
        let (kernel, n_out) = unsafe { (obj(k)?, Out::new(n_segments)?) };
        check_epsilon(epsilon)?;

        let hints = sve_hints(kernel, epsilon);
        let full = if y {
            hints.segments_y()
        } else {
            hints.segments_x()
        };
        let segs = sparse_ir::sve::half_domain_segments(&full);
        let n = (segs.len() - 1) as libc::c_int;

        // First call (segments NULL): only the number of intervals. Second
        // call: `*n_segments` is the caller's interval count, so the buffer
        // holds `*n_segments + 1` boundary points; it must hold all of them.
        if !segments.is_null() {
            // SAFETY: `n_segments` is non-null (checked by `Out::new`) and holds
            // the caller's count on the second call.
            if unsafe { n_segments.read() } < n {
                return Err(SPIR_INVALID_ARGUMENT);
            }
            // SAFETY: the caller allocated `*n_segments + 1 >= segs.len()`
            // doubles at `segments` (checked just above).
            unsafe { write_slice(segments, &segs) };
        }
        n_out.write(n);
        Ok(())
    })
}

/// Get x-segments for SVE discretization hints from a kernel
///
/// This function should be called twice:
/// 1. First call with segments=NULL: sets `*n_segments` to the number of segment intervals.
/// 2. Second call with segments allocated: fills `segments[0..n_segments]` with boundary
///    points (`n_segments + 1` values total). The caller must allocate at least
///    `n_segments + 1` elements.
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `segments` - Pointer to store segments array (NULL for first call)
/// * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segment intervals
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_segments_x(
    k: *const spir_kernel,
    epsilon: f64,
    segments: *mut f64,
    n_segments: *mut libc::c_int,
) -> StatusCode {
    sve_hint_segments(k, epsilon, segments, n_segments, false)
}

/// Get y-segments for SVE discretization hints from a kernel
///
/// This function should be called twice:
/// 1. First call with segments=NULL: sets `*n_segments` to the number of segment intervals.
/// 2. Second call with segments allocated: fills `segments[0..n_segments]` with boundary
///    points (`n_segments + 1` values total). The caller must allocate at least
///    `n_segments + 1` elements.
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `segments` - Pointer to store segments array (NULL for first call)
/// * `n_segments` - [IN/OUT] Input: ignored when segments is NULL. Output: number of segment intervals
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or n_segments is null, or segments array is too small
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_segments_y(
    k: *const spir_kernel,
    epsilon: f64,
    segments: *mut f64,
    n_segments: *mut libc::c_int,
) -> StatusCode {
    sve_hint_segments(k, epsilon, segments, n_segments, true)
}

/// Get the number of singular values hint from a kernel
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `nsvals` - Pointer to store the number of singular values
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or nsvals is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_nsvals(
    k: *const spir_kernel,
    epsilon: f64,
    nsvals: *mut libc::c_int,
) -> StatusCode {
    status_call(|| {
        // SAFETY: each pointer is null or, by the C contract, a live kernel
        // handle and a writable int.
        let (kernel, nsvals) = unsafe { (obj(k)?, Out::new(nsvals)?) };
        check_epsilon(epsilon)?;
        nsvals.write(sve_hints(kernel, epsilon).nsvals() as libc::c_int);
        Ok(())
    })
}

/// Get the number of Gauss points hint from a kernel
///
/// # Arguments
/// * `k` - Kernel object
/// * `epsilon` - Accuracy target for the basis
/// * `ngauss` - Pointer to store the number of Gauss points
///
/// # Returns
/// * `SPIR_COMPUTATION_SUCCESS` on success
/// * `SPIR_INVALID_ARGUMENT` if k or ngauss is null
/// * `SPIR_INTERNAL_ERROR` if internal panic occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_kernel_get_sve_hints_ngauss(
    k: *const spir_kernel,
    epsilon: f64,
    ngauss: *mut libc::c_int,
) -> StatusCode {
    status_call(|| {
        // SAFETY: as in `spir_kernel_get_sve_hints_nsvals`
        let (kernel, ngauss) = unsafe { (obj(k)?, Out::new(ngauss)?) };
        check_epsilon(epsilon)?;
        ngauss.write(sve_hints(kernel, epsilon).ngauss() as libc::c_int);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR};
    use std::ptr;

    #[test]
    fn test_logistic_kernel_creation() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_regularized_bose_kernel_creation() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(10.0, &mut status);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_lambda() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut lambda = 0.0;
        let status = spir_kernel_get_lambda(kernel, &mut lambda);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(lambda, 10.0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_compute() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut result = 0.0;
        let status = spir_kernel_compute(kernel, 0.5, 0.5, &mut result);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(result > 0.0); // Kernel should be positive

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_null_pointer_errors() {
        // Null status pointer
        let kernel = spir_logistic_kernel_new(10.0, ptr::null_mut());
        assert!(kernel.is_null());

        // Null kernel pointer
        let mut lambda = 0.0;
        let status = spir_kernel_get_lambda(ptr::null(), &mut lambda);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
    }

    #[test]
    fn test_invalid_lambda() {
        let mut status = SPIR_COMPUTATION_SUCCESS;

        // Zero lambda
        let kernel = spir_logistic_kernel_new(0.0, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());

        // Negative lambda
        let kernel = spir_logistic_kernel_new(-1.0, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());

        // NaN lambda must not pass the <= 0.0 guard
        let kernel = spir_logistic_kernel_new(f64::NAN, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());

        // Same for the RegularizedBose kernel
        let kernel = spir_reg_bose_kernel_new(f64::NAN, &mut status);
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert!(kernel.is_null());
    }

    #[test]
    fn test_kernel_domain() {
        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut xmin = 0.0;
        let mut xmax = 0.0;
        let mut ymin = 0.0;
        let mut ymax = 0.0;
        let status = spir_kernel_get_domain(kernel, &mut xmin, &mut xmax, &mut ymin, &mut ymax);

        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(xmin, -1.0);
        assert_eq!(xmax, 1.0);
        assert_eq!(ymin, -1.0);
        assert_eq!(ymax, 1.0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_nsvals() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &mut nsvals);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(nsvals > 0);
        assert!(nsvals >= 10);
        assert!(nsvals <= 1000);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_ngauss() {
        let lambda = 10.0;
        let epsilon_coarse = 1e-6;
        let epsilon_fine = 1e-10;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        let mut ngauss_coarse = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_coarse, &mut ngauss_coarse);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss_coarse > 0);
        assert_eq!(ngauss_coarse, 10); // For epsilon >= 1e-8, ngauss should be 10

        let mut ngauss_fine = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_fine, &mut ngauss_fine);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss_fine > 0);
        assert_eq!(ngauss_fine, 16); // For epsilon < 1e-8, ngauss should be 16

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_segments_x() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // First call: get the number of segments
        let mut n_segments = 0;
        let status =
            spir_kernel_get_sve_hints_segments_x(kernel, epsilon, ptr::null_mut(), &mut n_segments);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments > 0);

        // Second call: get the actual segments
        let mut segments = vec![0.0; (n_segments + 1) as usize];
        let mut n_segments_out = n_segments + 1;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            segments.as_mut_ptr(),
            &mut n_segments_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_segments_out, n_segments);

        // Verify segments are valid
        assert_eq!(segments.len(), (n_segments + 1) as usize);
        assert!((segments[0] - (0.0)).abs() < 1e-10);
        assert!((segments[n_segments as usize] - 1.0).abs() < 1e-10);

        // Verify segments are in ascending order
        for i in 1..segments.len() {
            assert!(segments[i] > segments[i - 1]);
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_segments_y() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // First call: get the number of segments
        let mut n_segments = 0;
        let status =
            spir_kernel_get_sve_hints_segments_y(kernel, epsilon, ptr::null_mut(), &mut n_segments);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments > 0);

        // Second call: get the actual segments
        let mut segments = vec![0.0; (n_segments + 1) as usize];
        let mut n_segments_out = n_segments + 1;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            segments.as_mut_ptr(),
            &mut n_segments_out,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert_eq!(n_segments_out, n_segments);

        // Verify segments are valid
        assert_eq!(segments.len(), (n_segments + 1) as usize);
        assert!((segments[0] - (0.0)).abs() < 1e-10);
        assert!((segments[n_segments as usize] - 1.0).abs() < 1e-10);

        // Verify segments are in ascending order
        for i in 1..segments.len() {
            assert!(segments[i] > segments[i - 1]);
        }

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_with_regularized_bose() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_reg_bose_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Test nsvals
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &mut nsvals);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(nsvals > 0);

        // Test ngauss
        let mut ngauss = 0;
        let status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &mut ngauss);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(ngauss > 0);

        // Test segments_x
        let mut n_segments_x = 0;
        let status = spir_kernel_get_sve_hints_segments_x(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_x,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments_x > 0);

        // Test segments_y
        let mut n_segments_y = 0;
        let status = spir_kernel_get_sve_hints_segments_y(
            kernel,
            epsilon,
            ptr::null_mut(),
            &mut n_segments_y,
        );
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(n_segments_y > 0);

        spir_kernel_release(kernel);
    }

    #[test]
    fn test_kernel_get_sve_hints_error_handling() {
        let lambda = 10.0;
        let epsilon = 1e-8;

        let mut status = SPIR_INTERNAL_ERROR;
        let kernel = spir_logistic_kernel_new(lambda, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
        assert!(!kernel.is_null());

        // Test with nullptr kernel
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(ptr::null(), epsilon, &mut nsvals);
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        // Test with nullptr output parameter
        let status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, ptr::null_mut());
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        // Test with invalid epsilon
        let mut nsvals = 0;
        let status = spir_kernel_get_sve_hints_nsvals(kernel, -1.0, &mut nsvals);
        assert_ne!(status, SPIR_COMPUTATION_SUCCESS);

        spir_kernel_release(kernel);
    }

    /// The status of each invalid argument, and that a failing call leaves
    /// the caller's buffers alone
    #[test]
    fn test_kernel_invalid_arguments() {
        use crate::SPIR_INVALID_ARGUMENT;
        let epsilon = 1e-8;

        // Constructors: NULL status gives NULL; an invalid lambda is reported
        assert!(spir_logistic_kernel_new(10.0, ptr::null_mut()).is_null());
        let mut status = SPIR_INTERNAL_ERROR;
        assert!(spir_logistic_kernel_new(-1.0, &mut status).is_null());
        assert_eq!(status, SPIR_INVALID_ARGUMENT);

        let kernel = spir_logistic_kernel_new(10.0, &mut status);
        assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

        let mut v = 7.0;
        assert_eq!(
            spir_kernel_compute(kernel, 0.5, 0.5, ptr::null_mut()),
            SPIR_INVALID_ARGUMENT
        );
        assert_eq!(
            spir_kernel_compute(ptr::null(), 0.5, 0.5, &mut v),
            SPIR_INVALID_ARGUMENT
        );
        let (mut a, mut b, mut c) = (7.0, 7.0, 7.0);
        let status = spir_kernel_get_domain(kernel, &mut a, &mut b, &mut c, ptr::null_mut());
        assert_eq!(status, SPIR_INVALID_ARGUMENT);
        assert_eq!((v, a, b, c), (7.0, 7.0, 7.0, 7.0));
        let mut n = 7;
        assert_eq!(
            spir_kernel_get_sve_hints_ngauss(kernel, f64::NAN, &mut n),
            SPIR_INVALID_ARGUMENT
        );
        assert_eq!(n, 7);

        assert!(spir_kernel_clone(ptr::null()).is_null());
        spir_kernel_release(ptr::null_mut());

        type SegmentsFn =
            extern "C" fn(*const spir_kernel, f64, *mut f64, *mut libc::c_int) -> StatusCode;
        let x: SegmentsFn = spir_kernel_get_sve_hints_segments_x;
        let y: SegmentsFn = spir_kernel_get_sve_hints_segments_y;
        for segments_fn in [x, y] {
            let mut n = 0;
            assert_eq!(
                segments_fn(kernel, epsilon, ptr::null_mut(), &mut n),
                SPIR_COMPUTATION_SUCCESS
            );
            assert!(n > 1);
            let mut buf = vec![-1.0; n as usize + 1];

            // NULL kernel or count, invalid epsilon
            let mut m = n;
            let status = segments_fn(ptr::null(), epsilon, buf.as_mut_ptr(), &mut m);
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
            let status = segments_fn(kernel, epsilon, buf.as_mut_ptr(), ptr::null_mut());
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
            let status = segments_fn(kernel, 0.0, buf.as_mut_ptr(), &mut m);
            assert_eq!(status, SPIR_INVALID_ARGUMENT);

            // A buffer one point too short is rejected before any write
            let mut short = n - 1;
            let status = segments_fn(kernel, epsilon, buf.as_mut_ptr(), &mut short);
            assert_eq!(status, SPIR_INVALID_ARGUMENT);
            assert_eq!(short, n - 1);
            assert!(buf.iter().all(|&s| s == -1.0));

            // The exact size is filled, from 0 to 1 (the half domain)
            let mut m = n;
            let status = segments_fn(kernel, epsilon, buf.as_mut_ptr(), &mut m);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(m, n);
            assert_eq!((buf[0], buf[n as usize]), (0.0, 1.0));
            assert!(buf.windows(2).all(|w| w[0] < w[1]));
        }

        spir_kernel_release(kernel);
    }
}
