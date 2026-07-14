//! Main SVE computation functions

use crate::fpu_check::FpuGuard;
use crate::kernel::{AbstractKernel, CentrosymmKernel, KernelProperties, SVEHints};
use crate::numeric::RealNumeric;
use mdarray::DTensor;

use super::result::SVEResult;
use super::strategy::{CentrosymmSVE, NonCentrosymmSVE, SVEStrategy};
use super::types::{SVDStrategy, TworkType, safe_epsilon};

/// Main SVE computation function for centrosymmetric kernels
///
/// Automatically chooses the appropriate SVE strategy based on kernel properties
/// and working precision based on epsilon.
///
/// # Arguments
///
/// * `kernel` - The centrosymmetric kernel to expand
/// * `epsilon` - Required accuracy
/// * `cutoff` - Relative tolerance for singular value truncation
/// * `max_num_svals` - Maximum number of singular values to keep
/// * `twork` - Working precision type (Auto for automatic selection)
///
/// # Returns
///
/// SVEResult containing singular functions and values
///
/// # FPU State Warning
///
/// This function checks for dangerous FPU settings (Flush-to-Zero and Denormals-Are-Zero)
/// that can cause incorrect results. If detected, it temporarily corrects the FPU state
/// and prints a warning. If you see this warning, add `-fp-model precise` flag when
/// compiling with Intel Fortran.
/// Release unused memory back to the OS.
///
/// SVE computation allocates large temporary buffers for SVD.
/// After computation, these are freed but the allocator may retain them.
/// This function asks the allocator to return unused memory to the OS.
///
/// Supported platforms:
/// - macOS: uses `malloc_zone_pressure_relief`
/// - Linux (glibc): uses `malloc_trim`
/// - Other platforms: no-op (memory is still freed, just not returned to OS immediately)
#[inline]
fn release_unused_memory() {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    }

    // Only use malloc_trim on Linux with glibc (not musl)
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe { malloc_trim(0) };
    }
}

pub fn compute_sve<K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
{
    // Protect computation from dangerous FPU settings (FZ/DAZ)
    // This temporarily disables FZ/DAZ and restores them after computation
    let _fpu_guard = FpuGuard::new_protect_computation();

    // Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual, _svd_strategy) =
        safe_epsilon(epsilon, twork, SVDStrategy::Auto);

    // Dispatch based on working precision
    let result = match twork_actual {
        TworkType::Float64 => {
            compute_sve_with_precision::<f64, K>(kernel, safe_epsilon, cutoff, max_num_svals)
        }
        TworkType::Float64X2 => compute_sve_with_precision::<crate::Df64, K>(
            kernel,
            safe_epsilon,
            cutoff.map(crate::Df64::from),
            max_num_svals,
        ),
        _ => panic!("Invalid TworkType: {:?}", twork_actual),
    };

    // Release temporary memory back to OS after SVE computation
    release_unused_memory();

    result
}

/// Main SVE computation function for general kernels (centrosymmetric or non-centrosymmetric)
///
/// Automatically chooses the appropriate SVE strategy based on kernel properties.
/// For centrosymmetric kernels, uses CentrosymmSVE for efficiency.
/// For non-centrosymmetric kernels, uses NonCentrosymmSVE.
///
/// # Arguments
///
/// * `kernel` - The kernel to expand (can be centrosymmetric or non-centrosymmetric)
/// * `epsilon` - Required accuracy
/// * `cutoff` - Relative tolerance for singular value truncation
/// * `max_num_svals` - Maximum number of singular values to keep
/// * `twork` - Working precision type (Auto for automatic selection)
///
/// # Returns
///
/// SVEResult containing singular functions and values
///
/// # FPU State Warning
///
/// This function checks for dangerous FPU settings (Flush-to-Zero and Denormals-Are-Zero)
/// that can cause incorrect results. If detected, it temporarily corrects the FPU state
/// and prints a warning. If you see this warning, add `-fp-model precise` flag when
/// compiling with Intel Fortran.
pub fn compute_sve_general<K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<f64>,
    max_num_svals: Option<usize>,
    twork: TworkType,
) -> SVEResult
where
    K: AbstractKernel + KernelProperties + Clone + 'static,
{
    // Protect computation from dangerous FPU settings (FZ/DAZ)
    // This temporarily disables FZ/DAZ and restores them after computation
    let _fpu_guard = FpuGuard::new_protect_computation();

    // Determine safe epsilon and working precision
    let (safe_epsilon, twork_actual, _svd_strategy) =
        safe_epsilon(epsilon, twork, SVDStrategy::Auto);

    // Dispatch based on working precision
    let result = match twork_actual {
        TworkType::Float64 => compute_sve_general_with_precision::<f64, K>(
            kernel,
            safe_epsilon,
            cutoff,
            max_num_svals,
        ),
        TworkType::Float64X2 => compute_sve_general_with_precision::<crate::Df64, K>(
            kernel,
            safe_epsilon,
            cutoff.map(crate::Df64::from),
            max_num_svals,
        ),
        _ => panic!("Invalid TworkType: {:?}", twork_actual),
    };

    // Release temporary memory back to OS after SVE computation
    release_unused_memory();

    result
}

/// Compute SVE with specific precision type
fn compute_sve_with_precision<T, K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<T>,
    max_num_svals: Option<usize>,
) -> SVEResult
where
    T: RealNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // 1. Determine SVE strategy (automatically chooses CentrosymmSVE for centrosymmetric kernels)
    let sve = determine_sve::<T, K>(kernel, epsilon);

    // 2. Compute matrices
    let matrices = sve.matrices();

    // 3. Compute SVD for each matrix
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();

    for matrix in matrices.iter() {
        let (u, s, v) = crate::tsvd::compute_svd_dtensor(matrix);
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }

    // 4. Truncate based on cutoff
    // Default cutoff is 2.0 * T::epsilon() which adapts to the precision type (f64 or Df64)
    let default_cutoff = T::from_f64_unchecked(2.0) * T::epsilon();
    let rtol_t = cutoff.unwrap_or(default_cutoff);
    let (u_trunc, s_trunc, v_trunc) = truncate(u_list, s_list, v_list, rtol_t, max_num_svals);

    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}

/// Compute SVE with specific precision type for general kernels
fn compute_sve_general_with_precision<T, K>(
    kernel: K,
    epsilon: f64,
    cutoff: Option<T>,
    max_num_svals: Option<usize>,
) -> SVEResult
where
    T: RealNumeric + Send + Sync + Clone + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // 1. Determine SVE strategy based on kernel symmetry
    let sve = determine_sve_general::<T, K>(kernel, epsilon);

    // 2. Compute matrices
    let matrices = sve.matrices();

    // 3. Compute SVD for each matrix
    let mut u_list = Vec::new();
    let mut s_list = Vec::new();
    let mut v_list = Vec::new();

    for matrix in matrices.iter() {
        let (u, s, v) = crate::tsvd::compute_svd_dtensor(matrix);
        u_list.push(u);
        s_list.push(s);
        v_list.push(v);
    }

    // 4. Truncate based on cutoff
    // Default cutoff is T::epsilon() which adapts to the precision type (f64 or Df64)
    let rtol_t = cutoff.unwrap_or_else(T::epsilon);
    let (u_trunc, s_trunc, v_trunc) = truncate(u_list, s_list, v_list, rtol_t, max_num_svals);

    // 5. Post-process to create SVEResult
    sve.postprocess(u_trunc, s_trunc, v_trunc)
}

/// Determine the appropriate SVE strategy
///
/// For centrosymmetric kernels, uses CentrosymmSVE for efficient computation
/// by exploiting even/odd symmetry.
fn determine_sve<T, K>(kernel: K, epsilon: f64) -> Box<dyn SVEStrategy<T>>
where
    T: RealNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    // CentrosymmKernel trait implies centrosymmetric
    Box::new(CentrosymmSVE::new(kernel, epsilon))
}

/// Determine the appropriate SVE strategy for general kernels
///
/// Automatically chooses between CentrosymmSVE and NonCentrosymmSVE
/// based on kernel symmetry.
fn determine_sve_general<T, K>(kernel: K, epsilon: f64) -> Box<dyn SVEStrategy<T>>
where
    T: RealNumeric + Send + Sync + Clone + 'static,
    K: AbstractKernel + KernelProperties + Clone + 'static,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    if kernel.is_centrosymmetric() {
        // Try to use CentrosymmSVE if kernel implements CentrosymmKernel
        // For now, we'll use NonCentrosymmSVE as a fallback
        // In practice, centrosymmetric kernels should implement CentrosymmKernel
        Box::new(NonCentrosymmSVE::new(kernel, epsilon))
    } else {
        Box::new(NonCentrosymmSVE::new(kernel, epsilon))
    }
}

/// Truncate SVD results based on cutoff and maximum size
///
/// # Arguments
///
/// * `u_list` - List of U matrices
/// * `s_list` - List of singular value vectors
/// * `v_list` - List of V matrices
/// * `rtol` - Relative tolerance for truncation
/// * `max_num_svals` - Maximum number of singular values to keep
///
/// # Returns
///
/// Tuple of (truncated_u_list, truncated_s_list, truncated_v_list)
pub fn truncate<T: RealNumeric>(
    u_list: Vec<DTensor<T, 2>>,
    s_list: Vec<Vec<T>>,
    v_list: Vec<DTensor<T, 2>>,
    rtol: T,
    max_num_svals: Option<usize>,
) -> (Vec<DTensor<T, 2>>, Vec<Vec<T>>, Vec<DTensor<T, 2>>) {
    let zero = T::zero();

    // Validate
    if let Some(max) = max_num_svals {
        if max == 0 {
            panic!("max_num_svals must be positive");
        }
    }
    if rtol < zero || rtol > T::from_f64_unchecked(1.0) {
        panic!("rtol must be in [0, 1]");
    }

    // Find global maximum singular value
    let mut all_svals = Vec::new();
    for s in &s_list {
        all_svals.extend(s.iter().copied());
    }

    let max_sval = all_svals
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(zero);

    // Determine cutoff
    let cutoff = if let Some(max_count) = max_num_svals {
        if max_count < all_svals.len() {
            let mut sorted = all_svals.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let nth = sorted[max_count - 1];
            if rtol * max_sval > nth {
                rtol * max_sval
            } else {
                nth
            }
        } else {
            rtol * max_sval
        }
    } else {
        rtol * max_sval
    };

    // Truncate each result
    let mut u_trunc = Vec::new();
    let mut s_trunc = Vec::new();
    let mut v_trunc = Vec::new();

    for i in 0..s_list.len() {
        let s = &s_list[i];
        let u = &u_list[i];
        let v = &v_list[i];

        // Count singular values above cutoff
        let mut n_keep = 0;
        for &val in s.iter() {
            if val >= cutoff {
                n_keep += 1;
            }
        }

        // Slice U: keep first n_keep columns. Preserve one output block per
        // input block even when all singular values in that block are removed.
        let u_shape = *u.shape();
        let u_sliced = DTensor::<T, 2>::from_fn([u_shape.0, n_keep], |idx| u[[idx[0], idx[1]]]);
        u_trunc.push(u_sliced);

        s_trunc.push(s[..n_keep].to_vec());

        // Slice V: keep first n_keep columns
        let v_shape = *v.shape();
        let v_sliced = DTensor::<T, 2>::from_fn([v_shape.0, n_keep], |idx| v[[idx[0], idx[1]]]);
        v_trunc.push(v_sliced);
    }

    (u_trunc, s_trunc, v_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Route the trusted (centrosymmetric) LogisticKernel through the
    /// non-centrosymmetric SVE path (`compute_sve_general` -> `NonCentrosymmSVE`,
    /// full-domain single matrix) and check that its singular values match the
    /// symmetric path (`compute_sve` -> `CentrosymmSVE`). The SVE is unique, so
    /// the two decomposition routes must agree — this validates the
    /// non-symmetric path against trusted output.
    #[test]
    fn noncentrosymm_path_matches_centrosymm_for_logistic() {
        use crate::kernel::LogisticKernel;

        let lambda = 10.0;
        let eps = 1e-6;

        let sym = compute_sve(
            LogisticKernel::new(lambda),
            eps,
            None,
            None,
            TworkType::Auto,
        );
        let general = compute_sve_general(
            LogisticKernel::new(lambda),
            eps,
            None,
            None,
            TworkType::Auto,
        );

        assert!(!sym.s.is_empty() && !general.s.is_empty());

        // Compare the leading singular values (same kernel, different route).
        let n = sym.s.len().min(general.s.len());
        for i in 0..n {
            let rel = (sym.s[i] - general.s[i]).abs() / sym.s[0];
            assert!(
                rel < 1e-6,
                "singular value {} differs between paths: centrosymm={:.6e}, general={:.6e} (rel={:.3e})",
                i,
                sym.s[i],
                general.s[i],
                rel
            );
        }

        // The two paths should keep a comparable number of singular values.
        let dn = (sym.s.len() as i64 - general.s.len() as i64).abs();
        assert!(
            dn <= 2,
            "singular-value counts differ too much: centrosymm={}, general={}",
            sym.s.len(),
            general.s.len()
        );

        // Reconstruct K(x,y) = Σ_l u_l(x) s_l v_l(y) from each path's singular
        // functions and check they agree at interior full-domain points. This
        // is gauge-invariant (per-mode sign flips cancel), so it validates the
        // polynomials produced by the non-centrosymmetric path, not just the
        // singular values.
        let recon = |sve: &SVEResult, x: f64, y: f64| -> f64 {
            let u = sve.u.get_polys();
            let v = sve.v.get_polys();
            (0..sve.s.len())
                .map(|l| u[l].evaluate(x) * sve.s[l] * v[l].evaluate(y))
                .sum()
        };
        for &x in &[-0.7_f64, -0.2, 0.3, 0.8] {
            for &y in &[-0.5_f64, 0.1, 0.6] {
                let rs = recon(&sym, x, y);
                let rg = recon(&general, x, y);
                assert!(
                    (rs - rg).abs() < 1e-5,
                    "kernel reconstruction differs at ({},{}): centrosymm={:.6e}, general={:.6e}",
                    x,
                    y,
                    rs,
                    rg
                );
            }
        }
    }

    #[test]
    fn test_truncate_by_rtol() {
        let u = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 0.1]];
        let v = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];

        // rtol = 0.1, max_sval = 10.0, cutoff = 1.0
        // Keep values >= 1.0: [10.0, 5.0]
        let (_, s_trunc, _) = truncate(u, s, v, 0.1, None);

        assert_eq!(s_trunc[0].len(), 2);
        assert_eq!(s_trunc[0][0], 10.0);
        assert_eq!(s_trunc[0][1], 5.0);
    }

    #[test]
    fn test_truncate_by_max_size() {
        let u = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];
        let s = vec![vec![10.0, 5.0, 2.0]];
        let v = vec![DTensor::<f64, 2>::from_elem([3, 3], 1.0)];

        // max_num_svals = 2
        let (_, s_trunc, _) = truncate(u, s, v, 0.0, Some(2));

        assert_eq!(s_trunc[0].len(), 2);
    }

    #[test]
    #[should_panic(expected = "max_num_svals must be positive")]
    fn test_truncate_rejects_zero_max_size() {
        let u = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];
        let s = vec![vec![1.0]];
        let v = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];

        truncate(u, s, v, 0.0, Some(0));
    }

    #[test]
    fn test_truncate_preserves_empty_blocks() {
        let u = vec![
            DTensor::<f64, 2>::from_elem([2, 1], 1.0),
            DTensor::<f64, 2>::from_elem([2, 1], 2.0),
        ];
        let s = vec![vec![10.0], vec![1.0]];
        let v = vec![
            DTensor::<f64, 2>::from_elem([2, 1], 1.0),
            DTensor::<f64, 2>::from_elem([2, 1], 2.0),
        ];

        let (u_trunc, s_trunc, v_trunc) = truncate(u, s, v, 0.5, None);

        assert_eq!(s_trunc.len(), 2);
        assert_eq!(s_trunc[0], vec![10.0]);
        assert!(s_trunc[1].is_empty());
        assert_eq!(*u_trunc[1].shape(), (2, 0));
        assert_eq!(*v_trunc[1].shape(), (2, 0));
    }

    #[test]
    #[should_panic(expected = "rtol must be in [0, 1]")]
    fn test_truncate_invalid_rtol() {
        let u = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];
        let s = vec![vec![1.0]];
        let v = vec![DTensor::<f64, 2>::from_elem([1, 1], 1.0)];

        truncate(u, s, v, 1.5, None);
    }
}
