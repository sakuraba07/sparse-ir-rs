//! ACA-accelerated truncated SVD for the non-symmetric SVE path.
//!
//! Partial-pivot Adaptive Cross Approximation (ACA) produces a low-rank
//! factorization `K̃ ≈ U_a V_aᵀ` in `O(k²(m+n))`, replacing the `O(r·m·n)`
//! ColPivQR front-end of [`crate::tsvd::tsvd`] for the (smooth, exponentially
//! rank-decaying) SVE kernel matrix. The factors are recompressed to a true
//! SVD by two thin QRs plus a small `k×k` SVD. Additive: only the
//! non-symmetric SVE path is rewired; the symmetric/real path is untouched.

use nalgebra::{ComplexField, DMatrix, RealField};
use num_traits::{One, ToPrimitive, Zero};

use crate::Df64;
use crate::numeric::{CustomNumeric, RealNumeric};
use crate::tsvd::{SVDResult, TSVDConfig, TSVDError, svd_decompose, tsvd};
use mdarray::DTensor;
use std::any::TypeId;
use std::mem;

/// Oversampling window: the total number of pivots collected from
/// the norm-based stopping trigger onward, *inclusive of the trigger iteration*.
/// With `OVERSAMPLE = 3` the loop takes the triggering pivot plus 2 further
/// pivots, then stops. (The trigger counts as the 1st of `OVERSAMPLE` because
/// the residual past the true rank settles near machine epsilon but not
/// bit-exact zero, so the degenerate-pivot guard does not bound the window on
/// its own — the counter does.) Oversampling defers the final rank decision to
/// the downstream small-SVD + rtol truncation in [`tsvd_aca`], rather than
/// letting a single marginal ACA stop-step drop a real singular value.
const OVERSAMPLE: usize = 3;

/// Partial-pivot ACA. See module docs. Generic over `ComplexField` so the same
/// primitive serves the real (`f64`/`Df64`) path now and the complex path later.
pub fn aca_low_rank<T>(
    matrix: &DMatrix<T>,
    rtol: T::RealField,
    max_rank: usize,
) -> Option<(DMatrix<T>, DMatrix<T>)>
where
    T: ComplexField + Copy,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    if m == 0 || n == 0 {
        return Some((DMatrix::zeros(m, 0), DMatrix::zeros(n, 0)));
    }
    let cap = max_rank.min(m).min(n);

    // Running factors, stored column-wise as Vec<column>.
    let mut u_cols: Vec<Vec<T>> = Vec::new(); // each length m
    let mut v_cols: Vec<Vec<T>> = Vec::new(); // each length n

    let mut used_rows = vec![false; m];
    let mut i_star = 0usize; // initial pivot row
    let mut approx_norm_sq = T::RealField::zero();
    let mut converged = false;
    // Number of extra pivots taken since the norm criterion first fired.
    // `None` means the criterion has not fired yet.
    let mut oversample_taken: Option<usize> = None;

    for _t in 0..cap {
        used_rows[i_star] = true;

        // Row residual a[j] = A[i*, j] - Σ_s u_s[i*] * v_s[j]
        let mut a: Vec<T> = (0..n).map(|j| matrix[(i_star, j)]).collect();
        for s in 0..u_cols.len() {
            let coeff = u_cols[s][i_star];
            for j in 0..n {
                a[j] -= coeff * v_cols[s][j];
            }
        }

        // Column pivot j* = argmax_j |a[j]|.
        let mut j_star = 0usize;
        let mut best = T::RealField::zero();
        for j in 0..n {
            let m_abs = a[j].modulus();
            if m_abs > best {
                best = m_abs;
                j_star = j;
            }
        }
        let delta = a[j_star];

        // Column residual b[i] = A[i, j*] - Σ_s u_s[i] * v_s[j*]
        let mut b: Vec<T> = (0..m).map(|i| matrix[(i, j_star)]).collect();
        for s in 0..u_cols.len() {
            let coeff = v_cols[s][j_star];
            for i in 0..m {
                b[i] -= u_cols[s][i] * coeff;
            }
        }

        // Degenerate pivot: nothing left to capture.
        if delta.modulus().is_zero() {
            converged = true;
            break;
        }

        // New rank-1 factor: u_t = b, v_t = a / delta  (so u_t v_tᵀ matches the
        // residual at row i* and column j*).
        let inv = T::one() / delta;
        let v_t: Vec<T> = a.iter().map(|&x| x * inv).collect();
        let u_t = b;

        // Incremental Frobenius norm of the approximation:
        // ‖A_k‖² = ‖A_{k-1}‖² + ‖u_t‖²‖v_t‖² + 2 Re Σ_{s<t}(u_sᴴu_t)(v_sᴴv_t)
        let u_t_nrm2 = u_t
            .iter()
            .fold(T::RealField::zero(), |acc, &x| acc + x.modulus_squared());
        let v_t_nrm2 = v_t
            .iter()
            .fold(T::RealField::zero(), |acc, &x| acc + x.modulus_squared());
        let self_term = u_t_nrm2 * v_t_nrm2;

        let mut cross = T::RealField::zero();
        for s in 0..u_cols.len() {
            let mut du = T::zero();
            for i in 0..m {
                du += u_cols[s][i].conjugate() * u_t[i];
            }
            let mut dv = T::zero();
            for j in 0..n {
                dv += v_cols[s][j].conjugate() * v_t[j];
            }
            let prod = du * dv;
            cross += prod.real();
        }
        approx_norm_sq += self_term.clone() + (T::RealField::one() + T::RealField::one()) * cross;

        u_cols.push(u_t);
        v_cols.push(v_t);

        // Stopping: ‖u_t‖‖v_t‖ = sqrt(self_term) ≤ rtol · ‖A_k‖_F.
        //
        // Oversampling: on first trigger, mark convergence but
        // keep pivoting unconditionally for OVERSAMPLE total iterations
        // counted from (and including) the triggering one — i.e. the trigger
        // pivot plus up to OVERSAMPLE-1 further pivots — regardless of
        // whether the criterion re-fires on later iterations. This captures
        // a few extra (typically tiny) rank-1 factors for the downstream
        // small-SVD + rtol truncation to decide on, rather than letting this
        // single stop-step decide the final rank. `oversample_taken` counts
        // completed iterations since (and including) the trigger; we break
        // once that count reaches OVERSAMPLE.
        if self_term.sqrt() <= rtol.clone() * approx_norm_sq.clone().sqrt() {
            converged = true;
        }
        if converged {
            let taken = oversample_taken.unwrap_or(0) + 1;
            oversample_taken = Some(taken);
            if taken >= OVERSAMPLE {
                break;
            }
        }

        // Next pivot row: largest |b[i]| among unused rows.
        let mut next = i_star;
        let mut best_row = T::RealField::zero();
        let mut found_unused = false;
        for i in 0..m {
            if used_rows[i] {
                continue;
            }
            found_unused = true;
            let m_abs = u_cols.last().unwrap()[i].modulus();
            if m_abs > best_row {
                best_row = m_abs;
                next = i;
            }
        }
        // No unused row remains: every row has already been pivoted on, so
        // there is nothing left to select and the loop must terminate here
        // rather than silently reusing `i_star`.
        if !found_unused {
            converged = true;
            break;
        }
        i_star = next;
    }

    if !converged {
        return None;
    }

    let k = u_cols.len();
    let u_a = DMatrix::from_fn(m, k, |i, t| u_cols[t][i]);
    let v_a = DMatrix::from_fn(n, k, |j, t| v_cols[t][j]);
    Some((u_a, v_a))
}

/// Rank-revealing truncated SVD via ACA + dual-QR recompression.
///
/// `K̃ ≈ U_a V_aᵀ = Q_u (R_u R_vᵀ) Q_vᵀ = Q_u M Q_vᵀ`; SVD of the small `k×k`
/// `M = Û Σ V̂ᵀ` gives `K̃ ≈ (Q_u Û) Σ (Q_v V̂)ᵀ`. Falls back to the CPQR
/// [`tsvd`] if ACA fails to converge, guaranteeing the existing accuracy floor.
pub fn tsvd_aca<T>(matrix: &DMatrix<T>, config: TSVDConfig<T>) -> Result<SVDResult<T>, TSVDError>
where
    T: ComplexField
        + RealField
        + Copy
        + nalgebra::RealField
        + std::fmt::Debug
        + ToPrimitive
        + RealNumeric,
{
    let (m, n) = matrix.shape();
    if m == 0 || n == 0 {
        return Err(TSVDError::EmptyMatrix);
    }
    if config.rtol <= Zero::zero() || config.rtol >= One::one() {
        return Err(TSVDError::InvalidTolerance(format!(
            "Tolerance must be in (0, 1), got {:?}",
            config.rtol
        )));
    }

    // Step 1: ACA low-rank factorization on the weighted matrix.
    // For a RealField T, T::RealField == T, so config.rtol is a valid rtol.
    let (u_a, v_a) = match aca_low_rank(matrix, config.rtol, m.min(n)) {
        Some(factors) => factors,
        None => return tsvd(matrix, config), // fallback: CPQR path
    };

    let k = u_a.ncols();
    if k == 0 {
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: nalgebra::DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 2: thin QR of both factors (economy form: Q is m×k / n×k, R is k×k).
    let qr_u = u_a.qr();
    let (q_u, r_u) = (qr_u.q(), qr_u.r());
    let qr_v = v_a.qr();
    let (q_v, r_v) = (qr_v.q(), qr_v.r());

    // Step 3: small SVD of M = R_u · R_vᵀ  (k×k).
    let small = r_u * r_v.transpose();
    let rtol_f64 = RealNumeric::to_f64(config.rtol);
    let svd = svd_decompose(&small, rtol_f64);
    if svd.rank == 0 {
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: nalgebra::DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 4: lift back. U = Q_u Û, V = Q_v V̂.
    let u_full = &q_u * &svd.u;
    let v_full = &q_v * &svd.v;

    Ok(SVDResult {
        u: u_full,
        s: svd.s,
        v: v_full,
        rank: svd.rank,
    })
}

/// Drop-in analogue of [`crate::tsvd::compute_svd_dtensor`] that routes the
/// decomposition through [`tsvd_aca`]. Same f64/Df64 dispatch and output.
pub fn compute_svd_dtensor_aca<T: RealNumeric + 'static>(
    matrix: &DTensor<T, 2>,
) -> (DTensor<T, 2>, Vec<T>, DTensor<T, 2>) {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let matrix_f64 = DMatrix::from_fn(matrix.shape().0, matrix.shape().1, |i, j| {
            RealNumeric::to_f64(matrix[[i, j]])
        });
        let rtol = 2.0 * f64::EPSILON;
        let result =
            tsvd_aca(&matrix_f64, TSVDConfig::new(rtol)).expect("TSVD (ACA) computation failed");

        let u = DTensor::<T, 2>::from_fn([result.u.nrows(), result.u.ncols()], |idx| {
            T::from_f64_unchecked(result.u[(idx[0], idx[1])])
        });
        let s: Vec<T> = result.s.iter().map(|x| T::from_f64_unchecked(*x)).collect();
        let v = DTensor::<T, 2>::from_fn([result.v.nrows(), result.v.ncols()], |idx| {
            T::from_f64_unchecked(result.v[(idx[0], idx[1])])
        });
        (u, s, v)
    } else if TypeId::of::<T>() == TypeId::of::<Df64>() {
        let matrix_df64: DMatrix<Df64> =
            DMatrix::from_fn(matrix.shape().0, matrix.shape().1, |i, j| unsafe {
                mem::transmute_copy(&matrix[[i, j]])
            });
        let rtol = Df64::from(2.0) * Df64::epsilon();
        let result =
            tsvd_aca(&matrix_df64, TSVDConfig::new(rtol)).expect("TSVD (ACA) computation failed");

        let u = DTensor::<T, 2>::from_fn([result.u.nrows(), result.u.ncols()], |idx| {
            T::convert_from(result.u[(idx[0], idx[1])])
        });
        let s: Vec<T> = result.s.iter().map(|x| T::convert_from(*x)).collect();
        let v = DTensor::<T, 2>::from_fn([result.v.nrows(), result.v.ncols()], |idx| {
            T::convert_from(result.v[(idx[0], idx[1])])
        });
        (u, s, v)
    } else {
        panic!("SVD (ACA) is only implemented for f64 and Df64");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tsvd::{TSVDConfig, tsvd};
    use nalgebra::DMatrix;

    /// A rank-3 matrix built as an explicit sum of 3 outer products must be
    /// recovered by ACA to `rtol·σ₀`, using at most a few extra pivots.
    #[test]
    fn recovers_known_rank3_f64() {
        let m = 40;
        let n = 30;
        // Three smooth, linearly independent outer products.
        let cols: Vec<(Vec<f64>, Vec<f64>)> = (0..3)
            .map(|t| {
                let u: Vec<f64> = (0..m)
                    .map(|i| ((i as f64 + 1.0) * (t as f64 + 1.0)).sin())
                    .collect();
                let v: Vec<f64> = (0..n)
                    .map(|j| ((j as f64 + 1.0) / (t as f64 + 2.0)).cos())
                    .collect();
                (u, v)
            })
            .collect();
        let a = DMatrix::<f64>::from_fn(m, n, |i, j| cols.iter().map(|(u, v)| u[i] * v[j]).sum());

        let (u_a, v_a) = aca_low_rank(&a, 1e-12, m.min(n)).expect("ACA should converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < 1e-10,
            "relative reconstruction error too large: {err:e}"
        );
        assert!(u_a.ncols() <= 6, "rank overshoot: {}", u_a.ncols());
    }

    /// Same recovery in extended precision (Df64) — the SVE work type for small eps.
    ///
    /// The matrix is built as an explicit sum of 3 outer products whose factor
    /// vectors are ratios of small integers, computed entirely in `Df64`
    /// arithmetic (no `f64` transcendental functions in the construction).
    /// Such ratios are not exactly representable in binary, so the resulting
    /// matrix is a genuine rank-3 matrix down to Df64 precision (~1e-32), not
    /// merely to f64 precision (~1e-16) as an `f64`-built-then-converted
    /// matrix would be. This lets rtol/error bounds far tighter than 1e-16
    /// actually exercise the extended-precision path instead of being
    /// trivially satisfied by f64-level agreement.
    #[test]
    fn recovers_known_rank3_df64() {
        use crate::Df64;
        let m = 24;
        let n = 20;
        // Three outer products u_t[i] = (i+1)/(t+3+offset), v_t[j] = (j+2)/(t+5+offset),
        // built directly in Df64 from small integer ratios.
        let cols: Vec<(Vec<Df64>, Vec<Df64>)> = (0..3)
            .map(|t| {
                let u: Vec<Df64> = (0..m)
                    .map(|i| Df64::from(i as f64 + 1.0) / Df64::from(t as f64 + 3.0))
                    .collect();
                let v: Vec<Df64> = (0..n)
                    .map(|j| Df64::from(j as f64 + 2.0) / Df64::from(t as f64 + 5.0))
                    .collect();
                (u, v)
            })
            .collect();
        let a = DMatrix::<Df64>::from_fn(m, n, |i, j| {
            cols.iter()
                .fold(Df64::from(0.0), |acc, (u, v)| acc + u[i] * v[j])
        });
        let (u_a, v_a) =
            aca_low_rank(&a, Df64::from(1e-28), m.min(n)).expect("ACA should converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < Df64::from(1e-24),
            "df64 reconstruction error too large: achieved {err:?}"
        );
    }

    /// Genericity guard: `aca_low_rank` must compile and run for `Complex<f64>`.
    /// v1 does not wire the complex path, but this keeps the primitive open.
    #[test]
    fn works_for_complex() {
        use num_complex::Complex64;
        let m = 16;
        let n = 12;
        let a = DMatrix::<Complex64>::from_fn(m, n, |i, j| {
            (0..2)
                .map(|t| {
                    Complex64::new(((i + t) as f64).cos(), ((j + t) as f64).sin())
                        * Complex64::new(((j * (t + 1)) as f64).sin(), 0.3)
                })
                .sum()
        });
        let (u_a, v_a) = aca_low_rank(&a, 1e-12, m.min(n)).expect("converge");
        let approx = &u_a * v_a.transpose();
        let err = (&a - &approx).norm() / a.norm();
        assert!(
            err < 1e-9,
            "complex reconstruction error too large: {err:e}"
        );
    }

    /// Build a smooth, exponentially rank-decaying matrix (a stand-in for the
    /// weighted SVE kernel) and check tsvd_aca's singular values match the
    /// CPQR tsvd's to rtol order, with orthonormal U,V.
    fn smooth_matrix(m: usize, n: usize) -> DMatrix<f64> {
        DMatrix::from_fn(m, n, |i, j| {
            let x = i as f64 / m as f64;
            let y = j as f64 / n as f64;
            (-(x - y).powi(2) * 8.0).exp() + 0.5 * (3.0 * x + 2.0 * y).cos()
        })
    }

    #[test]
    fn tsvd_aca_matches_cpqr_f64() {
        let a = smooth_matrix(60, 50);
        let rtol = 2.0 * f64::EPSILON;
        let cpqr = tsvd(&a, TSVDConfig::new(rtol)).unwrap();
        let aca = tsvd_aca(&a, TSVDConfig::new(rtol)).unwrap();

        let k = cpqr.rank.min(aca.rank);
        assert!(k > 0);
        for i in 0..k {
            let rel = (cpqr.s[i] - aca.s[i]).abs() / cpqr.s[0];
            // rtol requested above is 2·EPSILON (~4.4e-16); 1e-10 is a loose
            // bound well above that, to absorb recompression round-off
            // (dual-QR + small-SVD lift), not a tight rtol-order check.
            assert!(
                rel < 1e-10,
                "sv {i} differs: cpqr={} aca={} rel={rel:e}",
                cpqr.s[i],
                aca.s[i]
            );
        }
        // Orthonormal columns: UᵀU ≈ I.
        let utu = aca.u.transpose() * &aca.u;
        let ident = DMatrix::<f64>::identity(aca.u.ncols(), aca.u.ncols());
        assert!((utu - ident).norm() < 1e-9, "U not orthonormal");
        // Reconstruction error ≲ rtol·σ₀ (with margin).
        let recon = &aca.u * DMatrix::from_diagonal(&aca.s) * aca.v.transpose();
        assert!(
            (&a - &recon).norm() / aca.s[0] < 1e-8,
            "reconstruction too large"
        );
    }

    /// Test fallback: force `aca_low_rank` to genuinely return `None` and check
    /// that `tsvd_aca` then reproduces the CPQR `tsvd` result exactly.
    ///
    /// `aca_low_rank` returns `None` only if it exhausts all `min(m, n)` pivots
    /// without the Frobenius-norm stopping criterion ever firing, i.e. for a
    /// full-rank matrix whose singular values do not decay below `rtol·‖A‖`.
    /// The matrix below is dense (every entry nonzero, so no degenerate pivot
    /// can trigger an early, spuriously-converged `Some`), diagonally
    /// dominant (so it is well-conditioned and genuinely full rank), and has
    /// a flat-ish spectrum (off-diagonal decay is slow enough that the
    /// residual norm never drops below `rtol`). We assert `is_none()`
    /// directly so a future change to `aca_low_rank`'s convergence behavior
    /// that accidentally makes it converge here fails loudly instead of
    /// silently testing the ACA path again under the old name.
    #[test]
    fn tsvd_aca_falls_back_to_cpqr() {
        let m = 50;
        let n = 40;
        let a = DMatrix::from_fn(m, n, |i, j| {
            if i == j {
                3.0
            } else {
                1.0 / (2.0 + (i as f64 - j as f64).abs())
            }
        });
        let rtol = 1e-12;

        // Empirically verify this matrix forces the `None` branch before
        // relying on it: `aca_low_rank` must exhaust all min(m,n) pivots.
        assert!(
            aca_low_rank(&a, rtol, m.min(n)).is_none(),
            "test must exercise the None fallback"
        );

        let cpqr = tsvd(&a, TSVDConfig::new(rtol)).unwrap();
        let aca = tsvd_aca(&a, TSVDConfig::new(rtol)).unwrap();

        // Since aca_low_rank returned None, tsvd_aca took the `None => tsvd(...)`
        // branch verbatim, so the two results must agree to round-off (not
        // merely to rtol order as in the converging-ACA case).
        assert_eq!(cpqr.rank, aca.rank, "fallback must reproduce CPQR's rank");
        for i in 0..cpqr.rank {
            let rel = (cpqr.s[i] - aca.s[i]).abs() / cpqr.s[0];
            // rtol is 1e-12; 1e-10 is a loose bound above the requested rtol
            // to absorb round-off, kept consistent with the sibling f64 test.
            assert!(
                rel < 1e-10,
                "sv {i}: cpqr={} aca={} rel={rel:e}",
                cpqr.s[i],
                aca.s[i]
            );
        }
    }

    use mdarray::DTensor;

    #[test]
    fn dtensor_aca_matches_dtensor_cpqr_f64() {
        let m = 50;
        let n = 40;
        let base = smooth_matrix(m, n);
        let t = DTensor::<f64, 2>::from_fn([m, n], |idx| base[(idx[0], idx[1])]);

        let (_u_ref, s_ref, _v_ref) = crate::tsvd::compute_svd_dtensor(&t);
        let (u, s, _v) = compute_svd_dtensor_aca(&t);

        assert_eq!(s.len(), s_ref.len(), "rank differs");
        for i in 0..s.len() {
            let rel = (s[i] - s_ref[i]).abs() / s_ref[0];
            assert!(rel < 1e-10, "sv {i} differs: rel={rel:e}");
        }
        assert_eq!(u.shape().0, m);
    }
}
