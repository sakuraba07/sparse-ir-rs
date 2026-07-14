//! CPQR-based sampling-point selection (Cross-Interpolation row selection).
//!
//! Additive helper for the complex generalization of sparse sampling.  Given a
//! *candidate design matrix* whose rows are candidate sampling points and whose
//! columns are basis functions, select a well-conditioned subset of rows
//! (points) via column-pivoted QR on the adjoint.  This is the greedy
//! maxvol-via-CPQR strategy described in the design notes.
//!
//! It works for both real (`f64`) and complex (`Complex<f64>`) scalars because
//! [`ColPivQR`](crate::col_piv_qr::ColPivQR) is generic over `ComplexField`.
//! Nothing in the existing code calls this yet; it is introduced as a new path
//! so that the legacy root/sign-change selection remains untouched.

use crate::col_piv_qr::ColPivQR;
use nalgebra::{ComplexField, DMatrix, DVector};

/// Select up to `n` row indices of `a_cand` that form a well-conditioned
/// submatrix, where rows are candidate sampling points and columns are basis
/// functions.
///
/// The selection runs column-pivoted QR on the adjoint `A^H` (whose *columns*
/// are the candidate points) and returns the first `n` pivot columns. The
/// returned indices are in pivot order (most significant first); callers that
/// need spatial ordering should sort the result themselves.
///
/// If `n` exceeds the number of candidate points, it is clamped. `n == 0`
/// returns an empty vector.
pub fn select_sampling_rows_cpqr<T>(a_cand: &DMatrix<T>, n: usize) -> Vec<usize>
where
    T: ComplexField + Copy,
{
    let n_points = a_cand.nrows();
    let n = n.min(n_points);
    if n == 0 {
        return Vec::new();
    }

    // Adjoint has shape (n_basis, n_points); CPQR pivots its columns, i.e. the
    // candidate points.
    let adj = a_cand.adjoint();
    let qr = ColPivQR::new(adj);
    let perm = qr.p();

    // Replay the pivot swaps onto an index vector of length `n_points`. The
    // permutation records min(n_basis, n_points) swaps; applying them in order
    // to [0, 1, ..., n_points-1] puts the i-th pivot's original column index at
    // position i. The swap indices stay within [0, n_points), so a length
    // `n_points` vector is valid even though the permutation's nominal
    // dimension is min(n_basis, n_points).
    let mut idx: DVector<f64> = DVector::from_iterator(n_points, (0..n_points).map(|i| i as f64));
    perm.permute_rows(&mut idx);

    idx.rows(0, n).iter().map(|&x| x as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    /// Build a candidate matrix whose rows 0,1,2 are large and mutually
    /// orthogonal while the remaining rows are tiny combinations. CPQR must
    /// pick rows {0,1,2}.
    #[test]
    fn selects_dominant_orthogonal_rows_real() {
        // 6 candidate points (rows) x 3 basis functions (columns)
        let a = DMatrix::from_row_slice(
            6,
            3,
            &[
                10.0, 0.0, 0.0, // row 0  -> e1
                0.0, 10.0, 0.0, // row 1  -> e2
                0.0, 0.0, 10.0, // row 2  -> e3
                0.01, 0.01, 0.0, // row 3  tiny
                0.0, 0.01, 0.01, // row 4  tiny
                0.01, 0.0, 0.01, // row 5  tiny
            ],
        );

        let mut sel = select_sampling_rows_cpqr(&a, 3);
        sel.sort_unstable();
        assert_eq!(sel, vec![0, 1, 2], "CPQR should pick the dominant rows");
    }

    #[test]
    fn selects_dominant_orthogonal_rows_complex() {
        let c = |re: f64, im: f64| Complex64::new(re, im);
        let a = DMatrix::from_row_slice(
            6,
            3,
            &[
                c(10.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0), // row 0
                c(0.0, 0.0),
                c(0.0, 10.0),
                c(0.0, 0.0), // row 1 (purely imaginary)
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(7.0, 7.0), // row 2
                c(0.01, 0.0),
                c(0.01, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.01),
                c(0.0, 0.01),
                c(0.0, 0.0),
                c(0.01, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.01),
            ],
        );

        let mut sel = select_sampling_rows_cpqr(&a, 3);
        sel.sort_unstable();
        assert_eq!(
            sel,
            vec![0, 1, 2],
            "complex CPQR should pick the dominant rows"
        );
    }

    #[test]
    fn clamps_and_handles_empty() {
        let a = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        // n larger than the number of rows is clamped to the row count.
        let sel = select_sampling_rows_cpqr(&a, 10);
        assert_eq!(sel.len(), 2);
        // n == 0 yields an empty selection.
        assert!(select_sampling_rows_cpqr(&a, 0).is_empty());
    }
}
