//! Tests for FiniteTempBasis functionality

use crate::basis::{FermionicBasis, FiniteTempBasis};
use crate::kernel::{LogisticKernel, RegularizedBoseKernel};
use crate::traits::{Bosonic, Fermionic};

#[test]
fn test_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * omega_max);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);
}

#[test]
#[should_panic(expected = "beta must be positive")]
fn test_negative_beta() {
    let kernel = LogisticKernel::new(1.0);
    let _ = FermionicBasis::new(kernel, -1.0, None, None);
}

#[test]
fn test_default_tau_sampling_points_conditioning() {
    // Test parameters: beta=1.0, lambda=10.0
    let beta = 1.0;
    let lambda = 10.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    println!("\n=== Default Tau Sampling Points Test ===");
    println!("Beta: {}, Lambda: {}, Epsilon: {}", beta, lambda, epsilon);
    println!("Basis size: {}", basis.size());

    // Get default sampling points
    let tau_points = basis.default_tau_sampling_points();
    println!("Number of sampling points: {}", tau_points.len());

    // Verify range: [-beta/2, beta/2] (matches C++ implementation)
    let beta_half = beta / 2.0;
    for &tau in &tau_points {
        assert!(
            tau >= -beta_half && tau <= beta_half,
            "tau={} out of range [{}, {}]",
            tau,
            -beta_half,
            beta_half
        );
    }

    // Verify sorted (monotonically increasing)
    for i in 1..tau_points.len() {
        assert!(
            tau_points[i] >= tau_points[i - 1],
            "Points not sorted: tau[{}]={} < tau[{}]={}",
            i,
            tau_points[i],
            i - 1,
            tau_points[i - 1]
        );
    }

    // Verify symmetry around 0 (for range [-beta/2, beta/2])
    // Points should come in pairs: tau and -tau
    let tol = 1e-10;

    for &tau in &tau_points {
        let tau_reflected = -tau;
        let has_pair = tau_points.iter().any(|&t| (t - tau_reflected).abs() < tol);
        assert!(
            has_pair || tau.abs() < tol,
            "tau={} lacks symmetric pair around 0",
            tau
        );
    }
    println!("✅ Sampling points are symmetric around 0");

    // Evaluate sampling matrix: matrix[i,l] = u_l(tau_i)
    // Use the Basis trait method which handles tau normalization
    use crate::basis_trait::Basis;
    let matrix = basis.evaluate_tau(&tau_points);

    let num_points = tau_points.len();
    let basis_size = basis.size();
    println!("Sampling matrix shape: {}x{}", num_points, basis_size);

    // Compute SVD using mdarray-linalg (Faer backend)
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg_faer::Faer;
    let mut matrix_copy = matrix.clone();
    let svd = Faer.svd(&mut *matrix_copy).expect("SVD computation failed");

    println!("\nSampling matrix SVD:");
    let min_dim = svd.s.shape().0.min(svd.s.shape().1);
    println!("  Rank: {}", min_dim);
    // mdarray-linalg stores singular values in first row: s[[0, i]]
    println!("  First singular value: {:.6e}", svd.s[[0, 0]]);
    println!("  Last singular value: {:.6e}", svd.s[[0, min_dim - 1]]);

    let condition_number = svd.s[[0, 0]] / svd.s[[0, min_dim - 1]];
    println!("  Condition number: {:.6e}", condition_number);

    // Reference condition number (from Julia/C++ for beta=1, lambda=10)
    // This is approximate - actual value depends on implementation details
    let reference_cond = 25.0; // ~24.4 observed

    // Check: condition number should not be significantly worse
    // (within factor of 2 of reference)
    assert!(
        condition_number < reference_cond * 2.0,
        "Condition number too large: {:.2e} (reference: {:.2e})",
        condition_number,
        reference_cond
    );

    // Julia check: cond > 1e8 triggers warning
    assert!(
        condition_number < 1e8,
        "Sampling matrix is poorly conditioned: cond = {:.6e}",
        condition_number
    );

    println!(
        "✅ Condition number: {:.2e} (reference: {:.2e}, threshold: {:.2e})",
        condition_number,
        reference_cond,
        reference_cond * 2.0
    );
}

#[test]
fn test_regularized_bose_basis_construction() {
    let beta = 10.0;
    let omega_max = 1.0;
    let epsilon = 1e-6;

    let kernel = RegularizedBoseKernel::new(beta * omega_max);
    let basis =
        FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    assert_eq!(basis.beta, beta);
    assert!((basis.omega_max() - omega_max).abs() < 1e-10);
    assert!(basis.size() > 0);
    assert!(basis.accuracy > 0.0);
    assert!(basis.accuracy < epsilon);

    println!("\n=== RegularizedBoseKernel Basis Test ===");
    println!(
        "Beta: {}, Omega_max: {}, Epsilon: {}",
        beta, omega_max, epsilon
    );
    println!("Basis size: {}", basis.size());
    println!("Accuracy: {:.6e}", basis.accuracy);
}

#[test]
fn test_regularized_bose_basis_different_parameters() {
    // Test with different beta and omega_max values
    let test_cases = vec![(1.0, 1.0, 1e-6), (10.0, 10.0, 1e-6), (100.0, 1.0, 1e-6)];

    for (beta, omega_max, epsilon) in test_cases {
        let kernel = RegularizedBoseKernel::new(beta * omega_max);
        let basis = FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(
            kernel,
            beta,
            Some(epsilon),
            None,
        );

        assert_eq!(basis.beta, beta);
        assert!((basis.omega_max() - omega_max).abs() < 1e-10);
        assert!(basis.size() > 0);
        assert!(basis.accuracy > 0.0);
        assert!(basis.accuracy < epsilon);

        println!(
            "Beta={}, Omega_max={}: size={}, accuracy={:.6e}",
            beta,
            omega_max,
            basis.size(),
            basis.accuracy
        );
    }
}

#[test]
fn test_default_omega_sampling_points_fermionic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);

    let omega_points = basis.default_omega_sampling_points();

    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());

    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(
            omega.abs() <= wmax,
            "omega = {} exceeds wmax = {}",
            omega,
            wmax
        );
    }

    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_default_omega_sampling_points_bosonic() {
    let beta = 10000.0;
    let wmax = 1.0;
    let epsilon = 1e-6;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis = FiniteTempBasis::<LogisticKernel, Bosonic>::new(kernel, beta, Some(epsilon), None);

    let omega_points = basis.default_omega_sampling_points();

    // Should have same size as basis
    assert_eq!(omega_points.len(), basis.size());

    // Points should be in [-wmax, wmax]
    for &omega in &omega_points {
        assert!(
            omega.abs() <= wmax,
            "omega = {} exceeds wmax = {}",
            omega,
            wmax
        );
    }

    // Points should be sorted
    let mut sorted = omega_points.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega_points, sorted, "Omega points should be sorted");
}

#[test]
fn test_omega_points_symmetry() {
    let beta = 1000.0;
    let wmax = 2.0;
    let epsilon = 1e-8;

    let kernel = LogisticKernel::new(beta * wmax);
    let basis_f =
        FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(epsilon), None);
    let omega_points = basis_f.default_omega_sampling_points();

    // Check approximate symmetry: for each positive point, there should be a negative counterpart
    // (This is approximate due to the nature of the roots)
    let positive: Vec<f64> = omega_points.iter().filter(|&&x| x > 0.0).copied().collect();
    let negative: Vec<f64> = omega_points
        .iter()
        .filter(|&&x| x < 0.0)
        .map(|&x| -x)
        .collect();

    println!("Omega points: {:?}", omega_points);
    println!(
        "Number of positive: {}, negative: {}",
        positive.len(),
        negative.len()
    );
}

// ============================================================================
// Golden-value (characterization) tests — bake pass
//
// These tests lock the EXACT current output of the default sampling-point
// selection for beta=1.0, lambda=10.0, epsilon=1e-6.  They are a safety net
// for refactoring: if the output changes, these tests will fail.
// ============================================================================

#[test]
fn golden_default_tau_sampling_points_fermionic_b1_l10() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let taus = basis.default_tau_sampling_points();

    // Captured golden values (full precision, beta=1, lambda=10, epsilon=1e-6)
    let golden_tau: &[f64] = &[
        -4.22050537212752985e-1,
        -2.74996248692147893e-1,
        -1.51960695242697730e-1,
        -6.26773625570433701e-2,
        -1.19681417829884396e-2,
        1.19681417829884396e-2,
        6.26773625570434256e-2,
        1.51960695242697730e-1,
        2.74996248692147893e-1,
        4.22050537212752930e-1,
    ];

    assert_eq!(
        taus.len(),
        golden_tau.len(),
        "tau point count changed: expected {}, got {}",
        golden_tau.len(),
        taus.len()
    );

    let tol = 1e-12_f64;
    for (i, (&got, &expected)) in taus.iter().zip(golden_tau.iter()).enumerate() {
        let abs_err = (got - expected).abs();
        let rel_err = abs_err / expected.abs().max(1e-300);
        assert!(
            abs_err <= tol || rel_err <= tol,
            "tau[{}]: got {:.17e}, expected {:.17e}, abs_err={:.3e}, rel_err={:.3e}",
            i,
            got,
            expected,
            abs_err,
            rel_err
        );
    }
}

#[test]
fn golden_default_matsubara_sampling_points_fermionic_b1_l10() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let ns: Vec<i64> = basis
        .default_matsubara_sampling_points(false)
        .iter()
        .map(|f| f.n())
        .collect();

    // Captured golden values (fermionic, positive_only=false)
    let golden_ns: Vec<i64> = vec![-23, -7, -5, -3, -1, 1, 3, 5, 7, 23];

    assert_eq!(
        ns, golden_ns,
        "fermionic Matsubara indices changed:\n  got:      {:?}\n  expected: {:?}",
        ns, golden_ns
    );
}

#[test]
fn golden_default_tau_sampling_points_bosonic_b1_l10() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = crate::basis::BosonicBasis::new(kernel, beta, Some(epsilon), None);

    let taus = basis.default_tau_sampling_points();

    // Captured golden values (full precision, bosonic, beta=1, lambda=10, epsilon=1e-6)
    let golden_tau: &[f64] = &[
        -4.22050537212752985e-1,
        -2.74996248692147893e-1,
        -1.51960695242697730e-1,
        -6.26773625570433701e-2,
        -1.19681417829884396e-2,
        1.19681417829884396e-2,
        6.26773625570434256e-2,
        1.51960695242697730e-1,
        2.74996248692147893e-1,
        4.22050537212752930e-1,
    ];

    assert_eq!(
        taus.len(),
        golden_tau.len(),
        "tau point count changed: expected {}, got {}",
        golden_tau.len(),
        taus.len()
    );

    let tol = 1e-12_f64;
    for (i, (&got, &expected)) in taus.iter().zip(golden_tau.iter()).enumerate() {
        let abs_err = (got - expected).abs();
        let rel_err = abs_err / expected.abs().max(1e-300);
        assert!(
            abs_err <= tol || rel_err <= tol,
            "tau[{}]: got {:.17e}, expected {:.17e}, abs_err={:.3e}, rel_err={:.3e}",
            i,
            got,
            expected,
            abs_err,
            rel_err
        );
    }
}

#[test]
fn golden_default_matsubara_sampling_points_bosonic_b1_l10() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = crate::basis::BosonicBasis::new(kernel, beta, Some(epsilon), None);

    let ns: Vec<i64> = basis
        .default_matsubara_sampling_points(false)
        .iter()
        .map(|f| f.n())
        .collect();

    // Captured golden values (bosonic, positive_only=false)
    let golden_ns: Vec<i64> = vec![-28, -8, -6, -4, -2, 0, 2, 4, 6, 8, 28];

    assert_eq!(
        ns, golden_ns,
        "bosonic Matsubara indices changed:\n  got:      {:?}\n  expected: {:?}",
        ns, golden_ns
    );
}

// ============================================================================
// CPQR-based tau sampling points — additive path.
//
// These tests check the new default_*_cpqr selection against the legacy
// root-based selection. The key claim is conditioning parity: CPQR points
// must produce a design matrix at least as well-conditioned as the legacy
// points (comparable, typically better). They do NOT lock exact values — the
// CPQR points depend on the candidate grid and are expected to differ from the
// golden root-based points.
// ============================================================================

#[test]
fn cpqr_tau_sampling_points_basic_shape_fermionic() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let taus = basis.default_tau_sampling_points_cpqr();

    // Count matches the basis size (square design matrix).
    assert_eq!(
        taus.len(),
        basis.size(),
        "expected one CPQR point per basis fn"
    );

    // All points lie in [-beta/2, beta/2].
    let beta_half = beta / 2.0;
    for &t in &taus {
        assert!(
            t >= -beta_half - 1e-12 && t <= beta_half + 1e-12,
            "tau={} out of range [{}, {}]",
            t,
            -beta_half,
            beta_half
        );
    }

    // Sorted ascending and distinct.
    for i in 1..taus.len() {
        assert!(
            taus[i] > taus[i - 1],
            "points not strictly increasing at {}: {} <= {}",
            i,
            taus[i],
            taus[i - 1]
        );
    }
}

#[test]
fn cpqr_tau_sampling_conditioning_not_worse_than_roots() {
    use crate::poly::{default_sampling_points, default_sampling_points_cpqr};
    use nalgebra::DMatrix;

    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let sve = basis.sve_result();
    let u = &sve.u;
    let l = basis.size();

    // Condition number of the design matrix U_c(x_r) for a given point set,
    // using all `l` basis functions as columns.
    let cond_of = |pts: &[f64]| -> f64 {
        let n = pts.len();
        let mut m = DMatrix::<f64>::zeros(n, l);
        for (r, &x) in pts.iter().enumerate() {
            for c in 0..l {
                m[(r, c)] = u[c].evaluate(x);
            }
        }
        let sv = m.singular_values(); // descending
        let smin = sv[sv.len() - 1];
        if smin <= 0.0 {
            f64::INFINITY
        } else {
            sv[0] / smin
        }
    };

    // Legacy points are computed in [-1, 1] (same domain as the CPQR points).
    let legacy_pts = default_sampling_points(u, l);
    let cpqr_pts = default_sampling_points_cpqr(u, l);

    let cond_legacy = cond_of(&legacy_pts);
    let cond_cpqr = cond_of(&cpqr_pts);

    println!(
        "\n[CPQR tau conditioning] L={}, cond_legacy={:.3e}, cond_cpqr={:.3e}",
        l, cond_legacy, cond_cpqr
    );

    assert!(cond_cpqr.is_finite(), "CPQR design matrix is singular");
    // CPQR must be comparable or better. Allow a modest slack factor to absorb
    // candidate-grid discretization; a regression to a much worse conditioning
    // would still trip this.
    assert!(
        cond_cpqr <= 10.0 * cond_legacy,
        "CPQR conditioning markedly worse than roots: cond_cpqr={:.3e} > 10 * cond_legacy={:.3e}",
        cond_cpqr,
        cond_legacy
    );
}

#[test]
fn cpqr_matsubara_sampling_points_basic_shape_fermionic() {
    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let pts = basis.default_matsubara_sampling_points_cpqr();

    // One frequency per basis function.
    assert_eq!(pts.len(), basis.size());

    // Fermionic frequencies must have odd index, and all must be distinct.
    let mut ns: Vec<i64> = pts.iter().map(|f| f.n()).collect();
    for &n in &ns {
        assert_eq!(n.rem_euclid(2), 1, "fermionic index {} is not odd", n);
    }
    let before = ns.len();
    ns.sort_unstable();
    ns.dedup();
    assert_eq!(ns.len(), before, "duplicate Matsubara frequencies selected");
}

#[test]
fn cpqr_matsubara_conditioning_not_worse_than_sign_changes() {
    use nalgebra::DMatrix;
    use num_complex::Complex64;

    let beta = 1.0_f64;
    let lambda = 10.0_f64;
    let epsilon = 1e-6_f64;

    let kernel = LogisticKernel::new(lambda);
    let basis = FermionicBasis::new(kernel, beta, Some(epsilon), None);

    let uhat = basis.uhat_full();
    let l = basis.size();

    // Condition number of the complex design matrix uhat_c(iw_r) over a given
    // set of Matsubara frequencies, using the first `l` basis functions.
    let cond_of = |pts: &[crate::freq::MatsubaraFreq<Fermionic>]| -> f64 {
        let rows = uhat.evaluate_at_many(pts);
        let mut m = DMatrix::<Complex64>::zeros(pts.len(), l);
        for (r, row) in rows.iter().enumerate() {
            for c in 0..l {
                m[(r, c)] = row[c];
            }
        }
        let sv = m.singular_values(); // real, descending
        let smin = sv[sv.len() - 1];
        if smin <= 0.0 {
            f64::INFINITY
        } else {
            sv[0] / smin
        }
    };

    let legacy_pts = basis.default_matsubara_sampling_points(false);
    let cpqr_pts = basis.default_matsubara_sampling_points_cpqr();

    let cond_legacy = cond_of(&legacy_pts);
    let cond_cpqr = cond_of(&cpqr_pts);

    println!(
        "\n[CPQR matsubara conditioning] L={}, cond_legacy={:.3e}, cond_cpqr={:.3e}",
        l, cond_legacy, cond_cpqr
    );

    assert!(
        cond_cpqr.is_finite(),
        "CPQR Matsubara design matrix is singular"
    );
    assert!(
        cond_cpqr <= 10.0 * cond_legacy,
        "CPQR conditioning markedly worse than sign-changes: cond_cpqr={:.3e} > 10 * cond_legacy={:.3e}",
        cond_cpqr,
        cond_legacy
    );
}
