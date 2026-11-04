//! EXPERIMENT ONLY (not for upstream): effect of the CPQR pivot rule.
//! Run twice, with SPIR_CPQR_PIVOT unset (entry pivot, as main) and =norm.
//! Prints one line per quantity: `key<TAB>v1 v2 ...` (values {:.17e}).

use sparse_ir::sve::{SVEOptions, TworkType, compute_sve, compute_sve_dense, compute_sve_with};
use sparse_ir::tsvd::compute_svd_dtensor;
use sparse_ir::{
    Bosonic, ExpandableKernel, Fermionic, FiniteTempBasis, LogisticKernel, LogisticKernelWindowed,
    RegularizedBoseKernel, matrix::Mat,
};

fn emit(key: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{key}\t{}", s.join(" "));
}

fn sve_cases<K: ExpandableKernel + Clone>(name: &str, k: K, lam: f64) {
    for eps in [1e-6, 1e-8] {
        let r = compute_sve_with(
            k.clone(),
            &SVEOptions::new().epsilon(eps).twork(TworkType::Float64),
        )
        .unwrap();
        emit(&format!("f64\t{name}\t{lam:e}\t{eps:e}"), r.s());
    }
    // Df64 reference by the default (ACA) path; CPQR only in its fallback
    let r = compute_sve(k.clone(), None, None, None, TworkType::Float64X2).unwrap();
    emit(&format!("ref\t{name}\t{lam:e}\t-"), r.s());
    if lam <= 1e4 {
        let r = compute_sve_dense(k.clone(), None, TworkType::Float64X2, false).unwrap();
        emit(&format!("df64dense\t{name}\t{lam:e}\t-"), r.s());
    }
}

fn main() {
    let lams = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
    for &lam in &lams {
        sve_cases("logistic", LogisticKernel::new(lam).unwrap(), lam);
        sve_cases("regbose", RegularizedBoseKernel::new(lam).unwrap(), lam);
        sve_cases(
            "windowed",
            LogisticKernelWindowed::from_physical(lam, -0.3, 1.0).unwrap(),
            lam,
        );
        for eps in [1e-6, 1e-8] {
            let b = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
                LogisticKernel::new(lam).unwrap(),
                lam,
                Some(eps),
                None,
            )
            .unwrap();
            emit(&format!("basisF\tlogistic\t{lam:e}\t{eps:e}"), b.s());
            emit(
                &format!("tauF\tlogistic\t{lam:e}\t{eps:e}"),
                &b.default_tau_sampling_points().unwrap(),
            );
            let b = FiniteTempBasis::<RegularizedBoseKernel, Bosonic>::new(
                RegularizedBoseKernel::new(lam).unwrap(),
                lam,
                Some(eps),
                None,
            )
            .unwrap();
            emit(&format!("basisB\tregbose\t{lam:e}\t{eps:e}"), b.s());
            emit(
                &format!("tauB\tregbose\t{lam:e}\t{eps:e}"),
                &b.default_tau_sampling_points().unwrap(),
            );
        }
    }

    // Synthetic: a large isolated entry against a spread-out column of
    // larger norm, both near the 2ε cut of the QR.
    let m = 102;
    let delta = 4e-16;
    let a = Mat::<f64>::from_fn([m, 3], |idx| {
        let (i, j) = (idx[0], idx[1]);
        match j {
            0 => (i == 0) as u8 as f64,
            1 => {
                if i == 1 {
                    delta
                } else {
                    0.0
                }
            }
            _ => {
                if i >= 2 {
                    0.9 * delta
                } else {
                    0.0
                }
            }
        }
    });
    let (_, s, _) = compute_svd_dtensor(&a).unwrap();
    emit("synthetic\t-\t-\t-", &s);
}
