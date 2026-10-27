//! Comparison of the Matsubara sampling-point selections on fits of noisy
//! data (review of #342).
//!
//! Methods: `sign` (sign changes of û_L, centrosymmetric kernels only), `id`
//! (greedy pivoted QR on the (Re û, Im û) pair rows, no exchanges), `full`
//! (ID + exchanges minimizing ‖F_S⁻¹T_S‖² + ‖F_S⁻¹‖²) and `notail` (as
//! `full`, without the aliasing tail T_S).
//!
//! Data: G(τ) = -Σ_p c_p K(τ, ω_p) with the logistic kernel
//! K(τ, ω) = e^{-τω} / (1 + e^{-βω}), so G(iν) = Σ_p c_p / (iν - ω_p) for
//! fermions and Σ_p c_p tanh(βω_p/2) / (iν - ω_p) for bosons, and the exact
//! IR coefficients are g_l = -S_l Σ_p c_p V_l(ω_p). Gaussian noise of
//! relative size σ (times max |G(iν)|) on the real and imaginary parts, the
//! same draw for the same frequency in every method.
//!
//! Usage: cargo run --release -p sparse-ir --example bench_matsubara_selection -- OUT.csv [MAX_LAMBDA]

use num_complex::Complex64;
use sparse_ir::basis::{FiniteTempBasis, MatsubaraPointMethod};
use sparse_ir::freq::MatsubaraFreq;
use sparse_ir::kernel::{BasisKernel, LogisticKernel, LogisticKernelWindowed};
use sparse_ir::traits::{Bosonic, Fermionic, Statistics, StatisticsType};
use sparse_ir::{MatsubaraSampling, MatsubaraSamplingPositiveOnly};
use std::fmt::Write as _;
use std::io::Write as _;

const TRIALS: usize = 100;
const SIGMAS: [f64; 3] = [0.0, 1e-8, 1e-4];
const SEED: u64 = 0x5eed_2026_1017;

/// splitmix64
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Two standard normal numbers for (trial, n), the same in every method
fn normals(trial: usize, n: i64) -> (f64, f64) {
    let a = mix(SEED ^ mix(trial as u64) ^ mix(n as u64 ^ 0xabcd_ef01_2345_6789));
    let b = mix(a);
    let u1 = ((a >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let u2 = ((b >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f64::consts::PI * u2;
    (r * t.cos(), r * t.sin())
}

/// Logistic kernel e^{-τω} / (1 + e^{-βω}), evaluated stably
fn kernel(tau: f64, beta: f64, w: f64) -> f64 {
    if w >= 0.0 {
        (-tau * w).exp() / (1.0 + (-beta * w).exp())
    } else {
        ((beta - tau) * w).exp() / (1.0 + (beta * w).exp())
    }
}

/// Poles (ω_p, c_p) of a test spectrum in the window [a, b]
fn spectrum(name: &str, a: f64, b: f64) -> Vec<(f64, f64)> {
    let map = |x: f64| a + (x + 1.0) / 2.0 * (b - a);
    match name {
        "poles" => [(-0.8, 0.2), (-0.2, 0.3), (0.3, 0.4), (0.9, 0.1)]
            .iter()
            .map(|&(x, c)| (map(x), c))
            .collect(),
        "semicircle" => {
            // ρ(ω) = 2/(π r²) √(r² - (ω - m)²) by the trapezoidal rule in θ
            let (m, r) = ((a + b) / 2.0, 0.45 * (b - a));
            let k = 400;
            (1..k)
                .map(|j| {
                    let th = std::f64::consts::PI * j as f64 / k as f64;
                    let w = 2.0 / std::f64::consts::PI * th.sin().powi(2) * std::f64::consts::PI
                        / k as f64;
                    (m + r * th.cos(), w)
                })
                .collect()
        }
        "lowpoles" => vec![(0.02 * b, 0.5), (0.02 * a, 0.5)],
        _ => unreachable!(),
    }
}

const SPECTRA: [&str; 3] = ["poles", "semicircle", "lowpoles"];

/// Exact G(iν_n) of the poles
fn giw<S: StatisticsType>(poles: &[(f64, f64)], beta: f64, n: i64) -> Complex64 {
    let iv = Complex64::new(0.0, std::f64::consts::PI * n as f64 / beta);
    poles
        .iter()
        .map(|&(w, c)| {
            let weight = match S::STATISTICS {
                Statistics::Fermionic => 1.0,
                Statistics::Bosonic => (beta * w / 2.0).tanh(),
            };
            c * weight / (iv - w)
        })
        .sum()
}

fn mean_p90(mut xs: Vec<f64>) -> (f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.sort_by(f64::total_cmp);
    let i = ((xs.len() as f64 * 0.9).ceil() as usize).clamp(1, xs.len()) - 1;
    (mean, xs[i])
}

enum Sampler<S: StatisticsType> {
    Full(MatsubaraSampling<S>),
    Positive(MatsubaraSamplingPositiveOnly<S>),
}

#[allow(clippy::too_many_arguments)]
fn run_basis<K, S>(
    label: &str,
    kernel_k: K,
    beta: f64,
    eps: f64,
    window: (f64, f64),
    centrosymmetric: bool,
    out: &mut String,
) where
    K: BasisKernel,
    S: StatisticsType + 'static,
{
    let t0 = std::time::Instant::now();
    let basis = FiniteTempBasis::<K, S>::new(kernel_k, beta, Some(eps), None).unwrap();
    let l_size = basis.size();
    let zeta = match S::STATISTICS {
        Statistics::Fermionic => 1,
        Statistics::Bosonic => 0,
    };
    let lambda = basis.lambda();

    // Dense grids
    let taus: Vec<f64> = (0..=2000)
        .map(|i| beta * (1.0 - (std::f64::consts::PI * i as f64 / 2000.0).cos()) / 2.0)
        .collect();
    let mut ks: Vec<i64> = (0..=200).collect();
    let kmax = (1000.0 * lambda).max(1e4);
    for j in 0..=200 {
        ks.push((200.0 * (kmax / 200.0).powf(j as f64 / 200.0)) as i64);
    }
    ks.sort();
    ks.dedup();
    let mut ns: Vec<i64> = ks.iter().map(|k| 2 * k + zeta).collect();
    let neg: Vec<i64> = ns.iter().filter(|&&n| n != 0).map(|n| -n).collect();
    ns.extend(neg);
    let freqs: Vec<MatsubaraFreq<S>> = ns.iter().map(|&n| MatsubaraFreq::new(n).unwrap()).collect();
    let u_tau = basis.u().evaluate_at_many(&taus); // (L, ntau)
    let uhat_iw = basis.uhat().evaluate_at_many(&freqs); // [nfreq][L]
    let s = basis.s();
    let v = basis.v();

    // Spectra: exact coefficients and data on the dense grids
    struct Truth {
        name: &'static str,
        poles: Vec<(f64, f64)>,
        g: Vec<f64>,
        g_tau: Vec<f64>,
        g_iw: Vec<Complex64>,
        scale_iw: f64,
    }
    let truths: Vec<Truth> = SPECTRA
        .iter()
        .map(|&name| {
            let poles = spectrum(name, window.0, window.1);
            let g: Vec<f64> = (0..l_size)
                .map(|l| {
                    -s[l]
                        * poles
                            .iter()
                            .map(|&(w, c)| c * v[l].evaluate(w))
                            .sum::<f64>()
                })
                .collect();
            let g_tau: Vec<f64> = taus
                .iter()
                .map(|&t| {
                    -poles
                        .iter()
                        .map(|&(w, c)| c * kernel(t, beta, w))
                        .sum::<f64>()
                })
                .collect();
            let g_iw: Vec<Complex64> = ns.iter().map(|&n| giw::<S>(&poles, beta, n)).collect();
            let scale_iw = g_iw.iter().map(|z| z.norm()).fold(0.0, f64::max);
            Truth {
                name,
                poles,
                g,
                g_tau,
                g_iw,
                scale_iw,
            }
        })
        .collect();

    // Sanity check: the expansion with the exact coefficients reproduces the
    // data to about ε
    let mut check = String::new();
    for t in &truths {
        let scale_tau = t.g_tau.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let e_tau = (0..taus.len())
            .map(|i| {
                let fit: f64 = (0..l_size).map(|l| t.g[l] * u_tau[[l, i]]).sum();
                (fit - t.g_tau[i]).abs()
            })
            .fold(0.0, f64::max)
            / scale_tau;
        let e_iw = (0..ns.len())
            .map(|i| {
                let fit: Complex64 = (0..l_size).map(|l| t.g[l] * uhat_iw[i][l]).sum();
                (fit - t.g_iw[i]).norm()
            })
            .fold(0.0, f64::max)
            / t.scale_iw;
        let _ = write!(check, " {}: τ {e_tau:.1e} iν {e_iw:.1e};", t.name);
    }
    eprintln!(
        "{label} L={l_size} ({:.1?}) truncation check:{check}",
        t0.elapsed()
    );

    let mut methods: Vec<&str> = Vec::new();
    if centrosymmetric {
        methods.push("sign");
    }
    // `final`: the pair selection of the final code (ID, A-optimal
    // oversampling with mitigate), to be compared with `id` / `id-A` of the
    // benchmark branch
    methods.push("final");
    for &method in &methods {
        for mitigate in [false, true] {
            for positive_only in [false, true] {
                let points = match method {
                    "sign" => basis.matsubara_sampling_points(
                        positive_only,
                        mitigate,
                        MatsubaraPointMethod::SignChanges,
                    ),
                    _ => basis.matsubara_sampling_points(
                        positive_only,
                        mitigate,
                        MatsubaraPointMethod::PairSelection,
                    ),
                }
                .unwrap();
                let pn: Vec<i64> = points.iter().map(|f| f.n()).collect();
                let (sampler, cond) = if positive_only {
                    let smp = MatsubaraSamplingPositiveOnly::with_sampling_points(&basis, points)
                        .unwrap();
                    let c = smp.condition_number().unwrap();
                    (Sampler::Positive(smp), c)
                } else {
                    let smp = MatsubaraSampling::with_sampling_points(&basis, points).unwrap();
                    let c = smp.condition_number().unwrap();
                    (Sampler::Full(smp), c)
                };
                for t in &truths {
                    for &sigma in &SIGMAS {
                        let trials = if sigma == 0.0 { 1 } else { TRIALS };
                        let (mut ec, mut et, mut ei) = (Vec::new(), Vec::new(), Vec::new());
                        for trial in 0..trials {
                            let data: Vec<Complex64> = pn
                                .iter()
                                .map(|&n| {
                                    let (x, y) = normals(trial, n);
                                    giw::<S>(&t.poles, beta, n)
                                        + Complex64::new(x, y) * (sigma * t.scale_iw)
                                })
                                .collect();
                            let coef: Vec<Complex64> = match &sampler {
                                Sampler::Full(smp) => smp.fit(&data).unwrap(),
                                Sampler::Positive(smp) => smp
                                    .fit(&data)
                                    .unwrap()
                                    .into_iter()
                                    .map(|x| Complex64::new(x, 0.0))
                                    .collect(),
                            };
                            let gnorm = t.g.iter().map(|x| x * x).sum::<f64>().sqrt();
                            let dc = (0..l_size)
                                .map(|l| (coef[l] - t.g[l]).norm_sqr())
                                .sum::<f64>()
                                .sqrt();
                            ec.push(dc / gnorm);
                            let scale_tau = t.g_tau.iter().map(|x| x.abs()).fold(0.0, f64::max);
                            let dt = (0..taus.len())
                                .map(|i| {
                                    let fit: Complex64 =
                                        (0..l_size).map(|l| coef[l] * u_tau[[l, i]]).sum();
                                    (fit - t.g_tau[i]).norm()
                                })
                                .fold(0.0, f64::max);
                            et.push(dt / scale_tau);
                            let di = (0..ns.len())
                                .map(|i| {
                                    let fit: Complex64 =
                                        (0..l_size).map(|l| coef[l] * uhat_iw[i][l]).sum();
                                    (fit - t.g_iw[i]).norm()
                                })
                                .fold(0.0, f64::max);
                            ei.push(di / t.scale_iw);
                        }
                        let (cm, c9) = mean_p90(ec);
                        let (tm, t9) = mean_p90(et);
                        let (im, i9) = mean_p90(ei);
                        let _ = writeln!(
                            out,
                            "{label},{:?},{lambda:e},{eps:e},{l_size},{method},{mitigate},{positive_only},{},{sigma:e},{},{cond:e},{cm:e},{c9:e},{tm:e},{t9:e},{im:e},{i9:e}",
                            S::STATISTICS,
                            t.name,
                            pn.len()
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "{label} {:?} L={l_size} done ({:.1?})",
        S::STATISTICS,
        t0.elapsed()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("output CSV path");
    let max_lambda: f64 = args.get(2).map_or(1e6, |x| x.parse().unwrap());
    let mut file = std::fs::File::create(path).unwrap();
    writeln!(
        file,
        "kernel,stat,lambda,eps,L,method,mitigate,positive_only,spectrum,sigma,npts,cond,coef_mean,coef_p90,tau_mean,tau_p90,iw_mean,iw_p90"
    )
    .unwrap();
    let windows: [(&str, f64, f64, bool, bool); 3] = [
        ("win(-0.3;1)", -0.3, 1.0, true, true),
        ("win(-1;0.02)", -1.0, 0.02, true, true),
        ("win(-0.01;1)", -0.01, 1.0, false, true),
    ];
    for lambda in [1e2, 1e4, 1e6] {
        if lambda > max_lambda {
            continue;
        }
        for eps in [1e-6, 1e-10, 1e-15] {
            let mut out = String::new();
            let beta = lambda;
            let k = LogisticKernel::new(lambda).unwrap();
            run_basis::<_, Fermionic>("logistic", k, beta, eps, (-1.0, 1.0), true, &mut out);
            run_basis::<_, Bosonic>("logistic", k, beta, eps, (-1.0, 1.0), true, &mut out);
            for &(name, a, b, fermi, bose) in &windows {
                let k = LogisticKernelWindowed::from_physical(beta, a, b).unwrap();
                if fermi {
                    run_basis::<_, Fermionic>(name, k.clone(), beta, eps, (a, b), false, &mut out);
                }
                if bose {
                    run_basis::<_, Bosonic>(name, k, beta, eps, (a, b), false, &mut out);
                }
            }
            file.write_all(out.as_bytes()).unwrap();
            file.flush().unwrap();
        }
    }
}
