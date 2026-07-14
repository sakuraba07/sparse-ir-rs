//! Custom numeric traits for high-precision computation
//!
//! This module provides custom numeric traits that work with both f64 and the xprec Df64 backend
//! for high-precision numerical computation in gauss quadrature and matrix operations.

use crate::Df64;
use libm::expm1;
use simba::scalar::ComplexField;
use std::fmt::Debug;

/// Custom numeric trait for high-precision numerical computation
///
/// This trait provides the essential numeric operations needed for gauss module
/// and matrix_from_gauss functions. Supports both f64 and Df64 types.
///
/// Uses standard traits for common operations:
/// - num_traits::Zero for zero()
/// - simba::scalar::ComplexField for mathematical functions (abs, sqrt, cos, sin, exp, etc.)
/// - ComplexField::is_finite() for is_finite()
pub trait CustomNumeric:
    Copy
    + Debug
    + std::fmt::Display
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + num_traits::Zero                    // zero()
    + simba::scalar::ComplexField         // abs, cos, sin, exp, sqrt, etc.
{
    /// Convert from f64 to Self (direct conversion, no Option)
    ///
    /// This is a static method that can be called as T::from_f64_unchecked(x)
    /// Renamed to avoid conflict with num_traits::FromPrimitive::from_f64
    fn from_f64_unchecked(x: f64) -> Self;

    /// Convert from a real numeric type to Self (generic conversion)
    ///
    /// This allows conversion between different real numeric types while
    /// preserving precision. Can be called as T::convert_from(real_value).
    /// The source is bounded to [`RealNumeric`] because the conversion goes
    /// through an `f64` magnitude; complex `Self` types are still constructed
    /// from real sources via [`CustomNumeric::from_f64_unchecked`].
    fn convert_from<U: RealNumeric + 'static>(value: U) -> Self;

    /// Get machine epsilon
    fn epsilon() -> Self;

    /// Get high-precision PI constant
    fn pi() -> Self;

    /// Check if the value is valid (not NaN/infinite)
    fn is_valid(&self) -> bool;

    /// Get absolute value as the same type (convenience method)
    ///
    /// This method wraps ComplexField::abs() and converts the result back to Self
    /// to avoid type conversion issues in generic code.
    /// Note: Only abs() has this problem - other math functions (exp, cos, sin, sqrt)
    /// already return Self directly from ComplexField.
    fn abs_as_same_type(self) -> Self;

    /// Compute exp(self) - 1 with higher precision for small values
    ///
    /// This is more accurate than `self.exp() - 1` when self is close to zero,
    /// avoiding catastrophic cancellation.
    fn exp_m1(self) -> Self;
}

/// Real-only numeric operations, separated from [`CustomNumeric`] so that
/// complex scalar types can implement `CustomNumeric` without providing a total
/// order or a sign-preserving scalar projection.
///
/// `f64` and `Df64` implement both traits, so existing real code paths are
/// unchanged: generic functions that need ordering, `to_f64`, `max`, or `min`
/// simply bound `T: RealNumeric` instead of `T: CustomNumeric`.
pub trait RealNumeric: CustomNumeric + PartialOrd {
    /// Convert to f64 (sign-preserving; only meaningful for real types).
    fn to_f64(self) -> f64;

    /// Maximum of two values (not provided by ComplexField).
    fn max(self, other: Self) -> Self;

    /// Minimum of two values (not provided by ComplexField).
    fn min(self, other: Self) -> Self;
}

/// f64 implementation of CustomNumeric
impl CustomNumeric for f64 {
    fn from_f64_unchecked(x: f64) -> Self {
        x
    }

    fn convert_from<U: RealNumeric + 'static>(value: U) -> Self {
        // Use match to optimize conversion based on the source type
        // Note: Using TypeId for compile-time optimization, but falling back to safe conversion
        match std::any::TypeId::of::<U>() {
            // For f64 to f64, this is just a copy (no conversion needed)
            id if id == std::any::TypeId::of::<f64>() => {
                // Safe: f64 to f64 conversion
                // This is a no-op for f64
                value.to_f64()
            }
            // For Df64 to f64, use the conversion method
            id if id == std::any::TypeId::of::<Df64>() => {
                // Safe: Df64 to f64 conversion
                value.to_f64()
            }
            // Fallback: convert via f64 for unknown types
            _ => value.to_f64(),
        }
    }

    fn epsilon() -> Self {
        f64::EPSILON
    }

    fn pi() -> Self {
        std::f64::consts::PI
    }

    fn is_valid(&self) -> bool {
        num_traits::Float::is_finite(*self)
    }

    fn abs_as_same_type(self) -> Self {
        Self::convert_from(self.abs())
    }

    fn exp_m1(self) -> Self {
        expm1(self)
    }
}

/// f64 implementation of RealNumeric
impl RealNumeric for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
}

/// Df64 implementation of CustomNumeric
impl CustomNumeric for Df64 {
    fn from_f64_unchecked(x: f64) -> Self {
        Df64::from(x)
    }

    fn convert_from<U: RealNumeric + 'static>(value: U) -> Self {
        // Use match to optimize conversion based on the source type
        // Note: Using TypeId for compile-time optimization, but falling back to safe conversion
        match std::any::TypeId::of::<U>() {
            // For f64 to Df64, use the conversion method
            id if id == std::any::TypeId::of::<f64>() => {
                // Safe: f64 to Df64 conversion
                let f64_value = value.to_f64();
                Self::from_f64_unchecked(f64_value)
            }
            // For Df64 to Df64, this is just a copy (no conversion needed)
            id if id == std::any::TypeId::of::<Df64>() => {
                // Safe: Df64 to Df64 conversion (copy)
                // We know U is Df64 from TypeId check, and Df64 implements Copy
                // so we can safely copy the value without losing precision
                // Note: We can't just return `value` directly because Rust's type system
                // doesn't know that U == Df64 at compile time, even though we've verified
                // it at runtime with TypeId. So we need unsafe transmute_copy.
                unsafe {
                    // Safety: We've verified via TypeId that U == Df64, and Df64 is Copy
                    // so this is a no-op copy that preserves all precision (hi and lo parts)
                    std::mem::transmute_copy(&value)
                }
            }
            // Fallback: convert via f64 for unknown types
            _ => Self::from_f64_unchecked(value.to_f64()),
        }
    }

    fn epsilon() -> Self {
        // Df64::EPSILON = f64::EPSILON * f64::EPSILON / 2.0
        let epsilon_value = f64::EPSILON * f64::EPSILON / 2.0;
        Df64::from(epsilon_value)
    }

    fn pi() -> Self {
        Df64::from(std::f64::consts::PI)
    }

    fn is_valid(&self) -> bool {
        let value = *self;
        f64::from(value).is_finite() && !f64::from(value).is_nan()
    }

    fn abs_as_same_type(self) -> Self {
        Self::convert_from(self.abs())
    }

    fn exp_m1(self) -> Self {
        // Use ComplexField::exp_m1() for Df64
        ComplexField::exp_m1(self)
    }
}

/// Df64 implementation of RealNumeric
impl RealNumeric for Df64 {
    fn to_f64(self) -> f64 {
        // Use hi() and lo() methods for better precision
        let hi = self.hi();
        let lo = self.lo();
        hi + lo
    }

    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
}

/// Complex<f64> implementation of CustomNumeric.
///
/// Complex scalars deliberately do NOT implement [`RealNumeric`]: they have no
/// total order and no sign-preserving projection to `f64`. Real-only quantities
/// derived from a complex value (e.g. its modulus) are obtained through
/// [`simba::scalar::ComplexField`] (`abs`, `modulus`, ...), which returns the
/// real field type.
impl CustomNumeric for num_complex::Complex<f64> {
    fn from_f64_unchecked(x: f64) -> Self {
        num_complex::Complex::new(x, 0.0)
    }

    fn convert_from<U: RealNumeric + 'static>(value: U) -> Self {
        // Real source -> complex with zero imaginary part.
        num_complex::Complex::new(value.to_f64(), 0.0)
    }

    fn epsilon() -> Self {
        num_complex::Complex::new(f64::EPSILON, 0.0)
    }

    fn pi() -> Self {
        num_complex::Complex::new(std::f64::consts::PI, 0.0)
    }

    fn is_valid(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn abs_as_same_type(self) -> Self {
        // Modulus, represented as a (purely real) complex value.
        num_complex::Complex::new(self.norm(), 0.0)
    }

    fn exp_m1(self) -> Self {
        // No catastrophic-cancellation-safe complex expm1 here; the plain
        // definition is adequate for the complex paths that use this.
        self.exp() - num_complex::Complex::new(1.0, 0.0)
    }
}

// Note: ScalarOperand implementations for f64 and Df64 are provided by ndarray
// We cannot implement them here due to Orphan Rules, but they are already implemented
// in the ndarray crate for standard numeric types.

// Note: Df64ArrayOps trait and impl removed after ndarray migration
// Array operations should now be done using mdarray Tensor methods directly

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_custom_numeric() {
        let x = 1.5_f64;
        let y = -2.0_f64;

        // Test basic operations
        assert_eq!(x.abs(), 1.5);
        assert_eq!(y.abs(), 2.0);

        // Test mathematical functions
        let cos_x = x.cos();
        assert!(f64::from(cos_x).is_finite());

        let sqrt_x = x.sqrt();
        assert!(f64::from(sqrt_x).is_finite());

        // Test conversion
        assert_eq!(x.to_f64(), 1.5);
        assert_eq!(<f64 as CustomNumeric>::epsilon(), f64::EPSILON);
    }

    #[test]
    fn test_twofloat_custom_numeric() {
        let x = Df64::from_f64_unchecked(1.5);
        let y = Df64::from_f64_unchecked(-2.0);

        // Test basic operations
        assert_eq!(x.abs(), Df64::from_f64_unchecked(1.5));
        assert_eq!(y.abs(), Df64::from_f64_unchecked(2.0));

        // Test mathematical functions
        let cos_x = x.cos();
        assert!(f64::from(cos_x).is_finite());

        let sqrt_x = x.sqrt();
        assert!(f64::from(sqrt_x).is_finite());

        // Test conversion
        let x_f64 = x.to_f64();
        assert!((x_f64 - 1.5).abs() < 1e-15);

        // Test epsilon
        let eps = Df64::epsilon();
        assert!(eps > Df64::from_f64_unchecked(0.0));
        assert!(eps < Df64::from_f64_unchecked(1.0));
    }

    // Note: Df64ArrayOps tests removed after ndarray migration

    #[test]
    fn test_complex_f64_custom_numeric() {
        use num_complex::Complex;
        type C = Complex<f64>;

        // from_f64_unchecked builds a purely real complex.
        let z = <C as CustomNumeric>::from_f64_unchecked(2.5);
        assert_eq!(z, Complex::new(2.5, 0.0));

        // convert_from a real source (f64 and Df64) -> real complex.
        assert_eq!(
            <C as CustomNumeric>::convert_from(3.0_f64),
            Complex::new(3.0, 0.0)
        );
        assert_eq!(
            <C as CustomNumeric>::convert_from(Df64::from_f64_unchecked(-1.5)),
            Complex::new(-1.5, 0.0)
        );

        // abs_as_same_type returns the modulus as a real complex.
        let w = Complex::new(3.0, 4.0);
        assert_eq!(w.abs_as_same_type(), Complex::new(5.0, 0.0));

        // is_valid catches NaN/inf in either component.
        assert!(CustomNumeric::is_valid(&Complex::new(1.0, -2.0)));
        assert!(!CustomNumeric::is_valid(&Complex::new(f64::NAN, 0.0)));
        assert!(!CustomNumeric::is_valid(&Complex::new(0.0, f64::INFINITY)));

        // Arithmetic and ComplexField math are available through the bound.
        let a = Complex::new(1.0, 1.0);
        let b = Complex::new(2.0, -3.0);
        assert_eq!(a + b, Complex::new(3.0, -2.0));
        assert_eq!(a * b, Complex::new(5.0, -1.0));
        assert!(CustomNumeric::is_valid(&a.exp()));
        assert!(CustomNumeric::is_valid(&a.sqrt()));

        // pi/epsilon constants are real-valued complex.
        assert_eq!(<C as CustomNumeric>::pi().im, 0.0);
        assert!(<C as CustomNumeric>::epsilon().re > 0.0);
    }

    #[test]
    fn test_precision_comparison() {
        // Test that Df64 provides higher precision than f64
        let pi_f64 = std::f64::consts::PI;
        let pi_tf = Df64::from_f64_unchecked(pi_f64);

        // Both should be finite
        assert!(pi_f64.is_finite());
        assert!(f64::from(pi_tf).is_finite());

        // Df64 should convert back to f64 with minimal loss
        let pi_back = pi_tf.to_f64();
        assert!((pi_back - pi_f64).abs() < f64::EPSILON);
    }
}
