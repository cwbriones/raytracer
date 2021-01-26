use std::fmt::Display;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Sub;

use rand::distributions::Distribution;
use rand::distributions::Standard;
use rand::Rng;
use serde::Deserialize;

// TODO: This can probably be made simpler using a phantom type to distinguish absolute
// vs relative coordinates.

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Point3(f64, f64, f64);

impl Point3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Point3(x, y, z)
    }

    #[inline]
    pub fn min_pointwise(&self, other: &Self) -> Point3 {
        Point3(
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
        )
    }

    #[inline]
    pub fn max_pointwise(&self, other: &Self) -> Point3 {
        Point3(
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
        )
    }

    #[inline]
    pub fn x(&self) -> f64 {
        self.0
    }

    #[inline]
    pub fn y(&self) -> f64 {
        self.1
    }

    #[inline]
    pub fn z(&self) -> f64 {
        self.2
    }

    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        [self.0, self.1, self.2][i]
    }
}

impl From<[f64; 3]> for Point3 {
    fn from(arr: [f64; 3]) -> Self {
        Point3(arr[0], arr[1], arr[2])
    }
}

impl Into<Vec3> for Point3 {
    fn into(self) -> Vec3 {
        Vec3(self.0, self.1, self.2)
    }
}

impl Default for Point3 {
    fn default() -> Self {
        Point3(0.0, 0.0, 0.0)
    }
}

impl Display for Point3 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({}, {}, {})", self.0, self.1, self.2)
    }
}

impl Mul<Point3> for f64 {
    type Output = Point3;

    #[inline]
    fn mul(self, p: Point3) -> Point3 {
        Point3(p.0 * self, p.1 * self, p.2 * self)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Vec3(f64, f64, f64);

impl Display for Vec3 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({}, {}, {})", self.0, self.1, self.2)
    }
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3(x, y, z)
    }

    #[inline]
    pub fn ihat() -> Self {
        Vec3::new(1.0, 0.0, 0.0)
    }

    #[inline]
    pub fn jhat() -> Self {
        Vec3::new(0.0, 1.0, 0.0)
    }

    #[inline]
    pub fn khat() -> Self {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn x(&self) -> f64 {
        self.0
    }
    #[inline]
    pub fn y(&self) -> f64 {
        self.1
    }
    #[inline]
    pub fn z(&self) -> f64 {
        self.2
    }

    #[inline]
    pub fn negate(&self) -> Self {
        Vec3(-self.0, -self.1, -self.2)
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.square_length().sqrt()
    }

    #[inline]
    pub fn square_length(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    #[inline]
    pub fn mul_pointwise(&self, other: &Self) -> Vec3 {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Vec3(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }

    #[inline]
    pub fn unit(&self) -> Self {
        *self / self.length()
    }

    #[inline]
    pub fn rand_within<R: Rng, D: Distribution<f64>>(rng: &mut R, dist: D) -> Self {
        Vec3(dist.sample(rng), dist.sample(rng), dist.sample(rng))
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.0.abs() < s && self.1.abs() < s && self.2.abs() < s
    }

    // #[inline]
    pub fn get(&self, i: usize) -> f64 {
        [self.0, self.1, self.2][i]
    }

    pub fn lerp(&self, to: Vec3, t: f64) -> Vec3 {
        Vec3::new(
            (1.0 - t) * self.0 + t * to.0,
            (1.0 - t) * self.1 + t * to.1,
            (1.0 - t) * self.2 + t * to.2,
        )
    }

    #[inline]
    pub fn rel_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self.0 - other.0).abs() < epsilon
            && (self.1 - other.1).abs() < epsilon
            && (self.2 - other.2).abs() < epsilon
    }
}

impl Distribution<Vec3> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        Vec3(rng.gen(), rng.gen(), rng.gen())
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(arr: [f64; 3]) -> Self {
        Vec3(arr[0], arr[1], arr[2])
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3(0.0, 0.0, 0.0)
    }
}

impl Add<Vec3> for Point3 {
    type Output = Self;

    #[inline]
    fn add(self, other: Vec3) -> Self {
        Point3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl AddAssign<Vec3> for Point3 {
    fn add_assign(&mut self, other: Vec3) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
    }
}

impl Sub for Point3 {
    type Output = Vec3;

    #[inline]
    fn sub(self, other: Self) -> Vec3 {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;

    #[inline]
    fn sub(self, other: Vec3) -> Point3 {
        Point3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Add for Vec3 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
        self.1 += other.1;
        self.2 += other.2;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(v.0 * self, v.1 * self, v.2 * self)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    #[inline]
    fn mul(self, c: f64) -> Self {
        Vec3(self.0 * c, self.1 * c, self.2 * c)
    }
}

impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, c: f64) {
        self.0 *= c;
        self.1 *= c;
        self.2 *= c;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    #[inline]
    fn div(self, c: f64) -> Self {
        Vec3(self.0 / c, self.1 / c, self.2 / c)
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, c: f64) {
        self.0 /= c;
        self.1 /= c;
        self.2 /= c;
    }
}

/// A unit-length quarternion, which can be used for rotations.
#[derive(Debug, Clone, Copy)]
pub struct UnitQuaternion {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl UnitQuaternion {
    /// Create a new unit quarternion from the given components.
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        let inv_mag = (x * x + y * y + z * z + w * w).sqrt().recip();
        UnitQuaternion {
            x: inv_mag * x,
            y: inv_mag * y,
            z: inv_mag * z,
            w: inv_mag * w,
        }
    }

    /// A unit quaternion oriented along the x-axis.
    #[allow(unused)]
    pub const fn i() -> Self {
        UnitQuaternion {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        }
    }

    /// A unit quaternion oriented along the y-axis.
    #[allow(unused)]
    pub const fn j() -> Self {
        UnitQuaternion {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            w: 0.0,
        }
    }

    /// A unit quaternion oriented along the z-axis.
    #[allow(unused)]
    pub const fn k() -> Self {
        UnitQuaternion {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 0.0,
        }
    }

    /// Return the conjugate of this quaternion.
    #[inline]
    pub fn conj(&self) -> Self {
        UnitQuaternion {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Create a quaternion from a desired rotation around an axis.
    pub fn rotation(axis: Vec3, angle: f64) -> Self {
        let cos = (angle / 2.0).cos();
        let sin = (angle / 2.0).sin();

        Self::new(sin * axis.x(), sin * axis.y(), sin * axis.z(), cos)
    }

    /// Rotate a vector using this quaternion.
    #[allow(unused)]
    pub fn rotate_vec(&self, direction: Vec3) -> Vec3 {
        let lhs = UnitQuaternion {
            x: direction.x(),
            y: direction.y(),
            z: direction.z(),
            w: 0.0,
        } * self.conj();
        let out = *self * lhs;

        Vec3::new(out.x, out.y, out.z)
    }

    /// Rotate a point using this quaternion.
    pub fn rotate_point(&self, direction: Point3) -> Point3 {
        let lhs = UnitQuaternion {
            x: direction.x(),
            y: direction.y(),
            z: direction.z(),
            w: 0.0,
        } * *self;
        let out = self.conj() * lhs;

        Point3::new(out.x, out.y, out.z)
    }

    #[cfg(test)]
    #[inline]
    pub fn rel_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self.x - other.x).abs() < epsilon
            && (self.y - other.y).abs() < epsilon
            && (self.z - other.z).abs() < epsilon
            && (self.w - other.w).abs() < epsilon
    }
}

impl Mul<UnitQuaternion> for UnitQuaternion {
    type Output = UnitQuaternion;

    fn mul(self, other: UnitQuaternion) -> UnitQuaternion {
        // See: http://ariya.blogspot.com/2010/07/faster-quaternion-multiplication.html

        let ww = (self.z + self.x) * (other.x + other.y);
        let yy = (self.w - self.y) * (other.w + other.z);
        let zz = (self.w + self.y) * (other.w - other.z);
        let xx = ww + yy + zz;
        let qq = 0.5 * (xx + (self.z - self.x) * (other.x - other.y));

        let w = qq - ww + (self.z - self.y) * (other.y - other.z);
        let x = qq - xx + (self.x + self.w) * (other.x + other.w);
        let y = qq - yy + (self.w - self.x) * (other.y + other.z);
        let z = qq - zz + (self.z + self.y) * (other.w - other.x);

        UnitQuaternion { x, y, z, w }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const EPS: f64 = 1E-8;

    #[test]
    fn quaternion_multiplication_table() {
        let i = UnitQuaternion::i();
        let j = UnitQuaternion::j();
        let k = UnitQuaternion::k();

        let ii = i * i;
        let jj = j * j;
        let kk = k * k;

        let ij = i * j;
        let jk = j * k;
        let ki = k * i;

        let ijk = i * jk;

        let one = UnitQuaternion::new(0.0, 0.0, 0.0, 1.0);
        let neg_one = UnitQuaternion::new(0.0, 0.0, 0.0, -1.0);

        assert!(ii.rel_eq(&neg_one, EPS), "ii = {:?}", ii);
        assert!(jj.rel_eq(&neg_one, EPS), "jj = {:?}", jj);
        assert!(kk.rel_eq(&neg_one, EPS), "kk = {:?}", kk);

        assert!(ij.rel_eq(&k, EPS), "ij = {:?}", ij);
        assert!(jk.rel_eq(&i, EPS), "jk = {:?}", jk);
        assert!(ki.rel_eq(&j, EPS), "ki = {:?}", ki);

        assert!(ijk.rel_eq(&neg_one, EPS), "ijk = {:?}", ijk);
        assert!((one * one).rel_eq(&one, EPS), "1 * 1 = {:?}", one * one);
    }

    #[test]
    fn quaternion_rotation() {
        let rotate_around_z = UnitQuaternion::rotation(Vec3::khat(), ::std::f64::consts::FRAC_PI_2);
        let ihat = Vec3::ihat();

        let rotated = rotate_around_z.rotate_vec(ihat);
        assert!(
            rotated.rel_eq(&Vec3::jhat(), EPS),
            "i-hat rotates 90deg around z-hat to j-hat"
        );
    }

    #[test]
    fn quaternion_rotation_composition() {
        let rotate_around_z = UnitQuaternion::rotation(Vec3::khat(), ::std::f64::consts::FRAC_PI_2);
        let rotate_around_x = UnitQuaternion::rotation(Vec3::ihat(), ::std::f64::consts::FRAC_PI_2);

        // Rotate around z, then around x.
        // around z: i-hat -> j-hat
        // around x: j-hat -> k-hat
        let full_rotation = rotate_around_x * rotate_around_z;

        let ihat = Vec3::ihat();
        let rotated = full_rotation.rotate_vec(ihat);
        assert!(
            rotated.rel_eq(&Vec3::khat(), EPS),
            "Expected k-hat, got {:?}",
            rotated
        );
    }
}
