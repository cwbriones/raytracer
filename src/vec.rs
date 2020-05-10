use std::ops::Sub;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Div;
use std::ops::DivAssign;

use std::fmt::Display;

#[derive(Debug, Clone, Copy)]
pub struct Vec3(pub f64, pub f64, pub f64);

// TODO: Point + Vec = Point
//       Point - Point = Vec
//
//       Illegal:
//       Point + Point
pub type Point3 = Vec3;

impl Display for Vec3 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({}, {}, {})", self.0, self.1, self.2)
    }
}

impl Vec3 {
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
}

impl Default for Vec3 {
    fn default() -> Self {
        Vec3(0.0, 0.0, 0.0)
    }
}

impl Add for Vec3 {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Vec3(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2
        )
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
        Vec3(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2
        )
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(
            v.0 * self,
            v.1 * self,
            v.2 * self,
        )
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    #[inline]
    fn mul(self, c: f64) -> Self {
        Vec3(
            self.0 * c,
            self.1 * c,
            self.2 * c,
        )
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
        Vec3(
            self.0 / c,
            self.1 / c,
            self.2 / c,
        )
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, c: f64) {
        self.0 /= c;
        self.1 /= c;
        self.2 /= c;
    }
}
