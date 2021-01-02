use rand::Rng;
use rand::distributions::Standard;
use rand::distributions::Distribution;

use std::ops::Sub;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Div;
use std::ops::DivAssign;
use std::fmt::Display;

// TODO: This can probably be made simpler using a phantom type to distinguish absolute
// vs relative coordinates.

#[derive(Debug, Clone, Copy)]
pub struct Point3(f64, f64, f64);

impl Point3 {
    #[inline]
    pub fn at(x: f64, y: f64, z: f64) -> Self {
        Point3(x, y, z)
    }

    #[inline]
    pub fn origin_vec(self) -> Vec3 {
        Vec3::new(self.0, self.1, self.2)
    }

    // #[inline]
    // pub fn x(&self) -> f64 { self.0 }
    // #[inline]
    // pub fn y(&self) -> f64 { self.1 }
    // #[inline]
    // pub fn z(&self) -> f64 { self.2 }
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
#[derive(Debug, Clone, Copy)]
pub struct Vec3(f64, f64, f64);

impl Display for Vec3 {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "({}, {}, {})", self.0, self.1, self.2)
    }
}

impl Vec3 {
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
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
    pub fn zhat() -> Self {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[inline]
    pub fn x(&self) -> f64 { self.0 }
    #[inline]
    pub fn y(&self) -> f64 { self.1 }
    #[inline]
    pub fn z(&self) -> f64 { self.2 }

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

    #[inline]
    pub fn rand_within<R: Rng, D: Distribution<f64>>(rng: &mut R, dist: D) -> Self {
        Vec3(
            dist.sample(rng),
            dist.sample(rng),
            dist.sample(rng),
        )
    }
}

impl Distribution<Vec3> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        Vec3(
            rng.gen(),
            rng.gen(),
            rng.gen(),
        )
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
        Point3(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2
        )
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
        Vec3(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2
        )
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Point3;

    #[inline]
    fn sub(self, other: Vec3) -> Point3 {
        Point3(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2
        )
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
