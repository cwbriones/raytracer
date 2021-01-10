use std::cmp::{
    // Eq,
    // Ord,
    Ordering,
};

use rand::distributions::Uniform;
use rand::Rng;

use crate::geom::Vec3;

pub trait RandUtil: Rng {
    fn gen_in_unit_sphere(&mut self) -> Vec3;
    fn gen_in_unit_disk(&mut self) -> Vec3;
}

impl<R> RandUtil for R
where
    R: Rng,
{
    fn gen_in_unit_sphere(&mut self) -> Vec3 {
        let dist = Uniform::new(-1.0, 1.0);
        loop {
            let v = Vec3::rand_within(self, dist);
            let square_len = v.square_length();
            if square_len < 1.0 {
                return v;
            }
        }
    }

    fn gen_in_unit_disk(&mut self) -> Vec3 {
        loop {
            let x = self.gen_range(-1.0..1.0);
            let y = self.gen_range(-1.0..1.0);
            let v = Vec3::new(x, y, 0.0);
            if v.square_length() < 1.0 {
                return v;
            }
        }
    }
}

/// Extension trait to add a clamp method to floats.
///
/// This conflicts with a method to be stabilized in Rust 1.50, hence the odd spelling.
pub trait Klamp {
    fn klamp(self, min: Self, max: Self) -> Self;
}

impl Klamp for f64 {
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn klamp(self, min: f64, max: f64) -> f64 {
        // This is copied directly from std::f64::clamp,
        // with the exception of debug_assert!
        debug_assert!(min <= max);
        let mut x = self;
        if x < min {
            x = min;
        }
        if x > max {
            x = max;
        }
        x
    }
}

#[derive(PartialEq, Clone, Copy)]
pub struct NonNan(f64);

impl NonNan {
    pub fn new(val: f64) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
}

impl Eq for NonNan {}

impl PartialOrd for NonNan {
    fn partial_cmp(&self, other: &NonNan) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
