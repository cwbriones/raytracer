use std::cmp::Ordering;

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
