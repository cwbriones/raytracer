use crate::vec::{Point3, Vec3};

#[derive(Debug, Clone)]
pub struct Ray {
    origin: Point3,
    dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    pub fn origin(&self) -> Point3 {
        self.origin
    }

    pub fn dir(&self) -> Vec3 {
        self.dir
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin * t + self.dir
    }
}
