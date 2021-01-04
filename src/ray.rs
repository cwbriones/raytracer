use crate::vec::{Point3, Vec3};

#[derive(Debug, Clone)]
pub struct Ray {
    origin: Point3,
    dir: Vec3,
    inv_dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Self {
        let inv_dir = Vec3::new(dir.x().recip(), dir.y().recip(), dir.z().recip());
        Self { origin, dir, inv_dir }
    }

    pub fn origin(&self) -> Point3 {
        self.origin
    }

    pub fn dir(&self) -> Vec3 {
        self.dir
    }

    pub fn inv_dir(&self) -> Vec3 {
        self.inv_dir
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin + self.dir * t
    }
}
