use crate::ray::Ray;
use crate::vec::{
    Point3,
    Vec3, 
};

#[derive(Debug, Default)]
pub struct Camera {
    lower_left: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    origin: Point3,
}

pub struct CameraBuilder {
    camera: Camera,
}

impl Camera {
    pub fn builder() -> CameraBuilder {
        CameraBuilder {
            camera: Default::default(),
        }
    }

    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        let dir = (self.lower_left + u * self.horizontal + v * self.vertical).origin_vec();
        Ray::new(self.origin, dir)
    }
}

impl CameraBuilder {
    pub fn horizontal(mut self, h: Vec3) -> Self {
        self.camera.horizontal = h;
        self
    }

    pub fn vertical(mut self, v: Vec3) -> Self {
        self.camera.vertical = v;
        self
    }

    pub fn origin(mut self, origin: Point3) -> Self {
        self.camera.origin = origin;
        self
    }

    pub fn build(self) -> Camera {
        let mut camera = self.camera;
        camera.lower_left = camera.origin - camera.horizontal / 2.0 - camera.vertical / 2.0 - Vec3::zhat();
        camera
    }
}
