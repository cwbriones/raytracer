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
    from: Point3,
    towards: Point3,
    aspect_ratio: f64,
    vfov_radians: f64,
    vup: Vec3,
}

impl Camera {
    pub fn builder(vfov: f64, aspect_ratio: f64) -> CameraBuilder {
        CameraBuilder {
            aspect_ratio,
            vfov_radians: vfov.to_radians(),
            from: Point3::at(0., 0., 0.),
            towards: Point3::at(0., 0., 1.),
            vup: Vec3::new(0., 1., 0.),
        }
    }

    fn new(
        builder: CameraBuilder,
    ) -> Camera {
        let h = (builder.vfov_radians / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = builder.aspect_ratio * viewport_height;

        let focal_length = 1.0;
        
        // Form an orthonormal basis for our camera system.
        let w: Vec3 = (builder.from - builder.towards).unit();
        let u = builder.vup.cross(&w).unit();
        let v = w.cross(&u);

        let origin = builder.from;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - focal_length * w;
        Camera {
            origin,
            horizontal,
            vertical,
            lower_left
        }
    }

    pub fn get_ray(&self, s: f64, v: f64) -> Ray {
        let dir = (self.lower_left + s * self.horizontal + v * self.vertical) - self.origin;
        Ray::new(self.origin, dir)
    }
}

impl CameraBuilder {
    pub fn from(self, from: Point3) -> Self {
        CameraBuilder { from, ..self }
    }

    pub fn towards(self, towards: Point3) -> Self {
        CameraBuilder { towards, ..self }
    }

    pub fn build(self) -> Camera {
        Camera::new(self)
    }
}
