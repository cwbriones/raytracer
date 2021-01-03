use rand::Rng;

use crate::ray::Ray;
use crate::util::RandUtil;
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
    lens_radius: f64,
    basis: (Vec3, Vec3, Vec3),
}

pub struct CameraBuilder {
    from: Point3,
    towards: Point3,
    aspect_ratio: f64,
    vfov_radians: f64,
    vup: Vec3,
    aperture: f64,
    focus_dist: Option<f64>,
}

impl Camera {
    pub fn builder(vfov: f64, aspect_ratio: f64) -> CameraBuilder {
        CameraBuilder {
            aspect_ratio,
            vfov_radians: vfov.to_radians(),
            from: Point3::at(13., 2., 3.),
            towards: Point3::at(0., 0., 0.),
            vup: Vec3::new(0., 1., 0.),
            aperture: 5.0,
            focus_dist: None,
        }
    }

    fn new(
        builder: CameraBuilder,
    ) -> Camera {
        let h = (builder.vfov_radians / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = builder.aspect_ratio * viewport_height;

        // The distance from the camera where everything is in perfect focus.
        //
        // This is not the same as the focal length.
        // That is the distance between the projection and image planes.
        let focus_dist = builder.focus_dist.unwrap_or_else(|| {
            (builder.from - builder.towards).length()
        });
        
        // Form an orthonormal basis for our camera system.
        let w = (builder.from - builder.towards).unit();
        let u = builder.vup.cross(&w).unit();
        let v = w.cross(&u);

        let origin = builder.from;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;
        let lens_radius = builder.aperture / 2.0;
        Camera {
            origin,
            horizontal,
            vertical,
            lower_left,
            lens_radius,
            basis: (u, v, w),
        }
    }

    pub fn get_ray<R: Rng>(&self, rng: &mut R, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * rng.gen_in_unit_disk();
        let offset = self.basis.0 * rd.x() + self.basis.1 * rd.y();
        let offset_origin = self.origin - offset;

        Ray::new(
            offset_origin,
            (self.lower_left + s * self.horizontal + t * self.vertical) - offset_origin
        )
    }
}

impl CameraBuilder {
    pub fn from(self, from: Point3) -> Self {
        CameraBuilder { from, ..self }
    }

    pub fn towards(self, towards: Point3) -> Self {
        CameraBuilder { towards, ..self }
    }

    pub fn focus_dist(self, focus_dist: f64) -> Self {
        CameraBuilder { focus_dist: Some(focus_dist), ..self }
    }

    pub fn aperture(self, aperture: f64) -> Self {
        CameraBuilder { aperture, ..self }
    }

    pub fn build(self) -> Camera {
        Camera::new(self)
    }
}
