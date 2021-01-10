use crate::geom::{Point3, Vec3};
use crate::material::Material;
use crate::Hit;
use crate::trace::Ray;
use crate::AABB;

#[derive(Clone)]
pub struct Sphere {
    center: Point3,
    radius: f64,
    material: Material,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Material) -> Self {
        Sphere { center, radius, material }
    }

    #[inline(always)]
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {

        let oc = ray.origin() - self.center;
        let a = ray.dir().square_length();
        let half_b = oc.dot(&ray.dir());
        let c = oc.square_length() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant > 0.0 {
            // The point of intersection.
            let root = discriminant.sqrt();
            let temp = (-half_b - root) / a;
            if t_min < temp && temp < t_max {
                // First root
                let outward_normal = (ray.at(temp) - self.center) / self.radius;
                return Some(Hit::new(
                    ray,
                    temp,
                    outward_normal,
                    &self.material,
                ));
            }
            let temp = (-half_b + root) / a;
            if t_min < temp && temp < t_max {
                // Second root
                let outward_normal = (ray.at(temp) - self.center) / self.radius;
                return Some(Hit::new(
                    ray,
                    temp,
                    outward_normal,
                    &self.material,
                ));
            }
        }
        // Does not hit the sphere.
        None
    }

    pub fn bounding_box(&self) -> AABB {
        let min = self.center - Vec3::new(self.radius, self.radius, self.radius);
        let max = self.center + Vec3::new(self.radius, self.radius, self.radius);
        AABB::new(min, max)
    }
}

