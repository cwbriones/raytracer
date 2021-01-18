use crate::geom::{
    Point3,
    Vec3,
};
use crate::material::Material;

#[derive(Debug, Clone)]
pub struct Ray {
    origin: Point3,
    dir: Vec3,
    inv_dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Self {
        let inv_dir = Vec3::new(dir.x().recip(), dir.y().recip(), dir.z().recip());
        Self {
            origin,
            dir,
            inv_dir,
        }
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

#[derive(Clone)]
pub struct Hit<'m> {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: &'m Material,
}

impl<'m> Hit<'m> {
    pub fn new(ray: &Ray, t: f64, outward_normal: Vec3, material: &'m Material) -> Self {
        let point = ray.at(t);
        let front_face = ray.dir().dot(&outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            outward_normal.negate()
        };
        Hit {
            point,
            normal,
            t,
            front_face,
            material,
        }
    }
}

#[derive(Clone)]
pub struct AABB {
    min: Point3,
    max: Point3,
}

impl AABB {
    pub fn new(min: Point3, max: Point3) -> Self {
        AABB { min, max }
    }

    #[inline(always)]
    pub fn hit(&self, ray: &Ray, mut t_min: f64, mut t_max: f64) -> bool {
        // Maybe do this with SIMD intrinsics?
        // Check all directions at once
        for a in 0..3 {
            let inv_d = ray.inv_dir().get(a);
            let mut t0 = (self.min.get(a) - ray.origin().get(a)) * inv_d;
            let mut t1 = (self.max.get(a) - ray.origin().get(a)) * inv_d;
            if inv_d < 0.0 {
                ::std::mem::swap(&mut t0, &mut t1);
            }
            t_min = if t0 > t_min { t0 } else { t_min };
            t_max = if t1 < t_max { t1 } else { t_max };
            if t_max < t_min {
                return false;
            }
        }
        true
    }

    pub fn merge(&self, other: &AABB) -> AABB {
        let min = self.min.min_pointwise(&other.min);
        let max = self.max.max_pointwise(&other.max);
        AABB { min, max }
    }

    pub fn surface_area(&self) -> f64 {
        let lengths = self.max - self.min;
        2.0 * (lengths.x() * lengths.y() + lengths.y() * lengths.z() + lengths.z() * lengths.x())
    }

    pub fn centroid(&self) -> Point3 {
        let offset = 0.5 * (self.max - self.min);
        self.min + offset
    }

    pub fn min(&self) -> &Point3 {
        &self.min
    }

    #[cfg(test)]
    pub fn max(&self) -> &Point3 {
        &self.max
    }
}

/// An object within the scene that can be hit by rays.
pub trait Hittable {
    /// Attempt to hit object with `ray`, returning the hit that occurred, if any.
    ///
    /// The hit must not be returned if it occured at time t outside [t_min, t_max].
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

/// An object that has a bounding box (i.e. one that is finite in extent).
pub trait Bounded {
    /// Returns a bounding box for this object.
    ///
    /// Ideally this box is as small as possible, but that is not required.
    fn bounding_box(&self) -> AABB;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid() {
        let aabb = AABB::new(Point3::new(-1., -1., -1.), Point3::new(1., 1., 1.));
        let centroid = aabb.centroid();
        assert_eq!(centroid.get(0), 0.0);
        assert_eq!(centroid.get(1), 0.0);
        assert_eq!(centroid.get(2), 0.0);
    }

    #[test]
    fn bounding_box_surface_area() {
        let aabb = AABB::new(Point3::new(0., 0., 0.), Point3::new(1., 2., 3.));
        assert_eq!(aabb.surface_area(), 22.0);

        let aabb = AABB::new(Point3::new(-1., -2., -3.), Point3::new(4., 5., 6.));
        assert_eq!(aabb.surface_area(), 286.0);
    }
}
