use crate::geom::{
    Point3,
    UnitQuaternion,
    Vec3,
};
use crate::material::Material;

#[derive(Debug, Clone)]
pub struct Ray {
    origin: Point3,
    dir: Vec3,
    inv_dir: Vec3,
    time: f64,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3, time: f64) -> Self {
        let inv_dir = Vec3::new(dir.x().recip(), dir.y().recip(), dir.z().recip());
        Self {
            origin,
            dir,
            inv_dir,
            time,
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

    pub fn time(&self) -> f64 {
        self.time
    }
}

pub struct Hit<'m> {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: &'m Material,
    pub u: f64,
    pub v: f64,
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
            u: 0.,
            v: 0.,
        }
    }

    pub fn with_uv(&self, u: f64, v: f64) -> Self {
        Hit {
            point: self.point,
            normal: self.normal,
            t: self.t,
            front_face: self.front_face,
            material: self.material,
            u,
            v,
        }
    }
}

/// An axis-aligned bounding box.
#[derive(Debug, Clone)]
pub struct Aabb {
    min: Point3,
    max: Point3,
}

impl Aabb {
    pub fn new(min: Point3, max: Point3) -> Self {
        Aabb { min, max }
    }

    // Returns a new Aabb with each of its sides being at least
    // length delta.
    pub fn pad(&self, delta: f64) -> Self {
        let x = (self.max.x() - self.min.x()).abs();
        let y = (self.max.y() - self.min.y()).abs();
        let z = (self.max.z() - self.min.z()).abs();

        let dx = if x >= delta { 0.0 } else { delta / 2.0 };
        let dy = if y >= delta { 0.0 } else { delta / 2.0 };
        let dz = if z >= delta { 0.0 } else { delta / 2.0 };

        let min = self.min - Vec3::new(dx, dy, dz);
        let max = self.max + Vec3::new(dx, dy, dz);

        Aabb { min, max }
    }

    #[inline(always)]
    pub fn hit(&self, ray: &Ray, Interval(mut t_min, mut t_max): Interval) -> bool {
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

    pub fn merge(&self, other: &Aabb) -> Aabb {
        let min = self.min.min_pointwise(&other.min);
        let max = self.max.max_pointwise(&other.max);
        Aabb { min, max }
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

    pub fn translate(&self, offset: Vec3) -> Self {
        let min = self.min + offset;
        let max = self.max + offset;
        Aabb { min, max }
    }

    pub fn rotate(&self, axis: Vec3, angle: f64) -> Self {
        let delta = self.max - self.min;
        let midpoint = self.min + 0.5 * delta;

        // Iterate over all 8 points of the Aabb and rotate them about the midpoint.
        // Then compute the new rotated min/max points.
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let rot = UnitQuaternion::rotation(axis, angle);
        for ii in 0..8 {
            let i = ii & 1;
            let j = (ii >> 1) & 1;
            let k = (ii >> 2) & 1;
            let delta = Vec3::new(i as f64, j as f64, k as f64).mul_pointwise(&delta);

            let rotated = rot.rotate_point(midpoint, self.min + delta);
            min = min.min_pointwise(&rotated);
            max = max.max_pointwise(&rotated);
        }
        Aabb { min, max }
    }
}

#[derive(Clone, Copy)]
pub struct Interval(pub f64, pub f64);

impl Interval {
    /// Universe is the largest possible interval.
    pub const UNIVERSE: Interval = Interval(-f64::INFINITY, f64::INFINITY);

    #[inline(always)]
    pub fn contains(&self, t: f64) -> bool {
        self.0 <= t && t <= self.1
    }
}

/// An object within the scene that can be hit by rays.
pub trait Hittable {
    /// Attempt to hit object with `ray`, returning the hit that occurred, if any.
    ///
    /// The hit must not be returned if it occured at time t outside [t_min, t_max].
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit>;
}

/// An object that has a bounding box (i.e. one that is finite in extent).
pub trait Bounded {
    /// Returns a bounding box for this object.
    ///
    /// Ideally this box is as small as possible, but that is not required.
    fn bounding_box(&self) -> Aabb;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroid() {
        let aabb = Aabb::new(Point3::new(-1., -1., -1.), Point3::new(1., 1., 1.));
        let centroid = aabb.centroid();
        assert_eq!(centroid.get(0), 0.0);
        assert_eq!(centroid.get(1), 0.0);
        assert_eq!(centroid.get(2), 0.0);
    }

    #[test]
    fn bounding_box_surface_area() {
        let aabb = Aabb::new(Point3::new(0., 0., 0.), Point3::new(1., 2., 3.));
        assert_eq!(aabb.surface_area(), 22.0);

        let aabb = Aabb::new(Point3::new(-1., -2., -3.), Point3::new(4., 5., 6.));
        assert_eq!(aabb.surface_area(), 286.0);
    }
}
