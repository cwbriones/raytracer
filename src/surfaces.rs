use std::f64::consts::PI;
use std::sync::Arc;

use crate::geom::{
    Point3,
    Vec3,
};
use crate::material::Material;
use crate::trace::{
    Aabb,
    Bounded,
    Hit,
    Hittable,
    Interval,
    Ray,
};

#[derive(Debug, Clone)]
pub struct Sphere {
    center_start: Point3,
    center_end: Point3,
    radius: f64,
    material: Material,
}

impl Sphere {
    pub fn stationary(center: Point3, radius: f64, material: Material) -> Self {
        Sphere {
            center_start: center,
            center_end: center,
            radius,
            material,
        }
    }

    pub fn moving(start: Point3, end: Point3, radius: f64, material: Material) -> Self {
        Sphere {
            center_start: start,
            center_end: end,
            radius,
            material,
        }
    }

    fn center(&self, t: f64) -> Point3 {
        (1. - t) * self.center_start + (t * self.center_end).into()
    }
}

impl Bounded for Sphere {
    fn bounding_box(&self) -> Aabb {
        let min = self.center(0.0) - Vec3::new(self.radius, self.radius, self.radius);
        let max = self.center(0.0) + Vec3::new(self.radius, self.radius, self.radius);
        let b1 = Aabb::new(min, max);

        let min = self.center(1.0) - Vec3::new(self.radius, self.radius, self.radius);
        let max = self.center(1.0) + Vec3::new(self.radius, self.radius, self.radius);
        let b2 = Aabb::new(min, max);

        b1.merge(&b2)
    }
}

impl Hittable for Sphere {
    #[inline(always)]
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        let center = self.center(ray.time());
        let oc = ray.origin() - center;
        let a = ray.dir().square_length();
        let half_b = oc.dot(&ray.dir());
        let c = oc.square_length() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant > 0.0 {
            // The point of intersection.
            let root = discriminant.sqrt();
            let temp = (-half_b - root) / a;
            if interval.contains(temp) {
                // First root
                let point = ray.at(temp);
                let outward_normal = (point - center) / self.radius;
                let (u, v) = get_sphere_uv(outward_normal);
                let hit = Hit::new(ray, temp, outward_normal, &self.material).with_uv(u, v);
                return Some(hit);
            }
            let temp = (-half_b + root) / a;
            if interval.contains(temp) {
                // Second root
                let point = ray.at(temp);
                let outward_normal = (point - center) / self.radius;
                let (u, v) = get_sphere_uv(outward_normal);
                let hit = Hit::new(ray, temp, outward_normal, &self.material).with_uv(u, v);
                return Some(hit);
            }
        }
        // Does not hit the sphere.
        None
    }
}

fn get_sphere_uv(n: Vec3) -> (f64, f64) {
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    let theta = (-n.y()).acos();
    let phi = (-n.z()).atan2(n.x()) + PI;
    (phi / (2.0 * PI), theta / PI)
}

/// A triangular mesh.
///
/// A mesh encapulates a set of vertices along with a set of indices referencing those vertices in
/// order to efficiently store a set of triangles that form a complete model.
///
/// A mesh is not itself a `crate::surface::Hittable`, but provides a method to inject the triangles comprising the
/// mesh into the scene.
#[derive(Debug)]
pub struct Mesh {
    indices: Vec<usize>,
    vertices: Vec<Point3>,
    normals: Vec<Vec3>,
    triangles: usize,
}

#[derive(Debug)]
struct MeshTransform {
    material: Material,
}

impl Mesh {
    /// Create a new mesh given only its vertices.
    ///
    /// This will compute all of the vertex normals for the mesh. If they are already
    /// available through some other means (e.g. loaded from a file) you should instead use
    /// `new_with_normals` to save on unnecessary computation.
    pub fn new(indices: Vec<usize>, vertices: Vec<Point3>) -> Self {
        let mut normals = vec![Vec3::default(); vertices.len()];
        for chunk in indices.chunks_exact(3) {
            let v0 = vertices[chunk[0]];
            let v1 = vertices[chunk[1]];
            let v2 = vertices[chunk[2]];

            let e1 = v1 - v0;
            let e2 = v0 - v2;
            let n = e2.cross(&e1); // Normalizing this vector would make this a non-weighted vertex normal.

            for idx in chunk {
                normals[*idx] += n;
            }
        }
        for n in &mut normals {
            // This may or may not need a negative sign in front.
            // I don't believe it does, mostly based on how the
            // dielectrics end up looking like metals if it's accidentally
            // included (because of total "internal" reflection).
            *n = 1.0 * n.unit();
        }
        Self::new_with_normals(indices, vertices, normals)
    }

    /// Create a new mesh given its vertices and surface normals.
    pub fn new_with_normals(
        indices: Vec<usize>,
        vertices: Vec<Point3>,
        normals: Vec<Vec3>,
    ) -> Self {
        let triangles = indices.len() / 3;
        Self {
            indices,
            vertices,
            normals,
            triangles,
        }
    }

    /// Return an iterator over the triangles contained within this mesh.
    pub fn triangles(self: Arc<Self>, material: Material) -> impl Iterator<Item = Triangle> {
        let transform = Arc::new(MeshTransform { material });
        (0..self.triangles).map(move |idx| Triangle::new(self.clone(), transform.clone(), idx))
    }
}

/// A triangle primitive.
#[derive(Debug, Clone)]
pub struct Triangle {
    /// Location in the mesh.
    idx: usize,
    mesh: Arc<Mesh>,
    // This doesn't save any space now but should once support for generic
    // transformations is added while still reusing the underlying mesh.
    //
    // e.g. scaling, translation, rotation.
    transform: Arc<MeshTransform>,
}

impl Triangle {
    fn new(mesh: Arc<Mesh>, transform: Arc<MeshTransform>, idx: usize) -> Self {
        Triangle {
            mesh,
            transform,
            idx,
        }
    }

    #[inline]
    fn v0(&self) -> Point3 {
        self.mesh.vertices[self.mesh.indices[3 * self.idx]]
    }

    #[inline]
    fn v1(&self) -> Point3 {
        self.mesh.vertices[self.mesh.indices[3 * self.idx + 1]]
    }

    #[inline]
    fn v2(&self) -> Point3 {
        self.mesh.vertices[self.mesh.indices[3 * self.idx + 2]]
    }

    #[inline]
    fn n0(&self) -> Vec3 {
        self.mesh.normals[self.mesh.indices[3 * self.idx]]
    }

    #[inline]
    fn n1(&self) -> Vec3 {
        self.mesh.normals[self.mesh.indices[3 * self.idx + 1]]
    }

    #[inline]
    fn n2(&self) -> Vec3 {
        self.mesh.normals[self.mesh.indices[3 * self.idx + 2]]
    }

    #[inline(always)]
    pub fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        // TODO: Understand and annotate this code.
        let e1 = self.v1() - self.v0();
        let e2 = self.v2() - self.v0();
        let dir = ray.dir();

        let pvec = dir.cross(&e2);
        let det = e1.dot(&pvec);

        if -1E-8 < det && det < 1E-8 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = ray.origin() - self.v0();
        let u = tvec.dot(&pvec) * inv_det;

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let qvec = tvec.cross(&e1);
        let v = dir.dot(&qvec) * inv_det;

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = e2.dot(&qvec) * inv_det;
        if t < 1E-4 || !interval.contains(t) {
            return None;
        }
        let normal = u * self.n1() + v * self.n2() + (1f64 - u - v) * self.n0();
        Some(Hit::new(ray, t, normal, &self.transform.material))
    }

    pub fn bounding_box(&self) -> Aabb {
        let min = self
            .v0()
            .min_pointwise(&self.v1())
            .min_pointwise(&self.v2());
        let max = self
            .v0()
            .max_pointwise(&self.v1())
            .max_pointwise(&self.v2());
        Aabb::new(min, max)
    }
}

/// A quadrilateral primitive.
#[derive(Debug, Clone)]
pub struct Quad {
    // The lower left corner of this quadrilateral.
    pub q: Point3,
    // The first side of the quadrilateral.
    pub u: Vec3,
    // The second side of the quadrilateral.
    pub v: Vec3,
    pub material: Material,

    // cached unit normal
    normal: Vec3,
    d: f64,
    w: Vec3,
}

impl Quad {
    // Creates a new Quadrilateral.
    pub fn new(q: Point3, u: Vec3, v: Vec3, material: Material) -> Self {
        let n = u.cross(&v);
        let normal = n.unit();
        let d = normal.dot(&q.into());
        let w = n / n.dot(&n);
        Quad {
            q,
            u,
            v,
            material,
            normal,
            d,
            w,
        }
    }

    #[inline(always)]
    pub fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        let denom = self.normal.dot(&ray.dir());
        if denom.abs() < 1E-8 {
            return None;
        }
        let t = (self.d - self.normal.dot(&ray.origin().into())) / denom;
        if t < 1E-4 || !interval.contains(t) {
            return None;
        }

        // Determine whether or not the ray lies within the planar shape using its plain
        // coordinates.
        let intersection = ray.at(t);
        let planar_hitpt_vector = intersection - self.q;
        let alpha = self.w.dot(&planar_hitpt_vector.cross(&self.v));
        let beta = self.w.dot(&self.u.cross(&planar_hitpt_vector));

        if is_interior(alpha, beta) {
            // TODO: Set surface coordinates
            // Ray hits the 2d shape. Create the hit
            Some(Hit::new(ray, t, self.normal, &self.material))
        } else {
            None
        }
    }

    pub fn bounding_box(&self) -> Aabb {
        let b = self.q + self.u + self.v;
        Aabb::new(self.q.min_pointwise(&b), self.q.max_pointwise(&b)).pad(0.0001)
    }
}

fn is_interior(a: f64, b: f64) -> bool {
    (0.0..1.0).contains(&a) && (0.0..1.0).contains(&b)
}

#[derive(Debug, Clone)]
pub enum Surface {
    Sphere(Sphere),
    Triangle(Triangle),
    Quad(Quad),
}

impl Hittable for Surface {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        match *self {
            Self::Sphere(ref sphere) => sphere.hit(ray, interval),
            Self::Triangle(ref triangle) => triangle.hit(ray, interval),
            Self::Quad(ref quad) => quad.hit(ray, interval),
        }
    }
}

impl Hittable for Vec<Surface> {
    fn hit(&self, ray: &Ray, Interval(t_min, t_max): Interval) -> Option<Hit> {
        let mut closest = t_max;
        let mut closest_hit = None;
        for o in self {
            if let Some(hit) = o.hit(ray, Interval(t_min, closest)) {
                closest = hit.t;
                closest_hit = Some(hit);
            }
        }
        closest_hit
    }
}

impl Bounded for Surface {
    fn bounding_box(&self) -> Aabb {
        match *self {
            Self::Sphere(ref sphere) => sphere.bounding_box(),
            Self::Triangle(ref triangle) => triangle.bounding_box(),
            Self::Quad(ref quad) => quad.bounding_box(),
        }
    }
}

impl From<Sphere> for Surface {
    fn from(sphere: Sphere) -> Self {
        Surface::Sphere(sphere)
    }
}

impl From<Triangle> for Surface {
    fn from(triangle: Triangle) -> Self {
        Surface::Triangle(triangle)
    }
}

impl From<Quad> for Surface {
    fn from(quad: Quad) -> Self {
        Surface::Quad(quad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_bounding_box() {
        let sphere = Sphere::stationary(
            Point3::new(0., 0., 0.),
            1.0,
            Material::lambertian(Vec3::new(1.0, 1.0, 1.0)),
        );
        let aabb = sphere.bounding_box();

        assert_eq!(aabb.min().get(0), -1.0);
        assert_eq!(aabb.min().get(1), -1.0);
        assert_eq!(aabb.min().get(2), -1.0);

        assert_eq!(aabb.max().get(0), 1.0);
        assert_eq!(aabb.max().get(1), 1.0);
        assert_eq!(aabb.max().get(2), 1.0);
    }
}
