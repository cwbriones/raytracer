use std::f64::consts::PI;
use std::sync::Arc;

pub use bvh::Bvh;
pub use bvh::BvhBuilder;
use rand::Rng;

use crate::geom::{
    Point3,
    UnitQuaternion,
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

mod bvh;

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
    Rotated(Rotated),
    Translated(Translated),
    ConstantMedium(ConstantMedium),
    Bvh(Bvh),
}

impl Hittable for Surface {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        match *self {
            Self::Sphere(ref sphere) => sphere.hit(ray, interval),
            Self::Triangle(ref triangle) => triangle.hit(ray, interval),
            Self::Quad(ref quad) => quad.hit(ray, interval),
            Self::Translated(ref t) => t.hit(ray, interval),
            Self::Rotated(ref r) => r.hit(ray, interval),
            Self::ConstantMedium(ref r) => r.hit(ray, interval),
            Self::Bvh(ref bvh) => bvh.hit(ray, interval),
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
            Self::Translated(ref t) => t.bounding_box(),
            Self::Rotated(ref r) => r.bounding_box(),
            Self::ConstantMedium(ref c) => c.bounding_box(),
            Self::Bvh(ref bvh) => bvh.bounding_box(),
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

impl From<Translated> for Surface {
    fn from(value: Translated) -> Self {
        Surface::Translated(value)
    }
}

impl From<Rotated> for Surface {
    fn from(value: Rotated) -> Self {
        Surface::Rotated(value)
    }
}

impl From<ConstantMedium> for Surface {
    fn from(value: ConstantMedium) -> Self {
        Surface::ConstantMedium(value)
    }
}

impl From<Bvh> for Surface {
    fn from(value: Bvh) -> Self {
        Surface::Bvh(value)
    }
}

#[derive(Debug, Clone)]
pub struct Translated {
    hittable: Arc<Surface>,
    offset: Vec3,
    bound: Aabb,
}

impl Translated {
    pub fn new<T: Into<Surface>>(hittable: T, offset: Vec3) -> Self {
        let hittable = hittable.into();
        let bound = hittable.bounding_box();
        Translated {
            hittable: Arc::new(hittable),
            offset,
            bound: bound.translate(offset),
        }
    }
}

impl Hittable for Translated {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        // Transform the ray into object space
        let objray = Ray::new(ray.origin() - self.offset, ray.dir(), ray.time());
        let mut hit = self.hittable.hit(&objray, interval);

        if let Some(ref mut h) = hit {
            h.point += self.offset;
        }
        hit
    }
}

impl Bounded for Translated {
    fn bounding_box(&self) -> Aabb {
        self.bound.clone()
    }
}

#[derive(Debug, Clone)]
pub struct Rotated {
    hittable: Arc<Surface>,
    rot: UnitQuaternion,
    bound: Aabb,
    center: Point3,
}

impl Rotated {
    pub fn new<S: Into<Surface>>(hittable: S, axis: Vec3, angle: f64) -> Self {
        let hittable = hittable.into();
        let bound = hittable.bounding_box();
        Rotated {
            hittable: Arc::new(hittable),
            rot: UnitQuaternion::rotation(axis, angle),
            bound: bound.rotate(axis, angle),
            center: bound.centroid(),
        }
    }
}

impl Hittable for Rotated {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        // Transform the ray into object space
        let dir = self.rot.conj().rotate_vec(ray.dir());
        let origin = self.rot.conj().rotate_point(self.center, ray.origin());
        let objray = Ray::new(origin, dir, ray.time());

        let mut hit = self.hittable.hit(&objray, interval);
        if let Some(ref mut h) = hit {
            h.point = self.rot.rotate_point(self.center, h.point);
            h.normal = self.rot.rotate_vec(h.normal);
        }
        hit
    }
}

impl Bounded for Rotated {
    fn bounding_box(&self) -> Aabb {
        self.bound.clone()
    }
}

pub fn make_box(a: Point3, b: Point3, material: Material) -> Bvh {
    let min = a.min_pointwise(&b);
    let max = a.max_pointwise(&b);

    let dx = Vec3::new(max.x() - min.x(), 0.0, 0.0);
    let dy = Vec3::new(0.0, max.y() - min.y(), 0.0);
    let dz = Vec3::new(0.0, 0.0, max.z() - min.z());
    let surfaces = vec![
        // front
        Quad::new(
            Point3::new(min.x(), min.y(), max.z()),
            dx,
            dy,
            material.clone(),
        ),
        // right
        Quad::new(
            Point3::new(max.x(), min.y(), max.z()),
            dz.negate(),
            dy,
            material.clone(),
        ),
        // back
        Quad::new(
            Point3::new(max.x(), min.y(), min.z()),
            dx.negate(),
            dy,
            material.clone(),
        ),
        // left
        Quad::new(
            Point3::new(min.x(), min.y(), min.z()),
            dz,
            dy,
            material.clone(),
        ),
        // top
        Quad::new(
            Point3::new(min.x(), max.y(), max.z()),
            dx,
            dz.negate(),
            material.clone(),
        ),
        // bottom
        Quad::new(Point3::new(min.x(), min.y(), min.z()), dx, dz, material),
    ]
    .into_iter()
    .map(Into::into)
    .collect::<Vec<_>>();
    Bvh::new_leaf(surfaces)
}

#[derive(Debug, Clone)]
pub struct ConstantMedium {
    boundary: Arc<Surface>,
    neg_inv_density: f64,
    phase_function: Material,
}

impl ConstantMedium {
    pub fn new<S: Into<Surface>>(boundary: S, d: f64, albedo: Vec3) -> Self {
        ConstantMedium {
            boundary: Arc::new(boundary.into()),
            neg_inv_density: -(1.0 / d),
            phase_function: Material::isotropic(albedo),
        }
    }
}

impl Hittable for ConstantMedium {
    fn hit(&self, ray: &Ray, interval: Interval) -> Option<Hit> {
        // Calculate where the ray would hit the boundary
        let mut hit_in = match self.boundary.hit(ray, Interval::UNIVERSE) {
            Some(h) => h,
            None => return None,
        };
        let mut hit_out = match self
            .boundary
            .hit(ray, Interval(hit_in.t + 0.0001, f64::INFINITY))
        {
            Some(h) => h,
            None => return None,
        };

        hit_in.t = hit_in.t.max(interval.0);
        hit_out.t = hit_out.t.min(interval.1);

        // The ray must hit the front before it hits the back
        if hit_in.t >= hit_out.t {
            return None;
        }
        hit_in.t = hit_in.t.max(0.0);
        let ray_length = ray.dir().length();
        let distance_inside_boundary = (hit_out.t - hit_in.t) * ray_length;

        let mut rng = rand::thread_rng(); // FIXME
        let hit_distance = self.neg_inv_density * rng.gen::<f64>().ln();

        if hit_distance > distance_inside_boundary {
            // the ray made it entirely through the volume
            return None;
        }
        let t = hit_in.t + hit_distance / ray_length;
        // FIXME: front_face was always just inferred from normal
        // but in this case both the normal and face are arbitrary
        let normal = Vec3::new(1.0, 0.0, 0.0);
        let mut h = Hit::new(ray, t, normal, &self.phase_function);
        h.front_face = true;
        Some(h)
    }
}

impl Bounded for ConstantMedium {
    //     aabb bounding_box() const override { return boundary->bounding_box(); }
    fn bounding_box(&self) -> Aabb {
        self.boundary.bounding_box()
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
