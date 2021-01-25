use std::sync::Arc;

use crate::geom::{
    Point3,
    Vec3,
};
use crate::material::Material;
use crate::trace::{
    Bounded,
    Hit,
    Hittable,
    Ray,
    AABB,
};

#[derive(Debug, Clone)]
pub struct Sphere {
    center: Point3,
    radius: f64,
    material: Material,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Material) -> Self {
        Sphere {
            center,
            radius,
            material,
        }
    }
}

impl Bounded for Sphere {
    fn bounding_box(&self) -> AABB {
        let min = self.center - Vec3::new(self.radius, self.radius, self.radius);
        let max = self.center + Vec3::new(self.radius, self.radius, self.radius);
        AABB::new(min, max)
    }
}

impl Hittable for Sphere {
    #[inline(always)]
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
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
                return Some(Hit::new(ray, temp, outward_normal, &self.material));
            }
            let temp = (-half_b + root) / a;
            if t_min < temp && temp < t_max {
                // Second root
                let outward_normal = (ray.at(temp) - self.center) / self.radius;
                return Some(Hit::new(ray, temp, outward_normal, &self.material));
            }
        }
        // Does not hit the sphere.
        None
    }
}

/// A triangular mesh.
///
/// A mesh encapulates a set of vertices along with a set of indices referencing those vertices in
/// order to efficiently store a set of triangles that form a complete model.
///
/// A mesh is not itself an object, but provides a method to inject the triangles comprising the
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
    pub fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
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

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let qvec = tvec.cross(&e1);
        let v = dir.dot(&qvec) * inv_det;

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = e2.dot(&qvec) * inv_det;
        if t < 1E-4 || t < t_min || t > t_max {
            return None;
        }
        let normal = u * self.n1() + v * self.n2() + (1f64 - u - v) * self.n0();
        Some(Hit::new(ray, t, normal, &self.transform.material))
    }

    pub fn bounding_box(&self) -> AABB {
        let min = self
            .v0()
            .min_pointwise(&self.v1())
            .min_pointwise(&self.v2());
        let max = self
            .v0()
            .max_pointwise(&self.v1())
            .max_pointwise(&self.v2());
        AABB::new(min, max)
    }
}

#[derive(Debug, Clone)]
pub enum Surface {
    Sphere(Sphere),
    Triangle(Triangle),
}

impl Hittable for Surface {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        match *self {
            Self::Sphere(ref sphere) => sphere.hit(ray, t_min, t_max),
            Self::Triangle(ref triangle) => triangle.hit(ray, t_min, t_max),
        }
    }
}

impl Bounded for Surface {
    fn bounding_box(&self) -> AABB {
        match *self {
            Self::Sphere(ref sphere) => sphere.bounding_box(),
            Self::Triangle(ref triangle) => triangle.bounding_box(),
        }
    }
}

impl Into<Surface> for Sphere {
    fn into(self) -> Surface {
        Surface::Sphere(self)
    }
}

impl Into<Surface> for Triangle {
    fn into(self) -> Surface {
        Surface::Triangle(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_bounding_box() {
        let sphere = Sphere::new(
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
