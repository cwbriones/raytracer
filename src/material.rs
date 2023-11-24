use rand::Rng;

use crate::geom::Vec3;
use crate::trace::{
    Hit,
    Ray,
};
use crate::util::RandUtil;

#[derive(Debug, Clone)]
pub struct Material {
    albedo: Vec3,
    kind: MaterialKind,
}

impl Material {
    pub fn lambertian(albedo: Vec3) -> Self {
        Material {
            albedo,
            kind: MaterialKind::Lambertian,
        }
    }

    pub fn metal(albedo: Vec3, fuzz: f64) -> Self {
        Material {
            albedo,
            kind: MaterialKind::Metal(fuzz),
        }
    }

    pub fn dielectric(refractive_index: f64) -> Self {
        Material {
            albedo: Vec3::new(1.0, 1.0, 1.0),
            kind: MaterialKind::Dielectric(refractive_index),
        }
    }

    pub fn diffuse_light(albedo: Vec3) -> Self {
        Material {
            albedo,
            kind: MaterialKind::DiffuseLight,
        }
    }

    #[inline(always)]
    pub fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<(Vec3, Ray)> {
        self.kind.scatter(ray, hit, rng).map(|r| (self.albedo, r))
    }

    pub fn emit(&self, _: &Hit) -> Vec3 {
        match self.kind {
            MaterialKind::DiffuseLight => self.albedo,
            _ => Vec3::default(),
        }
    }
}

#[derive(Debug, Clone)]
enum MaterialKind {
    Lambertian,
    Metal(f64),
    Dielectric(f64),
    DiffuseLight,
}

impl MaterialKind {
    #[inline(always)]
    fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
        match *self {
            Self::Lambertian => Some(lambertian_scatter(ray, hit, rng)),
            Self::Metal(fuzz) => metallic_scatter(fuzz, ray, hit, rng),
            Self::Dielectric(refractive_index) => {
                Some(dielectric_scatter(refractive_index, ray, hit, rng))
            }
            Self::DiffuseLight => None,
        }
    }
}

#[inline(always)]
fn lambertian_scatter<R: Rng>(ray: &Ray, hit: &Hit, rng: &mut R) -> Ray {
    // True lambertian reflection utilizes vectors on the unit sphere,
    // not within it. However, the "approximation" with interior sampling
    // is somewhat more intuitive. The difference amounts to normalizing
    // the vector after sampling.
    //
    // Normalization results in a slightly darker surface since
    // rays are more uniformly scattered.
    //
    // See Section 8.5.
    let mut scatter_direction = hit.normal + rng.gen_in_unit_sphere().unit();

    // Catch degenerate scatter direction
    if scatter_direction.near_zero() {
        scatter_direction = hit.normal;
    }

    Ray::new(hit.point, scatter_direction, ray.time())
}

#[inline(always)]
fn metallic_scatter<R: Rng>(fuzz: f64, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
    let reflected = reflect(&ray.dir(), &hit.normal);
    let scattered = Ray::new(
        hit.point,
        reflected + fuzz * rng.gen_in_unit_sphere(),
        ray.time(),
    );
    if scattered.dir().dot(&hit.normal) > 1e-8 {
        Some(scattered)
    } else {
        None
    }
}

#[inline(always)]
fn dielectric_scatter<R: Rng>(refractive_index: f64, ray: &Ray, hit: &Hit, rng: &mut R) -> Ray {
    let refraction_ratio = if hit.front_face {
        1.0 / refractive_index
    } else {
        refractive_index
    };
    let unit_dir = ray.dir().unit();
    let cos_theta = unit_dir.negate().dot(&hit.normal).min(1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    let scatter_dir =
        if cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.gen::<f64>() {
            // Refraction impossible, must reflect.
            reflect(&unit_dir, &hit.normal)
        } else {
            refract(&unit_dir, &hit.normal, refraction_ratio)
        };
    Ray::new(hit.point, scatter_dir, ray.time())
}

fn reflectance(cosine: f64, refractive_index: f64) -> f64 {
    // Use Schlick's approximation for reflectance.
    //
    // https://en.wikipedia.org/wiki/Schlick%27s_approximation
    let mut r0 = (1.0 - refractive_index) / (1.0 + refractive_index);
    r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

/// Reflect an inbound ray v across a surface given the surface normal n.
fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    *v - (2.0 * v.dot(n)) * *n
}

fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = uv.negate().dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -1.0 * (1.0 - r_out_perp.square_length()).abs().sqrt() * *n;

    r_out_perp + r_out_parallel
}
