use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::geom::{
    Point3,
    Vec3,
};
use crate::trace::{
    Hit,
    Ray,
};
use crate::util::RandUtil;

#[derive(Debug, Clone)]
pub struct Material {
    kind: MaterialKind,
    texture: Texture,
}

impl Material {
    pub fn lambertian<T: Into<Texture>>(texture: T) -> Self {
        Material {
            texture: texture.into(),
            kind: MaterialKind::Lambertian,
        }
    }

    pub fn metal(albedo: Vec3, fuzz: f64) -> Self {
        Material {
            texture: Texture::SolidColor(albedo),
            kind: MaterialKind::Metal(fuzz),
        }
    }

    pub fn dielectric(refractive_index: f64) -> Self {
        Material {
            texture: Texture::SolidColor(Vec3::new(1.0, 1.0, 1.0)),
            kind: MaterialKind::Dielectric(refractive_index),
        }
    }

    pub fn diffuse_light(albedo: Vec3) -> Self {
        Material {
            texture: Texture::SolidColor(albedo),
            kind: MaterialKind::DiffuseLight,
        }
    }

    pub fn isotropic(albedo: Vec3) -> Self {
        Material {
            texture: Texture::SolidColor(albedo),
            kind: MaterialKind::Isotropic,
        }
    }

    #[inline(always)]
    pub fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<(Vec3, Ray)> {
        self.kind.scatter(ray, hit, rng).map(|r| {
            let albedo = self.texture.color_at(hit.u, hit.v, &hit.point);
            (albedo, r)
        })
    }

    pub fn emit(&self, hit: &Hit) -> Vec3 {
        if let MaterialKind::DiffuseLight = self.kind {
            self.texture.color_at(hit.u, hit.v, &hit.point)
        } else {
            Vec3::default()
        }
    }
}

#[derive(Debug, Clone)]
enum MaterialKind {
    Lambertian,
    Isotropic,
    Metal(f64),
    Dielectric(f64),
    DiffuseLight,
}

impl MaterialKind {
    #[inline(always)]
    fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
        match *self {
            Self::Lambertian => Some(lambertian_scatter(ray, hit, rng)),
            Self::Isotropic => Some(isotropic_scatter(ray, hit, rng)),
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

fn isotropic_scatter<R: Rng>(ray: &Ray, hit: &Hit, rng: &mut R) -> Ray {
    let dir = rng.gen_in_unit_sphere();
    Ray::new(hit.point, dir, ray.time())
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

#[derive(Debug, Clone)]
pub enum Texture {
    SolidColor(Vec3),
    Checker(Checker),
    UVChecker(UVChecker),
    Perlin(NoiseTexture),
}

impl Texture {
    pub fn checker(scale: f64, even: Vec3, odd: Vec3) -> Self {
        Self::Checker(Checker {
            inv_scale: scale.recip(),
            even,
            odd,
        })
    }

    pub fn uv_checker(scale: f64, even: Vec3, odd: Vec3) -> Self {
        Self::UVChecker(UVChecker { scale, even, odd })
    }

    pub fn noise(scale: f64, noise: Arc<PerlinNoise>) -> Self {
        Texture::Perlin(NoiseTexture::new(scale, noise))
    }

    pub fn color_at(&self, u: f64, v: f64, point: &Point3) -> Vec3 {
        match self {
            Texture::SolidColor(c) => *c,
            Texture::Checker(texture) => texture.color_at(point),
            Texture::UVChecker(texture) => texture.color_at(u, v),
            Texture::Perlin(texture) => texture.marbled(point),
        }
    }
}

impl From<Vec3> for Texture {
    fn from(value: Vec3) -> Self {
        Texture::SolidColor(value)
    }
}

#[derive(Debug, Clone)]
pub struct Checker {
    inv_scale: f64,
    even: Vec3,
    odd: Vec3,
}

impl Checker {
    fn color_at(&self, point: &Point3) -> Vec3 {
        let x = (self.inv_scale * point.x()).floor() as isize;
        let y = (self.inv_scale * point.y()).floor() as isize;
        let z = (self.inv_scale * point.z()).floor() as isize;
        if (x + y + z) % 2 == 0 {
            self.even
        } else {
            self.odd
        }
    }
}

#[derive(Debug, Clone)]
pub struct UVChecker {
    scale: f64,
    even: Vec3,
    odd: Vec3,
}

impl UVChecker {
    fn color_at(&self, u: f64, v: f64) -> Vec3 {
        let u = (self.scale * u).floor() as isize;
        let v = (self.scale * v).floor() as isize;
        if (u + v) % 2 == 0 {
            self.even
        } else {
            self.odd
        }
    }
}

#[derive(Debug, Clone)]
pub struct NoiseTexture {
    scale: f64,
    noise: Arc<PerlinNoise>,
}

impl NoiseTexture {
    pub fn new(scale: f64, noise: Arc<PerlinNoise>) -> Self {
        NoiseTexture { scale, noise }
    }

    // example in the book, but not used in final texture
    #[allow(unused)]
    fn standard(&self, p: &Point3) -> Vec3 {
        // Because the vecs have components in range -1..1,
        // we need to scale them back into the range 0..1 so that
        // there are only non-negative color values.
        let mut noise = self.noise.noise(&p.scaled(self.scale));
        noise = (noise + 1.0) * 0.5;
        Vec3::new(noise, noise, noise)
    }

    // example in the book, but not used in final texture
    #[allow(unused)]
    fn netting(&self, p: &Point3) -> Vec3 {
        let noise = self.noise.turb(&p.scaled(self.scale));
        Vec3::new(noise, noise, noise)
    }

    fn marbled(&self, p: &Point3) -> Vec3 {
        let noise = 0.5 * (1.0 + (self.scale * p.z() + 10.0 * self.noise.turb(p)).sin());
        Vec3::new(noise, noise, noise)
    }
}

#[derive(Debug)]
pub struct PerlinNoise {
    ranvec: Vec<Vec3>,
    permx: Vec<usize>,
    permy: Vec<usize>,
    permz: Vec<usize>,
}

impl PerlinNoise {
    const POINT_COUNT: usize = 256;

    pub fn new<T: Rng>(rng: &mut T) -> Self {
        let dist = rand::distributions::Uniform::new(-1.0, 1.0);
        let ranvec = (0..Self::POINT_COUNT)
            .map(|_| Vec3::rand_within(rng, dist).unit())
            .collect();
        let permx = Self::generate_perm(rng);
        let permy = Self::generate_perm(rng);
        let permz = Self::generate_perm(rng);
        PerlinNoise {
            ranvec,
            permx,
            permy,
            permz,
        }
    }

    fn turb(&self, p: &Point3) -> f64 {
        let depth = 7;
        let mut acc = 0.0;
        let mut temp_p = *p;
        let mut weight = 1.0;

        for _ in 0..depth {
            acc += weight * self.noise(&temp_p);
            weight *= 0.5;
            temp_p = temp_p.scaled(2.0);
        }
        acc.abs()
    }

    fn noise(&self, p: &Point3) -> f64 {
        let i = p.x().floor() as isize;
        let j = p.y().floor() as isize;
        let k = p.z().floor() as isize;
        let mut buf: [Vec3; 8] = Default::default();
        for (ii, v) in buf.iter_mut().enumerate() {
            let di = ii & 1;
            let dj = (ii & 2) >> 1;
            let dk = (ii & 4) >> 2;
            *v = self.ranvec[self.permx[((i + di as isize) & 255) as usize]
                ^ self.permy[((j + dj as isize) & 255) as usize]
                ^ self.permz[((k + dk as isize) & 255) as usize]];
        }
        let u = p.x() - p.x().floor();
        let v = p.y() - p.y().floor();
        let w = p.z() - p.z().floor();
        Self::trilinear_interp(&buf, u, v, w)
    }

    fn trilinear_interp(buf: &[Vec3; 8], u: f64, v: f64, w: f64) -> f64 {
        let uu = u * u * (3. - 2. * u);
        let vv = v * v * (3. - 2. * v);
        let ww = w * w * (3. - 2. * w);

        // trilinear interpolate
        let mut acc = 0.0;
        for (ii, f) in buf.iter().enumerate() {
            let i = (ii & 1) as f64;
            let j = ((ii & 2) >> 1) as f64;
            let k = ((ii & 4) >> 2) as f64;

            let weight = Vec3::new(u - i, v - j, w - k);

            acc += (i * uu + (1.0 - i) * (1.0 - uu))
                * (j * vv + (1.0 - j) * (1.0 - vv))
                * (k * ww + (1.0 - k) * (1.0 - ww))
                * (f.dot(&weight));
        }
        acc
    }

    fn generate_perm<T: Rng>(rng: &mut T) -> Vec<usize> {
        let mut perm = (0usize..Self::POINT_COUNT).collect::<Vec<_>>();
        perm.shuffle(rng);
        perm
    }
}
