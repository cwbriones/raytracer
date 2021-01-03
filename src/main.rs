mod vec;
mod ray;
mod camera;
mod util;

use std::time::Instant;
use std::sync::Mutex;
use std::sync::Arc;

use camera::Camera;
use vec::{Point3, Vec3};
use ray::Ray;
use util::RandUtil;

use argh::FromArgs;
use rand::thread_rng;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand::distributions::Uniform;
use image::{self, ImageBuffer};

#[derive(Clone)]
struct Hit {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: Material,
}

impl Hit {
    pub fn new(ray: &Ray, t: f64, outward_normal: Vec3, material: Material) -> Self {
        let point = ray.at(t);
        let front_face = ray.dir().dot(&outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { outward_normal.negate() };
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
struct Material {
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

    pub fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<(Vec3, Ray)> {
        self.kind.scatter(ray, hit, rng).map(|r| (self.albedo, r))
    }
}

#[derive(Clone)]
enum MaterialKind {
    Lambertian,
    Metal(f64),
    Dielectric(f64),
}

impl MaterialKind {
    fn scatter<R: Rng>(&self, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
        match *self {
            Self::Lambertian => {
                lambertian_scatter(ray, hit, rng)
            },
            Self::Metal(fuzz) => {
                metallic_scatter(fuzz, ray, hit, rng)
            },
            Self::Dielectric(refractive_index) => {
                dielectric_scatter(refractive_index, ray, hit, rng)
            },
        }
    }
}

fn lambertian_scatter<R: Rng>(_: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
    // True lambertian reflection utilizes vectors on the unit sphere,
    // not within it. However, the "approximation" with interior sampling
    // is somewhat more intuitive. The difference amounts to normalizing
    // the vector after sampling.
    //
    // Normalization results in a slightly darker surface since
    // rays are more uniformly scattered.
    //
    // See Section 8.5.
    //
    // FIXME: How can we pass in the current RNG?
    let mut scatter_direction = hit.normal + rng.gen_in_unit_sphere().unit();

    // Catch degenerate scatter direction
    if scatter_direction.near_zero() {
        scatter_direction = hit.normal;
    }

    let scattered = Ray::new(hit.point, scatter_direction);
    Some(scattered)
}

fn metallic_scatter<R: Rng>(fuzz: f64, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
    let reflected = reflect(&ray.dir(), &hit.normal);
    let scattered = Ray::new(hit.point, reflected + fuzz * rng.gen_in_unit_sphere());
    if scattered.dir().dot(&hit.normal) > 1e-8 {
        Some(Ray::new(hit.point, scattered.dir()))
    } else {
        None
    }
}

fn dielectric_scatter<R: Rng>(refractive_index: f64, ray: &Ray, hit: &Hit, rng: &mut R) -> Option<Ray> {
    let refraction_ratio = if hit.front_face {
        1.0 / refractive_index
    } else {
        refractive_index
    };
    let unit_dir = ray.dir().unit();
    let cos_theta = unit_dir
        .negate()
        .dot(&hit.normal)
        .min(1.0);
    let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
    let cannot_refract = refraction_ratio * sin_theta > 1.0;
    let scatter_dir = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.gen::<f64>() {
        // Refraction impossible, must reflect.
        reflect(&unit_dir, &hit.normal)
    } else {
        // Refract.
        refract(&unit_dir, &hit.normal, refraction_ratio)
    };
    Some(Ray::new(hit.point, scatter_dir))
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

#[derive(Clone)]
struct Scene {
    objects: Vec<Sphere>,
}

impl Scene {
    pub fn ray_color<R: Rng>(&self, mut ray: Ray, rng: &mut R, max_depth: usize) -> Option<Vec3> {
        // In the book this is done recursively, but I've refactored it into an
        // explicit accumulating loop to make profiling easier.
        //
        // This prevents additional tracing iterations from adding to the stack frame.
        let mut attenuation = Vec3::new(1.0, 1.0, 1.0);
        for _ in 0..max_depth {
            if let Some(hit) = self.scatter(&ray, 0.001, ::std::f64::INFINITY) {
                if let Some((cur_attenuation, ray_out)) = hit.material.scatter(&ray, &hit, rng) {
                    // The ray was scattered in a different direction. Continue following it.
                    attenuation = attenuation.mul_pointwise(&cur_attenuation);
                    ray = ray_out;
                } else {
                    // The ray was completely absorbed.
                    return None;
                }
            } else {
                // The ray escaped after bouncing.
                // Multiply the color with the sky.
                let unit_dir = ray.dir().unit();
                let t = 0.5 * (unit_dir.y() + 1.0);
                return Some(attenuation.mul_pointwise(&lerp(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.5, 0.7, 1.0), t)));
            }
        }
        // The ray hasn't resolved for the maximum allowed iterations,
        None
    }

    fn scatter(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>
    {
        let mut closest: Option<Hit> = None;
        for object in &self.objects {
            let max = closest.as_ref().map(|h| h.t).unwrap_or(t_max);
            closest = object.hit(ray, t_min, max).or(closest)
        }
        closest
    }
}

#[derive(Clone)]
struct Sphere {
    center: Point3,
    radius: f64,
    material: Material,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Material) -> Self {
        Sphere { center, radius, material }
    }
}

impl Sphere {
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
                return Some(Hit::new(
                    ray,
                    temp,
                    outward_normal,
                    self.material.clone(),
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
                    self.material.clone(),
                ));
            }
        }
        // Does not hit the sphere.
        None
    }
}

fn lerp(from: Vec3, to: Vec3, t: f64) -> Vec3 {
    Vec3::new(
        (1.0 - t) * from.x() + t*to.x(),
        (1.0 - t) * from.y() + t*to.y(),
        (1.0 - t) * from.z() + t*to.z(),
    )
}

fn average<T, F>(n: usize, mut f: F) -> T
    where F: FnMut() -> T,
          T: Default + ::std::ops::AddAssign + ::std::ops::Div<f64, Output=T>,
{
    let mut acc = <T as Default>::default();
    for _ in 0..n {
        acc += f();
    }
    acc / (n as f64)
}

/// Extension trait to add a clamp method to floats.
///
/// This conflicts with a method to be stabilized in Rust 1.50, hence the odd spelling.
trait KlampExt {
    fn klamp(self, min: Self, max: Self) -> Self;
}

impl KlampExt for f64 {
    #[must_use = "method returns a new number and does not mutate the original value"]
    #[inline]
    fn klamp(self, min: f64, max: f64) -> f64 {
        // This is copied directly from std::f64::clamp,
        // with the exception of debug_assert!
        debug_assert!(min <= max);
        let mut x = self;
        if x < min {
            x = min;
        }
        if x > max {
            x = max;
        }
        x
    }
}


#[derive(FromArgs)]
/// A simple ray tracer implementation in rust.
struct TracerConfig {
    #[argh(option, short='n', default="TracerConfig::default_samples()")]
    /// the number of samples to take per pixel.
    num_samples: usize,
    #[argh(option, short='o', default="TracerConfig::default_output()")]
    /// destination of the output image.
    ///
    /// supported formats: png, jpg
    output: String,
    #[argh(option, default="TracerConfig::default_threads()")]
    /// the number of threads to use.
    threads: usize,
    #[argh(option, default="TracerConfig::default_width()")]
    /// the output image width.
    width: u32,
}

impl TracerConfig {
    const fn default_threads() -> usize { 4 }
    const fn default_samples() -> usize { 100 }
    const fn default_width() -> u32 { 400 }

    fn default_output() -> String { "output.png".into() }
}

fn random_scene() -> Scene {
    let mut objects = Vec::new();

    let ground_material = Material::lambertian(Vec3::new(0.5, 0.5, 0.5));
    objects.push(Sphere::new(
        Point3::at(0., -1000., 0.),
        1000.,
        ground_material,
    ));

    let mut rng = thread_rng();
    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.gen::<f64>();
            let center = Point3::at(
                a as f64 + 0.9 * rng.gen::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.gen::<f64>(),
            );
            if (center - Point3::at(4.0, 0.2, 0.0)).length() > 0.9 {
                let material: Material;
                if choose_material < 0.8 {
                    // diffuse
                    let albedo = rng.gen::<Vec3>().mul_pointwise(&rng.gen::<Vec3>());
                    material = Material::lambertian(albedo);
                } else if choose_material < 0.95 {
                    // metal
                    let albedo = Vec3::rand_within(&mut rng, Uniform::new(0.5, 1.0));
                    let fuzz = rng.gen_range(0.0..0.5);
                    material = Material::metal(albedo, fuzz);
                } else {
                    // glass
                    material = Material::dielectric(1.5);
                }
                objects.push(Sphere::new(center, 0.2, material));
            }
        }
    }
    let material1 = Material::lambertian(Vec3::new(0.05, 0.2, 0.6));
    objects.push(Sphere::new(
        Point3::at(-4., 1., 0.),
        1.0,
        material1,
    ));
    let material2 = Material::dielectric(1.5);
    objects.push(Sphere::new(
        Point3::at(0., 1., 0.),
        1.0,
        material2,
    ));
    let material3 = Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
    objects.push(Sphere::new(
        Point3::at(4., 1., 0.),
        1.0,
        material3,
    ));

    Scene { objects }
}

fn main() {
    let config = argh::from_env::<TracerConfig>();

    const ASPECT_RATIO: f64 = 3.0 / 2.0;
    let image_width = config.width;
    let threads = config.threads;
    let image_height = (image_width as f64 / ASPECT_RATIO) as u32;
    let samples_per_pixel: usize = config.num_samples;
    let rays = (image_width * image_height) as usize * samples_per_pixel;
    let max_depth = 50;

    let start = Instant::now();

    // This is stupid but as a limitation of using trait objects, I can't create the scene
    // once and send it to all worker threads.
    //
    // Instead we generate the scene in each thread, in which case we need to ensure
    // the seed is identical.
    let scene = random_scene();

    let img = crossbeam::scope(|s| {
        let img = Arc::new(Mutex::new(ImageBuffer::new(image_width, image_height)));
        for worker_id in 0..threads {
            let scene = scene.clone();
            let img = img.clone();
            s.spawn(move |_| {
                let camera = Camera::builder(20.0, ASPECT_RATIO)
                    .from(Point3::at(13., 2., 3.))
                    .towards(Point3::at(0., 0., 0.))
                    .focus_dist(10.0)
                    .aperture(0.1)
                    .build();

                let mut rng = SmallRng::from_entropy();
                (worker_id..(image_height as usize))
                    .step_by(threads)
                    .flat_map(|j| (0..image_width).map(move |i| (i, j)))
                    .for_each(|(i, j)| {
                        let color_vec = average(samples_per_pixel, || {
                            let u = (i as f64 + rng.gen::<f64>()) / (image_width - 1) as f64;
                            let v = (j as f64 + rng.gen::<f64>()) / (image_height - 1) as f64;
                            let ray = camera.get_ray(&mut rng, u, v);
                            scene.ray_color(ray, &mut rng, max_depth).unwrap_or_else(Default::default)
                        });
                        let mut guard = img.lock().unwrap();
                        let r = 256. * (color_vec.x()).sqrt().klamp(0.0, 0.99);
                        let g = 256. * (color_vec.y()).sqrt().klamp(0.0, 0.99);
                        let b = 256. * (color_vec.z()).sqrt().klamp(0.0, 0.99);
                        guard.put_pixel(i, image_height - j as u32 - 1, image::Rgb([
                            r as u8,
                            g as u8,
                            b as u8,
                        ]));
                    });
            });
        }
        img
    }).unwrap();

    let img = Arc::try_unwrap(img)
        .expect("all other threads have been dropped")
        .into_inner()
        .expect("all other threads have been dropped");

    let elapsed_sec = start.elapsed().as_secs_f64();
    let rays_per_sec = (rays as f64) / elapsed_sec;
    eprintln!("\nDone in {:.2}s ({:.0} rays/s)", elapsed_sec, rays_per_sec);
    img.save(config.output).unwrap();
}
