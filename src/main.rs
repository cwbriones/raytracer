mod vec;
mod ray;
mod camera;

use std::time::Instant;
use std::sync::Mutex;
use std::sync::Arc;
use std::rc::Rc;

use argh::FromArgs;
use camera::Camera;
use vec::{Point3, Vec3};
use ray::Ray;

use rand::thread_rng;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand::distributions::Uniform;

use image::{self, ImageBuffer};

#[derive(Default)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Color{r, g, b}
    }

    pub fn write(&self) {
        let ir = (255.999 * self.r) as usize;
        let ig = (255.999 * self.g) as usize;
        let ib = (255.999 * self.b) as usize;

        println!("{} {} {}", ir, ig, ib);
    }
}

#[derive(Clone)]
struct Hit {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: Rc<dyn Material>,
}

impl Hit {
    pub fn new(ray: &Ray, t: f64, outward_normal: Vec3, material: Rc<dyn Material>) -> Self {
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

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

fn scatter<'a, I>(items: I, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: AsRef<dyn Hittable>,
{
    items.into_iter()
        .fold(None, |closest, h| {
            let max = closest.as_ref().map(|h| h.t).unwrap_or(t_max);
            h.as_ref().hit(ray, t_min, max).or(closest)
        })
}

trait Material {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<(Vec3, Ray)>;
}

struct Lambertian(Vec3);

impl Material for Lambertian {
    fn scatter(&self, _: &Ray, hit: &Hit) -> Option<(Vec3, Ray)> {
        // FIXME: How can we pass in the current RNG?
        let mut scatter_direction = hit.normal + random_in_unit_sphere(&mut thread_rng());

        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = hit.normal;
        }

        let scattered = Ray::new(hit.point, scatter_direction);
        Some((self.0, scattered))
    }
}

struct Metal(Vec3, f64);

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<(Vec3, Ray)> {
        let reflected = reflect(&ray.dir(), &hit.normal);
        let scattered = Ray::new(hit.point, reflected + self.1 * random_in_unit_sphere(&mut thread_rng()));
        if scattered.dir().dot(&hit.normal) > 1e-8 {
            Some((self.0, Ray::new(hit.point, scattered.dir())))
        } else {
            None
        }
    }
}

struct Dielectric {
    albedo: Vec3,
    refractive_index: f64,
}

impl Dielectric {
    fn reflectance(cosine: f64, refractive_index: f64) -> f64 {
        // Use Schlick's approximation for reflectance.
        //
        // https://en.wikipedia.org/wiki/Schlick%27s_approximation
        let mut r0 = (1.0 - refractive_index) / (1.0 + refractive_index);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &Hit) -> Option<(Vec3, Ray)> {
        let refraction_ratio = if hit.front_face {
            1.0 / self.refractive_index
        } else {
            self.refractive_index
        };
        let unit_dir = ray.dir().unit();
        let cos_theta = unit_dir
            .negate()
            .dot(&hit.normal)
            .min(1.0);
        let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let scatter_dir = if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > thread_rng().gen::<f64>() {
            // Refraction impossible, must reflect.
            reflect(&unit_dir, &hit.normal)
        } else {
            // Refract.
            refract(&unit_dir, &hit.normal, refraction_ratio)
        };
        Some((self.albedo, Ray::new(hit.point, scatter_dir)))
    }
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

struct Sphere {
    center: Point3,
    radius: f64,
    material: Rc<dyn Material>,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Rc<dyn Material>) -> Self {
        Sphere { center, radius, material }
    }
}

impl Hittable for Sphere {
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

fn ray_color<R: Rng>(ray: &Ray, world: &[Box<dyn Hittable>], rng: &mut R, depth: usize) -> Vec3 {
    if depth == 0 {
        // This is what the book does in section 8.2, but it seems like
        // we could avoid this entirely by not using recursion.
        return Default::default();
    }
    if let Some(hit) = scatter(world, ray, 0.001, ::std::f64::INFINITY) {
        return if let Some((attenutation, ref ray_out)) = hit.material.scatter(ray, &hit) {
            attenutation.mul_pointwise(&ray_color(ray_out, world, rng, depth - 1))
        } else {
            Default::default()
        }
    } else {
        let unit_dir = ray.dir().unit();
        let t = 0.5 * (unit_dir.y() + 1.0);
        lerp(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.5, 0.7, 1.0), t)
    }
}

fn random_in_unit_sphere<R: Rng>(rng: &mut R) -> Vec3 {
    let dist = Uniform::new(-1.0, 1.0);
    loop {
        let v = Vec3::rand_within(rng, dist);
        let square_len = v.square_length();
        if square_len < 1.0 {
            // True lambertian reflection utilizes vectors on the unit sphere,
            // not within it. However, the "approximation" is somewhat more intuitive
            // and noticeably more performant so we leave it in place.
            //
            // Normalization results in a slightly darker surface since
            // rays are more uniformly scattered.
            //
            // See Section 8.5.
            return v;
        }
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

fn main() {
    let config = argh::from_env::<TracerConfig>();

    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    let image_width = config.width;
    let threads = config.threads;
    let image_height = (image_width as f64 / ASPECT_RATIO) as u32;
    let samples_per_pixel: usize = config.num_samples;
    let rays = (image_width * image_height) as usize * samples_per_pixel;
    let max_depth = 50;

    let start = Instant::now();
    let img = crossbeam::scope(|s| {
        let img = Arc::new(Mutex::new(ImageBuffer::new(image_width, image_height)));
        for worker_id in 0..threads {
            let img = img.clone();
            s.spawn(move |_| {
                let material_ground = Rc::new(Lambertian(Vec3::new(0.8, 0.8, 0.0)));
                let material_center = Rc::new(Lambertian(Vec3::new(0.1, 0.2, 0.5)));
                let material_left = Rc::new(Dielectric {
                    albedo: Vec3::new(1.0, 1.0, 1.0),
                    refractive_index: 1.5,
                });
                let material_right = Rc::new(Metal(Vec3::new(0.8, 0.6, 0.2), 0.0));

                let world: Vec<Box<dyn Hittable>> = vec![
                    Box::new(Sphere::new(
                        Point3::at(0.0, -100.5, -1.0),
                        100.0,
                        material_ground
                    )),
                    Box::new(Sphere::new(
                        Point3::at(0.0, 0.0, -1.0),
                        0.5,
                        material_center
                    )),
                    Box::new(Sphere::new(
                        Point3::at(-1.0, 0.0, -1.0),
                        0.5,
                        material_left.clone()
                    )),
                    Box::new(Sphere::new(
                        Point3::at(-1.0, 0.0, -1.0),
                        -0.4,
                        material_left
                    )),
                    Box::new(Sphere::new(
                        Point3::at(1.0, 0.0, -1.0),
                        0.5,
                        material_right
                    )),
                ];

                let camera = Camera::builder()
                    .origin(Default::default())
                    .horizontal(Vec3::new(4.0, 0.0, 0.0))
                    .vertical(Vec3::new(0.0, 2.25, 0.0))
                    .build();

                let mut rng = SmallRng::from_entropy();
                (worker_id..(image_height as usize))
                    .step_by(threads)
                    .flat_map(|j| (0..image_width).map(move |i| (i, j)))
                    .for_each(|(i, j)| {
                        let color_vec = average(samples_per_pixel, || {
                            let u = (i as f64 + rng.gen::<f64>()) / (image_width - 1) as f64;
                            let v = (j as f64 + rng.gen::<f64>()) / (image_height - 1) as f64;
                            let ray = camera.get_ray(u, v);
                            ray_color(&ray, &world.as_slice(), &mut rng, max_depth)
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
