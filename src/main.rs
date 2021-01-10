mod bvh;
mod camera;
mod geom;
mod material;
mod surfaces;
mod trace;
mod util;

use std::time::Instant;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use argh::FromArgs;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rand::distributions::Uniform;
use image::{self, ImageBuffer};

use camera::Camera;
use surfaces::Sphere;
use geom::{Point3, Vec3};
use material::Material;
use trace::{Hit, Ray, AABB};
use util::Klamp;
use bvh::BVH;

#[derive(Clone)]
struct Scene {
    root: BVH,
}

impl Scene {
    pub fn new(mut spheres: Vec<Sphere>) -> Self {
        let root = BVH::new(&mut spheres);
        Scene { root }
    }

    fn scatter(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit>
    {
        self.root.hit(ray, t_min, t_max)
    }

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
    #[argh(option)]
    /// A seed to use for RNG. By default the RNG will be seed through the OS's entropy source.
    seed: Option<u64>,
}

impl TracerConfig {
    const fn default_threads() -> usize { 4 }
    const fn default_samples() -> usize { 100 }
    const fn default_width() -> u32 { 400 }

    fn default_output() -> String { "output.png".into() }
}

fn random_scene<R: Rng>(mut rng: R) -> Scene {
    let mut objects = Vec::new();

    let ground_material = Material::lambertian(Vec3::new(0.5, 0.5, 0.5));
    objects.push(Sphere::new(
        Point3::at(0., -1000., 0.),
        1000.,
        ground_material,
    ));

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

    Scene::new(objects)
}

fn progress_bar(pixels_remaining: Arc<AtomicUsize>, total: usize, samples_per_pixel: usize) {
    let start = Instant::now();
    let mut last_check = total + 1;
    let period = ::std::time::Duration::from_millis(1000);
    let mut rates = ::std::collections::VecDeque::new();
    eprint!("\n\n\n\n");
    loop {
        let remaining = pixels_remaining.load(Ordering::Relaxed);
        if remaining == 0 {
            break;
        }
        let current_rate = (last_check - remaining) as f32;
        if rates.len() == 60 {
            rates.pop_front();
        }
        rates.push_back(current_rate);
        let average_rate = rates.iter().sum::<f32>() / (rates.len() as f32);

        let estimated_time = period.mul_f32(remaining as f32 / average_rate);

        last_check = remaining;

        eprint!("\x1b[4A");
        eprintln!("    Elapsed Time: {}    ", format_duration(start.elapsed()));
        eprintln!("  Remaining Time: {}    ", format_duration(estimated_time));
        eprintln!("   Samples / sec: {}    ", average_rate * samples_per_pixel as f32);
        eprintln!("Remaining Pixels: {}    ", remaining);
        ::std::thread::sleep(period);
    }
    eprintln!("Done!");
}

fn format_duration(d: ::std::time::Duration) -> String {
    let hours = d.as_secs() / 3600;
    let minutes = (d.as_secs() - hours * 3600) / 60;
    let secs = d.as_secs() - minutes * 60 - hours * 3600;

    if hours > 0 {
        format!("{:0>2}:{:0>2}:{:0>2}", hours, minutes, secs)
    } else {
        format!("{:0>2}:{:0>2}", minutes, secs)
    }
}

fn small_rng(seed: Option<u64>) -> impl Rng {
    seed.map(SmallRng::seed_from_u64).unwrap_or_else(SmallRng::from_entropy)
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
    let scene = random_scene(small_rng(config.seed));

    let progress = Arc::new(AtomicUsize::new((image_width * image_height) as usize));
    let img = crossbeam::scope(|s| {
        let img = Arc::new(Mutex::new(ImageBuffer::new(image_width, image_height)));
        for worker_id in 0..threads {
            let scene = scene.clone();
            let progress = progress.clone();
            let img = img.clone();
            let mut rng = small_rng(config.seed);
            s.spawn(move |_| {
                let camera = Camera::builder(20.0, ASPECT_RATIO)
                    .from(Point3::at(13., 2., 3.))
                    .towards(Point3::at(0., 0., 0.))
                    .focus_dist(10.0)
                    .aperture(0.1)
                    .build();

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
                        progress.fetch_sub(1, Ordering::Relaxed);
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
        s.spawn(|_| progress_bar(progress.clone(), (image_width * image_height) as usize, samples_per_pixel));
        img
    }).unwrap();
    println!("{}", progress.load(Ordering::Relaxed));

    let img = Arc::try_unwrap(img)
        .expect("all other threads have been dropped")
        .into_inner()
        .expect("all other threads have been dropped");

    let elapsed_sec = start.elapsed().as_secs_f64();
    let rays_per_sec = (rays as f64) / elapsed_sec;
    eprintln!("\nDone in {:.2}s ({:.0} rays/s)", elapsed_sec, rays_per_sec);
    img.save(config.output).unwrap();
}
