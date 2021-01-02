mod vec;
mod ray;
mod camera;

use std::time::Instant;
use camera::Camera;
use vec::{Point3, Vec3};
use ray::Ray;

use rand::thread_rng;
use rand::Rng;

use image::{self, ImageBuffer, RgbImage};

use crossbeam::channel;

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

#[derive(Default, Clone, Copy)]
struct HitRecord {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
}

impl HitRecord {
    #[inline]
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.dir().dot(&outward_normal) < 0.0;
        self.normal = if self.front_face { outward_normal } else { outward_normal.negate() };
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool;
}

// FIXME: This should take an IntoIterator that turns into references of Hittables
fn hit_vec(items: &[Box<dyn Hittable>], ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
        let mut hit_anything = false;
        let mut closest = t_max;

        for h in items {
            let mut temp_rec = Default::default();
            if h.hit(ray, t_min, t_max, &mut temp_rec) {
                hit_anything = true;
                closest = temp_rec.t;
                *rec = temp_rec;
            }
        }
        return hit_anything;
}

struct Sphere {
    center: Point3,
    radius: f64,
}

impl Sphere {
    fn new(center: Point3, radius: f64) -> Self {
        Sphere { center, radius }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64, rec: &mut HitRecord) -> bool {
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
                rec.t = temp;
                rec.point = ray.at(rec.t);
                let outward_normal = (rec.point - self.center) / self.radius;
                rec.set_face_normal(ray, outward_normal);
                return true;
            }
            let temp = (-half_b + root) / a;
            if t_min < temp && temp < t_max {
                // Second root
                rec.t = temp;
                rec.point = ray.at(rec.t);
                let outward_normal = (rec.point - self.center) / self.radius;
                rec.set_face_normal(ray, outward_normal);
                return true;
            }
        }
        // Does not hit the sphere.
        return false;
    }
}

fn ray_color(ray: &Ray, world: &[Box<dyn Hittable>]) -> Vec3 {
    let mut hit_record = HitRecord::default();

    if hit_vec(world, ray, 0.0, ::std::f64::INFINITY, &mut hit_record) {
        let n = hit_record.normal;
        Vec3::new((n.x() + 1.0) / 2.0, (n.y() + 1.0) / 2.0, (n.z() + 1.0) / 2.0)
    } else {
        let unit_dir = ray.dir().unit();
        let t = 0.5 * (unit_dir.y() + 1.0);
        lerp(Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.5, 0.7, 1.0), t)
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

fn main() {
    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    const IMAGE_WIDTH: u32 = 768;
    const IMAGE_HEIGHT: u32 = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as u32;
    const SAMPLES_PER_PIXEL: usize = 100;
    const RAYS: usize = (IMAGE_WIDTH * IMAGE_HEIGHT) as usize * SAMPLES_PER_PIXEL;

    let start = Instant::now();
    let img = crossbeam::scope(|s| {
        let (send_coord, r_coord) = channel::bounded(1024);
        let (send_pxl, recv_pxl) = channel::bounded(1024);
        for _ in 0..4 {
            let send_pxl = send_pxl.clone();
            let r = r_coord.clone();
            s.spawn(move |_| {
                let world: Vec<Box<dyn Hittable>> = vec![
                    Box::new(Sphere::new(Point3::at(0.0, -100.5, -1.0), 100.0)),
                    Box::new(Sphere::new(Point3::at(0.0, 0.0, -1.0), 0.5)),
                ];

                let camera = Camera::builder()
                    .origin(Default::default())
                    .horizontal(Vec3::new(4.0, 0.0, 0.0))
                    .vertical(Vec3::new(0.0, 2.25, 0.0))
                    .build();

                let mut rng = thread_rng();
                while let Ok((i, j)) = r.recv() {
                    let color_vec = average(SAMPLES_PER_PIXEL, || {
                        let u = (i as f64 + rng.next_f64()) / (IMAGE_WIDTH - 1) as f64;
                        let v = (j as f64 + rng.next_f64()) / (IMAGE_HEIGHT - 1) as f64;
                        let ray = camera.get_ray(u, v);
                        ray_color(&ray, &world.as_slice())
                    });
                    send_pxl.send((i, j, image::Rgb([
                        (255.999 * color_vec.x()) as u8,
                        (255.999 * color_vec.y()) as u8,
                        (255.999 * color_vec.z()) as u8
                    ]))).unwrap();
                }
            });
        }
        drop(send_pxl);
        s.spawn(move |_| {
            (0..IMAGE_HEIGHT)
                .rev()
                .flat_map(|j| {
                    eprint!("\rScanlines remaining: {}     ", j);
                    (0..IMAGE_WIDTH).map(move |i| (i, j))
                })
                .for_each(|coords| {
                    send_coord.send(coords).unwrap()
                });
        });
        let mut img: RgbImage = ImageBuffer::new(IMAGE_WIDTH, IMAGE_HEIGHT);
        while let Ok((i, j, pixel)) = recv_pxl.recv() {
            img.put_pixel(i, j, pixel)
        }
        img
    }).unwrap();

    let elapsed_sec = start.elapsed().as_secs_f64();
    let rays_per_sec = (RAYS as f64) / elapsed_sec;
    eprintln!("\nDone in {:.2}s ({:.0} rays/s)", elapsed_sec, rays_per_sec);
    img.save("output.png").unwrap();
}
