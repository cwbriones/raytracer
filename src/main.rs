mod vec;
mod ray;

use vec::{Point3, Vec3};
use ray::Ray;

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

    pub fn lerp(&self, to: Self, t: f64) -> Self {
        Color {
            r: (1.0 - t) * self.r + t*to.r,
            g: (1.0 - t) * self.g + t*to.g,
            b: (1.0 - t) * self.b + t*to.b,
        }
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

fn ray_color(ray: &Ray, world: &[Box<dyn Hittable>]) -> Color {
    let mut hit_record = HitRecord::default();

    if hit_vec(world, ray, 0.0, ::std::f64::INFINITY, &mut hit_record) {
        let n = hit_record.normal;
        Color::new((n.x() + 1.0) / 2.0, (n.y() + 1.0) / 2.0, (n.z() + 1.0) / 2.0)
    } else {
        let unit_dir = ray.dir().unit();
        let t = 0.5 * (unit_dir.y() + 1.0);
        Color::new(1.0, 1.0, 1.0).lerp(Color::new(0.5, 0.7, 1.0), t)
    }
}

fn main() {
    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    const IMAGE_WIDTH: usize = 768;
    const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

    let origin: Point3 = Default::default();
    let horizontal = Vec3::new(4.0, 0.0, 0.0);
    let vertical = Vec3::new(0.0, 2.25, 0.0);
    let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::zhat();

    let world: Vec<Box<dyn Hittable>> = vec![
        Box::new(Sphere::new(Point3::at(0.0, -100.5, -1.0), 100.0)),
        Box::new(Sphere::new(Point3::at(0.0, 0.0, -1.0), 0.5)),
    ];

    println!("P3 {} {}", IMAGE_WIDTH, IMAGE_HEIGHT);
    println!("255");
    (0..IMAGE_HEIGHT)
        .rev()
        .flat_map(|j| {
            eprint!("\rScanlines remaining: {}     ", j);
            (0..IMAGE_WIDTH).map(move |i| (i, j))
        })
        .map(|(i, j)| {
            let u = i as f64 / (IMAGE_WIDTH - 1) as f64;
            let v = j as f64 / (IMAGE_HEIGHT - 1) as f64;
            let dir = (lower_left_corner + u * horizontal + v * vertical).origin_vec();
            let ray = Ray::new(origin, dir);
            ray_color(&ray, &world.as_slice())
        })
        .for_each(|c| c.write());
    eprintln!("\nDone.");
}
