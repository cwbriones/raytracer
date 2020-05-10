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

fn hit_sphere(center: Point3, radius: f64, ray: &Ray) -> f64 {
    let oc = ray.origin() - center;
    let a = ray.dir().square_length();
    let half_b = oc.dot(&ray.dir());
    let c = oc.square_length() - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        // Does not hit the sphere.
        -1.0
    } else {
        // The point of intersection.
        (-half_b - discriminant.sqrt()) / a
    }
}

fn ray_color(ray: &Ray) -> Color {
    let mut t = hit_sphere(Point3::at(0.0, 0.0, -1.0), 0.5, ray);
    if t > 0.0 {
        let n = (ray.at(t) - Vec3::new(0.0, 0.0, -1.0)).origin_vec().unit();
        debug_assert!(n.length() == 1.0);
        // Each n_i is in [-1, 1] which means that n_i + 1 is in [0, 2].
        return Color::new((n.x() + 1.0) / 2.0, (n.y() + 1.0) / 2.0, (n.z() + 1.0) / 2.0);
    }
    let unit_dir = ray.dir().unit();
    t = 0.5 * (unit_dir.y() + 1.0);
    return Color::new(1.0, 1.0, 1.0).lerp(Color::new(0.5, 0.7, 1.0), t)
}

fn main() {
    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    const IMAGE_WIDTH: usize = 768;
    const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

    let origin: Point3 = Default::default();
    let horizontal = Vec3::new(4.0, 0.0, 0.0);
    let vertical = Vec3::new(0.0, 2.25, 0.0);
    let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::zhat();

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
            ray_color(&ray)
        })
        .for_each(|c| c.write());
    eprintln!("\nDone.");
}
