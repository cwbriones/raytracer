mod vec;
mod ray;

use std::ops::Mul;

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

fn ray_color(ray: &Ray) -> Color {
    let unit_y = ray.dir().unit().y();
    let t = 0.5 * (unit_y + 1.0);
    return Color::new(1.0, 1.0, 1.0).lerp(Color::new(0.5, 0.7, 1.0), t)
}

fn main() {
    const ASPECT_RATIO: f64 = 16.0 / 9.0;
    const IMAGE_WIDTH: usize = 384;
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
