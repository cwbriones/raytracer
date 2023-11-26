/// Example scenes as shown in the Ray Tracing in One Weeked series.
use std::str::FromStr;
use std::sync::Arc;

use rand::distributions::Uniform;
use rand::Rng;

use super::Scene;
use crate::camera::Camera;
use crate::geom::{
    Point3,
    Vec3,
};
use crate::material::{
    Material,
    PerlinNoise,
    Texture,
};
use crate::surfaces::make_box;
use crate::surfaces::ConstantMedium;
use crate::surfaces::Quad;
use crate::surfaces::Rotated;
use crate::surfaces::Sphere;
use crate::surfaces::Translated;

#[derive(Debug, Clone)]
pub enum Example {
    OneWeekend,
    RandomSpheres,
    TwoSpheres,
    TwoPerlinSpheres,
    CornellSmoke,
}

#[derive(Debug, Clone, Copy)]
pub struct InvalidExample;

impl ::std::error::Error for InvalidExample {}

impl ::std::fmt::Display for InvalidExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid example")
    }
}

impl FromStr for Example {
    type Err = InvalidExample;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "one-weekend" => Ok(Example::OneWeekend),
            "random-spheres" => Ok(Example::RandomSpheres),
            "two-spheres" => Ok(Example::TwoSpheres),
            "two-perlin" => Ok(Example::TwoPerlinSpheres),
            "cornell-smoke" => Ok(Example::CornellSmoke),
            _ => Err(InvalidExample),
        }
    }
}

impl Example {
    pub fn scene(&self, aspect_ratio: f64) -> (Scene, Camera) {
        match self {
            Example::OneWeekend => one_weekend(aspect_ratio),
            Example::RandomSpheres => random_spheres(aspect_ratio),
            Example::TwoSpheres => two_spheres(aspect_ratio),
            Example::TwoPerlinSpheres => two_perlin_spheres(aspect_ratio),
            Example::CornellSmoke => cornell_smoke(aspect_ratio),
        }
    }
}

/// Create a random scene as shown in the final section of Ray Tracing in One Weekend.
fn one_weekend(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from(Point3::new(13., 2., 3.))
        .towards(Point3::new(0., 0., 0.))
        .focus_dist(10.)
        .aperture(0.1)
        .build();

    let mut rng = rand::thread_rng();
    let mut objects = Scene::builder();
    objects.set_background(Vec3::new(0.7, 0.8, 1.0));

    let ground_material = Material::lambertian(Vec3::new(0.5, 0.5, 0.5));
    objects.add(Sphere::stationary(
        Point3::new(0., -1000., 0.),
        1000.,
        ground_material,
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.gen::<f64>();
            let center = Point3::new(
                a as f64 + 0.9 * rng.gen::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.gen::<f64>(),
            );
            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
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
                objects.add(Sphere::stationary(center, 0.2, material));
            }
        }
    }
    let material1 = Material::lambertian(Vec3::new(0.05, 0.2, 0.6));
    objects.add(Sphere::stationary(Point3::new(-4., 1., 0.), 1.0, material1));
    let material2 = Material::dielectric(1.5);
    objects.add(Sphere::stationary(Point3::new(0., 1., 0.), 1.0, material2));
    let material3 = Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
    objects.add(Sphere::stationary(Point3::new(4., 1., 0.), 1.0, material3));

    (objects.build(), camera)
}

/// Create a random scene as shown in the final section of Ray Tracing in One Weekend.
fn random_spheres(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from(Point3::new(13., 2., 3.))
        .towards(Point3::new(0., 0., 0.))
        .focus_dist(10.)
        .aperture(0.1)
        .build();

    let mut rng = rand::thread_rng();
    let mut objects = Scene::builder();
    objects.set_background(Vec3::new(0.7, 0.8, 1.0));

    let ground_material = Material::lambertian(Texture::checker(
        0.32,
        Vec3::new(0.2, 0.3, 0.1),
        Vec3::new(0.9, 0.9, 0.9),
    ));
    objects.add(Sphere::stationary(
        Point3::new(0., -1000., 0.),
        1000.,
        ground_material,
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = rng.gen::<f64>();
            let center = Point3::new(
                a as f64 + 0.9 * rng.gen::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.gen::<f64>(),
            );
            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                let material: Material;
                if choose_material < 0.8 {
                    // diffuse
                    let albedo = rng.gen::<Vec3>().mul_pointwise(&rng.gen::<Vec3>());
                    material = Material::lambertian(albedo);
                    let center2 = center + Vec3::new(0., rng.gen_range(0.0..0.5), 0.);
                    objects.add(Sphere::moving(center, center2, 0.2, material));
                } else if choose_material < 0.95 {
                    // metal
                    let albedo = Vec3::rand_within(&mut rng, Uniform::new(0.5, 1.0));
                    let fuzz = rng.gen_range(0.0..0.5);
                    material = Material::metal(albedo, fuzz);
                    objects.add(Sphere::stationary(center, 0.2, material));
                } else {
                    // glass
                    material = Material::dielectric(1.5);
                    objects.add(Sphere::stationary(center, 0.2, material));
                }
            }
        }
    }
    let material1 = Material::lambertian(Vec3::new(0.05, 0.2, 0.6));
    objects.add(Sphere::stationary(Point3::new(-4., 1., 0.), 1.0, material1));
    let material2 = Material::dielectric(1.5);
    objects.add(Sphere::stationary(Point3::new(0., 1., 0.), 1.0, material2));
    let material3 = Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
    objects.add(Sphere::stationary(Point3::new(4., 1., 0.), 1.0, material3));

    (objects.build(), camera)
}

fn two_spheres(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from((13.0, 2.0, 3.0))
        .towards((0.0, 0.0, 0.0))
        .build();

    let checker = Material::lambertian(Texture::uv_checker(
        30.0,
        Vec3::new(0.2, 0.3, 0.1),
        Vec3::new(0.9, 0.9, 0.9),
    ));

    let mut world = Scene::builder();
    world.set_background(Vec3::new(0.7, 0.8, 1.0));
    world.add(Sphere::stationary(
        Point3::new(0.0, -10.0, 0.0),
        10.0,
        checker.clone(),
    ));
    world.add(Sphere::stationary(
        Point3::new(0.0, 10.0, 0.0),
        10.0,
        checker,
    ));

    (world.build(), camera)
}

fn two_perlin_spheres(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from((13.0, 2.0, 3.0))
        .towards((0.0, 0.0, 0.0))
        .build();

    let mut rng = rand::thread_rng();
    let noise = Arc::new(PerlinNoise::new(&mut rng));
    let perlin = Texture::noise(4.0, noise);

    let mut world = Scene::builder();
    world.set_background(Vec3::new(0.7, 0.8, 1.0));
    world.add(Sphere::stationary(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(perlin.clone()),
    ));
    world.add(Sphere::stationary(
        Point3::new(0.0, 2.0, 0.0),
        2.0,
        Material::lambertian(perlin),
    ));

    (world.build(), camera)
}

fn cornell_smoke(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(40.0, aspect_ratio)
        .from((278.0, 278.0, -800.0))
        .towards((278.0, 278.0, 0.0))
        .build();

    let red = Material::lambertian(Vec3::new(0.65, 0.05, 0.05));
    let white = Material::lambertian(Vec3::new(0.73, 0.73, 0.73));
    let green = Material::lambertian(Vec3::new(0.12, 0.45, 0.15));
    let light = Material::diffuse_light(Vec3::new(7.0, 7.0, 7.0));

    let mut scene = Scene::builder();
    scene.add(Quad::new(
        Point3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        green,
    ));
    scene.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        red,
    ));
    scene.add(Quad::new(
        Point3::new(113.0, 554.0, 127.0),
        Vec3::new(330.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 305.0),
        light,
    ));
    scene.add(Quad::new(
        Point3::new(0.0, 555.0, 0.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    scene.add(Quad::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 555.0),
        white.clone(),
    ));
    scene.add(Quad::new(
        Point3::new(0.0, 0.0, 555.0),
        Vec3::new(555.0, 0.0, 0.0),
        Vec3::new(0.0, 555.0, 0.0),
        white.clone(),
    ));
    // FIXME: Unlike the reference code, for whatever reason the volume cannot overlap the cornell
    // box itself at y=0 without a strange interaction at the bottom of the volumes. So each volume
    // has a y slightly more than zero to compensate.
    let box1 = make_box(
        Point3::new(0.0, 0.001, 0.0),
        Point3::new(165.0, 333.0, 165.0),
        white.clone(),
    );
    let box1 = ConstantMedium::new(box1, 0.01, Vec3::new(0.0, 0.0, 0.0));
    let box1 = Rotated::new(box1, Vec3::jhat(), 15.0f64.to_radians());
    let box1 = Translated::new(box1, Vec3::new(265.0, 0.0, 295.0));
    scene.add(box1);

    let box2 = make_box(
        Point3::new(0.0, 0.001, 0.0),
        Point3::new(165.0, 165.0, 165.0),
        white.clone(),
    );
    let box2 = ConstantMedium::new(box2, 0.01, Vec3::new(1.0, 1.0, 1.0));
    let box2 = Rotated::new(box2, Vec3::jhat(), (-18.0f64).to_radians());
    let box2 = Translated::new(box2, Vec3::new(130.0, 0.0, 65.0));
    scene.add(box2);

    (scene.build(), camera)
}
