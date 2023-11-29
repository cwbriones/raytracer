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
use crate::surfaces::Bvh;
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
    Cornell,
    CornellSmoke,
    Earth,
    FinalScene,
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
            "cornell" => Ok(Example::Cornell),
            "cornell-smoke" => Ok(Example::CornellSmoke),
            "earth" => Ok(Example::Earth),
            "final-scene" => Ok(Example::FinalScene),
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
            Example::Cornell => cornell(aspect_ratio),
            Example::CornellSmoke => cornell_smoke(aspect_ratio),
            Example::Earth => earth(aspect_ratio),
            Example::FinalScene => final_scene(aspect_ratio),
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
    let mut scene = Scene::builder();
    scene.set_background(Vec3::new(0.7, 0.8, 1.0));

    let mut surfaces = Bvh::builder();

    let ground_material = Material::lambertian(Vec3::new(0.5, 0.5, 0.5));
    scene.add(Sphere::stationary(
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
                surfaces.add(Sphere::stationary(center, 0.2, material));
            }
        }
    }
    let material1 = Material::lambertian(Vec3::new(0.05, 0.2, 0.6));
    surfaces.add(Sphere::stationary(Point3::new(-4., 1., 0.), 1.0, material1));
    let material2 = Material::dielectric(1.5);
    surfaces.add(Sphere::stationary(Point3::new(0., 1., 0.), 1.0, material2));
    let material3 = Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
    surfaces.add(Sphere::stationary(Point3::new(4., 1., 0.), 1.0, material3));

    scene.add(surfaces.build());

    (scene.build(), camera)
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

    let mut earth = Scene::builder();
    earth.set_background(Vec3::new(0.7, 0.8, 1.0));
    earth.add(Sphere::stationary(
        Point3::new(0.0, -10.0, 0.0),
        10.0,
        checker.clone(),
    ));
    earth.add(Sphere::stationary(
        Point3::new(0.0, 10.0, 0.0),
        10.0,
        checker,
    ));

    (earth.build(), camera)
}

fn two_perlin_spheres(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from((13.0, 2.0, 3.0))
        .towards((0.0, 0.0, 0.0))
        .build();

    let mut rng = rand::thread_rng();
    let noise = Arc::new(PerlinNoise::new(&mut rng));
    let perlin = Texture::noise(4.0, noise);

    let mut earth = Scene::builder();
    earth.set_background(Vec3::new(0.7, 0.8, 1.0));
    earth.add(Sphere::stationary(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Material::lambertian(perlin.clone()),
    ));
    earth.add(Sphere::stationary(
        Point3::new(0.0, 2.0, 0.0),
        2.0,
        Material::lambertian(perlin),
    ));

    (earth.build(), camera)
}

fn cornell(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(40.0, aspect_ratio)
        .from((278.0, 278.0, -800.0))
        .towards((278.0, 278.0, 0.0))
        .build();

    let red = Material::lambertian(Vec3::new(0.65, 0.05, 0.05));
    let white = Material::lambertian(Vec3::new(0.73, 0.73, 0.73));
    let green = Material::lambertian(Vec3::new(0.12, 0.45, 0.15));
    let light = Material::diffuse_light(Vec3::new(15.0, 15.0, 15.0));

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
        Point3::new(343., 554., 332.),
        Vec3::new(-130.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -105.0),
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
    let box1 = make_box(
        Point3::new(0.0, 0.001, 0.0),
        Point3::new(165.0, 333.0, 165.0),
        white.clone(),
    );
    let box1 = Rotated::new(box1, Vec3::jhat(), 15.0f64.to_radians());
    let box1 = Translated::new(box1, Vec3::new(265.0, 0.0, 295.0));
    scene.add(box1);

    let box2 = make_box(
        Point3::new(0.0, 0.001, 0.0),
        Point3::new(165.0, 165.0, 165.0),
        white.clone(),
    );
    let box2 = Rotated::new(box2, Vec3::jhat(), (-18.0f64).to_radians());
    let box2 = Translated::new(box2, Vec3::new(130.0, 0.0, 65.0));
    scene.add(box2);

    (scene.build(), camera)
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

fn earth(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(20.0, aspect_ratio)
        .from((0.0, 0.0, 12.0))
        .towards((0.0, 0.0, 0.0))
        .build();

    let mut scene = Scene::builder();
    scene.set_background(Vec3::new(0.7, 0.8, 1.0));

    let texture = earth_texture();
    let material = Material::lambertian(texture);
    let globe = Rotated::new(
        Rotated::new(
            Sphere::stationary(Point3::new(0.0, 0.0, 0.0), 2.0, material),
            Vec3::jhat(),
            (-80.0f64).to_radians(),
        ),
        Vec3::ihat(),
        (15.0f64).to_radians(),
    );
    scene.add(globe);

    (scene.build(), camera)
}

fn earth_texture() -> Texture {
    match read_image_data("./scenes/res/earth.png") {
        Ok((image, width, height)) => Texture::image(image.into(), width, height),
        Err(e) => {
            eprintln!("could not load earth texture: {}", e);
            Vec3::new(0.0, 1.0, 1.0).into()
        }
    }
}

fn read_image_data(path: &str) -> Result<(Vec<u8>, usize, usize), anyhow::Error> {
    use std::fs::File;

    use anyhow::anyhow;
    use png::ColorType::*;

    let mut decoder = png::Decoder::new(File::open(path)?);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info()?;
    let mut img_data = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut img_data)?;

    let img_data = match info.color_type {
        Rgb => img_data,
        Rgba => {
            // convert RGBA -> RGB by discarding the alpha channel
            let mut vec = Vec::with_capacity((img_data.len() / 4) * 3);
            let iter = img_data.chunks(4).flat_map(|p| &p[..3]).cloned();
            vec.extend(iter);
            vec
        }
        t => return Err(anyhow!("uncovered color type: {:?}", t)),
    };
    Ok((img_data, info.width as usize, info.height as usize))
}

fn final_scene(aspect_ratio: f64) -> (Scene, Camera) {
    let camera = Camera::builder(40.0, aspect_ratio)
        .from((478.0, 278.0, -600.0))
        .towards((278.0, 278.0, 0.0))
        .build();

    let mut rng = rand::thread_rng();

    let mut boxes1 = Vec::new();
    let ground = Material::lambertian(Vec3::new(0.48, 0.83, 0.53));
    let boxes_per_side = 20;
    for i in 0..boxes_per_side {
        for j in 0..boxes_per_side {
            let w = 100.0;
            let x0 = -1000.0 + i as f64 * w;
            let z0 = -1000.0 + j as f64 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = rng.gen_range(1.0..101.0);
            let z1 = z0 + w;
            boxes1.push(
                make_box(
                    Point3::new(x0, y0, z0),
                    Point3::new(x1, y1, z1),
                    ground.clone(),
                )
                .into(),
            );
        }
    }

    let mut scene = Scene::builder();
    scene.add(Bvh::new(boxes1));

    let light = Material::diffuse_light(Vec3::new(7.0, 7.0, 7.0));
    scene.add(Quad::new(
        Point3::new(123.0, 554.0, 147.0),
        Vec3::new(300.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 265.0),
        light,
    ));

    let center1 = Point3::new(400.0, 400.0, 200.0);
    let center2 = center1 + Vec3::new(30.0, 0.0, 0.0);
    let sphere_material = Material::lambertian(Vec3::new(0.7, 0.3, 0.1));
    scene.add(Sphere::moving(center1, center2, 50.0, sphere_material));

    scene.add(Sphere::stationary(
        Point3::new(260.0, 150.0, 45.0),
        50.0,
        Material::dielectric(1.5),
    ));
    scene.add(Sphere::stationary(
        Point3::new(0.0, 150.0, 145.0),
        50.0,
        Material::metal(Vec3::new(0.8, 0.8, 0.9), 1.0),
    ));

    let boundary = Sphere::stationary(
        Point3::new(360.0, 150.0, 145.0),
        70.0,
        Material::dielectric(1.5),
    );
    scene.add(boundary.clone());
    scene.add(ConstantMedium::new(boundary, 0.2, Vec3::new(0.2, 0.4, 0.9)));

    let boundary = Sphere::stationary(
        Point3::new(0.0, 0.0, 0.0),
        5000.0,
        Material::dielectric(1.5),
    );
    scene.add(ConstantMedium::new(
        boundary,
        0.0001,
        Vec3::new(1.0, 1.0, 1.0),
    ));

    let emat = Material::lambertian(earth_texture());
    scene.add(Sphere::stationary(
        Point3::new(400.0, 200.0, 400.0),
        100.0,
        emat,
    ));
    let pertext = Texture::noise(0.1, Arc::new(PerlinNoise::new(&mut rng)));
    scene.add(Sphere::stationary(
        Point3::new(220.0, 280.0, 300.0),
        80.0,
        Material::lambertian(pertext),
    ));

    let mut boxes2 = Vec::new();
    let white = Material::lambertian(Vec3::new(0.73, 0.73, 0.73));
    let ns = 1000;
    for _ in 0..ns {
        let p = Point3::new(
            rng.gen_range(0.0..165.0),
            rng.gen_range(0.0..165.0),
            rng.gen_range(0.0..165.0),
        );
        boxes2.push(Sphere::stationary(p, 10.0, white.clone()).into());
    }
    scene.add(Translated::new(
        Rotated::new(Bvh::new(boxes2), Vec3::jhat(), 15f64.to_radians()),
        Vec3::new(-100.0, 270.0, 395.0),
    ));

    (scene.build(), camera)
}
