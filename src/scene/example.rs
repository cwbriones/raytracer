/// Example scenes as shown in the Ray Tracing in One Weeked series.
use rand::distributions::Uniform;
use rand::Rng;

use super::Scene;
use crate::camera::Camera;
use crate::geom::{
    Point3,
    Vec3,
};
use crate::material::Material;
use crate::surfaces::Sphere;

/// Create a random scene as shown in the final section of Ray Tracing in One Weekend.
pub fn one_weekend(aspect_ratio: f64) -> (Scene, Camera) {
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
    objects.add(Sphere::new(
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
                objects.add(Sphere::new(center, 0.2, material));
            }
        }
    }
    let material1 = Material::lambertian(Vec3::new(0.05, 0.2, 0.6));
    objects.add(Sphere::new(Point3::new(-4., 1., 0.), 1.0, material1));
    let material2 = Material::dielectric(1.5);
    objects.add(Sphere::new(Point3::new(0., 1., 0.), 1.0, material2));
    let material3 = Material::metal(Vec3::new(0.7, 0.6, 0.5), 0.0);
    objects.add(Sphere::new(Point3::new(4., 1., 0.), 1.0, material3));

    (objects.build(), camera)
}
