use rand::Rng;

use crate::bvh::BVH;
use crate::geom::Vec3;
use crate::surfaces::Surface;
use crate::trace::Hittable;
use crate::trace::{
    Hit,
    Ray,
};

pub mod example;
mod load;

pub use load::load_scene;

#[derive(Clone)]
pub struct Scene {
    root: BVH<Surface>,
}

impl Scene {
    pub fn builder() -> SceneBuilder {
        SceneBuilder {
            surfaces: Vec::new(),
        }
    }
}

pub struct SceneBuilder {
    surfaces: Vec<Surface>,
}

impl SceneBuilder {
    pub fn add<S>(&mut self, surface: S)
    where
        S: Into<Surface>,
    {
        self.surfaces.push(surface.into());
    }

    pub fn build(mut self) -> Scene {
        let root = BVH::new(&mut self.surfaces);
        Scene { root }
    }
}

impl Scene {
    fn scatter(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        self.root.hit(ray, t_min, t_max)
    }

    pub fn ray_color<R: Rng>(&self, mut ray: Ray, rng: &mut R, max_depth: usize) -> Option<Vec3> {
        // In the book this is done recursively, but I've refactored it into an
        // explicit accumulating loop to make profiling easier.
        //
        // This prevents additional tracing iterations from adding stack frames.
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
                let color = &Vec3::new(1.0, 1.0, 1.0).lerp(Vec3::new(0.5, 0.7, 1.0), t);
                return Some(attenuation.mul_pointwise(color));
            }
        }
        // The ray hasn't resolved for the maximum allowed iterations,
        None
    }
}
