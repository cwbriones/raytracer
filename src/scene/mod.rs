use rand::Rng;

use crate::bvh::Bvh;
use crate::geom::Vec3;
use crate::surfaces::Surface;
use crate::trace::Hittable;
use crate::trace::Ray;

pub mod example;
mod load;

pub use load::load_scene;

#[derive(Clone)]
pub struct Scene {
    root: Bvh<Surface>,
    background: Vec3,
}

impl Scene {
    pub fn builder() -> SceneBuilder {
        SceneBuilder {
            surfaces: Vec::new(),
            background: Vec3::default(),
        }
    }
}

pub struct SceneBuilder {
    surfaces: Vec<Surface>,
    background: Vec3,
}

impl SceneBuilder {
    pub fn add<S>(&mut self, surface: S)
    where
        S: Into<Surface>,
    {
        self.surfaces.push(surface.into());
    }

    pub fn set_background(&mut self, background: Vec3) {
        self.background = background;
    }

    pub fn build(mut self) -> Scene {
        let root = Bvh::new(&mut self.surfaces);
        Scene {
            root,
            background: self.background,
        }
    }
}

impl Scene {
    pub fn ray_color<R: Rng>(
        &self,
        mut ray: Ray,
        rng: &mut R,
        max_depth: usize,
        stack: &mut Vec<(Vec3, Vec3)>,
    ) -> Vec3 {
        // In the book this is done recursively, but I've refactored it into an
        // explicit accumulating loop to make profiling easier.
        //
        // This prevents additional tracing iterations from adding stack frames.
        //
        // Because of the emissivity calculation, we need to keep a stack of light
        // emitted/attenuated at each iteration to then resolve once the ray has
        // stopped scattering.
        //
        // FIXME: Because the color calculation is inherently recursive, we need to
        // maintain a stack of light emitted/attenuated. This isn't inherently a problem
        // but is awkward because the caller has to provide a buffer so that we don't
        // allocate on every single call.
        stack.clear();
        let mut depth = 0;
        loop {
            if let Some(hit) = self.root.hit(&ray, 0.001, ::std::f64::INFINITY) {
                let emitted = hit.material.emit(&hit);
                if let Some((cur_attenuation, scattered)) = hit.material.scatter(&ray, &hit, rng) {
                    // The ray was scattered in a different direction. Continue following it.
                    stack.push((emitted, cur_attenuation));
                    ray = scattered;
                }
                depth += 1;
                if depth < max_depth {
                    continue;
                }
                // The ray was completely absorbed, or we have run out of iterations.
                //
                // Either way we stop scattering.
                stack.push((emitted, Vec3::default()));
                return resolve_color(stack);
            }
            // The ray escaped after bouncing.
            stack.push((Vec3::default(), self.background));
            return resolve_color(stack);
        }
    }
}

fn resolve_color(stack: &[(Vec3, Vec3)]) -> Vec3 {
    let mut acc = Vec3::new(1.0, 1.0, 1.0);
    for (emitted, attenuation) in stack.iter().rev() {
        acc = *emitted + attenuation.mul_pointwise(&acc);
    }
    acc
}
