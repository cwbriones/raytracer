use std::cell::RefCell;

use rand::Rng;

use crate::geom::Vec3;
use crate::surfaces::Bvh;
use crate::surfaces::BvhBuilder;
use crate::surfaces::Surface;
use crate::trace::Hittable;
use crate::trace::Interval;
use crate::trace::Ray;

pub mod example;
mod load;

pub use load::load_scene;

#[derive(Clone)]
pub struct Scene {
    root: Surface,
    background: Vec3,
}

impl Scene {
    pub fn builder() -> SceneBuilder {
        SceneBuilder::new()
    }
}

pub struct SceneBuilder {
    surfaces: BvhBuilder,
    background: Vec3,
    use_bvh: bool,
}

impl SceneBuilder {
    fn new() -> Self {
        SceneBuilder {
            surfaces: Bvh::builder(),
            background: Vec3::default(),
            use_bvh: true,
        }
    }

    pub fn add<S>(&mut self, surface: S)
    where
        S: Into<Surface>,
    {
        self.surfaces.add(surface);
    }

    pub fn set_background(&mut self, background: Vec3) {
        self.background = background;
    }

    pub fn set_use_bvh(&mut self, use_bvh: bool) {
        self.use_bvh = use_bvh;
    }

    pub fn build(self) -> Scene {
        let root = if self.use_bvh {
            self.surfaces.build()
        } else {
            self.surfaces.build_leaf()
        }
        .into();
        Scene {
            root,
            background: self.background,
        }
    }
}

impl Scene {
    thread_local! {
        static BUF: RefCell<Vec<(Vec3, Vec3)>> = const { RefCell::new(Vec::new()) };
    }

    pub fn ray_color<R: Rng>(&self, ray: Ray, rng: &mut R, max_depth: usize) -> Vec3 {
        Self::BUF.with_borrow_mut(|buf| {
            buf.clear();
            self.ray_color_impl(ray, rng, max_depth, buf)
        })
    }

    fn ray_color_impl<R: Rng>(
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
        // Because the color calculation is inherently recursive, we need to
        // maintain a stack of light emitted/attenuated. This is handled using
        // a thread_local so that the caller doesn't need to provide a fixed size
        // buffer on every single call.
        let mut depth = 0;
        loop {
            let interval = Interval(0.001, ::std::f64::INFINITY);
            if let Some(hit) = self.root.hit(&ray, interval) {
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
            } else {
                // The ray escaped after bouncing.
                stack.push((Vec3::default(), self.background));
            }
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
