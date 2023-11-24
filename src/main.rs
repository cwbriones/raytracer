#![allow(clippy::suspicious_operation_groupings)]
mod bvh;
mod camera;
mod geom;
mod material;
mod progress;
mod scene;
mod surfaces;
mod trace;
mod util;

use std::fs::File;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{
    Rng,
    SeedableRng,
};
use rayon::prelude::*;

#[derive(Parser)]
/// A simple ray tracer implementation in rust.
struct TracerOpt {
    /// Number of samples to take per pixel.
    #[arg(long, short, default_value = "4")]
    num_samples: usize,
    /// Destination of the output image.
    ///
    /// supported formats: png, jpg
    #[arg(long, short, default_value = "output.png")]
    output: String,
    #[arg(long, short, default_value_t = default_threads())]
    /// Number of render threads to use.
    threads: usize,
    /// Output image width.
    #[arg(long, default_value = "400")]
    width: usize,
    /// Output image height.
    #[arg(long)]
    height: Option<usize>,
    /// Output image aspect ratio [default: 1.5].
    #[arg(long, conflicts_with = "height")]
    aspect_ratio: Option<f64>,
    /// Seed to use for RNG.
    ///
    /// By default the RNG will be seeded through the OS-provided entropy source.
    #[arg(long)]
    seed: Option<u64>,
    /// A scene file to load from configuration.
    #[arg(long)]
    scene: Option<String>,
    /// An example scene from the book, generated at random.
    #[arg(long)]
    example: Option<scene::example::Example>,
    /// Disables diplaying the estimate of time remaining.
    #[arg(long)]
    no_progress: bool,
    /// Disables the use of the bvh acceleration structure.
    ///
    /// This significantly degrades performance for large scenes, but can be useful for debugging
    /// since it guarantees every surface intersection will be tested.
    #[arg(long)]
    no_bvh: bool,
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(Into::into)
        .unwrap_or(1)
}

impl TracerOpt {
    const DEFAULT_ASPECT_RATIO: f64 = 1.5;

    fn height(&self) -> usize {
        if let Some(height) = self.height {
            height
        } else if let Some(aspect_ratio) = self.aspect_ratio {
            (self.width as f64 / aspect_ratio) as usize
        } else {
            (self.width as f64 / TracerOpt::DEFAULT_ASPECT_RATIO) as usize
        }
    }

    fn aspect_ratio(&self) -> f64 {
        if let Some(height) = self.height {
            self.width as f64 / height as f64
        } else if let Some(aspect_ratio) = self.aspect_ratio {
            aspect_ratio
        } else {
            TracerOpt::DEFAULT_ASPECT_RATIO
        }
    }

    fn scene(&self) -> Result<(scene::Scene, camera::Camera), anyhow::Error> {
        let aspect_ratio = self.aspect_ratio();
        if let Some(ref path) = self.scene {
            return scene::load_scene(path, aspect_ratio, !self.no_bvh)
                .with_context(|| format!("load scene file '{path}'"));
        }
        Ok(self
            .example
            .clone()
            .unwrap_or(scene::example::Example::OneWeekend)
            .scene(aspect_ratio))
    }
}

fn main() -> anyhow::Result<()> {
    let config = TracerOpt::parse();

    let image_width = config.width;
    let _threads = config.threads;
    let image_height = config.height();
    let samples_per_pixel: usize = config.num_samples;
    let rays = image_width * image_height * samples_per_pixel;
    let max_depth = 50;

    let start = Instant::now();

    if config.threads != default_threads() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build_global()
            .expect("init global threadpool");
    }
    let (scene, camera) = config.scene()?;

    let mut buf = vec![0; 3 * image_width * image_height];
    let (progress, recorder) = progress::ProgressBar::new(image_width * image_height);
    let scene = scene.clone();
    let camera = camera.clone();

    if !config.no_progress {
        rayon::spawn(move || progress.run(samples_per_pixel));
    }

    buf.par_chunks_mut(3).enumerate().for_each(|(idx, pixel)| {
        let i = idx % image_width;
        let j = image_height - idx / image_width - 1;

        let mut rng = small_rng(config.seed);
        let color_vec = average(samples_per_pixel, || {
            let u = (i as f64 + rng.gen::<f64>()) / (image_width - 1) as f64;
            let v = (j as f64 + rng.gen::<f64>()) / (image_height - 1) as f64;
            let ray = camera.get_ray(&mut rng, u, v);
            scene.ray_color(ray, &mut rng, max_depth)
        });
        pixel[0] = (256. * (color_vec.x()).sqrt().clamp(0.0, 0.99)) as u8;
        pixel[1] = (256. * (color_vec.y()).sqrt().clamp(0.0, 0.99)) as u8;
        pixel[2] = (256. * (color_vec.z()).sqrt().clamp(0.0, 0.99)) as u8;
        recorder.record();
    });

    let elapsed_sec = start.elapsed().as_secs_f64();
    let rays_per_sec = (rays as f64) / elapsed_sec;
    eprintln!("\nDone in {elapsed_sec:.2}s ({rays_per_sec:.0} rays/s)");

    let f = File::create(config.output)?;
    let mut encoder = png::Encoder::new(f, image_width as u32, image_height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer
        .write_image_data(&buf[..])
        .with_context(|| "could not write image")?;
    writer.finish()?;
    Ok(())
}

fn small_rng(seed: Option<u64>) -> impl Rng {
    seed.map(SmallRng::seed_from_u64)
        .unwrap_or_else(SmallRng::from_entropy)
}

fn average<T, F>(n: usize, mut f: F) -> T
where
    F: FnMut() -> T,
    T: Default + ::std::ops::AddAssign + ::std::ops::Div<f64, Output = T>,
{
    let mut acc = <T as Default>::default();
    for _ in 0..n {
        acc += f();
    }
    acc / (n as f64)
}
