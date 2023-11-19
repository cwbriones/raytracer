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
use once_cell::sync::Lazy;
use rand::rngs::SmallRng;
use rand::{
    Rng,
    SeedableRng,
};
use rayon::prelude::*;
use structopt::StructOpt;

static NUM_CPUS: Lazy<String> = Lazy::new(|| rayon::current_num_threads().to_string());

static THREADS_HELP: Lazy<String> = Lazy::new(|| {
    format!(
        "Number of threads to use [default: {}]\n\n\
             Defaults to the number of logical cpus.",
        &*NUM_CPUS
    )
});

#[derive(StructOpt)]
/// A simple ray tracer implementation in rust.
struct TracerOpt {
    /// Number of samples to take per pixel.
    #[structopt(long, short, default_value = "4")]
    num_samples: usize,
    /// Destination of the output image.
    ///
    /// supported formats: png, jpg
    #[structopt(long, short, default_value = "output.png")]
    output: String,
    #[structopt(long, short = "t", help(&THREADS_HELP), default_value(&NUM_CPUS))]
    /// Number of render threads to use.
    threads: usize,
    /// Output image width.
    #[structopt(long, default_value = "400")]
    width: usize,
    /// Output image height.
    #[structopt(long)]
    height: Option<usize>,
    /// Output image aspect ratio [default: 1.5].
    #[structopt(long, conflicts_with = "height")]
    aspect_ratio: Option<f64>,
    /// Seed to use for RNG.
    ///
    /// By default the RNG will be seeded through the OS-provided entropy source.
    #[structopt(long)]
    seed: Option<u64>,
    /// A scene file to load from configuration.
    #[structopt(long)]
    scene: Option<String>,
    /// Disables diplaying the estimate of time remaining.
    #[structopt(long)]
    no_progress: bool,
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
}

fn main() -> anyhow::Result<()> {
    let config = TracerOpt::from_args();

    let aspect_ratio = config.aspect_ratio();
    let image_width = config.width;
    let _threads = config.threads;
    let image_height = config.height();
    let samples_per_pixel: usize = config.num_samples;
    let rays = image_width * image_height * samples_per_pixel;
    let max_depth = 50;

    let start = Instant::now();

    if config.threads != rayon::current_num_threads() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .build_global()
            .expect("init global threadpool");
    }

    let (scene, camera) = if let Some(ref path) = config.scene {
        scene::load_scene(path, aspect_ratio)
            .with_context(|| format!("load scene file '{path}'"))?
    } else {
        scene::example::one_weekend(aspect_ratio)
    };

    let mut buf = vec![0; 3 * image_width * image_height];
    let (progress, recorder) = progress::ProgressBar::new(image_width * image_height);
    let scene = scene.clone();
    let camera = camera.clone();

    if !config.no_progress {
        rayon::spawn(move || progress.run(samples_per_pixel));
    }

    buf.par_chunks_mut(3 * image_width)
        .enumerate()
        .for_each(|(j, row)| {
            let j = image_height - j - 1;
            let mut rng = small_rng(config.seed);
            for i in 0..image_width {
                let color_vec = average(samples_per_pixel, || {
                    let u = (i as f64 + rng.gen::<f64>()) / (image_width - 1) as f64;
                    let v = (j as f64 + rng.gen::<f64>()) / (image_height - 1) as f64;
                    let ray = camera.get_ray(&mut rng, u, v);
                    scene
                        .ray_color(ray, &mut rng, max_depth)
                        .unwrap_or_default()
                });
                let r = 256. * (color_vec.x()).sqrt().clamp(0.0, 0.99);
                let g = 256. * (color_vec.y()).sqrt().clamp(0.0, 0.99);
                let b = 256. * (color_vec.z()).sqrt().clamp(0.0, 0.99);

                row[3 * i..3 * i + 3].copy_from_slice(&[r as u8, g as u8, b as u8]);
                recorder.record();
            }
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
