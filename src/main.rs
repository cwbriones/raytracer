mod vec;
mod ray;

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
}

fn main() {
    const IMAGE_WIDTH: usize = 256;
    const IMAGE_HEIGHT: usize = 256;

    println!("P3 {} {}", IMAGE_WIDTH, IMAGE_HEIGHT);
    println!("255");
    (0..IMAGE_HEIGHT)
        .rev()
        .flat_map(|j| {
            eprint!("\rScanlines remaining: {}     ", j);
            (0..IMAGE_WIDTH).map(move |i| (i, j))
        })
        .map(|(i, j)|
            Color::new(
                i as f64 / (IMAGE_WIDTH - 1) as f64,
                j as f64 / (IMAGE_HEIGHT - 1) as f64,
                0.25
            )
        )
        .for_each(|c| c.write());
    eprintln!("\nDone.");
}
