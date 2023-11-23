use std::collections::VecDeque;
use std::sync::{
    atomic::AtomicUsize,
    atomic::Ordering,
    Arc,
    Mutex,
};
use std::time::{
    Duration,
    Instant,
};

const REPORT_PERIOD: Duration = Duration::from_millis(1000);
const AVERAGE_OVER_PERIODS: usize = 60;

const ANSI_CURSOR_UP: &str = "\x1b[4A";

/// A handle to a progress bar that allows for recording forward progress.
#[derive(Clone)]
pub struct ProgressRecorder {
    remaining: Arc<AtomicUsize>,
    state: Arc<Mutex<ProgressState>>,
}

impl ProgressRecorder {
    /// Create a new progress recorder with `total` units of work to process.
    ///
    /// The recorder can be cloned to observe progress being made from worker threads.
    pub fn new(total: usize) -> ProgressRecorder {
        let start = Instant::now();
        let remaining = Arc::new(AtomicUsize::new(total));
        let state = Arc::new(Mutex::new(ProgressState {
            rates: VecDeque::new(),
            start,
            last_remaining: total + 1,
            last_update: start,
        }));
        ProgressRecorder { remaining, state }
    }

    /// Record a single unit of work as completed and possibly print out progress.
    pub fn record(&self, samples_per_pixel: usize) {
        let remaining = self.remaining.fetch_sub(1, Ordering::Relaxed);
        if remaining == 0 {
            return;
        }
        if let Ok(mut state) = self.state.try_lock() {
            state.update(remaining, samples_per_pixel);
        }
    }
}

struct ProgressState {
    start: Instant,
    last_update: Instant,
    rates: VecDeque<f32>,
    last_remaining: usize,
}

impl ProgressState {
    fn update(&mut self, remaining: usize, samples_per_pixel: usize) {
        let now = Instant::now();
        if now - self.last_update < REPORT_PERIOD {
            return;
        }
        self.last_update = now;
        let current_rate = (self.last_remaining - remaining) as f32;
        self.last_remaining = remaining;
        if self.rates.len() == AVERAGE_OVER_PERIODS {
            self.rates.pop_front();
        }
        self.rates.push_back(current_rate);
        let sum = self.rates.iter().sum::<f32>();
        let average_rate = sum / (self.rates.len() as f32);
        let estimated_time = REPORT_PERIOD.mul_f32(remaining as f32 / average_rate);

        if self.rates.len() == 1 {
            eprint!("\n\n\n\n");
        }
        eprint!("{}", ANSI_CURSOR_UP);
        eprintln!(
            "    Elapsed Time: {}    ",
            format_duration(self.start.elapsed())
        );
        eprintln!("  Remaining Time: {}    ", format_duration(estimated_time));
        eprintln!(
            "   Samples / sec: {}    ",
            average_rate * samples_per_pixel as f32
        );
        eprintln!("Remaining Pixels: {}    ", remaining);
    }
}

fn format_duration(d: ::std::time::Duration) -> String {
    let hours = d.as_secs() / 3600;
    let minutes = (d.as_secs() - hours * 3600) / 60;
    let secs = d.as_secs() - minutes * 60 - hours * 3600;

    if hours > 0 {
        format!("{:0>2}:{:0>2}:{:0>2}", hours, minutes, secs)
    } else {
        format!("{:0>2}:{:0>2}", minutes, secs)
    }
}
