use std::collections::VecDeque;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{
    Duration,
    Instant,
};

const REPORT_PERIOD: Duration = Duration::from_millis(1000);
const AVERAGE_OVER_PERIODS: usize = 60;

const ANSI_CURSOR_UP: &'static str = "\x1b[4A";

/// A wrapper around a simple progress reporter.
///
/// This will report elapsed time, estimated time, as well as an
/// average rate over the last minute.
pub struct ProgressBar {
    total: usize,
    remaining: Arc<AtomicUsize>,
}

/// A handle to a progress bar that allows for recording forward progress.
pub struct ProgressRecorder {
    remaining: Arc<AtomicUsize>,
}

impl ProgressRecorder {
    /// Record a single unit of work as completed.
    pub fn record(&self) {
        self.remaining.fetch_sub(1, Ordering::Relaxed);
    }
}

impl ProgressBar {
    /// Create a new progress bar with `total` units of work to process.
    pub fn new(total: usize) -> Self {
        let remaining = Arc::new(AtomicUsize::new(total));
        ProgressBar { total, remaining }
    }

    /// Create a new recorder for this progress bar that can be used to log work being processed.
    pub fn create_recorder(&self) -> ProgressRecorder {
        ProgressRecorder {
            remaining: self.remaining.clone(),
        }
    }

    /// Run the progress bar.
    ///
    /// This will consume the struct and block the current thread until the operation is completed,
    /// so you'll generally want to spawn this on a separate thread.
    pub fn run(self, samples_per_pixel: usize) {
        let start = Instant::now();
        let mut last_check = self.total + 1;
        let mut rates = VecDeque::with_capacity(AVERAGE_OVER_PERIODS);
        eprint!("\n\n\n\n");
        loop {
            let remaining = self.remaining.load(Ordering::Relaxed);
            if remaining == 0 {
                break;
            }
            let current_rate = (last_check - remaining) as f32;
            if rates.len() == AVERAGE_OVER_PERIODS {
                rates.pop_front();
            }
            rates.push_back(current_rate);
            let average_rate = rates.iter().sum::<f32>() / (rates.len() as f32);

            let estimated_time = REPORT_PERIOD.mul_f32(remaining as f32 / average_rate);

            last_check = remaining;

            eprint!("{}", ANSI_CURSOR_UP);
            eprintln!("    Elapsed Time: {}    ", format_duration(start.elapsed()));
            eprintln!("  Remaining Time: {}    ", format_duration(estimated_time));
            eprintln!(
                "   Samples / sec: {}    ",
                average_rate * samples_per_pixel as f32
            );
            eprintln!("Remaining Pixels: {}    ", remaining);
            ::std::thread::sleep(REPORT_PERIOD);
        }
        eprintln!("Done!");
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
