//! Measure the native random derivation and permutation floors on one pinned core.

use std::env;
use std::fs;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use _hyperloader::rng::{block, feistel_permute, sample_seed_words};

const PERMUTATION_DOMAINS: [u64; 3] = [1 << 17, 300_000, 1_000_000_007];

struct Config {
    core: usize,
    iterations: u64,
    label: String,
    trials: usize,
    warmup_iterations: u64,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut core = None;
        let mut iterations = 2_000_000_u64;
        let mut label = None;
        let mut trials = 20_usize;
        let mut warmup_iterations = 200_000_u64;
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            if argument == "--bench" {
                continue;
            }
            let value = arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))?;
            match argument.as_str() {
                "--core" => core = Some(parse_value(&argument, &value)?),
                "--iterations" => iterations = parse_value(&argument, &value)?,
                "--label" => label = Some(value),
                "--trials" => trials = parse_value(&argument, &value)?,
                "--warmup-iterations" => {
                    warmup_iterations = parse_value(&argument, &value)?;
                }
                _ => return Err(format!("unknown argument {argument}")),
            }
        }
        let config = Self {
            core: core.ok_or_else(|| "--core is required".to_owned())?,
            iterations,
            label: label.ok_or_else(|| "--label is required".to_owned())?,
            trials,
            warmup_iterations,
        };
        if config.iterations == 0 || config.warmup_iterations == 0 || config.trials < 10 {
            return Err(
                "iterations and warmup must be positive, and trials must be at least 10".to_owned(),
            );
        }
        if config.label != "perf" && config.label != "eff" {
            return Err("label must be perf or eff".to_owned());
        }
        Ok(config)
    }
}

fn parse_value<T: std::str::FromStr>(argument: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{argument} received an invalid value"))
}

fn read_trimmed(path: &str) -> Result<String, String> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(|error| format!("cannot read {path}: {error}"))
}

fn allowed_cores() -> Result<String, String> {
    let status = read_trimmed("/proc/self/status")?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .map(str::to_owned)
        .ok_or_else(|| "cannot find Cpus_allowed_list in /proc/self/status".to_owned())
}

fn cpu_model(core: usize) -> Result<String, String> {
    let output = Command::new("lscpu")
        .arg("-e=CPU,MODELNAME")
        .output()
        .map_err(|error| format!("cannot execute lscpu: {error}"))?;
    if !output.status.success() {
        return Err("lscpu did not return the per-core model table".to_owned());
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("lscpu returned invalid UTF-8: {error}"))?
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            let listed_core = fields.next()?.parse::<usize>().ok()?;
            (listed_core == core).then(|| fields.collect::<Vec<_>>().join(" "))
        })
        .map(|value| value.replace(',', ";"))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("lscpu did not identify core {core}"))
}

fn frequency_path(core: usize, field: &str) -> String {
    format!("/sys/devices/system/cpu/cpu{core}/cpufreq/{field}")
}

fn current_frequency(core: usize) -> Result<u64, String> {
    parse_value(
        "scaling_cur_freq",
        &read_trimmed(&frequency_path(core, "scaling_cur_freq"))?,
    )
}

fn run_sample_derivation(iterations: u64) -> u64 {
    let mut checksum = 0_u64;
    for coordinate in 0..iterations {
        let (torch_seed, random_seed, numpy) =
            sample_seed_words(black_box(0x0123_4567_89AB_CDEF), 17, black_box(coordinate));
        checksum ^= torch_seed ^ random_seed ^ u64::from(numpy[coordinate as usize & 3]);
    }
    black_box(checksum)
}

fn run_native_draw(iterations: u64) -> u64 {
    let mut checksum = 0_u64;
    for coordinate in 0..iterations {
        let words = block(
            black_box(0x0123_4567_89AB_CDEF),
            17,
            black_box(coordinate),
            2,
            0,
        );
        checksum ^= u64::from(words[coordinate as usize & 3]);
    }
    black_box(checksum)
}

fn run_permutation(iterations: u64, domain: u64) -> u64 {
    let mut checksum = 0_u64;
    let mut position = 0_u64;
    for _ in 0..iterations {
        let index = feistel_permute(
            black_box(0x0123_4567_89AB_CDEF),
            17,
            domain,
            black_box(position),
        )
        .expect("the benchmark domain and position are valid");
        checksum ^= index;
        position += 1;
        if position == domain {
            position = 0;
        }
    }
    black_box(checksum)
}

fn measure(function: impl FnOnce() -> u64, iterations: u64) -> (u128, f64, u64) {
    let start = Instant::now();
    let checksum = function();
    let elapsed = start.elapsed().as_nanos();
    (elapsed, elapsed as f64 / iterations as f64, checksum)
}

fn emit_row(
    metric: &str,
    trial: usize,
    iterations: u64,
    measurement: (u128, f64, u64),
    frequency: u64,
) {
    let (elapsed, nanoseconds, checksum) = measurement;
    println!(
        "data,{metric},{trial},{iterations},{elapsed},{nanoseconds:.9},{checksum},{frequency}"
    );
}

fn main() -> Result<(), String> {
    let config = Config::parse()?;
    let allowed = allowed_cores()?;
    if allowed != config.core.to_string() {
        return Err(format!(
            "the process must be pinned only to core {}, but Cpus_allowed_list is {allowed}",
            config.core
        ));
    }
    let governor = read_trimmed(&frequency_path(config.core, "scaling_governor"))?;
    let maximum_frequency = read_trimmed(&frequency_path(config.core, "cpuinfo_max_freq"))?;
    let model = cpu_model(config.core)?;

    println!("meta,label,{}", config.label);
    println!("meta,core,{}", config.core);
    println!("meta,cpu_model,{model}");
    println!("meta,governor,{governor}");
    println!("meta,max_freq_khz,{maximum_frequency}");
    println!("meta,trials,{}", config.trials);
    println!("meta,iterations,{}", config.iterations);
    println!("meta,warmup_iterations,{}", config.warmup_iterations);
    println!("meta,sample_derivation_blocks,2");
    println!("meta,feistel_rounds,8");
    println!("kind,metric,trial,iterations,elapsed_ns,ns_per_op,checksum,freq_khz");

    black_box(run_sample_derivation(config.warmup_iterations));
    black_box(run_native_draw(config.warmup_iterations));
    for domain in PERMUTATION_DOMAINS {
        black_box(run_permutation(config.warmup_iterations, domain));
    }

    for trial in 0..config.trials {
        emit_row(
            "sample_derivation",
            trial,
            config.iterations,
            measure(
                || run_sample_derivation(config.iterations),
                config.iterations,
            ),
            current_frequency(config.core)?,
        );
        emit_row(
            "native_draw",
            trial,
            config.iterations,
            measure(|| run_native_draw(config.iterations), config.iterations),
            current_frequency(config.core)?,
        );
        for domain in PERMUTATION_DOMAINS {
            emit_row(
                &format!("permutation_{domain}"),
                trial,
                config.iterations,
                measure(
                    || run_permutation(config.iterations, domain),
                    config.iterations,
                ),
                current_frequency(config.core)?,
            );
        }
    }
    Ok(())
}
