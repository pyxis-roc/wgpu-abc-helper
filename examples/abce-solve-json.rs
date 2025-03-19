// SPDX-FileCopyrightText: 2025 University of Rochester
//
// SPDX-License-Identifier: MIT

// The main.rs will load a json file and attempt to solve it.
use abc_helper::{ConstraintHelper, ConstraintModule};
use anyhow::{Context, Result};
use clap::Parser;
use env_logger::{Builder, Target};
use std::{fs::File, path::PathBuf};

#[derive(Parser)]
struct Cli {
    /// The json path to the constraint module
    path: PathBuf,
    /// The index of the summary to solve. Defaults to the last summary.
    #[arg(long)]
    idx: Option<usize>,
    /// The log file to write to. Defaults to stdout.
    #[arg(long, short)]
    log: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let mut builder = Builder::new();
    builder.filter_level(log::LevelFilter::Trace);
    if let Some(log_path) = args.log {
        // Log to the specified file
        let log_file = File::create(log_path).with_context(|| "Failed to create log file")?;
        builder.target(Target::Pipe(Box::new(log_file)));
    } else {
        // Default to logging to stdout
        builder.target(Target::Stdout);
    }
    builder.parse_default_env().init();
    let contents = std::fs::read_to_string(&args.path)
        .with_context(|| format!("could not read file `{}`", args.path.display()))?;
    let module: ConstraintModule = serde_json::from_str(&contents).unwrap();

    let total_summaries = module.get_num_summaries();
    if total_summaries == 0 {
        return Err(anyhow::anyhow!(
            "No summaries found in the constraint module"
        ));
    }

    let idx = args.idx.unwrap_or(total_summaries - 1);
    if idx >= total_summaries {
        let msg = format!("Invalid summary index: {idx}. Total summaries: {total_summaries}");
        log::error!("{}", msg);
        return Err(anyhow::anyhow!(
            "Invalid summary index: {idx}. Total summaries: {total_summaries}"
        ));
    }

    let res = module.solve(idx.into());
    if let Err(e) = &res {
        log::error!("Error solving constraint: {e}");
        return Err(anyhow::anyhow!("Error solving constraint: {e}"));
    }
    let res = res.unwrap();

    log::info!("Constraint solution for summary {idx}:");
    for (i, a) in ConstraintHelper::solution_to_true_false(&res) {
        log::info!("Constraint id: {i}: {a}");
    }

    Ok(())
}
