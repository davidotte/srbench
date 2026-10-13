"""
Job Submission
--------------

This script automates the execution of machine learning experiments on multiple
datasets. It can run locally or submit jobs to a SLURM cluster using Singularity
containers.

Features:
- Avoids rerunning completed experiments unless --noskips is specified.
- Supports adding noise, scaling, and symbolic dataset handling.
- Parallel local execution with joblib.

Usage:
    python experiment/analyze.py <dataset_dir> [options]

Example:
    python experiment/analyze.py datasets/blackbox -ml rasr --slurm -q <partition>

The evals in the paper are launched through scripts/launch_evals.py, which
builds these commands.

"""

import os
import sys
import shlex
import tempfile
import hashlib
import argparse
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
from glob import glob
from joblib import Parallel, delayed
from yaml import load, Loader
from seeds import SEEDS


# Run once per GNU-parallel slot: pins the slot to its own GPU and its own block
# of cpus_per_task CPUs, since all slots share a single SLURM step.
RUN_SLOT_SH = r"""#!/usr/bin/bash
# Args: $1 = 1-based slot number (GNU parallel {%}); $2 = full shell command.
# Env:  SR_CPT = CPUs per slot (cpus_per_task).
set -u
slot="$1"
cmd="$2"
cpt="${SR_CPT:?SR_CPT not set}"

# Expand the allowed CPU list (e.g. "8-15,40-47") into an array of CPU ids.
allowed="$(taskset -cp $$ | awk -F': ' '{print $NF}')"
cpus=()
IFS=',' read -ra _parts <<< "$allowed"
for _p in "${_parts[@]}"; do
    if [[ "$_p" == *-* ]]; then
        for ((c=${_p%-*}; c<=${_p#*-}; c++)); do cpus+=("$c"); done
    elif [[ -n "$_p" ]]; then
        cpus+=("$_p")
    fi
done

export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$((slot - 1))"

# Pin to this slot's block of cpt CPUs, or run unpinned if the block is incomplete.
start="$(( (slot - 1) * cpt ))"
myset=("${cpus[@]:start:cpt}")
if [[ "${#myset[@]}" -eq "$cpt" ]]; then
    IFS=','; cpuset="${myset[*]}"; unset IFS
    exec taskset -c "$cpuset" bash -c "$cmd"
else
    echo "run_slot.sh: could not pin slot $slot to $cpt CPUs (allocation: $allowed); running unpinned" >&2
    exec bash -c "$cmd"
fi
"""


def parse_args():
    """Parse command line arguments for experiment configuration."""

    parser = argparse.ArgumentParser(
        description=("Run and manage machine learning experiments locally or on SLURM clusters. "
                     "Use -h/--help to see how to configure this file."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=True
    )

    # Script name
    parser.add_argument("-script", dest="script", type=str, default="optimize_model", help="Name of Python script to execute inside container.")

    # Directories and paths
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory (e.g., pmlb/datasets).")
    parser.add_argument("-pretrained_dir", dest="pretrained_dir", type=str, default="/", help="Folder with pre-trained models or checkpoints.")
    parser.add_argument("-results", dest="results_dir", type=str, default="results", help="Results output directory.")
    parser.add_argument("-images", dest="singularity_dir", type=str, default="../singularity", help="Directory containing Singularity (.sif) images. Required when using SLURM.")
    parser.add_argument("-sif", dest="sif_override", type=str, default=None, help="Path to a specific .sif image, overriding the default {images}/{ml}.sif.")

    # Experiment configuration
    parser.add_argument("-ml", dest="learners", type=str, default=None, help="Comma-separated list of ML methods (must match filenames in experiment/methods).")
    parser.add_argument(
        "-rasr_name_suffix",
        dest="rasr_name_suffix",
        type=str,
        default="",
        help="Optional suffix appended to RASR result names only (e.g., _ckpt2).",
    )
    parser.add_argument(
        "-rasr_candidates",
        dest="rasr_candidates",
        type=int,
        default=None,
        help="Override RASR's num_candidates via the RASR_NUM_CANDIDATES env var, "
             "without editing regressor.py. Leave unset to use its own default.",
    )
    parser.add_argument(
        "-rasr_temperature",
        dest="rasr_temperature",
        type=float,
        default=None,
        help="Override RASR's sampling temperature via the RASR_TEMPERATURE env var "
             "(0 = greedy decoding). Leave unset to use its own default.",
    )
    parser.add_argument(
        "-rasr_n_restarts",
        dest="rasr_n_restarts",
        type=int,
        default=None,
        help="Override RASR's constant-refinement n_restarts via the RASR_N_RESTARTS "
             "env var. Leave unset to use its own default.",
    )
    parser.add_argument(
        "-rasr_init_noise_scale",
        dest="rasr_init_noise_scale",
        type=float,
        default=None,
        help="Override RASR's constant-refinement init_noise_scale via the "
             "RASR_INIT_NOISE_SCALE env var. Leave unset to use its own default.",
    )
    parser.add_argument(
        "--rasr_no_refine",
        dest="rasr_no_refine",
        action="store_true",
        help="Disable RASR's constant refinement via the RASR_REFINE_CONSTANTS=0 env var.",
    )
    parser.add_argument(
        "-refiner_workers",
        dest="refiner_workers",
        type=int,
        default=None,
        help="Override RASR's and Flash-ANSR's number of constant-refinement worker "
             "processes via the REFINER_WORKERS env var. Leave unset to use their own "
             "default (0, i.e. serial refinement).",
    )
    parser.add_argument("--sym_data", action="store_true", help="Specify if datasets are symbolic.")
    parser.add_argument("--save_population", action="store_true", help="Save population at the end of the run.")
    parser.add_argument("--skip_tuning", action="store_true", help="Skip hyperparameter tuning step.")
    parser.add_argument("--tuned", action="store_true", help="Run tuned version of estimators.")
    parser.add_argument("--scale_x", action="store_true", help="Apply feature scaling.")
    parser.add_argument("--scale_y", action="store_true", help="Apply target scaling.")
    parser.add_argument('--ecotracker', action='store_true', default=False, help='Enables ecotracker. May conflict with SO.')

    # Noise injection
    parser.add_argument("-target_noise", dest="y_noise", type=float, default=0.0, help="Gaussian noise added to target variable.")
    parser.add_argument("-feature_noise", dest="x_noise", type=float, default=0.0, help="Gaussian noise added to input features.")

    # Execution mode and resources
    parser.add_argument("--no_docker", action="store_true", help="Run jobs locally without using docker images.")
    parser.add_argument("--local", action="store_true", help="Run jobs locally.")
    parser.add_argument("--slurm", action="store_true", help="Submit jobs to SLURM cluster.")

    # Resources
    parser.add_argument("-n_jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("-gpus_per_node", type=int, default=1, help="GPUs per SLURM job; one run per GPU at a time.")
    parser.add_argument("-cpus_per_task", type=int, default=32, help="CPUs per GPU slot.")
    parser.add_argument("-n_shards", type=int, default=1, help="Number of disjoint shards (single-node jobs) to split the sweep into.")
    parser.add_argument("-shard_index", type=int, default=None, help="Submit only this shard (0-based). If omitted, all shards are submitted.")
    parser.add_argument("-fit_time_limit", dest="fit_time", type=int, default=3600, help="Time limit (in seconds) for model fitting.")
    parser.add_argument("-job_time_limit", dest="job_time", type=str, default="8:00", help="Maximum job walltime (HH:MM).")

    # Reproducibility
    parser.add_argument("-seed", dest="seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-n_trials", dest="n_trials", type=int, default=30, help="Number of trials per learner.")
    parser.add_argument("-starting_seed", dest="start_seed", type=int, default=0, help="Starting seed index.")

    # SLURM-specific arguments. Ignored when running locally
    parser.add_argument("--noskips", action="store_true", help="Overwrite existing results.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with minimal configuration.")
    parser.add_argument("-max_samples", type=int, default=40000, help="Limit number of training samples.")
    parser.add_argument("-m", type=int, default=10000, help="Memory (MB) per run; each job requests m * gpus_per_node.")
    parser.add_argument("-q", type=str, default='', help="SLURM partition to submit each learner's sweep job to.")
    parser.add_argument("-x", "--exclude", dest="exclude_nodes", type=str, default='', help="Comma-separated list of SLURM nodes to exclude.")
    parser.add_argument("-dependency", dest="dependency", type=str, default=None, help="Comma-separated sbatch --dependency value per shard index.")

    return parser.parse_args()


# Feynman datasets without results for the baseline methods.
RASR_ONLY_FEYNMAN_DATASETS = {
    'feynman_I_26_2',
    'feynman_I_30_5',
    'feynman_test_10',
}


def detect_datasets(dataset_dir):
    """Return sorted list of dataset file paths based on dataset size."""

    if dataset_dir.endswith(".tsv.gz"):
        datasets = [dataset_dir]
    else:
        datasets = glob(os.path.join(dataset_dir, "*/*.tsv.gz"))

    kept_datasets = []
    dataset_sizes = []
    for dataset in datasets:
        dataset_name = os.path.basename(dataset)[:-len(".tsv.gz")]
        if dataset_name in RASR_ONLY_FEYNMAN_DATASETS:
            print(f"Skipping RASR-only Feynman dataset (no baseline results): {dataset}")
            continue

        # grab regression datasets
        metadata = load(
            open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
                Loader=Loader)
        if metadata['task'] != 'regression':
            print(f"Skipping non-regression dataset: {dataset}")
            continue

        kept_datasets.append(dataset)
        dataset_sizes.append(metadata["n_instances"] * metadata["n_features"])

    # Sort datasets by datapoints (so faster jobs get submitted first)
    return [kept_datasets[i] for i in np.argsort(dataset_sizes)]


def detect_learners(args):
    """Determine list of learners to use."""

    if args.learners is None:
        return [ml.split("/")[-2] for ml in glob("experiment/methods/*/regressor.py", recursive=True)]
    
    return [
        ("tuned" if (args.tuned or args.script == "optimize_model") else "") + ml
        for ml in args.learners.split(",")
    ]


def get_run_and_label_name(ml, rasr_name_suffix):
    """Return (run_name, label_name) for a learner."""
    run_name = ml.replace("tuned", "")
    label_name = ml
    if rasr_name_suffix:
        label_name = f"{ml}{rasr_name_suffix}"
    return run_name, label_name


def _rasr_env_vars(args) -> dict[str, str]:
    """RASR (and shared RASR/Flash-ANSR) hyperparameter overrides passed through as env vars."""
    env = {}
    if args.rasr_candidates is not None:
        env["RASR_NUM_CANDIDATES"] = str(args.rasr_candidates)
    if args.rasr_temperature is not None:
        env["RASR_TEMPERATURE"] = str(args.rasr_temperature)
    if args.rasr_n_restarts is not None:
        env["RASR_N_RESTARTS"] = str(args.rasr_n_restarts)
    if args.rasr_init_noise_scale is not None:
        env["RASR_INIT_NOISE_SCALE"] = str(args.rasr_init_noise_scale)
    if args.rasr_no_refine:
        env["RASR_REFINE_CONSTANTS"] = "0"
    if args.refiner_workers is not None:
        env["REFINER_WORKERS"] = str(args.refiner_workers)
    return env


def run_no_docker(commands, n_jobs):
    """Run all commands locally in parallel. Several processess will be spawned!"""

    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_local(commands, job_info, args):
    """Run experiment commands locally using Docker containers in parallel."""

    print("Running locally with Docker...")

    def _run(cmd, run_ml):
        docker_cmd = [
            "docker", "compose", "run", "--rm",
            "-v", f"{os.getcwd()}/experiment:/srbench",
            "-v", f"{os.getcwd()}/{args.dataset_dir}:/{args.dataset_dir}",
            "-v", f"{os.getcwd()}/{args.results_dir}:/{args.results_dir}",
            "-v", f"{args.pretrained_dir}:/srbench_pretrained",
        ] + [
            arg for k, v in _rasr_env_vars(args).items() for arg in ("-e", f"{k}={v}")
        ] + [
            f"{run_ml.lower()}",
            "python", "-u"
        ] + shlex.split(f"/srbench/{cmd}")

        print(" ".join(docker_cmd))
        subprocess.run(docker_cmd)

    Parallel(n_jobs=args.n_jobs)(delayed(_run)(cmd, info['run_ml'])
                                 for cmd, info in zip(commands, job_info))


def run_slurm(commands, job_info, args):
    """
    Submit a single learner's sweep to SLURM as ``-n_shards`` single-node jobs.
    Runs are split into shards by a stable hash of (dataset, seed), and GNU
    parallel runs one run per GPU inside each job.
    """

    label_mls = {info['label_ml'] for info in job_info}
    if len(label_mls) > 1:
        raise ValueError(
            f"run_slurm packs a whole sweep into one job and only supports a single "
            f"learner at a time, got: {sorted(label_mls)}. Run analyze.py separately "
            f"per learner."
        )

    n_shards = max(1, args.n_shards)
    if args.shard_index is not None:
        if not (0 <= args.shard_index < n_shards):
            raise ValueError(
                f"-shard_index must be in [0, {n_shards}) for -n_shards {n_shards}, "
                f"got {args.shard_index}."
            )
        shards_to_run = [args.shard_index]
    else:
        shards_to_run = list(range(n_shards))
        if n_shards > 1:
            print(f"Submitting all {n_shards} shards of {len(commands)} pending runs.")

    dependencies = args.dependency.split(',') if args.dependency else []

    for shard in shards_to_run:
        idxs = [i for i, info in enumerate(job_info) if _shard_of(info, n_shards) == shard]
        if not idxs:
            print(f"Shard {shard} of {n_shards}: no pending runs, nothing to submit.")
            continue
        dependency = dependencies[shard] if shard < len(dependencies) else ''
        _submit_sweep_job([commands[i] for i in idxs],
                          [job_info[i] for i in idxs],
                          args, shard, n_shards, dependency=dependency)


def _shard_of(info, n_shards):
    """Stable shard assignment for a run, independent of the pending set/order."""
    key = f"{info['dataset']}_{info['label_ml']}_{info['seed']}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n_shards


def _submit_sweep_job(commands, job_info, args, shard, n_shards, dependency=''):
    """Submit one single-node SLURM job for the given (sharded) run list."""
    label_ml = job_info[0]['label_ml']
    shard_tag = f"_shard{shard}of{n_shards}" if n_shards > 1 else ""

    partition = args.q.strip()
    n_gpus = max(1, args.gpus_per_node)

    job_mem = args.m * n_gpus

    def noise_suffix():
        s = ''
        if args.y_noise > 0: s += '_target-noise' + str(args.y_noise)
        if args.x_noise > 0: s += '_feature-noise' + str(args.x_noise)
        return s

    # Unique commands-file name per sweep, so queued jobs don't read overwritten files
    keys = sorted(f"{info['dataset']}_{info['label_ml']}_{info['seed']}" for info in job_info)
    h = hashlib.md5(('|'.join(keys) + noise_suffix()).encode()).hexdigest()[:8]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_label = label_ml[len("tuned"):] if label_ml.startswith("tuned") else label_ml
    job_name = f"eval_{job_label}{shard_tag}_{timestamp}"

    job_dir = os.path.join(args.results_dir, "slurm_jobs")
    os.makedirs(job_dir, exist_ok=True)

    commands_file = os.path.join(job_dir, f"{label_ml}{shard_tag}_{h}.commands.txt")

    lines = []
    for cmd, info in zip(commands, job_info):
        run_name = f"{info['dataset']}_{info['label_ml']}_{info['seed']}_{args.script}_{partition}{noise_suffix()}"
        out_file = f"{info['results_path']}/{run_name}.out"
        err_file = out_file.replace('.out', '.err')

        sif = args.sif_override if args.sif_override else f"{args.singularity_dir}/{info['run_ml']}.sif"
        singularity_cmd = (
            "singularity run --no-home --contain --nv "
            f"--bind $(pwd)/experiment:/srbench,$(pwd)/{args.dataset_dir}:/{args.dataset_dir} "
            f"--bind {args.pretrained_dir}:/srbench_pretrained/ "
            f"--bind $(pwd)/{args.results_dir}:/{args.results_dir} --writable-tmpfs "
            f"{sif} python /srbench/{cmd} "
            f"> {out_file} 2> {err_file}"
        )
        line = f"echo \"[$(date +%T)] START {run_name}\"; {singularity_cmd}; echo \"[$(date +%T)] DONE {run_name} (exit $?)\""
        lines.append(line)

    with open(commands_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    slot_script = os.path.abspath(os.path.join(job_dir, "run_slot.sh"))
    with open(slot_script, 'w') as f:
        f.write(RUN_SLOT_SH)
    os.chmod(slot_script, 0o755)

    # {%} is GNU parallel's slot number, {} the run's command line
    parallel_cmd = (
        f"parallel --will-cite -j {n_gpus} --line-buffer "
        f"'bash {slot_script} {{%}} {{}}' "
        f":::: {commands_file}"
    )

    job_out_file = os.path.join(job_dir, f"{job_name}.%J.job.out")
    job_err_file = job_out_file.replace('.job.out', '.job.err')

    batch_script_lines = [
        "#!/usr/bin/bash",
        f"#SBATCH -o {job_out_file}",
        f"#SBATCH --error={job_err_file}",
        "#SBATCH -N 1",
        f"#SBATCH --ntasks={n_gpus}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --time={args.job_time}:00",
        f"#SBATCH --mem={job_mem}",
        f"#SBATCH --gres=gpu:{n_gpus}",
    ]
    if args.exclude_nodes:
        batch_script_lines.append(f"#SBATCH --exclude={args.exclude_nodes}")
    if dependency:
        batch_script_lines.append(f"#SBATCH --dependency={dependency}")
    batch_script_lines += [
        "hostname",
        "nvidia-smi || echo 'nvidia-smi failed'",
        "",
        "TZ='America/New_York'",
        "export TZ",
        "",
        "timedatectl",
        "",
        f"export SR_CPT={args.cpus_per_task}",
        "",
    ] + [
        line
        for k, v in _rasr_env_vars(args).items()
        for line in (f"export SINGULARITYENV_{k}={v}", "")
    ] + [
        parallel_cmd,
    ]
    batch_script = "\n".join(batch_script_lines)

    print(f"Submitting sweep: {job_name} ({len(commands)} runs, {n_gpus} GPUs on 1 node)")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(batch_script)
            tmp_script = f.name
        sbatch_response = subprocess.check_output(["sbatch", tmp_script]).decode()
        print(f"Submitted: {sbatch_response.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {job_name}: {e}")
    finally:
        os.unlink(tmp_script)


def main():
    args = parse_args()
     
    print('script:', args.script)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Results directory: {args.results_dir}")

    learners = detect_learners(args)
    datasets = detect_datasets(args.dataset_dir)

    print(f"Found {len(datasets)} datasets and {len(learners)} learners.")

    if args.y_noise > 0: print('using target-noise', str(args.y_noise))
    if args.x_noise > 0: print('using feature-noise', str(args.x_noise))

    all_commands, job_info, jobs_w_results = [], [], []

    for t in range(args.start_seed, args.start_seed + args.n_trials):
        random_state = args.seed if args.seed and args.n_trials == 1 else SEEDS[t]

        for dataset in datasets:
            dataname = os.path.basename(dataset).replace(".tsv.gz", "")
            results_path = os.path.join(args.results_dir, dataname)
            os.makedirs(results_path, exist_ok=True)

            if (not args.sym_data and 'rethinking_feynman' in dataname):
                continue

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            for ml in learners:
                run_ml, label_ml = get_run_and_label_name(ml, args.rasr_name_suffix)
                # optimize_model and evaluate_model(--tuned) prepend "tuned" internally.
                label_for_script = label_ml
                if (args.script == "optimize_model" or args.tuned) and label_for_script.startswith("tuned"):
                    label_for_script = label_for_script[len("tuned"):]
                save_file = f"{results_path}/{dataname}_{label_ml}_{random_state}"

                if not args.noskips:
                    suffix = ''
                    if args.y_noise>0:
                        suffix += '_target-noise'+str(args.y_noise)
                    if args.x_noise>0:
                        suffix += '_feature-noise'+str(args.x_noise)
                    if 'optimize' in args.script:
                        suffix += '_cv_results'
                    suffix += '.json'
                        
                    if os.path.exists(save_file+suffix):
                        jobs_w_results.append([save_file, 'exists'])
                        continue

                # cant have 'tuned' prefix here because it will import the exact name
                if args.no_docker:
                    dataset_path = os.path.relpath(dataset, os.getcwd())
                    results_path_rel = os.path.relpath(results_path, os.getcwd())
                    env_prefix = "".join(f"{k}={v} " for k, v in _rasr_env_vars(args).items())
                    cmd = (
                        f"{env_prefix}python {args.script}.py {dataset_path} -ml {run_ml} "
                        f"-algorithm_label {label_for_script} "
                        f"-results_path {results_path_rel} -seed {random_state} "
                        f"-target_noise {args.y_noise} -feature_noise {args.x_noise} "
                        f"-fit_time_limit {args.fit_time} -max_samples {args.max_samples} "
                        f"{'--scale_x' if args.scale_x else ''} "
                        f"{'--scale_y' if args.scale_y else ''} "
                        f"{'--sym_data' if args.sym_data else ''} "
                        f"{'--skip_tuning' if args.skip_tuning else ''} "
                        f"{'--tuned' if args.tuned else ''} "
                        f"{'--ecotracker' if args.ecotracker else ''} "
                        f"{'--save_population' if args.save_population else ''} "
                        f"{'--test' if args.test else ''} "
                    )
                else:
                    cmd = (
                        f"{args.script}.py /{dataset} -ml {run_ml} "
                        f"-algorithm_label {label_for_script} "
                        f"-results_path /{results_path} -seed {random_state} "
                        f"-target_noise {args.y_noise} -feature_noise {args.x_noise} "
                        f"-fit_time_limit {args.fit_time} -max_samples {args.max_samples} "
                        f"{'--scale_x' if args.scale_x else ''} "
                        f"{'--scale_y' if args.scale_y else ''} "
                        f"{'--sym_data' if args.sym_data else ''} "
                        f"{'--skip_tuning' if args.skip_tuning else ''} "
                        f"{'--tuned' if args.tuned else ''}"
                        f"{'--ecotracker' if args.ecotracker else ''} "
                        f"{'--save_population' if args.save_population else ''} "
                        f"{'--test' if args.test else ''} "
                    )
                all_commands.append(cmd)
                job_info.append({
                    "run_ml": run_ml,
                    "label_ml": label_ml,
                    "dataset": dataname,
                    "seed": random_state,
                    "results_path": results_path,
                })

    print(f"Prepared {len(all_commands)} jobs...")

    if not args.noskips:
        print('skipped', len(jobs_w_results),'jobs with results. Override with --noskips.')
    
    if args.no_docker:
        run_no_docker(all_commands, args.n_jobs)
    elif args.local:
        run_local(all_commands, job_info, args)
    elif args.slurm:
        run_slurm(all_commands, job_info, args)
    else:
        print("Specify either --no_docker, --local or --slurm to execute jobs.")

    print("Finished submitting jobs.")


if __name__ == '__main__':
    main()