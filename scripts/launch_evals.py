#!/usr/bin/env python3
"""Launch the paper's srbench evaluations through experiment/analyze.py.

Usage:
    python scripts/launch_evals.py --list                         # see all evals across all tracks
    python scripts/launch_evals.py --group blackbox-main --dry-run # preview commands before submitting
    python scripts/launch_evals.py --group blackbox                # launch all blackbox groups (prefix match)
    python scripts/launch_evals.py --id 'blackbox/rasr_94m_c*'      # re-launch a specific eval (glob-matched)
    python scripts/launch_evals.py --group feynman -q <partition>   # switch partition/cluster
    python scripts/launch_evals.py --group feynman --shard-index 0  # split shards across partitions/clusters

Re-running a command skips runs whose result JSON already exists.
"""

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SRBENCH_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GPUS_PER_NODE = 4
DEFAULT_CPUS_PER_TASK = 1
DEFAULT_N_SHARDS = 4
DEFAULT_JOB_TIME = "24:00"
DEFAULT_SCRIPT = "optimize_model"
DEFAULT_PRETRAINED_ROOT = "pretrained"
DEFAULT_IMAGES = "images/"

# cluster -> partition -> -x exclude list ('' means no exclude for that partition)
CLUSTERS = {
    "default": {
        "<partition>": "",
    },
}


@dataclass(frozen=True)
class Track:
    name: str
    dataset_dir: str
    results_root: str
    sym_data: bool = False
    n_trials: int | None = None


TRACKS = {
    "blackbox": Track("blackbox", "datasets/blackbox/", "results_blackbox"),
    "principles": Track("principles", "datasets/firstprinciples/", "results_principles"),
    "feynman": Track("feynman", "datasets/feynman/", "results_feynman", sym_data=True, n_trials=10),
    "strogatz": Track("strogatz", "datasets/strogatz/", "results_strogatz", sym_data=True, n_trials=10),
}


@dataclass(frozen=True)
class Eval:
    id: str
    group: str
    track: str
    method: str  # rasr | flashansr | snip
    checkpoint: str  # folder under pretrained/
    results_name: str  # subdir under the track's results_root
    suffix: str = ""
    candidates: int | None = None
    temperature: float | None = None
    noise: float = 0.0
    scale: bool = True
    n_restarts: int | None = None
    init_noise_scale: float | None = None
    refine_constants: bool = True


EVALS: list[Eval] = []


def _add(
    evals, track, group, method, checkpoint, results_name, suffix="", candidates=None,
    temperature=None, noise=0.0, scale=True, n_restarts=None, init_noise_scale=None,
    refine_constants=True,
):
    # Noise levels share a results dir, so the id carries the noise level
    id_suffix = f"_noise{noise}" if noise > 0 else ""
    evals.append(Eval(
        id=f"{track}/{results_name}{id_suffix}",
        group=group,
        track=track,
        method=method,
        checkpoint=checkpoint,
        results_name=results_name,
        suffix=suffix,
        candidates=candidates,
        temperature=temperature,
        noise=noise,
        scale=scale,
        n_restarts=n_restarts,
        init_noise_scale=init_noise_scale,
        refine_constants=refine_constants,
    ))


def _blackbox_evals():
    evals = []
    track = "blackbox"

    main_models = [
        ("RASR-94M-L", "rasr_94m", "_94m"),
        ("RASR-94M-M", "rasr_94m_lambda0.15", "_94m_lambda0.15"),
        ("RASR-94M-S", "rasr_94m_lambda0.5", "_94m_lambda0.5"),
    ]
    for checkpoint, results_name, suffix in main_models:
        _add(evals, track, "blackbox-main", "rasr", checkpoint, results_name, suffix=suffix)

    _add(evals, track, "blackbox-baselines", "flashansr", "Flash-ANSR", "flashansr")
    _add(evals, track, "blackbox-baselines", "snip", "SNIP", "snip")

    # Test-time compute (RASR-94M-L with 1024 candidates is in blackbox-main)
    ttc_models = [
        ("RASR-94M-L", "rasr_94m", [256, 64, 16]),
        ("RASR-94M-GRPO", "rasr_94m_grpo", [1024, 256, 64, 16]),
        ("RASR-94M-base", "rasr_94m_base", [1024, 256, 64, 16]),
    ]
    for checkpoint, name_base, candidate_counts in ttc_models:
        name_tag = name_base[len("rasr_"):]  # e.g. "94m", "94m_grpo", "94m_base"
        for c in candidate_counts:
            _add(
                evals, track, "blackbox-ttc", "rasr", checkpoint,
                results_name=f"{name_base}_c{c}",
                suffix=f"_{name_tag}_c{c}",
                candidates=c,
            )
        _add(
            evals, track, "blackbox-ttc", "rasr", checkpoint,
            results_name=f"{name_base}_greedy",
            suffix=f"_{name_tag}_greedy",
            candidates=1,
            temperature=0.0,
        )

    # 24M ablation.
    _add(evals, track, "blackbox-24m", "rasr", "RASR-24M-base", "rasr_24m_base", suffix="_24m_base")
    _add(evals, track, "blackbox-24m", "rasr", "RASR-24M", "rasr_24m", suffix="_24m")

    # Prior ablation, intentionally without --scale_x/--scale_y
    _add(
        evals, track, "blackbox-24m-prior", "rasr", "RASR-24M-simple_prior-base",
        "rasr_24m_simple_prior_base", suffix="_24m_simple_prior_base", scale=False,
    )
    _add(
        evals, track, "blackbox-24m-prior", "rasr", "RASR-24M-simple_prior",
        "rasr_24m_simple_prior", suffix="_24m_simple_prior", scale=False,
    )

    # Constant-refinement ablation
    _add(
        evals, track, "blackbox-24m-constant", "rasr", "RASR-24M",
        "rasr_24m_no_refine", suffix="_24m_no_refine", refine_constants=False,
    )
    _add(
        # n_restarts=1 also disables the init noise
        evals, track, "blackbox-24m-constant", "rasr", "RASR-24M",
        "rasr_24m_no_noise", suffix="_24m_no_noise", n_restarts=1,
    )
    _add(
        evals, track, "blackbox-24m-constant", "rasr", "RASR-24M-skeleton",
        "rasr_24m_skeleton", suffix="_24m_skeleton", init_noise_scale=5.0,
    )

    return evals


def _principles_evals():
    evals = []
    track = "principles"

    main_models = [
        ("RASR-94M-L", "rasr_94m", "_94m"),
        ("RASR-94M-M", "rasr_94m_lambda0.15", "_94m_lambda0.15"),
        ("RASR-94M-S", "rasr_94m_lambda0.5", "_94m_lambda0.5"),
    ]
    for checkpoint, results_name, suffix in main_models:
        _add(evals, track, "principles-main", "rasr", checkpoint, results_name, suffix=suffix)

    _add(evals, track, "principles-baselines", "flashansr", "Flash-ANSR", "flashansr")
    _add(evals, track, "principles-baselines", "snip", "SNIP", "snip")

    return evals


def _symbolic_evals(track):
    """Feynman/Strogatz: RASR-94M-L and baselines, each x 4 noise levels."""
    evals = []
    noise_levels = [0.0, 0.001, 0.01, 0.1]

    for noise in noise_levels:
        _add(
            evals, track, f"{track}-main", "rasr", "RASR-94M-L",
            results_name="rasr_94m", suffix="_94m", noise=noise,
        )
    for method, checkpoint in [("flashansr", "Flash-ANSR"), ("snip", "SNIP")]:
        for noise in noise_levels:
            _add(
                evals, track, f"{track}-baselines", method, checkpoint,
                results_name=method, noise=noise,
            )

    return evals


EVALS.extend(_blackbox_evals())
EVALS.extend(_principles_evals())
EVALS.extend(_symbolic_evals("feynman"))
EVALS.extend(_symbolic_evals("strogatz"))

_ids = [e.id for e in EVALS]
assert len(_ids) == len(set(_ids)), f"duplicate eval ids: {[i for i in _ids if _ids.count(i) > 1]}"


def build_command(ev: Eval, args) -> list[str]:
    track = TRACKS[ev.track]
    results_dir = f"{track.results_root}{args.results_suffix}/{ev.results_name}"

    cmd = [
        "python", "-u", "experiment/analyze.py", track.dataset_dir,
        "-script", DEFAULT_SCRIPT,
        "-ml", ev.method,
        "--slurm",
        "-q", args.partition,
        "-images", args.images,
        "-pretrained_dir", f"{args.pretrained_root}/{ev.checkpoint}",
        "-results", results_dir,
        "-gpus_per_node", str(args.gpus_per_node),
        "-cpus_per_task", str(args.cpus_per_task),
        "-n_shards", str(args.n_shards),
        "-job_time_limit", args.job_time,
        "-sif", f"{args.images.rstrip('/')}/{ev.method}.sif",
        "-max_samples", "0",  # Disable analyze.py's default 40000-row training-set cap.
    ]

    if ev.suffix:
        cmd += ["-rasr_name_suffix", ev.suffix]

    if args.refine_workers is not None:
        cmd += ["-refiner_workers", str(args.refine_workers)]

    exclude = args.exclude if args.exclude is not None else CLUSTERS.get(args.cluster, {}).get(args.partition, "")
    if exclude:
        cmd += ["-x", exclude]

    if ev.method == "rasr":
        if ev.scale:
            cmd += ["--scale_x", "--scale_y"]
        if ev.candidates is not None:
            cmd += ["-rasr_candidates", str(ev.candidates)]
        if ev.temperature is not None:
            cmd += ["-rasr_temperature", str(ev.temperature)]
        if ev.n_restarts is not None:
            cmd += ["-rasr_n_restarts", str(ev.n_restarts)]
        if ev.init_noise_scale is not None:
            cmd += ["-rasr_init_noise_scale", str(ev.init_noise_scale)]
        if not ev.refine_constants:
            cmd += ["--rasr_no_refine"]

    if track.sym_data:
        cmd += ["--sym_data"]
    if track.n_trials is not None:
        cmd += ["-n_trials", str(track.n_trials)]
    if ev.noise > 0:
        cmd += ["-target_noise", str(ev.noise)]

    if args.shard_index is not None:
        cmd += ["-shard_index", str(args.shard_index)]
    if args.noskips:
        cmd += ["--noskips"]
    if args.extra:
        cmd += args.extra

    return cmd


def select_evals(args) -> list[Eval]:
    if not args.group and not args.id and not args.all:
        return []

    selected = []
    for ev in EVALS:
        if args.all:
            selected.append(ev)
            continue
        if args.group and any(ev.group.startswith(g) for g in args.group):
            selected.append(ev)
            continue
        if args.id and any(fnmatch.fnmatch(ev.id, pat) for pat in args.id):
            selected.append(ev)
            continue
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List all evals (id, group, method, checkpoint, results dir) and exit.")
    parser.add_argument("--group", nargs="+", default=[], help="Select evals by group name (prefix match, e.g. 'blackbox' matches all blackbox-* groups).")
    parser.add_argument("--id", nargs="+", default=[], help="Select evals by id, fnmatch glob patterns allowed.")
    parser.add_argument("--all", action="store_true", help="Select every eval in the matrix. Required if neither --group nor --id is given and you really want everything.")
    parser.add_argument("--dry-run", action="store_true", help="Print the analyze.py commands without submitting.")
    parser.add_argument("-q", "--partition", default="<partition>", help="SLURM partition.")
    parser.add_argument("--cluster", default="default", choices=list(CLUSTERS), help="Cluster, used to look up the default node exclude list per partition.")
    parser.add_argument("--exclude", default=None, help="Override the partition's default -x exclude list.")
    parser.add_argument("--shard-index", type=int, default=None, dest="shard_index", help="Submit only this shard (see -shard_index in analyze.py).")
    parser.add_argument("--n-shards", type=int, default=None, dest="n_shards", help=f"Default: {DEFAULT_N_SHARDS}.")
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE, dest="gpus_per_node")
    parser.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK, dest="cpus_per_task")
    parser.add_argument(
        "--refine-workers", type=int, default=None, dest="refine_workers",
        help=f"Use this many CPUs per task and this many RASR/Flash-ANSR constant-refinement "
             f"worker processes, overriding --cpus-per-task (default: {DEFAULT_CPUS_PER_TASK} "
             f"CPUs, 0 refinement workers).",
    )
    parser.add_argument("--job-time", default=DEFAULT_JOB_TIME, dest="job_time")
    parser.add_argument("--pretrained-root", default=DEFAULT_PRETRAINED_ROOT, dest="pretrained_root")
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--results-suffix", default="", dest="results_suffix", help="Suffix appended to each track's results root dir name, e.g. '_rerun'.")
    parser.add_argument("--noskips", action="store_true", help="Pass --noskips through to analyze.py, forcing reruns of existing results.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra raw arguments appended to every analyze.py command.")
    args = parser.parse_args()

    if args.refine_workers is not None:
        args.cpus_per_task = args.refine_workers

    if args.n_shards is None:
        args.n_shards = DEFAULT_N_SHARDS

    if args.list:
        width = max(len(e.id) for e in EVALS)
        for ev in EVALS:
            results_dir = f"{TRACKS[ev.track].results_root}/{ev.results_name}"
            print(f"{ev.id:<{width}}  {ev.group:<22}  {ev.method:<10}  {ev.checkpoint:<24}  -> {results_dir}")
        return

    selected = select_evals(args)
    if not selected:
        print("No evals selected. Use --group NAME, --id PATTERN, or --all. Run --list to see all eval ids/groups.", file=sys.stderr)
        sys.exit(1)

    print(f"Selected {len(selected)} eval(s):")
    for ev in selected:
        print(f"  {ev.id}")
    print()

    failed = []
    for ev in selected:
        cmd = build_command(ev, args)
        print(f"[{ev.id}] {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=SRBENCH_ROOT)
        if result.returncode != 0:
            failed.append(ev.id)

    if not args.dry_run:
        print(f"\n{len(selected) - len(failed)} submitted / {len(failed)} failed.")
        if failed:
            print("Failed ids (retry with --id):")
            for fid in failed:
                print(f"  {fid}")
            sys.exit(1)


if __name__ == "__main__":
    main()

