# RASR evaluation (fork of SRBench 2.0)

This is a fork of [SRBench 2.0](https://github.com/cavalab/srbench) with wrappers for RASR, Flash-ANSR
and SNIP and a Slurm launcher for the runs in the paper. It is used as the `srbench/` submodule of the
[RASR repository](https://github.com/automl/rasr), referred to as the root repo below.
The rest of this README is the upstream one.
All commands below run from this `srbench/` directory unless noted.

## Setup

Everything that runs outside the containers (launching, merging, figures) uses the root repo's
environment with the `notebook` and `eval` extras. From the repo root:
`uv sync --group cpu --extra notebook --extra eval`, or with pip (after installing torch)
`pip install -e ./torchtitan -e ".[notebook,eval]"`.
The cluster also needs GNU `parallel` and Apptainer/Singularity.

1. Download the datasets: `cd datasets && python download_data.py`.
2. Build the RASR Apptainer image into `images/`. `alg.def` resolves its paths from the parent repo
   root (the image copies the parent's `src/`), hence the `cd ..`:
   ```bash
   (cd .. && apptainer build --fakeroot --build-arg ALGORITHM=rasr srbench/images/rasr.sif srbench/alg.def)
   ```
   On a Slurm cluster, `sbatch scripts/build_sif.sh rasr` does the same (set the partition first).
3. Download the checkpoints `RASR-94M-L`, `RASR-94M-M` and `RASR-94M-S` (see the
   main `README.md` for the download location), extract the `.tar.gz` archives and put the resulting folders under
   `pretrained/`.

## Running

`scripts/launch_evals.py` defines every eval in the paper and submits it through
`experiment/analyze.py`:

```bash
python scripts/launch_evals.py --list                            # all evals and groups
python scripts/launch_evals.py --group blackbox-main --dry-run   # print the commands
python scripts/launch_evals.py --group blackbox-main -q <partition>
```

The `*-main` groups (tracks `blackbox`, `principles`, `feynman`, `strogatz`) run with the released
checkpoints. The `*-baselines` groups additionally need `flashansr`/`snip` images (built as above)
and the Flash-ANSR and SNIP checkpoints from their authors under `pretrained/Flash-ANSR/` and
`pretrained/SNIP/`; the other groups use ablation checkpoints that are not released. Each eval is
split into `-n_shards` single-node Slurm jobs, and rerunning a command skips runs whose result JSON
already exists. Partition names and resources are set with `-q`, `--gpus-per-node` and
`--cpus-per-task`.

## Merging results

The per-run JSON files are merged into a `results.feather` with:

```bash
# blackbox (principles: results_principles/... into results/first-principles-tuning)
python postprocessing/scripts/merge_results.py results_blackbox/rasr_94m results/black-box
# feynman / strogatz (noise level is part of the run key)
python postprocessing/scripts/merge_groundtruth_results.py results_feynman/rasr_94m results/ground-truth
```

To compare against the other SRBench methods, first place the upstream SRBench 2.0 `results/`
feather files there; otherwise a new file is created.

## Figures

The paper's figures are made by these notebooks in `postprocessing/notebooks/`, which read the
feather files above and write to `postprocessing/figs/`:

- `blackbox_pointplots.ipynb`: black-box test $R^2$ rank, expression size and training time
- `blackbox_pareto.ipynb`: black-box accuracy vs. expression size Pareto plots
- `firstprinciples_results.ipynb`: Pareto fronts per first-principles dataset
- `groundtruth_results.ipynb`: Feynman and Strogatz results across noise levels
- `blackbox_improvability_tables.ipynb`: number of sampled expressions and ablations

---


# SRBench: A Living Benchmark for Symbolic Regression

-----

The methods for symbolic regression (SR) have come a long way since the days of Koza-style genetic programming (GP).

Our goal with this project is to keep a living benchmark of modern symbolic regression, in the context of state-of-the-art ML methods.

Currently these are the challenges, as we see it:
- Aggregated results obscure the current state-of-the-art.
- A one-size-fits-all approach is not ideal for benchmarking SR methods.
- The benchmark needs continuous updates with new datasets and algorithms.

We introduce new visualizations to highlight where each algorithm excels or struggles.
We nearly double the number of SR methods evaluated under a unified experimental setup.
We propose a deprecation scheme to guide the selection of methods for future editions.
We propose a call for action for the community to think about what it takes for providing a good SR benchmark.

As before, we are making these comparisons open source, reproduceable and public, and hoping to share them widely with the entire ML research community.

> When SRBench started, its challanges were:
> - Lack of cross-pollination between the GP community and the ML community (different conferences, journals, societies etc)
> - Lack of strong benchmarks in SR literature (small problems, toy datasets, weak comparator methods)
> - Lack of a unified framework for SR, or GP

# Benchmarked Methods

This benchmark currently consists of **26** symbolic regression methods, including the original **14** methods from previous SRBench, plus all staged methods for benchmarking, plus recent methods that have been published and included the SRBench results in their paper.
We are using **24** datasets from [PMLB](https://github.com/EpistasisLab/penn-ml-benchmarks), including real-world and synthetic datasets from processes with and without ground-truth models.
We perform **30** independent runs for robust results comparisons.

There are 26 methods currently benchmarked:

| Method |  |  |
|:--|:--|:--|
| **AFP** - [paper](http://dx.doi.org/10.1007/978-1-4419-7747-2_8) | **AFP_fe** | **AFP_ehc** - [paper](https://www.sciencedirect.com/science/article/abs/pii/S0952197616301294) |
| **Bingo** - [paper](https://dl.acm.org/doi/10.1145/3520304.3534031) | **Brush** - [code](https://github.com/cavalab/brush/tree/multi_armed_bandits) | **BSR** - [paper](https://arxiv.org/abs/1910.08892) |
| **E2E** - [paper](https://papers.neurips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf) | **EPLEX** - [paper](https://direct.mit.edu/evco/article-pdf/27/3/377/1858632/evco_a_00224.pdf) | **EQL** - [paper](http://proceedings.mlr.press/v80/sahoo18a/sahoo18a.pdf) |
| **FEAT** - [paper](https://openreview.net/pdf?id=Hke-JhA9Y7) | **FFX** - [paper](https://link.springer.com/chapter/10.1007/978-1-4614-1770-5_13) | **Genetic Engine** - [paper](https://dl.acm.org/doi/10.1145/3564719.3568697) |
| **GPGomea** - [paper](http://dx.doi.org/10.1162/evco_a_00278) | **GPlearn** - [paper]() | **GPZGD** - [paper](https://doi.org/10.1145/3377930.3390237) |
| **ITEA** - [paper](https://direct.mit.edu/evco/article-pdf/29/3/367/1959462/evco_a_00285.pdf) | **NeSymRes** - [paper](http://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf) | **Operon** - [paper](https://link.springer.com/article/10.1007/s10710-019-09371-3) |
| **Ps-Tree** - [paper](https://www.sciencedirect.com/science/article/pii/S2210650222000335) | **PySR** - [paper](https://arxiv.org/abs/2305.01582) | **Qlattice** - [paper](https://arxiv.org/abs/2104.05417) |
| **Rils-rols** - [paper](http://dx.doi.org/10.1186/s40537-023-00743-2) | **TIR** - [paper](https://doi.org/10.1145/3597312) | **TPSR** - [paper](https://openreview.net/forum?id=0rVXQEeFEL) |
| **uDSR** - [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/dbca58f35bddc6e4003b2dd80e42f838-Paper-Conference.pdf) |  |  |

## Benchmark results

We made available all of our experiments' results as `feather` files inside `/results/`.

## Contributing and running it locally

Check out [`CONTRIBUTING.md`](./CONTRIBUTING.md) file to see how to set up your algorithm. This guide will detail all the requirements in order to submit a pull request with a compatible interface with SRBench.

The `analyze` file is the main entry point as it will parse the flags and create specific python commands to run each experiment independently. Some examples on how to invoke the experiments are available at the [`docs/user_guide.md`](./docs/user_guide.md).

## Reproducing the experiments

A detailed guide on how to reproduce the experiments by yourself is provided in [`docs/user_guide.md`](./docs/user_guide.md).

Once you get all the results, you nee to collate the results using the `collate` scripts in [`./postprocessing/scripts`](./postprocessing/scripts/collate_experiments_results.py)

# References

> **A paper containing the results from this repository is under review at GECCO 2025 Symbolic Regression Workshop.**

A _Call for action_ was reported in the GECCO 2025 paper:

> Imai Aldeia, G. S., Zhang, H., Bomarito, G., Cranmer, M., Fonseca, A., Burlacu, B., La Cava, W., and de França, F. 2025. Call for Action: towards the next generation of symbolic regression benchmark. _Proceedings of the Genetic and Evolutionary Computation Conference Companion_
[doi](https://dl.acm.org/doi/10.1145/3712255.3734309),
[preprint](https://arxiv.org/abs/2505.03977)

[SRBench](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v2.0) was reported in the Neurips 2021 paper: 

> La Cava, W., Orzechowski, P., Burlacu, B., de França, F. O., Virgolin, M., Jin, Y., Kommenda, M., & Moore, J. H. (2021). 
Contemporary Symbolic Regression Methods and their Relative Performance. 
_Neurips Track on Datasets and Benchmarks._
[arXiv](https://arxiv.org/abs/2107.14351),
[neurips.cc](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c0c7c76d30bd3dcaefc96f40275bdc0a-Abstract-round1.html)

[v1.0](https://github.com/EpistasisLab/regression-benchmark/releases/tag/v1.0) was reported in the GECCO 2018 paper: 

> Orzechowski, P., La Cava, W., & Moore, J. H. (2018). 
Where are we now? A large benchmark study of recent symbolic regression methods. 
GECCO 2018. [DOI](https://doi.org/10.1145/3205455.3205539), [Preprint](https://www.researchgate.net/profile/Patryk_Orzechowski/publication/324769381_Where_are_we_now_A_large_benchmark_study_of_recent_symbolic_regression_methods/links/5ae779b70f7e9b837d392dc9/Where-are-we-now-A-large-benchmark-study-of-recent-symbolic-regression-methods.pdf)