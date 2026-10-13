# User Guide

This documentation is intented to work with the `srbench_2025` branch. Make sure to clone and checkout to this specific branch if you want to follow this user guide.

**NOTE**: if you are trying to reproduce the results from the 2022 Neurips Benchmark paper, you 
should check out the [v2.0 release](https://github.com/cavalab/srbench/releases/tag/v2.0) version of this repo.  

## Installation

### Local install

We have provided a [conda environment](../base_environment.yml) that should make installation easier.
We've currently tested this on Ubuntu and CentOS. 
Steps:

0. (Optional) Using libmamba as the solver to speedup the installation process:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

1. Install the base conda environment. This will set up a python installation with some basic things needed to submit jobs and analyze the results:

```bash
conda env create -f base_environment.yml
conda activate srbench
```

2. Pull the benchmark algorithm images with

```bash
docker build --pull --rm -f "Dockerfile" -t srbench:latest "."
```

3. Download the PMLB datasets:

```bash
cd datasets
python download_data.py
``` 

## Reproducing the benchmark results

**NOTE**: these instructions are for the the [v2.0 release](https://github.com/cavalab/srbench/releases/tag/v2.0) version of this repo.  

Experiments are launched from the `experiments/` folder via the script `analyze.py`.
The script can be configured to run the experiment in parallel locally, or on a SLURM job scheduler. 
To see the full set of options, run `python analyze.py --help`, or check out the [source code](../experiment/analyze.py). 

**WARNING**: running some of the commands below will submit tens of thousands of experiments. 
Use accordingly. 

### Black-box experiment

After installing and configuring the conda environment, the complete black-box experiment was executed in a SLURM cluster via the command below executed from the root of the repository:

```bash
################################################################################
# 1. Black-box experiments - with gridsearch for each dataset-run
################################################################################

# Define algorithm groups
cpu_ml="afp,afp_fe,afp_ehc,bingo,brush,bsr,eplex,eql,feat,ffx,geneticengine,gpgomea,gplearn,gpzgd,itea,operon,ps-tree,pysr,qlattice,rils-rols,tir"
gpu_ml="e2et,flashansr,rasr,nesymres,snip,tpsr,udsr"  # Pretrained models or checkpoints for NN-based methods

# Common experiment parameters
COMMON_ARGS="datasets/blackbox/ \
    -script optimize_model \
    -results results_blackbox_tuning/ \
    -images /path-to-your-images/ \
    -pretrained_dir /path-to-pretrained-model-checkpoints/ \
    -n_trials 30 -job_time_limit 8:00 -fit_time_limit 3600 \
    -m 10000 -max_samples 40000 \
    --scale_x --scale_y --slurm --ecotracker"

# Submit jobs for CPU-based algorithms
python experiment/analyze.py $COMMON_ARGS -ml $cpu_ml

# Submit jobs for GPU-based algorithms
python experiment/analyze.py $COMMON_ARGS -ml $gpu_ml
```

After running the experiments, you can glue them with:

```bash
# Glue black-box results with 
python postprocessing/scripts/collate_experiments_results.py './results_blackbox_tuning/' './results/black-box-tuning/'

# Glue eco2ai with
python postprocessing/scripts/collate_blackbox_eco2ai_stats.py './results_blackbox_tuning/' './results/black-box-tuning/'
```

Alternativelly, you can run jobs locally, using Docker images instead of Singularity images. For example:

```bash
python experiment/analyze.py datasets/blackbox/     -script optimize_model     -results results_blackbox_tuning/     -images /path-to-your-images/     -pretrained_dir /path-to-pretrained-model-checkpoints/     -ml feat     -n_trials 1     -job_time_limit 8:00     -fit_time_limit 3600     -m 10000     -max_samples 40000     --scale_x --scale_y --local --ecotracker
```

The experiments are designed to run either with Docker locally or Singularity on a cluster. You may be able to bypass it by creating a conda environment with your code installed and passing `--no_docker` as argument. Make sure your custom environment is compatible with all dependencies. 

### First-principles experiment

```bash
################################################################################
# 3. first principles experiments
################################################################################
# Same procedure as ground-truth experiments, but with no noise addition.

python experiment/analyze.py datasets/firstprinciples \
    -script optimize_model \
    -results results_first_principles_tuning/ \
    -images /path-to-your-images/ \
    -pretrained_dir /path-to-pretrained-model-checkpoints/ \
    -n_trials 30 -job_time_limit 8:00 -fit_time_limit 3600 \
    -m 10000 -max_samples 40000 \
    --scale_x --scale_y --slurm \
    -ml $cpu_ml

python experiment/analyze.py datasets/firstprinciples \
    -script optimize_model \
    -results results_first_principles_tuning/ \
    -images /path-to-your-images/ \
    -pretrained_dir /path-to-pretrained-model-checkpoints/ \
    -n_trials 30 -job_time_limit 8:00 -fit_time_limit 3600 \
    -m 10000 -max_samples 40000 \
    --scale_x --scale_y --slurm \
    -ml $gpu_ml
```

Glue them with 

```bash
python postprocessing/scripts/collate_experiments_results.py './results_first_principles_tuning/' './results/first-principles-tuning/'
```

### Building docker locally

When a new algorithm is submitted to SRBench, a GitHub workflow will generate a docker image and push it to [Docker Hub](hub.docker.com). Ths means that you can also easily pull the images, without having to deal with local installations.

To build the docker images locally, first run `bash scripts/make_docker_compose_file.sh` in the root directory to create a `docker-compose.yml` file describing all images that will be created.
Then `docker compose up` should create all the images.
Instead of creating all images, you can build the image of a specific algorithm with the name of the service (_e.g._ `docker compose build feat`). 

The file `alg-Dockerfile` specifies the steps used to install the algorithm - you can check it out to see how it will be installed. 
The build relies on `install_algorithm.sh`, so do not delete this file.

You can now submit arbitrary python commands to the image, _e.g._ `docker compose run feat bash test.sh`. This command will run the test script to check if the algorithm is compatible.

Or you can enter bash mode using an image with `docker compose run feat bash`.

### Pushing to dockerhub on different account

If you make changes to the images and want to upload, first you create a docker account.

Then, change the image name `srbench` in `scripts/make_docker_compose_file.sh` to your docker hub ID before running it.

Now you can run `bash scripts/make_docker_compose_file.sh`, then `docker compose build`. 

On the dockerhub website, create a repository with the name of the service you want to push (that is, the name of the algorithm, _e.g._ `feat`) Login in the with `docker login`, and run `docker push docker_hub_id/alg_name:latest`, where `latest` is the tag.

### Using singularity

To use singularity (_e.g._, you need to use your images on a cluster without docker support) you may need to use singularity, which requires you to push your to dockerhub first.

First, create a folder for the `.sif` files.

Pull the image with `singularity pull folder/name.sif docker://user_id/image`. It will convert the docker container into something singularity can use.

**OBS:** the scripts are designed to run on a cluster using singularity.

### Post-processing

Navigate to the `postprocessing` folder to begin postprocessing the experiment results. 
The following two scripts collate the `.json` files into two `.feather` files to share results more easily. 
You will notice these `.feather` files are loaded to generate figures in the notebooks. 
They also perform some cleanup like shortening algorithm names, etc.

```
python collate_blackbox_results.py
python collate_groundtruth_results.py
```

**Visualization**

- [groundtruth_results.ipynb](../postprocessing/groundtruth_results.ipynb): ground-truth results comparisons
- [blackbox_results.ipynb](../postprocessing/blackbox_results.ipynb): ground-truth results comparisons
- [statistical_comparisons.ipynb](../postprocessing/statistical_comparisons.ipynb): post-hoc statistical comparisons
- [pmlb_plots](../postprocessing/pmlb_plots.ipynb): the [PMLB](https://github.com/EpistasisLab/pmlb) datasets visualization 


## Using your own datasets

To use your own datasets, you want to check out / modify read_file in read_file.py: https://github.com/cavalab/srbench/blob/4cc90adc9c450dad3cb3f82c93136bc2cb3b1a0a/experiment/read_file.py

If your datasets follow the convention of https://github.com/EpistasisLab/pmlb/tree/master/datasets, i.e. they are in a pandas DataFrame with the target column labelled "target", you can call `read_file` directly just passing the filename like you would with any of the PMLB datasets. 
The file should be stored and compressed as a `.tsv.gz` file. 
