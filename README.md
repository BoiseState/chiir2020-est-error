# piret-recsys-eval-errors

Repository for Quantifying Recommender Evaluation Errors

# Environment
Install Anaconda and packages in environment.yml

# Run the program with shell commands:

To list available invoke tasks, run command under the project directory:
`invoke --list`

## Calibration

1.  We calibrate each of 4 models for each of 4 statistics on each of 3 datasets.

    Example commands:
    `invoke gp-minimize-lda-unif-csr --dataname=az_music_5core --metric=ucorr`

    Task options are:
    `gp-minimize-lda-unif-csr`, `gp-minimize-lda-pop-csr`, `gp-minimize-ibp-unif-csr`, `gp-minimize-ibp-pop-csr`

    `dataname` options:
    `az_music_5core`, `ml_1m`, `steam_video_game`

    `metric` options are:
    `user-act`, `item-pop`, `icorr`, `ucorr`
 
2.  After finishing all the 48 tasks in 1, we calibrate relative average loss for each of 4 models on each of 3 datasets.

    Example commands:
    `invoke skopt-ibp-pop --dataname=ml_1m`

    Task options are:
    `skopt-ibp-unif`, `skopt-ibp-pop`, `skopt-lda-unif`, `skopt-lda-pop`.

    `dataname` options:
    `az_music_5core`, `ml_1m`, `steam_video_game`

3.  To generate the calibration table in the paper, we re-run data generation for 20 times with tuned parameters in 1 and 2.

    Example commands:
    `invoke calibration-table-ibp-pop-csr --dataname=ml_1m --metric=ucorr --ntimes=20`

    Task options are:
    `calibration-table-lda-unif-csr`, `calibration-table-lda-pop-csr`, `calibration-table-ibp-unif-csr`, `calibration-table-ipb-pop-csr`

    `dataname` options:
    `az_music_5core`, `ml_1m`, `steam_video_game`.

    `metric` options are:
    `user-act`, `item-pop`, `icorr`, `ucorr`, `avg_loss`.

## Simulation

Simulator generates datasets with parameters tuned by the relative average loss and conduct offline evaluation experiment.

Example commands:
`invoke simulate-lda-unif-csr --dataname=ml_1m`

Task options are:
`simulate-lda-unif-csr`, `simulate-lda-pop-csr`, `simulate-ibp-unif-csr`, `simulate-ipb-pop-csr`

`dataname` options:
`az_music_5core`, `ml_1m`, `steam_video_game`

## Result analysis
Experiments output are saved under `build` directory under the project directory, then run the notebook `Results-Analysis-CHIIR-2020.ipynb`.

A archived outputs from history runs are under folder `chiir-2020`.

## Run the program in Slurm cluster
Under the project directory, run commands in `batch-tasks-calibration.txt`, `batch-tasks-simulation.txt`

## Files and folders for archive

Python Module and env files: `simulation_utils`, `environment-mac.yml`, `environment.yml`, `tasks.py`

Experiment output: `chiir-2020`

Result analysis notebooks: `Results-Analysis-CHIIR-2020.ipynb`, `steam-kcore.ipynb`.

Slurm bash scripts and task examples: `batch-sims.sh`, `run-sims.sh`, `batch-tasks-calibration.txt`, `batch-tasks-simulation.txt`.

