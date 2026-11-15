# Tensor product model transformation for spatial interpolation

## Introduction

This MATLAB project compares Tensor Product (TP) model transformation for spatial interpolation with ordinary kriging, global Inverse Distance Weighting (IDW), Delaunay-based scattered interpolation, and a neural network.

For scattered observations, the TP pipeline constructs a grid, aggregates samples, and imputes missing cells before applying Higher-Order Singular Value Decomposition (HOSVD) and convexification. The resulting model is equivalent to a zero-order Takagi-Sugeno (TS) fuzzy model.

Experiments use synthetic benchmark functions and a monthly NOAA temperature reference dataset. Numerical checks concern the fitted convex representation, not guaranteed predictive accuracy.

## Repository Structure

| Path                       | Purpose                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `main.m`                   | Single experiment with plots, prediction errors, and model checks.                           |
| `monte_carlo_simulation.m` | Repeated comparisons with CSV summaries and raw MAT results.                                 |
| `tp-model-transformation/` | HOSVD, tensor operations, and convexification.                                               |
| `interpolation-algorithm/` | Kriging, IDW, scattered interpolation, and neural-network predictors.                        |
| `auxiliary/`               | Dataset loading/generation, sampling, grid construction, tensor completion, and diagnostics. |
| `formal-guarantees/`       | Numerical checks of output bounds, robustness, truncation, and error decomposition.          |
| `model-analyzes/`          | Convex TP model validation.                                                                  |
| `datasets/`                | NOAA NetCDF reference data.                                                                  |
| `experiments/`             | Complementary MATLAB analyses/figure exporters and Python table exporters/auditors.          |

## Requirements

- **MATLAB R2026a**: Reference release used for execution.
- **Deep Learning Toolbox**: Neural-network training (`fitnet`, `trainbr`).
- **Statistics and Machine Learning Toolbox**: Spatial distances and statistical utilities.
- **Image Processing Toolbox**: `stdfilt`, called by `calculate_erros` in both entry scripts.
- **NOAA data file**: Keep `datasets/air.mon.v6.ltm.1991.2020.nc` available when using `wind_temperature`. It is included in the repository.
- **Python 3 (optional)**: Only for the table exporters in `experiments/`, using the standard library. Neither entry script requires Python.

Run MATLAB from the repository root so relative paths resolve correctly. No external TP Tool installation is required.

## Running the Project

Set MATLAB's Current Folder to the repository root. Both scripts clear the current workspace and close existing figures when started.

### Single experiment: `main.m`

1. Open `main.m` and adjust the **Experiment settings** section:
   - Select a dataset with `chosen_dataset = avaliable_datasets(...)`; `chosen_mount` selects the NOAA month (1–12).
   - Set `n_samples_per_input` and `train_ratio`.
   - Select `"random"` or `"grid"` with `sampling_method = sampling_methods(...)`, and an imputation method with `fill_method = fill_methods(...)`.
   - Set `convexification_type` to `"SNNN"`, `"CNO"`, or `"RNO-INO"`; `cno_options` applies only to CNO.
   - Set `global_disable_plots = true` to suppress figures.
2. Execute:

   ```matlab
   run('main.m')
   ```

**Expected output:**

- Console diagnostics for grid size, rule count, convexification, and model checks.
- A comparison of TP, Kriging, IDW, Scattered interpolation, and Neural Net using relative error (RE), mean absolute error (MAE), root mean square error (RMSE), signed-error standard deviation (StdErr), maximum absolute error (MaxAE), and TS rule count.
- Figures showing the reference field, training samples/grid, membership functions, reconstruction, and error comparison when plots are enabled.
- Results retained in the workspace, including `Results_Table` and `formal_guarantee_report`. This script does not automatically save result files.

### Repeated comparisons: `monte_carlo_simulation.m`

1. Open `monte_carlo_simulation.m` and configure:
   - `chosen_datasets`: a subset of `avaliable_datasets`.
   - `chosen_months`: NOAA months (1–12), ignored for synthetic datasets.
   - `train_ratios_list`, `sampling_methods_list`, and `n_mc_iterations`.
   - `convexification_type`, optional `cno_options`, and `verify_guarantees`.
   - `export_results` and `results_dir` to control file exports.
2. Execute:

   ```matlab
   run('monte_carlo_simulation.m')
   ```

For a short execution check, change the assignments inside the script to `sampling_methods_list = "random"` and `n_mc_iterations = 10`. This automatically selects a separate `random_n10` output folder. Setting variables only in the Command Window will not override the script. Ten repetitions check execution, not statistical convergence.

**Expected output:**

- Console tables of mean ± standard deviation for seven methods: the four reference predictors and TS with global-mean, interpolation, or tensor-completion filling.
- Error metrics, TS rule counts, grid-occupancy diagnostics, and counts of successful repetitions (`N_ok`). When checks are enabled, additional tables report pass rates and quantitative bounds.
- No figures. With exports enabled, files are written under `results/monte_carlo/<convexification>/<sampling>_n<repetitions>/`, for example `cno/grid_n1/` with the current defaults or `cno/random_n10/` for a ten-repetition check:
  - `errors_<tag>.csv`: error and grid summaries.
  - `guarantees_<tag>.csv` and `guarantees_quant_<tag>.csv`: model-check summaries when enabled.
  - `mc_raw_<dataset-or-month>.mat`: raw results and configuration for each dataset/month.

For example, a random-sampling run can produce `errors_wind_temperature_month02_ratio30_random.csv` and `mc_raw_wind_temperature_month02.mat`. Matching files within the same run folder are overwritten; different sampling schemes or repetition counts use separate folders.

Complementary analyses and audit exports use `results/reviewer1/cno/` through `results/reviewer4/cno/`. Readers can fall back to archived reference folders when the new records are absent; new outputs never target those archived folders. Table/figure reproduction still requires the full 50-repetition Random references, not a reduced validation batch.

## Authors & License

This repository is the codebase for an academic doctoral study.

This project is open-source. Anyone is authorized and encouraged to download, use, modify, and distribute this code.
