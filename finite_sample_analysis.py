"""Sample complexity of interventional causal representation learning -- analysis.

This file can be run either as a script or notebook.
- When running as a script, it requires 1 positional argument: The path to the run directory to analyze.
- When running as a notebook, one needs to manually input the run directory to the 2nd cell.

Minor comment: The scatter plot in the last cell is hard-coded to provide different markers
ONLY when there are 3 distinct n values in the experiment. Otherwise, the differentiation is only
based on hue, and not B/W printing and color blindness friendly.
"""

#%%
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from matplotlib.ticker import ScalarFormatter

xticks_size = 16 * 1.5
yticks_size = 16 * 1.5
xlabel_size = 18 * 1.5
ylabel_size = 18 * 1.5
legend_size = 16 * 1.5
legend_loc = 'lower right'
linewidth = 3 * 1.5
linestyle = '--'
markersize = 10 * 1.5


is_notebook = hasattr(sys, 'ps1')

sns.set_theme()
sns.set_style("whitegrid")
if not is_notebook:
    pd.set_option('display.max_columns', None)

#%%
# Result dir setup: Take it from command line arguments
if not is_notebook:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to the specific run directory")
    args = parser.parse_args()
    run_dir = args.run_dir
else:
    run_dir = r"results\linear_hard_nr100_pr_2024-10-29_01-42-25"

if not os.path.exists(run_dir):
    raise FileNotFoundError("Run directory doesn't exist, exiting.")

if run_dir[-1] != '/': run_dir += '/'

#%%
# Read and load config

with open(run_dir + "config.pkl", "rb") as f:
    config = pickle.load(f)

nruns                   = config["nruns"]
nd_list                 = config["nd_list"]
nsamples_list           = config["nsamples_list"]
ATOL_EIGV               = config["ATOL_EIGV"]
ATOL_ORTH               = config["ATOL_ORTH"]
scm_type                = config["scm_type"]
hard_intervention       = config["hard_intervention"]
hard_graph_postprocess  = config["hard_graph_postprocess"]
var_change_mech         = config["var_change_mech"]
var_change              = config["var_change"]
estimate_score_fns      = config["estimate_score_fns"]
n_score_epochs          = config["n_score_epochs"]

#%%
# Read models, model constants, results and prior analyses
model_df      = dict()
model_cons_df = dict()
results_df    = dict()
analysis_df   = dict()
for nsamples in nsamples_list:
    for (n, d) in nd_list:
        for run_idx in range(nruns):
            with open(run_dir + f"{nsamples}_{n}_{d}_{run_idx}.pkl", "rb") as f:
                run_data = pickle.load(f)
                model_df     [nsamples, n, d, run_idx] = run_data["model"]
                model_cons_df[nsamples, n, d, run_idx] = run_data["model_cons"]
                results_df   [nsamples, n, d, run_idx] = run_data["results"]
                analysis_df  [nsamples, n, d, run_idx] = run_data["analysis"]

model_df      = pd.DataFrame(model_df     ).T
model_cons_df = pd.DataFrame(model_cons_df).T
results_df    = pd.DataFrame(results_df   ).T
analysis_df   = pd.DataFrame(analysis_df  ).T

model_df     .index = model_df     .index.set_names(["nsamples", "n", "d", "run_idx"])
model_cons_df.index = model_cons_df.index.set_names(["nsamples", "n", "d", "run_idx"])
results_df   .index = results_df   .index.set_names(["nsamples", "n", "d", "run_idx"])
analysis_df  .index = analysis_df  .index.set_names(["nsamples", "n", "d", "run_idx"])

correct_graph = analysis_df["shd"] == 0
correct_graph.name = "correct_graph"
analysis_df = analysis_df.assign(correct_graph=correct_graph)

#%%
# Print the original run's results

ok_runs = (results_df.count(1) == results_df.columns.size).groupby(level=["nsamples", "n", "d"]).sum()
ok_runs.name = "ok_runs"
fail_rates = 1.0 - ok_runs / results_df.groupby(level=["nsamples", "n", "d"]).size()
fail_rates.name = "fail_rates"

correct_graph_rate = (analysis_df['shd'] == 0).groupby(level=["nsamples", "n", "d"]).mean()
correct_graph_rate.name = "correct_graph_rate"

dsx_mse = (analysis_df["dsx_rmse"] ** 2).groupby(level=("nsamples", "n", "d")).mean()
dsx_mse.name = "dsx_mse_over_nruns"

shd_mean             = analysis_df["shd"            ].groupby(level=["nsamples", "n", "d"]).mean()
edge_precision_mean  = analysis_df["edge_precision" ].groupby(level=["nsamples", "n", "d"]).mean()
edge_recall_mean     = analysis_df["edge_recall"    ].groupby(level=["nsamples", "n", "d"]).mean()
encoder_est_err_mean = analysis_df["encoder_est_err"].groupby(level=["nsamples", "n", "d"]).mean()

shd_mean            .name = "shd_mean"
edge_precision_mean .name = "edge_precision_mean"
edge_recall_mean    .name = "edge_recall_mean"
encoder_est_err_mean.name = "encoder_est_err_mean"

shd_std             = analysis_df["shd"            ].groupby(level=["nsamples", "n", "d"]).std()
edge_precision_std  = analysis_df["edge_precision" ].groupby(level=["nsamples", "n", "d"]).std()
edge_recall_std     = analysis_df["edge_recall"    ].groupby(level=["nsamples", "n", "d"]).std()
encoder_est_err_std = analysis_df["encoder_est_err"].groupby(level=["nsamples", "n", "d"]).std()

shd_std            .name = "shd_std"
edge_precision_std .name = "edge_precision_std"
edge_recall_std    .name = "edge_recall_std"
encoder_est_err_std.name = "encoder_est_err_std"

results_to_print = pd.DataFrame([
    ok_runs,
    correct_graph_rate,
    shd_mean,
    shd_std,
    edge_precision_mean,
    edge_precision_std,
    edge_recall_mean,
    edge_recall_std,
    encoder_est_err_mean,
    encoder_est_err_std,
]).T

print("Results\n")
print(results_to_print)

#%%
# Model constants vs (n, d)

eta_star_mean   = model_cons_df["eta_star"  ].groupby(level=["n", "d"]).mean()
gamma_star_mean = model_cons_df["gamma_star"].groupby(level=["n", "d"]).mean()
beta_mean       = model_cons_df["beta"      ].groupby(level=["n", "d"]).mean()
beta_min_mean   = model_cons_df["beta_min"  ].groupby(level=["n", "d"]).mean()

eta_star_mean  .name = "eta_star_mean"
gamma_star_mean.name = "gamma_star_mean"
beta_mean      .name = "beta_mean"
beta_min_mean  .name = "beta_min_mean"

eta_star_std   = model_cons_df["eta_star"  ].groupby(level=["n", "d"]).std()
gamma_star_std = model_cons_df["gamma_star"].groupby(level=["n", "d"]).std()
beta_std       = model_cons_df["beta"      ].groupby(level=["n", "d"]).std()
beta_min_std   = model_cons_df["beta_min"  ].groupby(level=["n", "d"]).std()

eta_star_std  .name = "eta_star_std"
gamma_star_std.name = "gamma_star_std"
beta_std      .name = "beta_std"
beta_min_std  .name = "beta_min_std"

model_constants_stats = pd.DataFrame([
    eta_star_mean,
    eta_star_std,
    gamma_star_mean,
    gamma_star_std,
    beta_mean,
    beta_std,
    beta_min_mean,
    beta_min_std,
]).T

print("\nModel constants vs (n, d)\n")
print(model_constants_stats)

#%%
# Visualize model constants

model_cons_df.columns = [r'$\eta^{*}$', r'$\gamma^{*}$', r'$\beta$', r'$\beta_{\rm min}$']

g = sns.relplot(
    data=model_cons_df.melt(ignore_index=False),
    kind = "line",
    x="n",
    y="value",
    hue="variable",
    style="variable",
    markersize=markersize,
    linewidth=linewidth,
    linestyle=linestyle,
    aspect=1.5,
)
g.ax.tick_params(axis='both', which='major', labelsize=xticks_size)
g.legend.remove()
# g.ax.set_title('Model constants vs $(n, d)$')
g.ax.set_xlabel('Number of nodes',size=xlabel_size)
g.ax.set_ylabel('Constants',size=ylabel_size)
# g.ax.legend(fontsize=legend_size)
g.ax.legend(fontsize=legend_size, loc='lower left')
g.figure.tight_layout()
# g.ax.grid()
g.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

g.ax.set_yscale('log')
if not is_notebook:
    g.figure.show()
g.figure.savefig(run_dir + "model_constants.png")

#%%
# Visualize graph recovery vs nsamples

g = sns.relplot(
    data=analysis_df,
    kind="line",
    x="nsamples",
    y="correct_graph",
    hue="n",
    style="n",
    markersize=markersize,
    linewidth=linewidth,
    linestyle=linestyle,
    aspect=1.5,
)
g.ax.tick_params(axis='both', which='major', labelsize=xticks_size)
g.legend.remove()
# g.ax.set_title('Graph recovery probability vs number of samples')
g.ax.set_ylabel('Graph recovery rate', size=ylabel_size)
g.ax.set_xlabel('Number of samples',size=xlabel_size)
g.ax.legend(fontsize=legend_size, loc='upper left')
g.figure.tight_layout()
# g.ax.grid()
g.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

g.ax.set_xscale('log')
if not is_notebook:
    g.figure.show()
g.figure.savefig(run_dir + "graph_rec_vs_nsamples.png")

#%%
# Visualize score estimation error vs nsamples

g = sns.relplot(
    data=analysis_df,
    kind="line",
    x="nsamples",
    y="dsx_rmse",
    hue="n",
    style="n",
    markersize=markersize,
    linewidth=linewidth,
    linestyle=linestyle,
    aspect=1.5,
)
g.ax.tick_params(axis='both', which='major', labelsize=xticks_size)
g.legend.remove()
# g.ax.set_title('Score estimation error vs number of samples')
g.ax.set_xlabel('Number of samples', size=xlabel_size)
g.ax.set_ylabel('Score estimation MSE', size=ylabel_size)
g.ax.legend(fontsize=legend_size, loc='upper right')
g.figure.tight_layout()
# g.ax.grid()
g.ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

g.ax.set_xscale('log')
if not is_notebook:
    g.figure.show()
g.figure.savefig(run_dir + "dsx_rmse_vs_nsamples.png")

#%%
# Visualize performance vs dsx_mse

plt.figure(figsize=(7.5,5))
ax = sns.scatterplot(
    data=pd.DataFrame([correct_graph_rate, dsx_mse]).T.reset_index(level="n").filter(like="15", axis=0),
    x="dsx_mse_over_nruns",
    y="correct_graph_rate",
    hue="n",
    style="n",
    s=markersize ** 2,
    palette=sns.color_palette(),
    markers=["D", "o", "X"] if len(analysis_df.index.levels[1]) == 3 else "o",
)
g = ax.get_figure()
assert g is not None

ax.tick_params(axis='both', which='major', labelsize=xticks_size)
ax.get_legend().remove()
# ax.set_title('Graph recovery probability vs score estimation error')
ax.set_xlabel('Score estimation MSE', size=xlabel_size)
ax.set_ylabel('Graph recovery rate', size=ylabel_size)
ax.legend(fontsize=legend_size, loc='upper right')

for t in ax.get_legend().get_texts():
    t.set_text("$n=" + t.get_text() + "$")

g.figure.tight_layout()
# ax.grid()
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

ax.set_ylim(bottom=-0.1, top=1.1)
ax.set_xscale('log')
if not is_notebook:
    g.show()
g.savefig(run_dir + "graph_rec_vs_dsx_mse_d_eq_15.png")

#%%
if not is_notebook:
    input("Please enter key to exit.")
