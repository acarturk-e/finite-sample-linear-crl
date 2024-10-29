"""Sample complexity of interventional causal representation learning -- Main experiments file.

This file is built to be run as a script without any arguments.
1 complete run took ~8 minutes to run on my laptop CPU.
The generated results in `results` directory is ~50MB per run.
See the output for the relative path to the run directory.

Portions of this file is taken from the main file of
https://github.com/acarturk-e/score-based-crl
with author's permission.
"""

import datetime
import os
import logging
import pickle
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from finite_sample_algos import *
import utils

scm_type = "linear"

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")

ATOL_EIGV = 1e-2
ATOL_ORTH = 1e-1

DECODER_MIN_COND_NUM = 1e-1


if __name__ == "__main__":
    nd_list = [
        (3, 3),
        (3, 15),
        (5, 5),
        (5, 15),
        (10, 10),
        (10, 15),
    ]
    nsamples_list = list(map(lambda ex: int(10 ** ex), [2.5, 3, 3.5, 4, 4.5, 5]))

    pd.set_option('display.max_columns', None)

    fill_rate = 0.5
    nruns = 100
    np_rng = np.random.default_rng()

    # Score computation/estimation settings
    estimate_score_fns = True
    enable_gaussian_score_est = True
    n_score_epochs = 20
    add_noise_to_ssm = False

    # SCM settings
    hard_intervention = True
    hard_graph_postprocess = True
    type_int = "hard int"
    var_change_mech = "scale"
    var_change = 0.25

    # DEBUG:
    randomize_top_order = True
    randomize_intervention_order = True

    # Result dir setup
    run_name = (
        scm_type
        + "_"
        + ("hard" if hard_intervention else "soft")
        + "_"
        + f"nr{nruns}"
        + "_"
        + ("gt" if not estimate_score_fns else ("ss" if not enable_gaussian_score_est or scm_type != "linear" else "pr"))
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    run_dir = os.path.join("results", run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s"
    )
    log_file_h = logging.FileHandler(log_file)
    log_file_h.setFormatter(log_formatter)
    log_file_h.setLevel(logging.DEBUG)
    log_console_h = logging.StreamHandler()
    log_console_h.setFormatter(log_formatter)
    log_console_h.setLevel(logging.INFO)
    log_root_l = logging.getLogger()
    log_root_l.setLevel(logging.DEBUG)
    log_root_l.addHandler(log_file_h)
    log_root_l.addHandler(log_console_h)

    logging.info(f"Logging to {log_file}")

    config = {
        "nruns" : nruns,
        "nd_list": nd_list,
        "nsamples_list": nsamples_list,
        "ATOL_EIGV": ATOL_EIGV,
        "ATOL_ORTH": ATOL_ORTH,
        "scm_type": scm_type,
        "hard_intervention": hard_intervention,
        "hard_graph_postprocess": hard_graph_postprocess,
        "var_change_mech": var_change_mech,
        "var_change": var_change,
        "estimate_score_fns": estimate_score_fns,
        "n_score_epochs": n_score_epochs
    }
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    model_df = {
        (nsamples, n, d, run_idx): {
            "scm": SCM.__new__(SCM),
            "intervention_order": np.empty(n, dtype=np.int_),
            "decoder": np.empty((d, n)),
            "encoder": np.empty((n, d)),
        }
        for run_idx in range(nruns)
        for (n, d) in nd_list
        for nsamples in nsamples_list
    }
    model_cons_df = {
        (nsamples, n, d, run_idx): {
            "eta_star": 0.0,
            "gamma_star": 0.0,
            "beta": 0.0,
            "beta_min": 0.0,
        }
        for run_idx in range(nruns)
        for (n, d) in nd_list
        for nsamples in nsamples_list
    }
    results_df = {
        (nsamples, n, d, run_idx): {}
        for run_idx in range(nruns)
        for (n, d) in nd_list
        for nsamples in nsamples_list
    }
    analysis_df = {
        (nsamples, n, d, run_idx): {
            "shd": float('nan'),
            "edge_precision": float('nan'),
            "edge_recall": float('nan'),
            "encoder_est_err": float('nan'),
            "dsx_rmse": float('nan'),
            "rxs_2e": float('nan'),
        }
        for run_idx in range(nruns)
        for (n, d) in nd_list
        for nsamples in nsamples_list
    }

    for nsamples_idx, nsamples in enumerate(nsamples_list):
        logging.info(f"Starting {nsamples = }")
        for nd_idx, (n, d) in enumerate(nd_list):
            logging.info(f"Starting {(n, d) = }")

            for run_idx in range(nruns):
                if run_idx % 10 == 10 - 1:
                    logging.info(f"{(nsamples, n, d) = }, {run_idx = }")

                run_data = {
                    "model": {},
                    "model_cons": {},
                    "results": {},
                    "analysis": {},
                }

                # Build the decoder in two steps:
                # 1: Uniformly random selection of column subspace
                # TODO: Theoretically ensure this is indeed uniform
                import scipy.stats  # type: ignore
                decoder_q: npt.NDArray[np.floating] = scipy.stats.ortho_group(d, np_rng).rvs()[:, :n]  # type: ignore

                # 2: Random mixing within the subspace
                decoder_r = np_rng.random((n, n)) - 0.5
                decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)
                while decoder_r_svs[-1] / decoder_r_svs[0] < DECODER_MIN_COND_NUM:
                    decoder_r = np_rng.random((n, n)) - 0.5
                    decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)

                # Then, the full decoder is the composition of these transforms
                decoder = decoder_q @ decoder_r
                encoder = np.linalg.pinv(decoder)

                scm = SCM(
                    n,
                    fill_rate,
                    randomize_top_order=randomize_top_order,
                    np_rng=np_rng,
                )

                intervention_order = np_rng.permutation(n) if randomize_intervention_order else np.arange(n)
                envs = [list[int]()] + [[i] for i in intervention_order]

                # Also form the inverse of the intervention order
                inv_intervention_order = np.arange(n)
                inv_intervention_order[intervention_order] = np.arange(n)

                z_samples = np.stack(
                    [
                        scm.sample(
                            (nsamples,),
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )
                x_samples = decoder @ z_samples

                gt_sz_samples = np.stack(
                    [
                        scm.score_fn(
                            z_samples[0, ...],
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )
                gt_sx_samples = encoder.T @ gt_sz_samples

                # Model parameters
                gt_dsx_samples = gt_sx_samples[0, ...] - gt_sx_samples[1:, ...]
                gt_rxs = utils.cov(gt_dsx_samples[..., 0], center_data=False)

                eta_star = float("+inf")
                for mcm in map(list, utils.powerset(range(n))):
                    if len(mcm) == 0: continue
                    cm_eigval = np.linalg.eigh(gt_rxs[mcm].sum(0)).eigenvalues
                    cm_rank = np.sum(cm_eigval > ATOL_EIGV, dtype=np.int_)
                    eta_star = min(eta_star, cm_eigval[d - cm_rank])
                gamma_star = float("+inf")
                for k in range(n):
                    gamma_star = min(gamma_star, np.linalg.norm(encoder[k, :]) * np.linalg.norm(decoder[:, k]))
                beta = 1 / (2 * ((gt_dsx_samples ** 2).sum((-1, -2)).mean(-1)).max(0))
                beta_min = 4 * ((gt_dsx_samples ** 2).sum((-1, -2)).mean(-1)).min(0)

                run_data["model"]["scm"]                = scm
                run_data["model"]["intervention_order"] = intervention_order
                run_data["model"]["decoder"]            = decoder
                run_data["model"]["encoder"]            = encoder
                run_data["model"]["gt_rxs"]             = gt_rxs
                run_data["model_cons"]["eta_star"] = eta_star
                run_data["model_cons"]["gamma_star"] = gamma_star
                run_data["model_cons"]["beta"]     = beta
                run_data["model_cons"]["beta_min"] = beta_min

                # Evaluate score functions on the same data points
                if estimate_score_fns:
                    # Use estimated (noisy) score functions
                    # We denote the `d` dimensional samples projected down to `n`
                    # dimensional support of `x_samples` by suffix `_n`.

                    x_samples_cov = utils.cov(x_samples[0, : n + d, :, 0])
                    xsc_eigval, xsc_eigvec = np.linalg.eigh(x_samples_cov)
                    basis_of_x_supp = xsc_eigvec[:, -n:]
                    x_samples_n = basis_of_x_supp.T @ x_samples

                    hat_sx_fns = list[Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]]()
                    for i in range(len(envs)):
                        # If we know the latent model is Linear Gaussian, score estimation
                        # is essentially just precision matrix --- a parameter --- estimation
                        hat_sx_fn_i_n = utils.gaussian_score_est(x_samples_n[i])
                        def hat_sx_fn_i(
                            x_in: npt.NDArray[np.floating],
                            # python sucks... capture value with this since loops are NOT scopes
                            hat_sx_fn_i_n: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] = hat_sx_fn_i_n,
                        ) -> npt.NDArray[np.floating]:
                            """Reduce input down to support of x, compute estimate, transform the result back up."""
                            return basis_of_x_supp @ hat_sx_fn_i_n(basis_of_x_supp.T @ x_in)
                        hat_sx_fns.append(hat_sx_fn_i)

                    sx_samples = np.stack(
                        [
                            hat_sx_fns[env_idx](x_samples[0, ...])
                            for env_idx in range(len(envs))
                        ]
                    )

                else:
                    # Use ground truth score functions
                    sx_samples = gt_sx_samples

                dsx_samples = sx_samples[0, ...] - sx_samples[1:, ...]
                rxs = utils.cov(dsx_samples[..., 0], center_data=False)

                graph = scm.adj_mat
                graph_tc = utils.dag.transitive_closure(graph)
                assert graph_tc is not None

                dsx_rmse = np.max(((dsx_samples - gt_dsx_samples) ** 2).mean((1, 2, 3)) ** 0.5)
                rxs_2e = np.max(np.linalg.norm(rxs - gt_rxs, 2, (-1, -2)))

                run_data["analysis"]["dsx_rmse"]    = dsx_rmse
                run_data["analysis"]["rxs_2e"]      = rxs_2e

                # Run algorithm steps
                try:
                    top_order = causal_order_estimation(rxs)
                    graph_est = graph_estimation(rxs, top_order, ATOL_EIGV, ATOL_ORTH)
                    encoder_est = encoder_estimation(rxs, ATOL_EIGV)
                except Exception as err:
                    logging.error(f"Unexpected {err=}, masking entry out")
                    continue

                # This run succeeded
                run_data["results"]["top_order"]    = top_order
                run_data["results"]["graph_est"]    = graph_est
                run_data["results"]["encoder_est"]  = encoder_est

                ### ANALYSIS

                # Both the graph and the encoder are subject to permutation with intervention order.
                # Invert this to compare to ground truth
                graph_est = graph_est[inv_intervention_order, :][:, inv_intervention_order]
                encoder_est = encoder_est[inv_intervention_order, :]
                eff_transform_s = encoder_est @ decoder
                eff_transform_s *= (np.sign(np.diagonal(eff_transform_s)) / np.linalg.norm(eff_transform_s, ord=2, axis=1))[:, None]

                # Graph accuracy metrics: SHD, precision, recall. Compare transitive closures.
                edge_cm_s = [
                    [
                        ( graph_tc &  graph_est).sum(dtype=np.int_),
                        (~graph_tc &  graph_est).sum(dtype=np.int_),
                    ], [
                        ( graph_tc & ~graph_est).sum(dtype=np.int_),
                        (~graph_tc & ~graph_est).sum(dtype=np.int_),
                    ]
                ]
                shd            = edge_cm_s[0][1] + edge_cm_s[1][0]
                edge_precision = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[0][1])) if (edge_cm_s[0][0] + edge_cm_s[0][1]) != 0 else 1.0
                edge_recall    = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[1][0])) if (edge_cm_s[0][0] + edge_cm_s[1][0]) != 0 else 1.0

                # Variables recovery metric: Encoder error
                # Construct the maximal theoretically allowed mixing pattern (up to parents)
                max_mixing_mat = graph.T | np.eye(n, dtype=bool)
                encoder_est_err = np.linalg.norm(eff_transform_s * (~max_mixing_mat), 2)

                run_data["analysis"]["shd"]             = shd
                run_data["analysis"]["edge_precision"]  = edge_precision
                run_data["analysis"]["edge_recall"]     = edge_recall
                run_data["analysis"]["encoder_est_err"] = encoder_est_err
                run_data["analysis"]["dsx_rmse"]        = dsx_rmse
                run_data["analysis"]["rxs_2e"]          = rxs_2e

                # Save run
                model_df     [nsamples, n, d, run_idx] = run_data["model"]
                model_cons_df[nsamples, n, d, run_idx] = run_data["model_cons"]
                results_df   [nsamples, n, d, run_idx] = run_data["results"]
                analysis_df  [nsamples, n, d, run_idx] = run_data["analysis"]
                with open(os.path.join(run_dir, f"{nsamples}_{n}_{d}_{run_idx}.pkl"), "wb") as f:
                    pickle.dump(run_data, f)

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

    print(f"Run directory: {run_dir}")
    print("\nResults\n")
    print(results_to_print)
