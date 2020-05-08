# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import tdepps.utils.stats as stats


def make_bg_pdf_scan_plots(fname, emp_dist, thresh_vals, pvals, scales,
                           pval_thresh):
    """
    Make BG PDF scan plots and fitted and empirical PDF, SF comparisons.
    Saves a PNG plot to the given file path.

    Parameters
    ----------
    fname : str
        Absolute filename to where the plot is saved.
    emp_dist : ``tdepps.utils.stats.emp_with_exp_tail_dist`` instance
        PDF object with the best fit threshold stored.
    thresh_vals : array-like
        Scanned threshold values.
    pvals : array-like
        p-values for each scanned threshold.
    scales : array-like
        Fitted scales for each scanned threshold.
    pval_thresh : float
        p-value used to decide which is the best fit threshold.
    """
    def _plot_sigma_lines(ax, sigmas):
        sigmas = np.sort(sigmas)
        q = 100. * np.atleast_1d(stats.sigma2prob(sigmas))
        _p = stats.percentile_nzeros(emp_dist.data, emp_dist.nzeros,
                                     q=q, sorted=True)
        for i, pi in enumerate(_p):
            ax.axvline(pi, 0, 1, ls="--", c="C7",
                       alpha=sigmas[i] / np.amax(sigmas),
                       label=r"{:.1f}$\sigma$".format(sigmas[i]))

    fig, (axl, axc, axr) = plt.subplots(1, 3, figsize=(17.5, 5))

    # ## Left: Plot the selected combined PDF ##
    _plot_sigma_lines(axl, [3., 4., 5., 5.5])
    # Plot empirical PDF part
    h, b, err, _ = emp_dist.data_hist(dx=.25, density=True, which="emp")
    axl.plot(b, np.r_[h[0], h], drawstyle="steps-pre", color="k")
    mids = 0.5 * (b[:-1] + b[1:])
    axl.errorbar(mids, h, yerr=err, fmt=",", color="k")
    # Plot exponential data part
    h, b, err, _ = emp_dist.data_hist(dx=.25, density=True, which="exp")
    axl.plot(b, np.r_[h[0], h], drawstyle="steps-pre", color="C7")
    mids = 0.5 * (b[:-1] + b[1:])
    axl.errorbar(mids, h, yerr=err, fmt=",", color="C7")
    # Plot the exponetial PDF part
    x = np.linspace(emp_dist.thresh, np.amax(emp_dist.data), 100)
    axl.plot(x, emp_dist.pdf(x), color="C3",
             label=("exp tail\n" +
                    r"$\lambda$={:.2f}".format(1. / emp_dist.scale)))
    axl.axvline(emp_dist.thresh, 0, 1, ls=":", color="C3")
    axl.set_yscale("log", nonposy="clip")
    axl.set_xlabel("ts")
    axl.set_title("Test Statitics")
    axl.legend()

    # ## Center: Plot the selected combined p-values ##
    x = np.linspace(0, np.amax(emp_dist.data), 500)
    cdf_emp = 1. - stats.cdf_nzeros(emp_dist.data, emp_dist.nzeros, vals=x,
                                    sorted=True)
    cdf_dist = emp_dist.sf(x)
    _plot_sigma_lines(axc, [3., 4., 5., 5.5])
    axc.plot(x, cdf_emp, color="k")
    axc.plot(x, cdf_dist, color="C3")
    axc.axvline(emp_dist.thresh, 0, 1, ls=":", color="C3", label="threshold")
    axc.set_yscale("log", nonposy="clip")
    axc.set_xlabel("ts")
    axc.set_title("p-values")
    axc.legend()

    # ## Right: Threshold scan
    _plot_sigma_lines(axr, [3., 4., 5., 5.5])
    axr.axhline(1, 0, 1, ls="--", c="C7")
    axr.axhline(pval_thresh, 0, 1, ls="-", c="C7")      # Rejection line
    axr.axvline(emp_dist.thresh, 0, 1, ls="-", c="k")   # Best thresh
    axr.plot(thresh_vals, pvals, c="C1", label="KS pval")
    axr.plot(thresh_vals, 1. / scales, c="C2", label="lambdas")
    axr.set_xlabel("ts")
    axr.set_title("Best thresh: {:.2f}".format(emp_dist.thresh))
    axr.legend()

    for axi in [axl, axc, axr]:
        axi.set_xlim(0, 40)

    fig.tight_layout()
    plt.savefig(fname + ".png", dpi=200, bbox_inches="tight")
