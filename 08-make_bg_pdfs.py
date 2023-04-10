# coding: utf-8

"""
Makes single BG PDF objects from the BG trials. PDFs are composed from an
empirical part with good statistics and a fitted exponential tail to get
continious p-values at the tails.
"""

import os
import json
import gzip
from glob import glob
import numpy as np

import tdepps.utils.stats as stats
from _paths import PATHS
from _plots import make_bg_pdf_scan_plots


inpath = os.path.join(PATHS.data, "bg_trials_combined", "tw_??.json.gz")
outpath = os.path.join(PATHS.local, "bg_pdfs")
plotpath = os.path.join(PATHS.plots, "bg_pdfs")

# Uncomment to process the independent ones from LIDO
# inpath = os.path.join(PATHS.data, "bg_trials_combined_lido", "tw_??.json.gz")
# outpath = os.path.join(PATHS.local, "bg_pdfs_lido")
# plotpath = os.path.join(PATHS.plots, "bg_pdfs_lido")

if not os.path.isdir(outpath):
    os.makedirs(outpath)
if not os.path.isdir(plotpath):
    os.makedirs(plotpath)

files = sorted(glob(inpath))
print(files)
print(files[-1])
#files = [files[-1]]
for fpath in files:
    fname = os.path.basename(fpath)
    print("Making PDF from BG trial file: {}".format(fname))

    with gzip.open(fpath) as inf:
        trials = json.load(inf)
        print("- Loaded:\n    {}".format(fpath))

    # Create PDF object and scan the best threshold
    print("- Scanning best threshold")
    print("ts: ", trials["ts"])
    emp_dist = stats.ExpTailEmpiricalDist(trials["ts"], trials["nzeros"],
                                            thresh=np.amax(trials["ts"]))
    # Scan in a range with still good statistics, but leave the really good
    # statistics part to the empirical PDF
    lo, hi = emp_dist.ppf(q=100. * stats.sigma2prob([3., 5.5]))
    thresh_vals = np.arange(lo, hi, 0.1)
    # Best fit: KS test p-value is larger than `pval_thresh` the first time
    pval_thresh = 0.5
    best_thresh, best_idx, pvals, scales = stats.scan_best_thresh(
        emp_dist, thresh_vals, pval_thresh=pval_thresh)

    # Save whole PDF object to recoverable JSON file. Save stored data with
    # float16 precision, which is sufficient and saves space
    pdf_name = os.path.join(outpath, "bg_pdf_" + fname)
    print("- Saving PDF object to:\n    {}".format(pdf_name))
    with gzip.open(pdf_name, "w") as f:
	# f.write was missing here, to_json creates json str (json.dumps),
	# which has to be saved later on (contrary to json.dump)
        f.write(emp_dist.to_json(dtype=np.float16, indent=0,
                         separators=(",", ":")))
        print("    Done")

    # Make scan plots
    plot_name = os.path.join(plotpath,
                             "bg_pdf_" + fname.replace(".json.gz", ""))
    make_bg_pdf_scan_plots(plot_name, emp_dist, thresh_vals, pvals, scales,
                           pval_thresh)
    print("- Saved plot to:\n    {}".format(plot_name))
