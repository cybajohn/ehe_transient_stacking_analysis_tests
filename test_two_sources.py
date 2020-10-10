import os
import json
import gzip
from glob import glob

from _paths import PATHS
from _plots import make_bg_pdf_scan_plots
import numpy as np
import matplotlib.pyplot as plt
import _loader

SECINDAY = 60. * 60. * 24.

inpath = os.path.join(PATHS.data, "bg_trials_combined", "tw_??.json.gz")

files = sorted(glob(inpath))

nzeros = []
ntrials = []
dt = []

for fpath in files:
    fname = os.path.basename(fpath)
    with gzip.open(fpath) as inf:
        trials = json.load(inf)
        print("- Loaded:\n    {}".format(fpath))
    nzeros.append(trials["nzeros"])
    ntrials.append(trials["ntrials"])
    dt.append(trials["time_window"][1]/SECINDAY)

non_zero_percent = 100. * (np.array(ntrials) - np.array(nzeros))/np.array(ntrials)

sample_names = _loader.source_list_loader()
sample_names = [sample_names[2]] # just 2012-2014
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        all_srcs.extend(_loader.source_list_loader(key)[key])

src_1 = all_srcs[2]["mjd"]
src_2 = all_srcs[6]["mjd"]

when_overlap = (src_2 - src_1)/2.

plt.plot(dt,non_zero_percent,"x")
#plt.axvline(when_overlap)
plt.xlabel("window in days")
plt.ylabel("non_zero trials in percent")
plt.savefig("plot_stash/time/one_source_trials_time.pdf")

plt.clf()
