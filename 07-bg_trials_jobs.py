# coding: utf-8

"""
Create jobfiles for `07-bg_trials.py`.

##############################################################################
# Used seed range for bg trial jobs: [0, 100000]
##############################################################################
"""

import os
import numpy as np

from dagman import dagman
from _paths import PATHS
from _loader import time_window_loader


job_creator = dagman.DAGManJobCreator(mem=4)
job_name = "ehe_transient_stacking"

job_dir = os.path.join(PATHS.jobs, "bg_trials")
script = os.path.join(PATHS.repo, "07-bg_trials.py")

# Get time windows
all_tw_ids = time_window_loader()
ntime_windows = len(all_tw_ids)

# Timing tests: For 6 year pass2 HESE, 5 years PS tracks data, 1 year GFU
# tw00: 1e6 trials in ~661s -> ~2770 trials / sec
# tw10: 1e5 trials in ~193s -> ~ 518 trials / sec
# tw20: 1e4 trials in ~430s -> ~  23 trials / sec
# Need 1e8 trials, because we have many zero trials
# Worst case: 1e8trials / 23trials/s / 3600s/h / 1000jobs ~ 1.2 h/job
ntrials = 1e5
njobs_per_tw = int(125)
ntrials_per_job = int(ntrials / float(njobs_per_tw))
njobs_tot = njobs_per_tw * ntime_windows
if int(ntrials) != ntrials_per_job * njobs_per_tw:
    raise ValueError("Job settings does not lead to exactly " +
                     "{} trials".format(int(ntrials)))
print("Preparing {} total trials per time window".format(int(ntrials)))
print("  - {} jobs per time window".format(njobs_per_tw))
print("  - {} trials per job".format(ntrials_per_job))
print("Creating {} total jobfiles for all time windows".format(int(njobs_tot)))
print("Worst runtime per job ~{:.2f}h".format(ntrials_per_job / 20. / 3600.))

# Make unique job identifiers:
# job_ids: 000 ... 999, 000 ... 999, ...
lead_zeros = int(np.ceil(np.log10(njobs_per_tw)))
job_ids = np.array(ntime_windows * ["{1:0{0:d}d}".format(lead_zeros, i) for i
                   in range(njobs_per_tw)])
# tw_ids: 00, ..., 00, 01, .., 01, ..., 20, ..., 20
tw_ids = np.concatenate([njobs_per_tw * [tw_id] for tw_id in all_tw_ids])

job_args = {
    "rnd_seed": np.arange(10000, 10000 + njobs_tot).astype(int),
    "ntrials": njobs_tot * [ntrials_per_job],
    "job_id": job_ids,
    "tw_id": tw_ids,
    }

job_creator.create_job(script=script, job_args=job_args,
                       job_name=job_name, job_dir=job_dir, overwrite=True)
