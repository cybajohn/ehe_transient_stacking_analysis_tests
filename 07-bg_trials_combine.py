# coding: utf-8

"""
Combine output for each time window to a single file containing all trials.
"""

import os
import sys
import json
import gzip
from glob import glob

try:
    from tqdm import tqdm
except ImportError:
    print("If you want fancy status bars: `pip install --user tqdm` ;)")
    tqdm = iter

from _paths import PATHS
from _loader import time_window_loader


inpath = os.path.join(PATHS.data, "bg_trials")

outpath = os.path.join(PATHS.data, "bg_trials_combined")
if os.path.isdir(outpath):
    res = raw_input("'{}' already exists. ".format(outpath) +
                    "\nAllow overwrites (y/n)? ")
    if not res.lower() in ("y", "yes"):
        print("Abort. Script has done nothing.")
        sys.exit()
    print("  Using output directory '{}'.".format(outpath))
else:
    os.makedirs(outpath)
    print("Created output directory '{}'.".format(outpath))

# Collect for all time windows
all_tw_ids = time_window_loader()
for tw_id in all_tw_ids:
    file_path = os.path.join(inpath, "tw_{:02d}_job_*.json.gz".format(tw_id))
    files = sorted(glob(file_path))
    print("Time window {:02d}, found {} trial files:".format(tw_id, len(files)))
    if len(files) > 0:
        print("  {}\n  ...\n  {}".format(files[0], files[-1]))
        # Build output dict
        trials = {
            "ns": [],
            "ts": [],
            "time_window": None,
            "time_window_id": -1,
            "nzeros": 0,
            "rnd_seed": [],
            "ntrials": 0,
            "ntrials_per_batch": [],
            }
        # Concatenate all files
        for _file in tqdm(files):
            with gzip.open(_file) as infile:
                trial_i = json.load(infile)
            trials["ns"] += trial_i["ns"]
            trials["ts"] += trial_i["ts"]
            trials["nzeros"] += trial_i["nzeros"]
            trials["ntrials"] += trial_i["ntrials"]
            trials["rnd_seed"].append(trial_i["rnd_seed"])
            trials["ntrials_per_batch"].append(trial_i["ntrials"])
        trials["time_window"] = trial_i["time_window"]
        trials["time_window_id"] = trial_i["time_window_id"]
        # Save it
        fpath = os.path.join(outpath, "tw_{:02d}.json.gz".format(tw_id))
        with gzip.open(fpath, "w") as outf:
            json.dump(trials, fp=outf, indent=0, separators=(",", ":"))
            print("  - Saved to:\n    {}".format(fpath))
    else:
        print("  - no trials found")
