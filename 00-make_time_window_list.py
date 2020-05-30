# coding: utf-8

"""
Dump a list of tested time windows to load them in the analysis scripts.
"""

import os
import numpy as np

from _paths import PATHS


SECINMIN = 60.
SECINHOUR = 60. * SECINMIN
SECINDAY = 24. * SECINHOUR


def sec2timestr(sec):
    sec = np.around(sec, decimals=3)
    d = int(sec / SECINDAY)
    sec -= d * SECINDAY
    h = int(sec / SECINHOUR)
    sec -= h * SECINHOUR
    m = int(sec / SECINMIN)
    s = sec - m * SECINMIN
    return "{:d}d : {:02d}h : {:02d}m : {:06.3f}s".format(d, h, m, s)


# Time window lower and upper times relative to sourve time in seconds.
# Time windows increase logarithmically from +-1 sec to +-2.5 days.
dt = np.logspace(0, np.log10(2.5 * SECINDAY), 20 + 1)
dt = np.vstack((-dt, dt)).T

print("Total time window lengths:")
for i, dti in enumerate(np.diff(dt).ravel()):
    print("{:2d}:  {}".format(i, sec2timestr(dti)))


outpath = os.path.join(PATHS.local, "time_window_list")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Saving with millisecond precision is enough
delimiter = " "
header = "#{:>11s}{}{:>12s}".format("dt0 [s]", delimiter, "dt1 [s]")
np.savetxt(X=dt, fname=os.path.join(outpath, "time_window_list.txt"),
           comments="", header=header, fmt="%+12.3f", delimiter=delimiter)

print("Saved time windows to:\n{}".format(outpath))
