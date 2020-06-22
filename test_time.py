import tdepps
from tdepps.backend import time_box
import numpy as np

time_window_half = 2

secinday = 60*60*24

srcs_t = np.linspace(1,9,9)/secinday

print("srcs_t: ",srcs_t)

srcs_dt0 = time_window_half * np.ones_like(srcs_t)*(-1.)
srcs_dt1 = time_window_half * np.ones_like(srcs_t)

print("srcs_dt0: ",srcs_dt0)
print("srcs_dt1: ",srcs_dt1)

event_time = np.array([0.5,1])/secinday

print("event_t: ",event_time)

print("time_box_values: ",time_box(event_time,srcs_t,srcs_dt0,srcs_dt1))

