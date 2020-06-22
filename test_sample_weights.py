import numpy as np
import _loader

sample_start_endtime = np.array([[2,8],[12,18],[22,28]])
sample_names = _loader.source_list_loader()
sample_names = sample_names[1:]
for name in sample_names:
	exp_off = _loader.off_data_loader(name)[name]
	print(name)
	print(np.amin(exp_off["time"]))
	print(np.amax(exp_off["time"]))
