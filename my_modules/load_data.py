from glob import glob as _glob
import os as _os
import json
import numpy as np

def load_mc_data(names, folder):
	"""
	Loads mc data
	---------
	Parameters:
	names: list
		list of sample names
	folder: str
		path to folder
	---------
	Returns:
	data: dict
		data
	"""
        files = sorted(_glob(_os.path.join(folder, "*")))
        file_names = map(lambda s: _os.path.splitext(_os.path.basename(s))[0],files)
        data = {}
        for name in names:
                idx = file_names.index(name)
                fname = files[idx]
                data[name] = np.load(fname)
        return data

def load_sources(s_file):
	"""
	Loads sources
	---------
	Parameters:
	s_file: str
		path to source file
	---------
	Returns:
	sources: dict
		dict of sources and information
	"""
        with open(s_file) as _file:
                sources = json.load(_file)
        return sources



