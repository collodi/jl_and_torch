import os
import torch

def read_bods(d):
	fns = sorted([fn for fn in os.listdir(d) if fn.endswith('.bod')])
	return [read_bod(f'{d}/{fn}') for fn in fns]

def read_caps(d):
	fns = sorted([fn for fn in os.listdir(d) if fn.endswith('.cap')])
	return [read_cap(f'{d}/{fn}') for fn in fns]

def read_bod(fn):
	with open(fn) as f:
		lns = [list(map(float, ln.strip().split()[1:])) for ln in f]
		return torch.tensor(lns)

def read_cap(fn):
	with open(fn) as f:
		ln = f.readlines()[1]
		return torch.tensor(float(ln.strip().split()[0]))
