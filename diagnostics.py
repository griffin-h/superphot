#!/usr/bin/env python

import os.path
from util import light_curve_event_data
from fit_model import setup_model
from classify import flux_model
import pymc3 as pm
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Path to PyMC3 trace directory')
parser.add_argument('--snana-path', type=str, default='.', help='Path to SNANA files')
parser.add_argument('--show', action='store_true', help='Show the plots instead of saving to a file')
args = parser.parse_args()

_, _, ps1id, fltr = os.path.split(args.filename.strip('/'))[-1].split('_')
snana_file = os.path.join(args.snana_path, f'PS1_PS1MD_{ps1id}.snana.dat')
t = light_curve_event_data(snana_file)
for fltr in 'griz':
    obs = t[t['FLT'] == fltr]
    model, varnames = setup_model(obs)
    trace = pm.load_trace(args.filename, model)

pm.traceplot(trace, textsize=6, figsize=(6., 7.))
pm.pairplot(trace, textsize=6, figsize=(6., 6.))
pm.plot_posterior(trace, textsize=6, figsize=(6., 4.))
summary = pm.summary(trace)

x = np.arange(obs['PHASE'].min(), obs['PHASE'].max())
plt.figure()
for i in np.random.randint(0, len(trace), size=100):
    params = [trace[j][i] for j in varnames]
    y = flux_model(x, *params)
    plt.plot(x, y, 'k', alpha=0.1)
plt.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o')
plt.xlabel('Phase')
plt.ylabel('Flux')

if args.show:
    print(summary)
    plt.show()
else:
    with open(os.path.join(args.filename, 'summary.txt'), 'w') as f:
        f.write(summary.to_string() + '\n')
    figure_filename = os.path.join(args.filename, 'Figure_{:d}.pdf')
    for i in range(4):
        fig = plt.figure(i + 1)
        fig.savefig(figure_filename.format(i + 1))
