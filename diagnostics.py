#!/usr/bin/env python

import os.path
from util import light_curve_event_data
from fit_model import setup_model, diagnostics
import pymc3 as pm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to PyMC3 trace directory')
    parser.add_argument('--snana-path', type=str, default='.', help='Path to SNANA files')
    parser.add_argument('--show', action='store_true', help='Show the plots instead of saving to a file')
    args = parser.parse_args()

    _, _, ps1id, fltr = os.path.split(args.filename.strip('/'))[-1].split('_')
    snana_file = os.path.join(args.snana_path, f'PS1_PS1MD_{ps1id}.snana.dat')
    t = light_curve_event_data(snana_file)
    obs = t[t['FLT'] == fltr]
    model, varnames = setup_model(obs)
    trace = pm.load_trace(args.filename, model)
    diagnostics(obs, trace, varnames, args.filename, args.show)
