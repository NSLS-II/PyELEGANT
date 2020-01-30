import argparse
from pathlib import Path
import os
import time
import datetime
from subprocess import Popen, PIPE
import shlex

import numpy as np
import matplotlib.pylab as plt

import pyelegant as pe

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check status of geneticOptimizer results')
    parser.add_argument(
        'rootname', metavar='rootname', type=str,
        help=('Root name for optimization input SDDS file, '
              'e.g., "optim1" for input file named "optim1.sdds".'))

    args = parser.parse_args()
    #print(args)

    rootname = args.rootname
    #print(rootname)

    dot_all_filepath = f'{rootname}.all'
    dot_best_filepath = f'{rootname}.best'
    dot_sort_filepath = f'{rootname}.sort'

    if not (Path(dot_all_filepath).exists() and
            Path(dot_best_filepath).exists()):
        raise RuntimeError(
            (f'Not all necessary files ({dot_all_filepath} '
             f'& {dot_best_filepath}) exist.'))

    print(f'Present Time: {pe.util.get_current_local_time_str()}')

    last_update_timestamp = os.stat(dot_all_filepath).st_mtime
    print('Last updated: {}'.format(
        datetime.datetime.fromtimestamp(
            last_update_timestamp).strftime('%Y-%m-%dT%H-%M-%S')))

    cmd = f'sdds2stream -rows {dot_all_filepath}'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    print('Total completed runs: {}'.format(out.replace('rows', '').strip()))

    cmd = f'sdds2stream -rows {dot_best_filepath}'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    print('Rank 1 runs: {}'.format(out.replace('rows', '').strip()))

    for state in ['Running', 'Pending']:
        cmd = f'squeue --user={os.environ["USER"]} --states={state.upper()} --format="%.9i"'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        n = len([s for s in out.split() if s.strip() != 'JOBID'])
        print(f'{state}: {n:d}')

    all_data, meta = pe.sdds.sdds2dicts(dot_all_filepath)
    best_data, best_meta = pe.sdds.sdds2dicts(dot_best_filepath)

    keys_formats = [
        ('runID', '{:6d}'), ('Rank', '{:4d}'), ('tLifetime', '{:9.2f}'),
        ('Area1', '{:9.2g}'), ('ChromPenalty', '{:12.3g}'),
        ('nux', '{:10.3f}'), ('nuy', '{:10.3f}'),
        ('dnux/dp', '{:10.3f}'), ('dnuy/dp', '{:10.3f}'),
        ('deltaLimitLT', '{:6.2g}')]
    fmt_widths = [int(float(fmt[2:-2])) for _, fmt in keys_formats]
    print(' | '.join([f'{{:^{w}}}'.format(k) for (k, _), w in
                      zip(keys_formats, fmt_widths)]))
    print(' | '.join([f'{{:^{w}}}'.format(best_meta['columns'][k]['UNITS'])
                      for (k, _), w in zip(keys_formats, fmt_widths)]))
    vals = zip(*[best_data['columns'][k] for k, _ in keys_formats])
    for row in vals:
        print(' | '.join([fmt.format(v) for v, (_, fmt) in zip(row, keys_formats)]))


    all_col = all_data['columns']
    best_col = best_data['columns']

    sort_data, _ = pe.sdds.sdds2dicts(dot_sort_filepath)
    sort_col = sort_data['columns']

    t0 = np.min(sort_col['Time'])
    dt = sort_col['Time'] - t0
    dt_hr = dt / (60 * 60)

    label_size = 16

    tz = datetime.datetime.now().astimezone().tzinfo

    plt.figure()
    plt.subplot(211)
    plt.plot(dt_hr, sort_col['Rank'], 'b.')
    rank1s = (sort_col['Rank'] == 1)
    plt.plot(dt_hr[rank1s], sort_col['Rank'][rank1s], 'r*')
    plt.xlabel(r'$\Delta t \mathrm{{ [hr] from {}}}$'.format(datetime.datetime.fromtimestamp(
        t0).strftime('%H:%M:%S on %m/%d/%Y')).replace(' ', '\,'), size=label_size)
    plt.ylabel(r'$\mathrm{{Rank}}$', size=label_size)
    plt.subplot(212)
    plt.title(r'$\mathrm{{{} ({})}}$'.format(datetime.datetime.fromtimestamp(
        time.time()).strftime('%H:%M:%S on %m/%d/%Y'), tz).replace(' ', '\,'), size=label_size)
    plt.plot(all_col['Area1'], all_col['tLifetime'], 'b.')
    plt.plot(best_col['Area1'], best_col['tLifetime'], 'r*')
    plt.grid(True, ls='--')
    for runID, x, y in zip(best_col['runID'], best_col['Area1'], best_col['tLifetime']):
        plt.annotate(runID, xy=(x, y))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.xlabel(
        r'$\mathrm{{Area1 [{}]}}$'.format(best_meta['columns']['Area1']['UNITS']).replace(' ', '\,'),
        size=label_size)
    plt.ylabel(
        r'$\mathrm{{tLifetime [{}]}}$'.format(best_meta['columns']['tLifetime']['UNITS']).replace(' ', '\,'),
        size=label_size)
    plt.tight_layout()

    plt.show()



