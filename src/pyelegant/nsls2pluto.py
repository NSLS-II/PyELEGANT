import sys
import os
import time
import datetime
from copy import deepcopy
from subprocess import Popen, PIPE, STDOUT
import shlex
import tempfile
import glob
from pathlib import Path
import json
import re
from collections import defaultdict
import gzip
import pickle

import numpy as np
import dill
from ruamel import yaml

from . import facility_name
from .local import sbatch_std_print_enabled
from . import util

_IMPORT_TIMESTAMP = time.time()

_SLURM_CONFIG_FILEPATH = Path.home().joinpath('.pyelegant', 'slurm_config.yaml')
_SLURM_CONFIG_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
if not _SLURM_CONFIG_FILEPATH.exists():
    _SLURM_CONFIG_FILEPATH.write_text(
        '''\
exclude: []
abs_time_limit: {}
'''
    )
#
SLURM_PARTITIONS = {}
SLURM_EXCL_NODES = None
SLURM_ABS_TIME_LIMIT = {}

_MISC_CONFIG_FILEPATH = Path.home().joinpath('.pyelegant', 'misc.yaml')
if not _MISC_CONFIG_FILEPATH.exists():
    _MISC_CONFIG_FILEPATH.write_text(
        '''\
functions:
  wait_for_completion:
    ErrLogAppearCheck:
      nMaxTry: 10
      interval: 1.0
'''
    )
#
MISC_CONFIG = {}

p = Popen(shlex.split('which elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
out, err = p.communicate()
if out.strip():
    path_tokens = out.split('/')
    if 'elegant' in path_tokens:
        __elegant_version__ = path_tokens[path_tokens.index('elegant') + 1]
    else:
        __elegant_version__ = 'unknown'
    del path_tokens
else:
    print('\n*** pyelegant:WARNING: ELEGANT not available.')
    __elegant_version__ = None
del p, out, err

if facility_name == 'nsls2apcluster':
    DEFAULT_REMOTE_OPTS = dict(
        sbatch={'use': False, 'wait': True},
        pelegant=False,
        # -------------
        # SLURM options
        job_name='job',
        output='job.%J.out',
        error='job.%J.err',
        partition='normal',
        ntasks=1,
        time=None,
        nodelist=None,
        exclude=None,
        # ---------------------------------
        # SBATCH error check decistion tree
        sbatch_err_check_tree=[
            ['exists', ('semaphore_file',), {}],
            [['not_empty', ('%s.newlte',), {}], 'no_error', 'retry'],
            [
                ['check_slurm_err_log', ('slurm_err_filepath', 'abort_info'), {}],
                'retry',
                'abort',
            ],
        ],
    )
elif facility_name == 'nsls2pluto':
    DEFAULT_REMOTE_OPTS = dict(
        sbatch={'use': False, 'wait': True},
        pelegant=False,
        # -------------
        # SLURM options
        job_name='job',
        output='job.%J.out',
        error='job.%J.err',
        partition='normal',
        qos='normal',
        ntasks=1,
        time=None,
        nodelist=None,
        exclude=None,
    )
else:
    raise ValueError(f'Invalid facility_name: {facility_name}')


def clear_slurm_excl_nodes():
    """"""

    global SLURM_EXCL_NODES

    if SLURM_EXCL_NODES is None:
        SLURM_EXCL_NODES = []
    else:
        SLURM_EXCL_NODES.clear()


def set_slurm_excl_nodes(node_name_list):
    """
    Examples:

    SLURM_EXCL_NODES = ['apcpu-005',]
    SLURM_EXCL_NODES = ['cpu-019', 'cpu-020', 'cpu-021',]
    SLURM_EXCL_NODES = [f'cpu-{n:03d}' for n in range(19, 26+1)] # exclude GPFS nodes
    SLURM_EXCL_NODES = [f'cpu-{n:03d}' for n in
                        list(range(2, 5+1)) + list(range(7, 15+1))] # exclude NFS nodes
    SLURM_EXCL_NODES = [
        f'cpu-{n:03d}' for n in list(range(19, 26+1)) + list(range(2, 5+1)) +
        list(range(7, 15+1))] # exclude both GPFS & NFS nodes
    SLURM_EXCL_NODES = [f'apcpu-{n:03d}' for n in range(1, 5+1)] + [
        f'cpu-{n:03d}' for n in list(range(2, 5+1)) +
        list(range(7, 15+1))] # exclude both apcpu & NFS nodes, i.e., including only GPFS nodes
    """

    clear_slurm_excl_nodes()
    SLURM_EXCL_NODES.extend(node_name_list)


def load_slurm_excl_nodes_from_config_file(force=False):
    """"""

    if (not force) and (
        not util.is_file_updated(_SLURM_CONFIG_FILEPATH, _IMPORT_TIMESTAMP)
    ):
        return

    yml = yaml.YAML()
    config = yml.load(_SLURM_CONFIG_FILEPATH.read_text())

    exclude_list = config.get('exclude', None)
    if exclude_list is not None:
        set_slurm_excl_nodes(list(exclude_list))


def _make_sure_slurm_excl_nodes_initialized():
    """"""

    global SLURM_EXCL_NODES

    if SLURM_EXCL_NODES is None:
        SLURM_EXCL_NODES = []

        load_slurm_excl_nodes_from_config_file(force=True)


def _get_slurm_partition_info():
    """"""

    cmd = 'scontrol show partition'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    parsed = {}
    for k, v in re.findall('([\w\d]+)=([^\s]+)', out):
        if k == 'PartitionName':
            d = parsed[v] = {}
        else:
            d[k] = v

    partition_info = parsed

    return partition_info


def _init_SLURM_ABS_TIME_LIMIT():
    """"""

    if not SLURM_PARTITIONS:
        SLURM_PARTITIONS.update(_get_slurm_partition_info())

    for _partition in list(SLURM_PARTITIONS):
        SLURM_ABS_TIME_LIMIT[_partition] = None


def load_misc_from_config_file(force=False):
    """"""

    if (
        (MISC_CONFIG != {})
        and (not force)
        and (not util.is_file_updated(_MISC_CONFIG_FILEPATH, _IMPORT_TIMESTAMP))
    ):
        return

    yml = yaml.YAML()
    config = yml.load(_MISC_CONFIG_FILEPATH.read_text())

    MISC_CONFIG.clear()
    for k, v in config.items():
        MISC_CONFIG[k] = v


def convertLocalTimeStrToSecFromUTCEpoch(time_str, frac_sec=False, time_format=None):
    """"""

    if time_format is None:
        DEF_FILENAME_TIMESTAMP_STR_FORMAT = '%Y-%m-%dT%H-%M-%S'

        time_format = DEF_FILENAME_TIMESTAMP_STR_FORMAT

    if not frac_sec:
        return time.mktime(time.strptime(time_str, time_format))
    else:
        if '.' not in time_str:
            time_str += '.0'
        _sec_str, _frac_str = time_str.split('.')
        t = time.mktime(time.strptime(_sec_str, time_format))
        t += float(_frac_str) / (10 ** len(_frac_str))
        return t


def _convert_slurm_time_duration_str_to_seconds(slurm_time_duration_str):
    """"""

    s_list = slurm_time_duration_str.split(':')
    if len(s_list) == 1:
        s_list = ['00', '00'] + s_list
    elif len(s_list) == 2:
        s_list = ['00'] + s_list
    elif (len(s_list) >= 4) or (len(s_list) == 0):
        raise RuntimeError('Unexpected number of splits')

    if '-' in s_list[0]:
        days_str, hrs_str = s_list[0].split('-')
        s_list[0] = hrs_str

        days_in_secs = int(days_str) * 60.0 * 60.0 * 24.0
    else:
        days_in_secs = 0.0

    d = time.strptime(':'.join(s_list), '%H:%M:%S')

    duration_in_sec = (
        days_in_secs
        + datetime.timedelta(
            hours=d.tm_hour, minutes=d.tm_min, seconds=d.tm_sec
        ).total_seconds()
    )

    return duration_in_sec

def _expand_node_range_pattern(index_str):
    """
    Examples for "index_str":
        '[019-025]'
    """

    tokens = index_str[1:-1].split(',')
    pat_list = []
    for tok in tokens:
        if '-' in tok:
            iStart, iEnd = tok.split('-')
            pat_list.extend([f'{i:03d}' for i in range(
                int(iStart), int(iEnd)+1)])
        else:
            pat_list.append(tok)
    pat = '|'.join(pat_list)

    return pat


def _get_unexpected_prefix(nodes_str):

    try:
        int(nodes_str)
        raise ValueError('All numbers')
    except:
        pass

    last_i = -1
    while True:
        try:
            int(nodes_str[last_i:])
            last_i -= 1
        except:
            last_i += 1
            prefix = nodes_str[:last_i]
            #print(f'Adding unexpected prefix: {prefix}')
            break

    return prefix


def _sinfo_parsing(parsed, partition, state, nodes_str, _unexpected_node_prefixes):

    nodes_tuple = tuple(re.findall('[\w\-]+[\d\-\[\],]+(?<!,)', nodes_str))

    nMaxNodeIndex = 100
    avail_prefixes = ['gpu', 'hpc'] + _unexpected_node_prefixes

    for nodes_str in nodes_tuple:
        if '[' not in nodes_str:
            for prefix in avail_prefixes:
                if nodes_str.startswith(prefix):
                    break
            else:
                #print(f'Unexpected node str: {nodes_str}')
                prefix = _get_unexpected_prefix(nodes_str)
                _unexpected_node_prefixes.append(prefix)
        else:
            prefix = nodes_str.split('[')[0]
            assert prefix in avail_prefixes
        index_str = nodes_str[len(prefix):]
        #print((prefix, index_str))
        if ',' in index_str:
            assert index_str.startswith('[') and index_str.endswith(']')
            pat = _expand_node_range_pattern(index_str)
        elif index_str.startswith('[') and index_str.endswith(']'):
            pat = _expand_node_range_pattern(index_str)
        else:
            pat = index_str
        matched_indexes = re.findall(pat, ','.join(
            [f'{i:03d}' for i in range(nMaxNodeIndex)]))
        #print(matched_indexes)
        node_list = [f'{prefix}{s}' for s in matched_indexes]
        #print(nodes_str)
        #print(prefix, node_list)
        parsed[partition][state].extend(node_list)
        
def get_n_free_cores(partition='normal'):

    # update partition info
    cmd = 'scontrol show partition'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    parsed = {}
    for k, v in re.findall('([\w\d]+)=([^\s]+)', out):
        if k == 'PartitionName':
            d = parsed[v] = {}
        else:
            d[k] = v

    # update sinfo
    cmd = 'sinfo -h -o "%P#%a#%l#%D#%T#%N"'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    _unexpected_node_prefixes = []

    parsed = defaultdict(dict)
    for line in out.strip().split('\n'):
        _partition, avail, tlim, n_nodes, state, nodes_str = line.split('#')
        if _partition.endswith('*'):
            _partition = _partition[:-1]
        if state not in parsed[_partition]:
            parsed[_partition][state] = []

        _sinfo_parsing(parsed, _partition, state, nodes_str, _unexpected_node_prefixes)

    sinfo = parsed
    
    ok_states = ['allocated', 'completing', 'idle', 'mixed']

    non_ok_nodes = defaultdict(list)
    for k, v in sinfo.items():
        for st in list(v):
            if st not in ok_states:
                for node_name in v[st]:
                    non_ok_nodes[node_name].append(st)

    node_names = np.unique([e for k, v in sinfo[partition].items()
                            for e in v
                            if e not in non_ok_nodes])

    cmd = 'scontrol show node'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    parsed = re.findall(
        'NodeName=([\w\d\-]+)\s+[\w=\s]+CPUAlloc=(\d+)\s+CPUTot=(\d+)\s+CPULoad=([\d\.N/A]+)',
        out)

    _nAlloc = _nTot = 0
    for node_name, nAlloc, nTot, cpu_load in parsed:
        if node_name not in node_names:
            continue
        _nAlloc += int(nAlloc)
        _nTot += int(nTot)
    n_free = _nTot - _nAlloc

    return n_free


def get_avail_adjusted_ncores(ncores_max, ncores_min, partition=None):
    
    assert ncores_max >= ncores_min >= 1
    
    if partition is None:
        _kwargs = {}
    else:
        _kwargs = dict(partition=partition)

    for _ in range(3):
        try:
            n_free_cores = get_n_free_cores(**_kwargs)
            break
        except:
            time.sleep(5.0)
    else:
        print(f'* Failed to obtain # of available cores. Will assume max. requested cores ({ncores_max}) available.')
        n_free_cores = ncores_max
        
    if n_free_cores < ncores_max:        
        msg = f'Only {n_free_cores} cores available. '
        if n_free_cores >= ncores_min:
            msg += 'Will use all available cores.'
            n_final = n_free_cores    
        else:
            msg += f'Will use min. requested cores ({ncores_min}).'
            n_final = ncores_min
        print(msg)
    else:
        n_final = ncores_max

    return n_final


def get_cluster_time_limits():
    """"""

    if not SLURM_PARTITIONS:
        SLURM_PARTITIONS.update(_get_slurm_partition_info())

    timelimit_d = {}
    for partition, info_d in SLURM_PARTITIONS.items():
        time_limit_str = info_d['MaxTime']
        if time_limit_str == 'UNLIMITED':
            time_limit_str = '1000-00:00:00'  # set the limit to 1000 days (or any other arbitrarily large number)
        timelimit_d[partition] = _convert_slurm_time_duration_str_to_seconds(
            time_limit_str
        )

    return timelimit_d


def get_constrained_timelimit_str(abs_timelimit, partition):
    """
    "abs_timelimit" must be in the format of '%Y-%m-%dT%H-%M-%S'.
    """

    dt_till_timelimit = (
        convertLocalTimeStrToSecFromUTCEpoch(
            abs_timelimit, time_format='%Y-%m-%dT%H-%M-%S'
        )
        - time.time()
    )

    def_timelimit = get_cluster_time_limits()[partition]

    if dt_till_timelimit > def_timelimit:
        timelimit_str = None
    elif dt_till_timelimit < 0.0:
        raise RuntimeError('It is already past specified absolute time limit.')
    else:
        sec = datetime.timedelta(seconds=dt_till_timelimit)
        dobj = datetime.datetime(1, 1, 1) + sec
        if dobj.day - 1 != 0:
            timelimit_str = '{:d}-'.format(dobj.day - 1)
        else:
            timelimit_str = ''
        timelimit_str += '{:02d}:{:02d}:{:02d}'.format(
            dobj.hour, dobj.minute, dobj.second
        )

    return timelimit_str


SRUN_OPTION_MAP = {
    'p': 'partition',
    'q': 'qos',
    'o': 'output',
    'e': 'error',
    't': 'time',
    'n': 'ntasks',
    'c': 'cpus-per-task',
    'cpus_per_task': 'cpus-per-task',
    'J': 'job-name',
    'job_name': 'job-name',
    'mail_user': 'mail-user',
    'mail_type': 'mail-type',
    'spread_job': 'spread-job',
    'w': 'nodelist',
    'x': 'exclude',
}


def extract_slurm_opts(remote_opts):
    """"""

    slurm_opts = {}

    _make_sure_slurm_excl_nodes_initialized()

    partition = remote_opts.get('partition', None)
    if partition is None:
        partition = remote_opts.get('p', None)
    if partition:
        try:
            abs_timelimit = SLURM_ABS_TIME_LIMIT[partition]
        except:
            _init_SLURM_ABS_TIME_LIMIT()
            abs_timelimit = SLURM_ABS_TIME_LIMIT[partition]

        if abs_timelimit:
            timelimit = get_constrained_timelimit_str(abs_timelimit, partition)
            if timelimit:
                slurm_opts['time'] = f'--time={timelimit}'
                # ^ Obviously this will be overwritten if a user specifies "time".
                #   If an invalid time limit is specified, it should fail.

    need_mail_user = False

    for k, v in remote_opts.items():

        # Retrieve the full option name, if it's a shortcut name or alternative name
        if k in SRUN_OPTION_MAP:
            k = SRUN_OPTION_MAP[k]

        if k in ('output', 'error', 'partition', 'qos', 'time'):

            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, v)

        elif k == 'gres':

            if v is None:
                slurm_opts[k] = ''
            else:
                if v not in ('gpu', 'gpu:2'):
                    raise ValueError('--gres must be "gpu" for 1 GPU or "gpu:2" for 2 GPUs')
                slurm_opts[k] = '--{}={}'.format(k, v)

        elif k in ('ntasks',):

            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={:d}'.format(k, v)

        elif k == 'cpus-per-task':

            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '-c {:d}'.format(v)

        elif k in ('job-name', 'mail-user'):

            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, v)

        elif k == 'mail-type':

            if v is None:
                slurm_opts[k] = ''
            else:
                valid_types = [
                    'BEGIN',
                    'END',
                    'FAIL',
                    'REQUEUE',
                    'ALL',
                    'INVALID_DEPEND',
                    'STAGE_OUT',
                    'TIME_LIMIT',
                    'TIME_LIMIT_90',
                    'TIME_LIMIT_80',
                    'TIME_LIMIT_50',
                ]

                if isinstance(v, str):
                    v = [v]

                if not all([_type in valid_types for _type in v]):
                    raise ValueError(
                        f'Invalid "mail-type" input: {v}. Valid types are {valid_types}'
                    )
                slurm_opts[k] = f'--mail-type={",".join(v)}'

                need_mail_user = True

        elif k == 'exclude':
            if v is None:
                v = SLURM_EXCL_NODES
            elif isinstance(v, str):
                v = list(set([v] + SLURM_EXCL_NODES))
            else:
                v = list(set(v + SLURM_EXCL_NODES))

            if v == []:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, ','.join(v))

        elif k == 'nodelist':
            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, ','.join(v))

        elif k in ('x11', 'spread-job'):
            if v:  # True or False
                slurm_opts[k] = f'--{k}'
            else:
                slurm_opts[k] = ''

        elif k in (
            'sbatch',
            'pelegant',
            'diag_mode',
            'abort_filepath',
            'status_check_interval',
            'del_input_file',
            'del_output_file',
        ):
            pass
        else:
            raise ValueError(f'Unknown slurm option keyword: {k}')

    if need_mail_user:
        if len([s for s in list(slurm_opts) if s == 'mail-user']) != 1:
            raise ValueError(
                '"mail-user" option must be specified when '
                '"mail-type" is given as a list.'
            )

    return slurm_opts


def write_sbatch_shell_file(
    sbatch_sh_filepath, slurm_opts, srun_cmd, nMaxTry=10, sleep=10.0
):
    """"""

    # CRITICAL: The line "#!/bin/bash" must come on the first line, not the second or later.
    contents = ['#!/bin/bash']

    contents += [' ']

    for v in slurm_opts.values():
        if v.strip() == '':
            continue
        contents += ['#SBATCH ' + v]

    contents += [' ']

    # contents += ['env']

    contents += [srun_cmd]

    contents += [' ']

    util.robust_text_file_write(
        sbatch_sh_filepath, '\n'.join(contents), nMaxTry=nMaxTry, sleep=sleep
    )


def check_unable_to_open_mode_w_File_exists(err_log_contents):
    """"""

    if ('unable to open' in err_log_contents) and (
        'in mode w: File exists' in err_log_contents
    ):
        abort = True
        print('\n##### Error: unable to open in mode w: File exists #####')
    else:
        abort = False

    return abort


def check_remote_err_log_exit_code(err_log_contents):
    """"""

    m = re.findall('srun: error:.+Exited with exit code', err_log_contents)

    if len(m) != 0:
        abort = True
        print('\n##### Error #####')
        print(err_log_contents)
    else:
        abort = False

    return abort


def _get_min_err_log_check():
    """"""

    return dict(
        funcs=[check_unable_to_open_mode_w_File_exists, check_remote_err_log_exit_code]
    )


def get_human_friendly_time_duration_str(dt, fmt='.2f'):
    """"""

    template = f'{{val:{fmt}}} [{{unit}}]'

    if dt < 60:
        return template.format(val=dt, unit='s')
    elif dt < 60 * 60:
        return template.format(val=dt / 60, unit='min')
    elif dt < 60 * 60 * 24:
        return template.format(val=dt / 60 / 60, unit='hr')
    else:
        return template.format(val=dt / 60 / 60 / 24, unit='day')


def wait_for_completion(
    job_ID_str,
    status_check_interval,
    timelimit_action=None,
    out_log_check=None,
    err_log_check=None,
):
    """"""

    load_misc_from_config_file(force=False)
    func_opts = MISC_CONFIG['functions'].get('wait_for_completion', {})
    optsErrLogAppearCheck = func_opts.get('ErrLogAppearCheck', {})

    t0 = time.time()

    dt_not_running = 0.0
    not_running_t0 = None

    if timelimit_action is not None:
        assert callable(timelimit_action['abort_func'])
        timelimit_sec = (
            get_cluster_time_limits()[timelimit_action['partition_name']]
            - timelimit_action['margin_sec']
        )
    else:
        timelimit_sec = None

    err_counter = 0

    cmd = (
        f'squeue --noheader --job={job_ID_str} -o "%.{len(job_ID_str)+1}i %.3t %.4C %R"'
    )
    num_cores = float('nan')
    used_nodes = 'unknown'
    err_log = ''
    err_found = False
    if err_log_check is not None:
        err_log_filename = f"{err_log_check['job_name']}.{job_ID_str}.err"
        prev_err_log_file_size = 0
    while True:
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        out, err = out.strip(), err.strip()

        if err:
            err_counter += 1

            if err == 'slurm_load_jobs error: Invalid job id specified':
                # The job is finished.
                break

            if err_counter >= 10:
                print(err)
                raise RuntimeError('Encountered error while waiting for job completion')
            else:
                print(err)
                sys.stdout.flush()
                time.sleep(10.0)
                continue

        else:
            err_counter = 0

        if out == '':
            # The job is finished.
            break

        L = out.splitlines()
        assert len(L) == 1
        tokens = L[0].split()
        state = tokens[1]
        num_cores = int(tokens[2])
        used_nodes = ' '.join(tokens[3:])

        if state == 'R':
            if not_running_t0 is not None:
                dt_not_running += time.time() - not_running_t0
                not_running_t0 = None
        else:
            if not_running_t0 is None:
                not_running_t0 = time.time()

        if timelimit_sec is not None:

            dt = time.time() - t0

            if dt >= timelimit_sec:

                print(
                    'Time limit of selected partition is nearing, '
                    'so performing a graceful abort task now.'
                )
                timelimit_action['abort_func'](
                    *timelimit_action.get('abort_func_args', ())
                )

                break

        if (state == 'R') and (err_log_check is not None):

            # Check if err log file size has increased. If not, there is no
            # need to check the contents.
            # print(f'Current directory is "{os.getcwd()}"')
            _err_log_file_missing = False
            for _ in range(optsErrLogAppearCheck.get('nMaxTry', 3)):
                try:
                    curr_err_log_file_size = os.stat(err_log_filename).st_size
                    if _err_log_file_missing:
                        print(f'File "{err_log_filename}" has appeared.')
                    break
                except FileNotFoundError:
                    print(f'Waiting for the file "{err_log_filename}" to appear...')
                    _err_log_file_missing = True
                    if False:
                        time.sleep(10.0)
                    else:
                        time.sleep(optsErrLogAppearCheck.get('interval', 1.0))
                        ls_cmd = f'ls {Path(err_log_filename).resolve().parent} > /dev/null 2>&1'
                        print('$ ' + ls_cmd)
                        _p = Popen(ls_cmd, shell=True)
                        # ^ Needed for NFS cache flushing (https://stackoverflow.com/questions/3112546/os-path-exists-lies)
                        # If the code is run on GPFS or Lustre, this caching problem
                        # does not occur.
                        _tStart = time.time()
                        _p.communicate()
                        print(f'ls took {time.time()-_tStart:.3f} [s]')
            else:
                raise FileNotFoundError(err_log_filename)
            if curr_err_log_file_size > prev_err_log_file_size:

                prev_err_log_file_size = curr_err_log_file_size

                err_log = Path(err_log_filename).read_text()
                # print(err_log)

                err_found = False
                for _check_func in err_log_check['funcs']:
                    # print(_check_func)
                    # print(_check_func(err_log))
                    # sys.stdout.flush()
                    if _check_func(err_log):

                        err_found = True

                        # Cancel the job
                        cmd = f'scancel {job_ID_str}'
                        p = Popen(
                            shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8'
                        )
                        out, err = p.communicate()
                        if err:
                            print(f'Tried cancelling Job {job_ID_str}')
                            print(f'\n*** stderr: command: {cmd}')
                            print(err)

                        break

                if err_found:
                    break

        time.sleep(status_check_interval)

    last_timestamp = time.time()
    dt_total = last_timestamp - t0
    if not_running_t0 is not None:
        dt_not_running += last_timestamp - not_running_t0
    dt_running = dt_total - dt_not_running

    ret = dict(
        total=dt_total,
        running=dt_running,
        nodes=used_nodes,
        ncores=num_cores,
        err_log=err_log,
        err_found=err_found,
    )

    return ret


def _sbatch(
    sbatch_sh_filepath,
    job_name,
    exit_right_after_submission=False,
    status_check_interval=5.0,
    err_log_check=None,
    print_stdout=True,
    print_stderr=True,
    nMaxTry=3,
):
    """"""

    # Re-try functionality in case of
    #   "sbatch: error: Slurm controller not responding, sleeping and retrying."
    for iTry in range(nMaxTry):
        p = Popen(
            shlex.split(f'sbatch {sbatch_sh_filepath}'),
            stdout=PIPE,
            stderr=PIPE,
            encoding='utf-8',
        )
        # ^ In order for this to properly work when launched from Wing IDE, the
        #   following $PATH modification (or similar, depending on the path to
        #   the conda env you want to use) is needed in its Project Properties:
        #     PATH=/nsls2/users/yhidaka/.conda/envs/py38tf/bin:$PATH
        #   Otheriwse, the srun command will not pick up the correct conda env's
        #   Python executable.
        out, err = p.communicate()
        if print_stdout:
            print('\n' + out.strip())
        if err and print_stderr:
            print('\n*** stderr ***')
            print(err)
            print('\n** Encountered error during main job submission.')
            if iTry != nMaxTry - 1:
                print('Will retry sbatch.\n')
                sys.stdout.flush()
                time.sleep(20.0)
            else:
                raise RuntimeError('Encountered error during main job submission')
        else:
            break

    job_ID_str = out.replace('Submitted batch job', '').strip()

    if print_stdout or print_stderr:
        sys.stdout.flush()

    sbatch_info = None
    if not exit_right_after_submission:

        _min_err_log_check = _get_min_err_log_check()
        if err_log_check is None:
            err_log_check = _min_err_log_check
        else:
            for _func in _min_err_log_check['funcs']:
                if _func not in err_log_check['funcs']:
                    err_log_check['funcs'].append(_func)

        err_log_check['job_name'] = job_name

        sbatch_info = wait_for_completion(
            job_ID_str, status_check_interval, err_log_check=err_log_check
        )

        if print_stdout:
            h_dt_total = get_human_friendly_time_duration_str(
                sbatch_info['total'], fmt='.2f'
            )
            h_dt_running = get_human_friendly_time_duration_str(
                sbatch_info['running'], fmt='.2f'
            )
            print(f'Elapsed: Total = {h_dt_total}; Running = {h_dt_running}')

    slurm_out_filename = '{job_name}.{job_ID_str}.out'.format(
        job_name=job_name, job_ID_str=job_ID_str
    )
    slurm_err_filename = '{job_name}.{job_ID_str}.err'.format(
        job_name=job_name, job_ID_str=job_ID_str
    )

    return job_ID_str, slurm_out_filename, slurm_err_filename, sbatch_info


def run(
    remote_opts,
    ele_filepath,
    macros=None,
    print_cmd=False,
    print_stdout=True,
    print_stderr=True,
    tee_to=None,
    tee_stderr=True,
    output_filepaths=None,
    err_log_check=None,
):
    """"""

    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    slurm_opts = extract_slurm_opts(remote_opts)

    output = None

    if ('sbatch' in remote_opts) and remote_opts['sbatch']['use']:

        if 'job-name' not in slurm_opts:
            print(
                '* Using `sbatch` requires "job-name" option to be specified. Using default.'
            )
            slurm_opts['job-name'] = '--job-name={}'.format(
                DEFAULT_REMOTE_OPTS['job_name']
            )

        # Make sure output/error log filenames conform to expectations.
        # If "err_log_check" is not None, it will try to look for the ".err"
        # file, so "--error" must be defined here.
        job_name = slurm_opts['job-name'].split('=')[-1]
        slurm_opts['output'] = f'--output={job_name}.%J.out'
        slurm_opts['error'] = f'--error={job_name}.%J.err'

        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpSbatch_', suffix='.sh'
        )
        sbatch_sh_filepath = os.path.abspath(tmp.name)

        if macros is None:
            macro_str = ''
        else:
            macro_str_list = []
            for k, v in macros.items():
                macro_str_list.append('='.join([k, v]))
            macro_str = '-macro=' + ','.join(macro_str_list)

        if remote_opts['pelegant']:
            srun_cmd = 'srun --mpi=pmi2 Pelegant {} {}'.format(ele_filepath, macro_str)
        else:
            srun_cmd = 'srun elegant {} {}'.format(ele_filepath, macro_str)

        write_sbatch_shell_file(
            sbatch_sh_filepath,
            slurm_opts,
            srun_cmd,  # ele_filepath, remote_opts['pelegant'], macros=macros,
            nMaxTry=10,
            sleep=10.0,
        )

        if 'abort_filepath' in remote_opts:
            abort_info = dict(
                filepath=remote_opts['abort_filepath'], ref_timestamp=time.time()
            )
        else:
            abort_info = None

        nMaxReTry = 5
        for _ in range(nMaxReTry):

            if (abort_info is not None) and util.is_file_updated(
                abort_info['filepath'], abort_info['ref_timestamp']
            ):
                print('\n\n*** Immediate abort requested. Aborting now.')
                raise RuntimeError('Abort requested.')

            wait_after_sbatch = remote_opts['sbatch'].get('wait', True)

            (job_ID_str, slurm_out_filepath, slurm_err_filepath, sbatch_info) = _sbatch(
                sbatch_sh_filepath,
                job_name,
                exit_right_after_submission=(not wait_after_sbatch),
                err_log_check=err_log_check,
                print_stdout=sbatch_std_print_enabled['out'],
                print_stderr=sbatch_std_print_enabled['err'],
            )

            if not wait_after_sbatch:
                output = dict(
                    sbatch_sh_filepath=sbatch_sh_filepath,
                    job_ID_str=job_ID_str,
                    slurm_out_filepath=slurm_out_filepath,
                    slurm_err_filepath=slurm_err_filepath,
                )
                return output
            else:
                output = sbatch_info
                if not output['err_found']:
                    break
        else:
            print('*** ERROR: Max number of sbatch runs exceeded.')
            return output

        if not remote_opts.get('diag_mode', False):
            try:
                os.remove(sbatch_sh_filepath)
            except IOError:
                print(
                    f'* Failed to delete temporary sbatch shell file "{sbatch_sh_filepath}"'
                )

            for fp in [slurm_out_filepath, slurm_err_filepath]:
                try:
                    os.remove(fp)
                except IOError:
                    print(f'* Failed to delete SLURM file "{fp}"')

    else:
        # "Elegant" module must be already loaded.

        cmd_list = ['srun'] + [v for v in slurm_opts.values() if v != '']

        if remote_opts['pelegant']:
            cmd_list += ['--mpi=pmi2', 'Pelegant', ele_filepath]
        else:
            cmd_list += ['elegant', ele_filepath]

        if macros is not None:
            macro_str_list = []
            for k, v in macros.items():
                macro_str_list.append('='.join([k, v]))
            cmd_list.append('-macro=' + ','.join(macro_str_list))

        if tee_to is None:
            if print_cmd:
                print('$ ' + ' '.join(cmd_list))
            p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
        else:
            if tee_stderr:
                p1 = Popen(cmd_list, stdout=PIPE, stderr=STDOUT)
            else:
                p1 = Popen(cmd_list, stdout=PIPE, stderr=PIPE)

            if isinstance(tee_to, str):
                cmd_list_2 = ['tee', tee_to]
            else:
                cmd_list_2 = ['tee'] + list(tee_to)

            if print_cmd:
                if tee_stderr:
                    equiv_cmd_connection = ['2>&1', '|']
                else:
                    equiv_cmd_connection = ['|']
                print('$ ' + ' '.join(cmd_list + equiv_cmd_connection + cmd_list_2))

            p = Popen(
                cmd_list_2, stdin=p1.stdout, stdout=PIPE, stderr=PIPE, encoding='utf-8'
            )
        out, err = p.communicate()

        if out and print_stdout:
            print(out)

        if err and print_stderr:
            print('stderr:')
            print(err)

    return output


def gen_mpi_submit_script(remote_opts):
    """"""

    slurm_opts = extract_slurm_opts(remote_opts)

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=False, prefix='tmpInput_', suffix='.pgz'
    )
    input_filepath = os.path.abspath(tmp.name)
    output_filepath = os.path.abspath(tmp.name.replace('tmpInput_', 'tmpOutput_'))
    mpi_sub_sh_filepath = os.path.abspath(
        tmp.name.replace('tmpInput_', 'tmpMpiSub_').replace('.pgz', '.sh')
    )

    # MPI_COMPILER_OPT_STR = '--mpi=pmi2'
    MPI_COMPILER_OPT_STR = ''

    main_script_path = f'{__file__[:-3]}_mpi_script.py'
    srun_cmd_list = [
        'srun',
        MPI_COMPILER_OPT_STR,
        'python -m mpi4py.futures',
        main_script_path,
        '_mpi_starmap',
        input_filepath,
    ]
    srun_cmd = ' '.join([s for s in srun_cmd_list if s.strip() != ''])

    if slurm_opts.get('x11', False):
        srun_cmd = 'export MPLBACKEND="agg"\n' + srun_cmd

    write_sbatch_shell_file(mpi_sub_sh_filepath, slurm_opts, srun_cmd)

    return input_filepath, output_filepath, mpi_sub_sh_filepath

    
def run_mpi_python(
    remote_opts,
    module_name,
    func_name,
    param_list,
    args,
    paths_to_prepend=None,
    err_log_check=None,
    ret_slurm_info=False,
    print_stdout=True,
    print_stderr=True,
):
    """
    Example:
        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
    """

    d = dict(
        module_name=module_name, func_name=func_name, param_list=param_list, args=args
    )

    _min_err_log_check = _get_min_err_log_check()
    if err_log_check is None:
        err_log_check = _min_err_log_check
    else:
        for _func in _min_err_log_check['funcs']:
            if _func not in err_log_check['funcs']:
                err_log_check['funcs'].append(_func)

    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    if ('job_name' not in remote_opts) and ('job-name' not in remote_opts):
        remote_opts['job_name'] = DEFAULT_REMOTE_OPTS['job_name']
    job_name = remote_opts['job_name']
    #
    # Must define these, as "err_log_check" will need to read the ".err" file.
    remote_opts['output'] = f'{job_name}.%J.out'
    remote_opts['error'] = f'{job_name}.%J.err'
    #
    remote_opts['partition'] = remote_opts.get('partition', 'normal')
    remote_opts['ntasks'] = remote_opts.get('ntasks', 50)

    input_filepath, output_filepath, mpi_sub_sh_filepath = gen_mpi_submit_script(
        remote_opts
    )

    d['output_filepath'] = output_filepath

    with open(input_filepath, 'wb') as f:

        if paths_to_prepend is None:
            paths_to_prepend = []
        dill.dump(paths_to_prepend, f, protocol=-1)

        dill.dump(d, f, protocol=-1)

    # Add re-try functionality in case of "sbatch: error: Slurm controller not responding
    # , sleeping and retrying."
    job_ID_str, slurm_out_filepath, slurm_err_filepath, sbatch_info = _sbatch(
        mpi_sub_sh_filepath,
        job_name,
        exit_right_after_submission=False,
        status_check_interval=5.0,
        err_log_check=err_log_check,
        print_stdout=print_stdout,
        print_stderr=print_stderr,
        nMaxTry=3,
    )

    if not sbatch_info['err_found']:
        results = util.load_pgz_file(output_filepath)
    else:
        results = sbatch_info['err_log']

    if remote_opts.get('del_input_file', True):
        try:
            os.remove(input_filepath)
        except:
            pass
    if remote_opts.get('del_sub_sh_file', True):
        try:
            os.remove(mpi_sub_sh_filepath)
        except:
            pass
    if remote_opts.get('del_output_file', True):
        try:
            os.remove(output_filepath)
        except:
            pass
    if remote_opts.get('del_job_out_file', True):
        try:
            os.remove(slurm_out_filepath)
        except:
            pass
    if remote_opts.get('del_job_err_file', True):
        try:
            os.remove(slurm_err_filepath)
        except:
            pass

    if not ret_slurm_info:
        return results
    else:
        return results, sbatch_info


def monitor_simplex_log_progress(job_ID_str, simplex_log_prefix, optimizer_remote_opts):
    """"""

    ntasks = optimizer_remote_opts['ntasks']

    optimizer_slurm_opts = extract_slurm_opts(optimizer_remote_opts)
    monitor_slurm_opts = {}
    for k in ['partition', 'qos', 'time']:
        if k in optimizer_slurm_opts:
            monitor_slurm_opts[k] = optimizer_slurm_opts[k]

    tmp = tempfile.NamedTemporaryFile(
        dir=os.getcwd(), delete=True, prefix='tmpSbatchSddsplot_'
    )
    job_name = os.path.basename(tmp.name)
    tmp.close()

    # Wait until all the ".simlog" files appear. If the sddsplot command is
    # issued before they appear, it will crash.
    for _ in range(10):
        simlog_list = glob.glob(f'{simplex_log_prefix}-*')
        if len(simlog_list) == ntasks:
            break
        else:
            time.sleep(5.0)

    srun_cmd_prefix = f'srun --x11 --job-name={job_name}'
    for k, v in monitor_slurm_opts.items():
        if v is None:
            continue
        srun_cmd_prefix += f' {v}'
    srun_cmd_prefix += ' sddsplot "-device=motif,-movie true -keep 1"'
    srun_cmd_prefix += ' -repeat -column=Step,best*'
    srun_cmd_suffix = ' -graph=line,vary -mode=y=autolog'

    # use_shell = True
    use_shell = False

    if use_shell:
        if False:
            srun_cmd = (
                f'srun --x11 --job-name={job_name} sddsplot "-device=motif,-movie true -keep 1" '
                f'-repeat -column=Step,best* {simplex_log_prefix}-* -graph=line,vary '
                '-mode=y=autolog'
            )
        else:
            srun_cmd_mid = f' {simplex_log_prefix}-*'
            srun_cmd = srun_cmd_prefix + srun_cmd_mid + srun_cmd_suffix

        # Launch sddsplot
        p_sddsplot = Popen(
            srun_cmd, shell=True, stdout=PIPE, stderr=PIPE, encoding='utf-8'
        )
    else:
        # If we're not going to use "shell", then the wildcard * must be expanded
        # by the glob module for the list of file names.
        if False:
            srun_cmd = (
                f'srun --x11 --job-name={job_name} sddsplot "-device=motif,-movie true -keep 1" '
                f'-repeat -column=Step,best* {" ".join(simlog_list)} -graph=line,vary '
                '-mode=y=autolog'
            )
        else:
            srun_cmd_mid = f' {" ".join(simlog_list)}'
            srun_cmd = srun_cmd_prefix + srun_cmd_mid + srun_cmd_suffix

        # Launch sddsplot
        p_sddsplot = Popen(
            shlex.split(srun_cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8'
        )

    # Get the launched job ID
    for _ in range(10):
        p = Popen(
            shlex.split('squeue -o "%.12i %.50j"'),
            stdout=PIPE,
            stderr=PIPE,
            encoding='utf-8',
        )
        out, err = p.communicate()

        if err:
            err_counter += 1

            if err_counter >= 10:
                print(err)
                raise RuntimeError('Encountered error while waiting for job completion')
            else:
                print(err)
                sys.stdout.flush()
                time.sleep(10.0)
                continue

        else:
            err_counter = 0

        job_ID_res_list = [L.split() for L in out.split('\n')[1:] if L.strip() != '']
        job_ID_list, job_name_list = zip(*job_ID_res_list)

        if job_name not in job_name_list:
            time.sleep(5.0)
            continue

        index = job_name_list.index(job_name)
        sddsplot_job_ID_str = job_ID_list[index]

        break

    if p_sddsplot.poll() is not None:
        out, err = p_sddsplot.communicate()
        print(f'$ {srun_cmd}')
        print('** stdout **')
        print(out)
        print('** stderr **')
        print(err)
        raise RuntimeError('The process for sddsplot appeared to have failed.')

    status_check_interval = 5.0
    wait_for_completion(job_ID_str, status_check_interval)

    # Kill the sddsplot job
    try:
        p = Popen(
            shlex.split(f'scancel {sddsplot_job_ID_str}'),
            stdout=PIPE,
            stderr=PIPE,
            encoding='utf-8',
        )
        out, err = p.communicate()
    except:
        pass


def start_hybrid_simplex_optimizer(
    elebuilder_obj, remote_opts, show_progress_plot=True
):
    """"""

    ed = elebuilder_obj

    req_remote_opts = dict(
        sbatch={'use': True},
        pelegant=True,
    )
    for k, v in req_remote_opts.items():
        if k in remote_opts:
            print(f'WARNING: `remote_opts["{k}"]` will be ignored.')
        remote_opts[k] = v
    default_remote_opts = dict(job_name='hybrid_simplex', ntasks=20)
    for k, v in default_remote_opts.items():
        if k not in remote_opts:
            remote_opts[k] = v

    if (
        not show_progress_plot
    ):  # If you don't care to see the progress of the optimization

        remote_opts['sbatch']['wait'] = True

        # Run Pelegant
        run(remote_opts, ed.ele_filepath)
        # ^ This will block until the optimization is completed.

    else:  # If you want to see the progress of the optimization

        _fp_list = [
            _v for _v in ed.actual_output_filepath_list if _v.endswith('.simlog')
        ]
        if len(_fp_list) == 0:
            raise RuntimeError(
                'simplex_log is NOT specified. Cannot monitor simplex progress.'
            )
        elif len(_fp_list) == 1:
            simlog_filepath = _fp_list[0]
        else:
            raise RuntimeError('More than one simplex_log file found.')

        remote_opts['sbatch']['wait'] = False

        # Run Pelegant
        job_info = run(remote_opts, ed.ele_filepath)

        # Start plotting simplex optimization progress
        monitor_simplex_log_progress(
            job_info['job_ID_str'], simlog_filepath, remote_opts
        )

        try:
            os.remove(job_info['sbatch_sh_filepath'])
        except IOError:
            print(
                '* Failed to delete temporary sbatch shell file "{}"'.format(
                    job_info['sbatch_sh_filepath']
                )
            )


def _tmp_glob(pattern, sort_order='mtime'):
    """
    This function is NOT meant to be directly run. The contents of this function
    is inspected and copied into a temporary Python file, which will be
    executed on a worker node.
    """

    import os
    import datetime
    from pathlib import Path

    if sort_order == 'mtime':
        sort_key = os.path.getmtime
    elif sort_order == 'size':
        sort_key = os.path.getsize
    else:
        raise ValueError(f'Invalid "sort_order": {sort_order}')

    for f in sorted(Path('/tmp').glob(pattern), key=sort_key):
        stat = f.stat()
        if stat.st_size < int(1024):
            size_str = f'{stat.st_size:>4.0f}'
        elif stat.st_size < int(1024 ** 2):
            size = stat.st_size / 1024
            size_str = f'{size:>3.0f}K'
        elif stat.st_size < int(1024 ** 3):
            size = stat.st_size / 1024 / 1024
            size_str = f'{size:>3.0f}M'
        elif stat.st_size < int(1024 ** 4):
            size = stat.st_size / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}G'
        elif stat.st_size < int(1024 ** 5):
            size = stat.st_size / 1024 / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}T'
        else:
            size = stat.st_size / 1024 / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}T'

        time_str = datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
            '%Y-%m-%d %H:%M:%S'
        )

        try:
            group = f.group()
        except KeyError:
            group = f.stat().st_gid

        print(f'{f.owner()} {group} {size_str} {time_str} {f.name}')


def tmp_glob(node_name, pattern, sort_order='mtime'):
    """"""

    tmp = tempfile.NamedTemporaryFile(
        dir=Path.cwd(), delete=False, prefix='tmpGlob_', suffix='.py'
    )
    temp_py = Path(tmp.name)
    tmp.close()

    import inspect

    func_lines = inspect.getsource(_tmp_glob).split('\n')
    for i, line in enumerate(func_lines):
        if line.strip().startswith(('import ', 'from ')):
            func_body_start_index = i
            break

    py_contents = [
        'if __name__ == "__main__":',
        f'    pattern = "{pattern}"',
        f'    sort_order = "{sort_order}"',
    ] + func_lines[func_body_start_index:]

    temp_py.write_text('\n'.join(py_contents))

    cmd = (
        f'srun --nodelist={node_name} --partition=normal --qos=debug '
        f'--time=0:30 python {str(temp_py)}'
    )
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out.strip())

    if err.strip():
        print('\n*** stderr ***')
        print(err.strip())

    # Remove temp Python file
    temp_py.unlink()


def _tmp_rm(pattern):
    """
    This function is NOT meant to be directly run. The contents of this function
    is inspected and copied into a temporary Python file, which will be
    executed on a worker node.
    """

    from pathlib import Path
    import shutil

    for f in sorted(Path('/tmp').glob(pattern)):
        try:
            if not f.is_dir():
                f.unlink()
            else:
                shutil.rmtree(f)
        except:
            pass


def tmp_rm(node_name, pattern):
    """
    Remove files/directories in /tmp on specified node.
    """

    tmp = tempfile.NamedTemporaryFile(
        dir=Path.cwd(), delete=False, prefix='tmpRm_', suffix='.py'
    )
    temp_py = Path(tmp.name)
    tmp.close()

    import inspect

    func_lines = inspect.getsource(_tmp_rm).split('\n')
    for i, line in enumerate(func_lines):
        if line.strip().startswith(('import ', 'from ')):
            func_body_start_index = i
            break

    py_contents = [
        'if __name__ == "__main__":',
        f'    pattern = "{pattern}"',
    ] + func_lines[func_body_start_index:]

    temp_py.write_text('\n'.join(py_contents))

    cmd = (
        f'srun --nodelist={node_name} --partition=normal --qos=debug '
        f'--time=0:30 python {str(temp_py)}'
    )
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out.strip())

    if err.strip():
        print('\n*** stderr ***')
        print(err.strip())

    # Remove temp Python file
    temp_py.unlink()


def srun_python_func(
    remote_opts, module_name, func_name, func_args, func_kwargs, paths_to_prepend=None
):
    """
    Example:
        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
    """

    d = dict(
        module_name=module_name,
        func_name=func_name,
        func_args=func_args,
        func_kwargs=func_kwargs,
    )

    tmp = tempfile.NamedTemporaryFile(
        dir=Path.cwd(), delete=False, prefix='tmpInput_', suffix='.pgz'
    )
    input_filepath = Path(tmp.name).resolve()
    output_filepath = Path(tmp.name.replace('tmpInput_', 'tmpOutput_')).resolve()
    tmp.close()

    d['output_filepath'] = str(output_filepath)

    slurm_opts = extract_slurm_opts(remote_opts)
    if 'job-name' not in slurm_opts:
        slurm_opts['job-name'] = '--job-name=srun_py_func'
    if 'partition' not in slurm_opts:
        slurm_opts['partition'] = '--partition=normal'
    if 'qos' not in slurm_opts:
        slurm_opts['qos'] = '--qos=normal'
    if 'cpus-per-task' not in slurm_opts:
        slurm_opts['cpus-per-task'] = '--cpus-per-task=1'
        # ^ This is not the case for nsls2pluto, which doesn't use hyper-threading,
        # but if the cluster uses hyper-threading (i.e., 2 logical cores for 1
        # physical core), then don't use "-c 1". If you do, it will still uses 2
        # logical cores, but runs 2 process of the same simultaneously (you can
        # tell from the print statements.) With "-c 2", there will be only one
        # process but can utilize 2 cores.
    if 'x11' in slurm_opts:
        raise NotImplementedError('x11 needs `export MPLBACKEND="agg"`')

    with open(input_filepath, 'wb') as f:

        if paths_to_prepend is None:
            paths_to_prepend = []
        dill.dump(paths_to_prepend, f, protocol=-1)

        dill.dump(d, f, protocol=-1)

    main_script_path = __file__[:-3] + '_srun_py_func_script.py'
    slurm_opts_str = ' '.join([v for v in slurm_opts.values()])
    srun_cmd = f'srun {slurm_opts_str} python {main_script_path} {input_filepath}'
    print(f'$ {srun_cmd}')
    p = Popen(shlex.split(srun_cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out.strip())

    if err.strip():
        print('\n*** stderr ***')
        print(err.strip())

    tFileWait = time.time()
    for _ in range(30):
        if not output_filepath.exists():
            if False:
                p = Popen(
                    # shlex.split(f'ls {output_filepath}'), # <= This will NOT refresh NFS cache
                    shlex.split('ls'),  # <= This WILL refresh NFS cache
                    stdout=PIPE,
                    stderr=PIPE,
                    encoding='utf-8',
                )
                out, err = p.communicate()
                print(out)
                if err.strip():
                    print('## stderr ##')
                    print(err.strip())
            else:
                p = Popen('ls > /dev/null 2>&1', shell=True)
                # ^ Needed for NFS cache flushing (https://stackoverflow.com/questions/3112546/os-path-exists-lies)
                # If the code is run on GPFS or Lustre, this caching problem
                # does not occur.
                p.communicate()
            continue

        print(f'Output file existence wait for {time.time()-tFileWait:.3f} [s]')
        results = util.load_pgz_file(output_filepath)
        break
    else:
        print(f'Waited output file to show up for {time.time()-tFileWait:.3f} [s]')
        raise IOError(f'Expected output file "{output_filepath}" not found.')

    if remote_opts.get('del_input_file', True):
        try:
            input_filepath.unlink()
        except:
            pass
    if remote_opts.get('del_output_file', True):
        try:
            output_filepath.unlink()
        except:
            pass

    return results


def starmap_async(
    remote_opts,
    module_name,
    func_name,
    func_args_iterable,
    paths_to_prepend=None,
    err_log_check=None,
):
    """"""

    _min_err_log_check = _get_min_err_log_check()
    if err_log_check is None:
        err_log_check = _min_err_log_check
    else:
        for _func in _min_err_log_check['funcs']:
            if _func not in err_log_check['funcs']:
                err_log_check['funcs'].append(_func)

    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    if 'abort_filepath' in remote_opts:
        abort_info = dict(
            filepath=remote_opts['abort_filepath'], ref_timestamp=time.time()
        )
    else:
        abort_info = None

    slurm_opts = extract_slurm_opts(remote_opts)
    if 'job-name' not in slurm_opts:
        print(
            '* Using `sbatch` requires "job_name" option to be specified. Using default.'
        )
        slurm_opts['job-name'] = f'--job-name={DEFAULT_REMOTE_OPTS["job_name"]}'
    if 'partition' not in slurm_opts:
        slurm_opts['partition'] = '--partition=normal'
    if 'qos' not in slurm_opts:
        slurm_opts['qos'] = '--qos=normal'
    if 'x11' in slurm_opts:
        raise NotImplementedError('x11 needs `export MPLBACKEND="agg"`')

    # Make sure output/error log filenames conform to expectations
    job_name = slurm_opts['job-name'].split('=')[-1]
    slurm_opts['output'] = f'--output={job_name}.%J.out'
    slurm_opts['error'] = f'--error={job_name}.%J.err'

    err_log_check['job_name'] = job_name

    job_info_list = []
    for func_args in func_args_iterable:

        d = dict(
            module_name=module_name,
            func_name=func_name,
            func_args=func_args,
            func_kwargs={},
        )

        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpInput_', suffix='.pgz'
        )
        input_filepath = os.path.abspath(tmp.name)
        output_filepath = os.path.abspath(tmp.name.replace('tmpInput_', 'tmpOutput_'))
        tmp.close()

        d['output_filepath'] = output_filepath

        with open(input_filepath, 'wb') as f:

            if paths_to_prepend is None:
                paths_to_prepend = []
            dill.dump(paths_to_prepend, f, protocol=-1)

            dill.dump(d, f, protocol=-1)

        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpSbatch_', suffix='.sh'
        )
        sbatch_sh_filepath = os.path.abspath(tmp.name)

        main_script_path = __file__[:-3] + '_srun_py_func_script.py'
        srun_cmd = f'srun python {main_script_path} {input_filepath}'

        write_sbatch_shell_file(
            sbatch_sh_filepath, slurm_opts, srun_cmd, nMaxTry=10, sleep=10.0
        )

        (job_ID_str, slurm_out_filepath, slurm_err_filepath, _) = _sbatch(
            sbatch_sh_filepath, job_name, exit_right_after_submission=True
        )

        job_info = dict(
            input_filepath=input_filepath,
            output_filepath=output_filepath,
            sbatch_sh_filepath=sbatch_sh_filepath,
            job_ID_str=job_ID_str,
            slurm_out_filepath=slurm_out_filepath,
            slurm_err_filepath=slurm_err_filepath,
        )
        job_info_list.append(job_info)

    status_check_interval = 5.0  # 10.0

    results_list = []
    for job_d in job_info_list:

        if (abort_info is not None) and util.is_file_updated(
            abort_info['filepath'], abort_info['ref_timestamp']
        ):
            print('\n\n*** Immediate abort requested. Aborting now.')
            raise RuntimeError('Abort requested.')

        sbatch_info = wait_for_completion(
            job_d['job_ID_str'], status_check_interval, err_log_check=err_log_check
        )

        if not sbatch_info['err_found']:
            results = util.load_pgz_file(job_d['output_filepath'])
        else:
            results = sbatch_info['err_log']
            return results

        results_list.append(results)

        if not remote_opts.get('diag_mode', False):
            if remote_opts.get('del_input_file', True):
                try:
                    os.remove(job_d['input_filepath'])
                except:
                    pass
            if remote_opts.get('del_output_file', True):
                try:
                    os.remove(job_d['output_filepath'])
                except:
                    pass
            if remote_opts.get('del_sub_sh_file', True):
                try:
                    os.remove(job_d['sbatch_sh_filepath'])
                except:
                    pass
            if remote_opts.get('del_job_out_file', True):
                try:
                    os.remove(job_d['slurm_out_filepath'])
                except:
                    pass
            if remote_opts.get('del_job_err_file', True):
                try:
                    os.remove(job_d['slurm_err_filepath'])
                except:
                    pass

    return results_list


def file_exists_w_nfs_cache_flushing(filepath, nMaxTry=5, interval=3.0):
    
    if not isinstance(filepath, Path):
        filepath = Path(filepath)
    
    tFileWait = time.time()
    for _ in range(nMaxTry):
        if not filepath.exists():
            p = Popen('ls > /dev/null 2>&1', shell=True, cwd=filepath.parent)
            # ^ Needed for NFS cache flushing (https://stackoverflow.com/questions/3112546/os-path-exists-lies)
            # If the code is run on GPFS or Lustre, this caching problem
            # does not occur.
            p.communicate()            
            print(f'File "{filepath}" hasn\'t appeared. NFS cache flushing attempted.')
            time.sleep(interval)
        else:
            break        
    else:
        print(f'Waited file "{filepath}" to show up for {time.time()-tFileWait:.3f} [s]')
        raise IOError(f'Expected file "{filepath}" not found.')
    

def get_file_size(filepath):

    file_exists_w_nfs_cache_flushing(filepath)
    return os.stat(filepath).st_size


def _gen_mpi_executor_submit_script(remote_opts, paths=None):
    """"""
    
    slurm_opts = extract_slurm_opts(remote_opts)

    tmp_dir = tempfile.TemporaryDirectory(
        suffix=f'_{datetime.datetime.now():%Y%m%dT%H%M%S}', 
        prefix='tmpMpiExec_', 
        dir=Path.cwd())
    tmp_dirpath = Path(tmp_dir.name)
    
    mpi_sub_sh_filepath = tmp_dirpath.joinpath('submit.sh')
    paths_filepath = tmp_dirpath.joinpath('paths.dill')
    
    with open(paths_filepath, 'wb') as f:
        dill.dump(paths, f)    
    
    # MPI_COMPILER_OPT_STR = '--mpi=pmi2'
    MPI_COMPILER_OPT_STR = ''
    
    main_script_path = f'{__file__[:-3]}_mpi_executor.py'
    srun_cmd_list = [
        'srun',
        MPI_COMPILER_OPT_STR,
        'python -m mpi4py.futures',
        main_script_path,
        f'{paths_filepath}'
    ]
    srun_cmd = ' '.join([s for s in srun_cmd_list if s.strip() != ''])

    if slurm_opts.get('x11', False):
        srun_cmd = 'export MPLBACKEND="agg"\n' + srun_cmd

    write_sbatch_shell_file(mpi_sub_sh_filepath, slurm_opts, srun_cmd)
    
    return tmp_dir, mpi_sub_sh_filepath, paths_filepath


def launch_mpi_python_executor(
    remote_opts, paths=None,
    err_log_check=None,
    print_stdout=True,
    print_stderr=True
    ):
        
    _min_err_log_check = _get_min_err_log_check()
    if err_log_check is None:
        err_log_check = _min_err_log_check
    else:
        for _func in _min_err_log_check['funcs']:
            if _func not in err_log_check['funcs']:
                err_log_check['funcs'].append(_func)
    
    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    if ('job_name' not in remote_opts) and ('job-name' not in remote_opts):
        remote_opts['job_name'] = DEFAULT_REMOTE_OPTS['job_name']
    job_name = remote_opts['job_name']
    #
    # Must define these, as "err_log_check" will need to read the ".err" file.
    remote_opts['output'] = f'{job_name}.%J.out'
    remote_opts['error'] = f'{job_name}.%J.err'
    #
    remote_opts['partition'] = remote_opts.get('partition', 'normal')
    remote_opts['ntasks'] = remote_opts.get('ntasks', 50)
    
    err_log_check['job_name'] = job_name
    
    tmp_dir, mpi_sub_sh_filepath, paths_filepath = _gen_mpi_executor_submit_script(
        remote_opts, paths=paths
    )
        
    # Add re-try functionality in case of "sbatch: error: Slurm controller not responding
    # , sleeping and retrying."
    job_ID_str, slurm_out_filename, slurm_err_filename, sbatch_info = _sbatch(
        mpi_sub_sh_filepath,
        job_name,
        exit_right_after_submission=True,
        print_stdout=print_stdout,
        print_stderr=print_stderr,
        nMaxTry=3,
    )
    
    # Must wait until the requested job starts running
    cmd = (
        f'squeue --noheader --job={job_ID_str} -o "%.{len(job_ID_str)+1}i %.3t %.4C %R"'
    )
    err_counter = 0
    t0 = time.perf_counter()
    while True:        
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        out, err = out.strip(), err.strip()

        if err:
            err_counter += 1
    
            if err == 'slurm_load_jobs error: Invalid job id specified':
                # The job has not been registered yet.
                time.sleep(3.0)
                continue

            if err_counter >= 10:
                print(err)
                msg = 'Encountered error while waiting for MPI executor job startup'
                raise RuntimeError(msg)
            else:
                print(err)
                sys.stdout.flush()
                time.sleep(10.0)
                continue

        else:
            err_counter = 0
            
        L = out.splitlines()
        assert len(L) == 1
        tokens = L[0].split()
        state = tokens[1]
        
        if state == 'R':
            # The job started running. Ready to exit this function.
            dt_pending = time.perf_counter() - t0
            break
        elif state == 'PD':
            time.sleep(5.0)
        else:
            raise RuntimeError(f'Unexpected job state: {state}')
    
    return dict(
        tmp_dir=tmp_dir, err_log_check=err_log_check,
        remote_opts=remote_opts,
        job_ID_str=job_ID_str, 
        slurm_out_filepath=Path(slurm_out_filename),
        slurm_err_filepath=Path(slurm_err_filename),
        dt_pending=dt_pending,
        job_counter=0)

    
def stop_mpi_executor(executor_d, del_log_files=True):
        
    tmp_dir = executor_d['tmp_dir']
    tmp_dirpath = Path(tmp_dir.name)
    
    tmp_dirpath.joinpath('stop_requested').write_text('')

    stopped_fp = tmp_dirpath.joinpath('stopped')

    t0 = time.time()
    while True:
        if stopped_fp.exists():
            break
        else:
            time.sleep(5.0)
            
        if time.time() - t0 >= 5 * 60:
            break

    # Delete the temp folder
    tmp_dir.cleanup() 
    
    if del_log_files:
        # Delete the slurm log files
        for fp in [executor_d['slurm_out_filepath'],
                   executor_d['slurm_err_filepath']]:
            try:
                fp.unlink()
            except:
                pass


def submit_job_to_mpi_executor(executor_d, module_name, func_name, param_list,
    args, check_interval=5.0, print_stdout=True, print_stderr=True,
):
    """
    Example:
        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
    """
    
    input_d = dict(
        module_name=module_name, func_name=func_name, param_list=param_list, args=args
    )
    
    prefix = f'job_{executor_d["job_ID_str"]}_{executor_d["job_counter"]:d}'

    tmp_dir = executor_d['tmp_dir']
    tmp_dirpath = Path(tmp_dir.name)

    input_filepath = tmp_dirpath.joinpath(f'{prefix}_input.dill')
    input_d['output_filepath'] = tmp_dirpath.joinpath(f'{prefix}_output.pgz')
    input_ready_filepath = tmp_dirpath.joinpath(f'{prefix}.ready')

    with open(input_filepath, 'wb') as f:
        dill.dump(input_d, f)

    file_exists_w_nfs_cache_flushing(input_filepath)
    
    input_ready_filepath.write_text('')
    file_exists_w_nfs_cache_flushing(input_ready_filepath)

    # Wait for the job to finish
    output_ready_filepath = _wait_for_mpi_exec_job_completion(
        executor_d, input_d['output_filepath'], check_interval=check_interval,
        print_stdout=print_stdout, print_stderr=print_stderr)
    
    with gzip.GzipFile(input_d['output_filepath'], 'rb') as f:
        results, dt = pickle.load(f)

    if print_stdout:
        h_dt_setup = get_human_friendly_time_duration_str(
            dt['setup'], fmt='.3f'
        )
        h_dt_run = get_human_friendly_time_duration_str(
            dt['run'], fmt='.3f'
        )
        print(f'Elapsed: Setup = {h_dt_setup}; Run = {h_dt_run}')

    if print_stdout or print_stderr:
        sys.stdout.flush()

    for fp in [input_filepath, input_ready_filepath, output_ready_filepath, 
               input_d['output_filepath']]:
        try:
            fp.unlink()
        except:
            pass
    
    executor_d["job_counter"] += 1

    return results, dt
    
    
def _wait_for_mpi_exec_job_completion(
    executor_d, 
    output_filepath, 
    check_interval=5.0, 
    print_stdout=True, 
    print_stderr=True):

    job_ID_str = executor_d['job_ID_str']
    err_log_check = executor_d['err_log_check']
    tmp_dir = executor_d['tmp_dir']

    tmp_dirpath = Path(tmp_dir.name)
    
    output_ready_filename = f'{output_filepath.name}.done'

    sq_cmd = (
        f'squeue --noheader --job={job_ID_str} -o "%.{len(job_ID_str)+1}i %.3t %.4C %R"'
    )
    err_log = ''
    err_found = False
    if err_log_check is not None:
        err_log_filename = f"{err_log_check['job_name']}.{job_ID_str}.err"
        err_log_filepath = Path(err_log_filename)
        prev_err_log_file_size = 0

    finished = False
    err_counter = 0

    while not finished:
        for fp in tmp_dirpath.glob(output_ready_filename):
            finished = True
            output_ready_filepath = fp
            break
        else:
            # Check the status of the job running the executor for any sign of errors
            p = Popen(shlex.split(sq_cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
            out, err = p.communicate()
            out, err = out.strip(), err.strip()
            
            if err:
                err_counter += 1
    
                if err == 'slurm_load_jobs error: Invalid job id specified':
                    msg = 'Error while waiting for job completion: Invalid job id'
                    raise RuntimeError(msg)
    
                if err_counter >= 10:
                    print(err)
                    raise RuntimeError('squeue error while waiting for job completion')
                else:
                    print(err)
                    sys.stdout.flush()
                    time.sleep(10.0)
                    continue
            else: # Reset counter since there was no error.
                err_counter = 0
                
            if out.strip() == '':
                raise RuntimeError('MPI executor job apparently stopped.')

            L = out.splitlines()
            assert len(L) == 1
            tokens = L[0].split()
            state = tokens[1]
            if state != 'R':
                raise RuntimeError(f'MPI executor job NOT in the running state: {state}')
            
            if err_log_check is not None:
                # Check if err log file size has increased. If not, there is no
                # need to check the contents.
                curr_err_log_file_size = get_file_size(err_log_filepath)
                
                if curr_err_log_file_size > prev_err_log_file_size:
    
                    prev_err_log_file_size = curr_err_log_file_size
    
                    err_log = err_log_filepath.read_text()
    
                    err_found = False
                    for _check_func in err_log_check['funcs']:
                        # print(_check_func)
                        # print(_check_func(err_log))
                        # sys.stdout.flush()
                        if _check_func(err_log):
    
                            err_found = True
    
                            # Cancel the job
                            cmd = f'scancel {job_ID_str}'
                            p = Popen(
                                shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8'
                            )
                            out, err = p.communicate()
                            if err:
                                print(f'Tried cancelling Job {job_ID_str}')
                                print(f'\n*** stderr: command: {cmd}')
                                print(err)
    
                            break
    
                    if err_found:
                        break
            
            time.sleep(check_interval)    

    return output_ready_filepath


def sendRunCompleteMail(subject, content):
    """"""

    import getpass
    import smtplib
    from email.message import EmailMessage

    username = getpass.getuser()

    s = smtplib.SMTP('localhost')
    msg = EmailMessage()
    msg['From'] = f'{username}@{s.local_hostname}'
    msg['To'] = f'{username}@bnl.gov'
    msg['Subject'] = subject
    msg.set_content(content)
    s.send_message(msg)
    s.quit()
