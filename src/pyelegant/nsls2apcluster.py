import sys
import os
import time
import datetime
from copy import deepcopy
from subprocess import Popen, PIPE
import shlex
import tempfile
import glob
from pathlib import Path
import json
import re

import numpy as np
import dill
from ruamel import yaml

from . import util

_IMPORT_TIMESTAMP = time.time()

_SLURM_CONFIG_FILEPATH = Path.home().joinpath('.pyelegant', 'slurm_config.yaml')
_SLURM_CONFIG_FILEPATH.parent.mkdir(parents=True, exist_ok=True)
if not _SLURM_CONFIG_FILEPATH.exists():
    _SLURM_CONFIG_FILEPATH.write_text('''\
exclude: []
abs_time_limit: {}
''')
#
SLURM_PARTITIONS = {}
SLURM_EXCL_NODES = None
SLURM_ABS_TIME_LIMIT = {}

if False:
    MODULE_LOAD_CMD_STR, MPI_COMPILER_OPT_STR = 'elegant-latest', ''
    #MODULE_LOAD_CMD_STR, MPI_COMPILER_OPT_STR = 'elegant-latest elegant/2020.2.0', ''
    #MODULE_LOAD_CMD_STR, MPI_COMPILER_OPT_STR = 'elegant/2020.1.1-1', '--mpi=pmi2'
else:
    _this_folder = os.path.dirname(os.path.abspath(__file__))
    _facility_json_filepath = os.path.join(_this_folder, 'facility.json')
    with open(_facility_json_filepath, 'r') as f:
        _facility_info = json.load(f)
    MODULE_LOAD_CMD_STR = _facility_info['MODULE_LOAD_CMD_STR']
    MPI_COMPILER_OPT_STR = _facility_info['MPI_COMPILER_OPT_STR']

_p = Popen(shlex.split('which elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
_out, _err = _p.communicate()
if _out.strip() == '':
    nMaxTry = 3
    for iTry in range(nMaxTry):
        try:
            # Load "Elegant" module (for Lmod v.8.1)
            sys.path.insert(0, os.path.join(os.environ['MODULESHOME'], 'init'))
            from env_modules_python import module
            os.environ['LMOD_REDIRECT'] = 'yes'
            #module('load', 'gcc', 'mpich', 'elegant')
            module('load', *MODULE_LOAD_CMD_STR.split())
            #print('Elegant module loaded successfully')
            break
        except:
            if iTry != nMaxTry - 1:
                time.sleep(5.0)
            else:
                raise RuntimeError('# Loading module "elegant" failed.')

p = Popen(shlex.split('which elegant'), stdout=PIPE, stderr=PIPE, encoding='utf-8')
out, err = p.communicate()
if out.strip():
    path_tokens = out.split('/')
    if 'elegant' in path_tokens:
        __elegant_version__ = path_tokens[path_tokens.index('elegant')+1]
    else:
        __elegant_version__ = 'unknown'
    del path_tokens
else:
    print('\n*** pyelegant:WARNING: ELEGANT not available.')
    __elegant_version__ = None
del p, out, err

# CRITICAL: The line "#!/bin/bash" must come on the first line, not the second or later.
__sbatch_sh_example = '''#!/bin/bash
#SBATCH --job-name={job_name}

#SBATCH --error={job_name}.%J.err
#SBATCH --output={job_name}.%J.out

#SBATCH --partition={partition}

# #SBATCH --time=10:00

#SBATCH --ntasks={ntasks:d}
# #SBATCH --cpus-per-task=1
# SBATCH --ntasks-per-core=1

# #SBATCH --nodelist=apcpu-001,apcpu-002,apcpu-003,apcpu-005

# SBATCH --exclude=apcpu-004

# send email when task begins
#SBATCH --mail-type=begin

# send email when task ends
#SBATCH --mail-type=end

#SBATCH --mail-user=yhidaka@bnl.gov

# module load elegant #---# for Environment Modules v.4.2.4
module load {MODULE_LOAD_CMD_STR} #---# for Lmod v.8.1

# hostname

srun --mpi=pmi2 Pelegant {ele_filepath}
'''

DEFAULT_REMOTE_OPTS = dict(
    sbatch={'use': False, 'wait': True}, pelegant=False,
    # -------------
    # SLURM options
    job_name='job', output='job.%J.out', error='job.%J.err',
    partition='normal', ntasks=1, time=None, nodelist=None, exclude=None,
    # ---------------------------------
    # SBATCH error check decistion tree
    sbatch_err_check_tree=[
        ['exists', ('semaphore_file',), {}],
        [
            ['not_empty', ('%s.newlte',), {}],
            'no_error',
            'retry'
        ],
        [
            ['check_slurm_err_log', ('slurm_err_filepath', 'abort_info'), {}],
            'retry',
            'abort'
        ]
    ],
)

def get_slurm_config_filepath():
    """"""
    return str(_SLURM_CONFIG_FILEPATH.resolve())

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

def _make_sure_slurm_excl_nodes_initialized():
    """"""

    global SLURM_EXCL_NODES

    if SLURM_EXCL_NODES is None:
        SLURM_EXCL_NODES = []

        load_slurm_excl_nodes_from_config_file(force=True)

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

def _init_SLURM_ABS_TIME_LIMIT():
    """"""

    if not SLURM_PARTITIONS:
        SLURM_PARTITIONS.update(_get_slurm_partition_info())

    for _partition in list(SLURM_PARTITIONS):
        SLURM_ABS_TIME_LIMIT[_partition] = None

def clear_slurm_abs_time_limit(partition=None):
    """"""

    if partition:
        SLURM_ABS_TIME_LIMIT[partition] = None
    else:
        for k in list(SLURM_ABS_TIME_LIMIT):
            SLURM_ABS_TIME_LIMIT[k] = None

def set_slurm_abs_time_limit(abs_time_limit_str, partition=None):
    """
    "abs_time_limit_str" should be either `None` or a string in the form of:
      '2020-07-02T14-50-00'
    """

    try:
        if abs_time_limit_str is not None:
            time.mktime(time.strptime(abs_time_limit_str, '%Y-%m-%dT%H-%M-%S'))
    except:
        raise ValueError(('Invalid format for "abs_time_limit_str". '
                          'A valid example is "2020-07-02T14-50-00"'))

    # Initialize if not done yet
    if SLURM_ABS_TIME_LIMIT == {}:
        _init_SLURM_ABS_TIME_LIMIT()

    if partition:
        SLURM_ABS_TIME_LIMIT[partition] = abs_time_limit_str
    else:
        for k in list(SLURM_ABS_TIME_LIMIT):
            SLURM_ABS_TIME_LIMIT[k] = abs_time_limit_str

def load_slurm_excl_nodes_from_config_file(force=False):
    """"""

    if (not force) and (
        not util.is_file_updated(_SLURM_CONFIG_FILEPATH, _IMPORT_TIMESTAMP)):
        return

    yml = yaml.YAML()
    config = yml.load(_SLURM_CONFIG_FILEPATH.read_text())

    exclude_list = config.get('exclude', None)
    if exclude_list is not None:
        set_slurm_excl_nodes(list(exclude_list))

def save_slurm_excl_nodes_to_config_file():
    """"""

    yml = yaml.YAML()
    try:
        config = yml.load(_SLURM_CONFIG_FILEPATH.read_text())
        if config is None:
            config = {}
    except:
        config = {}

    config['exclude'] = SLURM_EXCL_NODES

    with open(_SLURM_CONFIG_FILEPATH, 'w') as f:
        yml.dump(config, f)

def load_slurm_abs_time_limit_from_config_file(force=False):
    """"""

    if (not force) and (
        not util.is_file_updated(_SLURM_CONFIG_FILEPATH, _IMPORT_TIMESTAMP)):
        return

    yml = yaml.YAML()
    config = yml.load(_SLURM_CONFIG_FILEPATH.read_text())

    abs_time_limit_str_dict = config.get('abs_time_limit', None)
    if abs_time_limit_str_dict is None:
        return

    abs_time_limit_str_list = list(abs_time_limit_str_dict.values())
    if len(set(abs_time_limit_str_list)) == 1:
        set_slurm_abs_time_limit(abs_time_limit_str_list[0])
    else:
        for partition, abs_time_limit_str in abs_time_limit_str_dict.items():
            set_slurm_abs_time_limit(abs_time_limit_str, partition=partition)

def save_slurm_abs_time_limit_from_config_file():
    """"""

    yml = yaml.YAML()
    try:
        config = yml.load(_SLURM_CONFIG_FILEPATH.read_text())
        if config is None:
            config = {}
    except:
        config = {}

    config['abs_time_limit'] = SLURM_ABS_TIME_LIMIT

    with open(_SLURM_CONFIG_FILEPATH, 'w') as f:
        yml.dump(config, f)


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
            'sbatch_err_check_tree',
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
    sbatch_sh_filepath, slurm_opts, srun_cmd, nMaxTry=10, sleep=10.0):
    """"""

    # CRITICAL: The line "#!/bin/bash" must come on the first line, not the second or later.
    contents = ['#!/bin/bash']

    contents += [' ']

    for v in slurm_opts.values():
        contents += ['#SBATCH ' + v]

    contents += [' ']

    contents += ['module load {} #---# for Lmod v.8.1'.format(MODULE_LOAD_CMD_STR)]

    contents += [' ']

    contents += [srun_cmd]

    contents += [' ']

    util.robust_text_file_write(
        sbatch_sh_filepath, '\n'.join(contents), nMaxTry=nMaxTry, sleep=sleep)

def run(
    remote_opts, ele_filepath, macros=None, print_cmd=False,
    print_stdout=True, print_stderr=True, tee_to=None, tee_stderr=True,
    output_filepaths=None, err_log_check=None):
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

        # Make sure output/error log filenames conform to expectations
        job_name = slurm_opts['job-name'].split('=')[-1]
        slurm_opts['output'] = '--output={}.%J.out'.format(job_name)
        slurm_opts['error'] = '--error={}.%J.err'.format(job_name)

        tmp = tempfile.NamedTemporaryFile(
            dir=os.getcwd(), delete=False, prefix='tmpSbatch_', suffix='.sh')
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
            sbatch_sh_filepath, slurm_opts, srun_cmd, #ele_filepath, remote_opts['pelegant'], macros=macros,
            nMaxTry=10, sleep=10.0)

        if 'abort_filepath' in remote_opts:
            abort_info = dict(filepath=remote_opts['abort_filepath'],
                              ref_timestamp=time.time())
        else:
            abort_info = None

        if ('sbatch_err_check_tree' in remote_opts) and \
           (remote_opts['sbatch_err_check_tree'] == 'default'):
            remote_opts['sbatch_err_check_tree'] = deepcopy(
                DEFAULT_REMOTE_OPTS['sbatch_err_check_tree'])

        nMaxReTry = 5
        for _ in range(nMaxReTry):

            if (abort_info is not None) and util.is_file_updated(
                abort_info['filepath'], abort_info['ref_timestamp']):
                print('\n\n*** Immediate abort requested. Aborting now.')
                raise RuntimeError('Abort requested.')

            wait_after_sbatch = remote_opts['sbatch'].get('wait', True)

            (job_ID_str, slurm_out_filepath, slurm_err_filepath, sbatch_info
             ) = _sbatch(sbatch_sh_filepath, job_name,
                         exit_right_after_submission=(not wait_after_sbatch),
                         err_log_check=err_log_check)

            if not wait_after_sbatch:
                output = dict(
                    sbatch_sh_filepath=sbatch_sh_filepath, job_ID_str=job_ID_str,
                    slurm_out_filepath=slurm_out_filepath,
                    slurm_err_filepath=slurm_err_filepath)
                return output
            else:
                output = sbatch_info

            if 'sbatch_err_check_tree' not in remote_opts:
                # Will NOT check whether Elegant/Pelegant finished its run
                # without errors or not.
                break
            else:
                if remote_opts['sbatch_err_check_tree'] is None:
                    # Will NOT check whether Elegant/Pelegant finished its run
                    # without errors or not.
                    break
                else:
                    _update_sbatch_err_check_tree_kwargs(
                        remote_opts['sbatch_err_check_tree'], output_filepaths,
                        slurm_err_filepath, abort_info)

                    time.sleep(5.0)

                    flag = _sbatch_error_check(remote_opts['sbatch_err_check_tree'])

                    if flag == 'no_error':
                        break
                    elif flag == 'retry':
                        time.sleep(10.0)
                        continue
                    elif flag == 'abort':
                        raise RuntimeError(
                            'Unrecoverable error occurred or abort requested. Aborting.')
                    else:
                        raise ValueError('Unexpected flag: {}'.format(flag))

        if not remote_opts.get('diag_mode', False):
            try:
                os.remove(sbatch_sh_filepath)
            except IOError:
                print('* Failed to delete temporary sbatch shell file "{}"'.format(
                    sbatch_sh_filepath))

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
            p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
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

            p = Popen(cmd_list_2, stdin=p1.stdout, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        out, err = out.decode('utf-8'), err.decode('utf-8')

        if out and print_stdout:
            print(out)

        if err and print_stderr:
            print('stderr:')
            print(err)

    return output

def _sbatch(sbatch_sh_filepath, job_name, exit_right_after_submission=False,
            err_log_check=None):
    """"""

    #mpi_rank_header = get_mpi_rank_header()

    nMaxSbatch = 3
    for iSbatch in range(nMaxSbatch):
        p = Popen(['sbatch', sbatch_sh_filepath], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        out = out.decode('utf-8')
        err = err.decode('utf-8')
        #print(mpi_rank_header + out)
        print(out)
        if err:
            print(err)
            print('\n** Encountered error during main job submission.')
            if iSbatch != nMaxSbatch - 1:
                print('Will retry sbatch.\n')
                time.sleep(20.0)
            else:
                raise RuntimeError('Encountered error during main job submission')
        else:
            break

    job_ID_str = out.replace('Submitted batch job', '').strip()

    #print('* {}'.format(mpi_rank_header))

    sys.stdout.flush()

    sbatch_info = None
    if not exit_right_after_submission:

        status_check_interval = 5.0 #10.0

        _min_err_log_check = _get_min_err_log_check()
        if err_log_check is None:
            err_log_check = _min_err_log_check
        else:
            for _func in _min_err_log_check['funcs']:
                if _func not in err_log_check['funcs']:
                    err_log_check['funcs'].append(_func)

        err_log_check['job_name'] = job_name

        sbatch_info = wait_for_completion(
            job_ID_str, status_check_interval, err_log_check=err_log_check)

    slurm_out_filepath = '{job_name}.{job_ID_str}.out'.format(
        job_name=job_name, job_ID_str=job_ID_str)
    slurm_err_filepath = '{job_name}.{job_ID_str}.err'.format(
        job_name=job_name, job_ID_str=job_ID_str)

    return job_ID_str, slurm_out_filepath, slurm_err_filepath, sbatch_info

def _update_sbatch_err_check_tree_kwargs(
    check_tree, output_filepaths_in_ele, slurm_err_filepath, abort_info):
    """"""

    check_type, args, kwargs = check_tree[0]
    for arg in args:
        if arg in output_filepaths_in_ele:
            kwargs[arg] = output_filepaths_in_ele[arg]
        elif arg == 'abort_info':
            kwargs[arg] = abort_info
        elif arg == 'slurm_err_filepath':
            kwargs[arg] = slurm_err_filepath
        else:
            raise ValueError('Unexpected arg: "{}"'.format(arg))

    for sub_check_tree in check_tree[1:]:
        _update_sbatch_err_check_tree_kwargs(
            sub_check_tree, output_filepaths_in_ele, slurm_err_filepath, abort_info)

def _sbatch_error_check(check_tree):
    """"""

    check_type, args, kwargs = check_tree[0]
    outcomes = check_tree[1:]

    i = _get_outcome_index(check_type, *args, **kwargs)

    if isinstance(i, int):
        return _sbatch_error_check(outcomes[i])
    else:
        error_status = outcomes[i]
        return error_status

def _get_outcome_index(check_type, *args, **kwargs):
    """"""

    if check_type == 'exists':
        filepath = kwargs[args[0]]
        if os.path.exists(filepath):
            return 0
        else:
            return 1
    elif check_type == 'not_empty':
        filepath = kwargs[args[0]]
        if os.stat(filepath).st_size != 0: # File NOT empty
            return 0
        else:
            return 1
    elif check_type == 'check_slurm_err_log':
        slurm_err_filepath = kwargs[args[0]]
        abort_info = kwargs[args[1]]
        if _check_slurm_err_log(slurm_err_filepath, abort_info):
            return 0
        else:
            return 1
    else:
        raise ValueError('Unexpected "check_type": {}'.format(check_type))

def _check_slurm_err_log(err_contents, abort_info):
    """"""

    should_retry = True

    abort_requested = False

    if "double free or corruption" in err_contents:

        sys.stderr.write(
            '\n* Pelegant crashed with "double free or corruption". Will retry.\n')

        used_node_info = [L for L in err_contents.split('\n')
                          if L.startswith('srun: error:')]
        sys.stderr.write(' ')
        sys.stderr.write(used_node_info)
        sys.stderr.write(' ')

        p = Popen('squeue -o "%.7i %.9P %.8j %.8u %.2t %.10M %.10L %.6D %.4C %R"',
                  stdout=PIPE, stderr=PIPE, shell=True)
        out, err = p.communicate()
        sys.stderr.write(out)
        if err:
            sys.stderr.write('\n* ERROR:')
            sys.stderr.write(err)
            sys.stderr.write(' ')

    elif ('Unable to open file' in err_contents) and \
         ('for writing (SDDS_InitializeOutput)' in err_contents):

        sys.stderr.write(
            '\n* Pelegant crashed with "Unable to write SDDS file". Will retry.\n')

    elif ('unable to open' in err_contents) and \
         ('in mode w: File exists' in err_contents):  #and \
            #('Fatal error in PMPI_Barrier' in err_contents):

        sys.stderr.write(
            '\n* Pelegant crashed with "Unable to open in mode w: File exists". Will retry.\n')

    elif 'Fatal error in PMPI_Init' in err_contents:

        sys.stderr.write('\n* Pelegant crashed with "Fatal error in PMPI_Init". Will retry.\n')

    elif 'Unable to confirm allocation for job' in err_contents:
        # Example:
        # run: error: Unable to confirm allocation for job 6307055: Socket timed out on send/recv operation
        # srun: Check SLURM_JOB_ID environment variable. Expired or invalid job 6307055
        sys.stderr.write('\n* Pelegant crashed with "Unable to confirm allocation for job". Will retry.\n')

    elif (not err_contents.startswith('Error:')) and (
        'CANCELLED AT' in err_contents) and ('DUE TO TIME LIMIT' in err_contents):

        sys.stderr.write('\n* Pelegant crashed with SLURM time limit. Will retry.\n')

    else: # Unexpected or unhandled errors

        sys.stderr.write('\n*** Unexpected Elegant error encountered:')

        tmp = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False,
            prefix='unexpected_elegant_error.',
            suffix='.txt')
        err_filepath = os.path.abspath(tmp.name)
        with open(err_filepath, 'w') as f:
            f.write(err_contents)

        should_retry = False

    if is_file_updated(abort_info['filepath'], abort_info['ref_timestamp']):
        print('\n\n*** Immediate abort requested. Aborting now.')

        abort_requested = True

        should_retry = False

    return should_retry, abort_requested

def get_mpi_rank_header():
    """"""

    try:
        from mpi4py import MPI

        mpi_rank = MPI.COMM_WORLD.Get_rank() + 1
        mpi_rank_header = 'MPI Rank #{:d}: '.format(mpi_rank)
    except:
        mpi_rank = 0 # MPI NOT being used
        mpi_rank_header = ''

    return mpi_rank_header

def check_unable_to_open_mode_w_File_exists(err_log_contents):
    """"""

    if ('unable to open' in err_log_contents) and \
       ('in mode w: File exists' in err_log_contents):
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

    return dict(funcs=[check_unable_to_open_mode_w_File_exists,
                       check_remote_err_log_exit_code])

def wait_for_completion(
    job_ID_str, status_check_interval, timelimit_action=None,
    out_log_check=None, err_log_check=None):
    """"""

    t0 = time.time()

    dt_not_running = 0.0
    not_running_t0 = None

    if timelimit_action is not None:
        assert callable(timelimit_action['abort_func'])
        timelimit_sec = \
            get_cluster_time_limits()[timelimit_action['partition_name']] \
            - timelimit_action['margin_sec']
    else:
        timelimit_sec = None

    err_counter = 0

    cmd = f'squeue --noheader --job={job_ID_str} -o "%.{len(job_ID_str)+1}i %.3t %.4C %R"'
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

                print('Time limit of selected partition is nearing, '
                      'so performing a graceful abort task now.')
                timelimit_action['abort_func'](
                    *timelimit_action.get('abort_func_args', ()))

                break

        if (state == 'R') and (err_log_check is not None):

            # Check if err log file size has increased. If not, there is no
            # need to check the contents.
            #print(f'Current directory is "{os.getcwd()}"')
            for _ in range(3):
                try:
                    curr_err_log_file_size = os.stat(err_log_filename).st_size
                    break
                except FileNotFoundError:
                    time.sleep(5.0)
            else:
                raise FileNotFoundError(err_log_filename)
            if curr_err_log_file_size > prev_err_log_file_size:

                prev_err_log_file_size = curr_err_log_file_size

                err_log = Path(err_log_filename).read_text()
                #print(err_log)

                err_found = False
                for _check_func in err_log_check['funcs']:
                    #print(_check_func)
                    #print(_check_func(err_log))
                    #sys.stdout.flush()
                    if _check_func(err_log):

                        err_found = True

                        # Cancel the job
                        cmd = f'scancel {job_ID_str}'
                        p = Popen(shlex.split(cmd),
                                  stdout=PIPE, stderr=PIPE, encoding='utf-8')
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

    h_dt_total = get_human_friendly_time_duration_str(dt_total, fmt='.2f')
    h_dt_running = get_human_friendly_time_duration_str(dt_running, fmt='.2f')
    print(f'Elapsed: Total = {h_dt_total}; Running = {h_dt_running}')

    ret = dict(
        total=dt_total, running=dt_running, nodes=used_nodes, ncores=num_cores,
        err_log=err_log, err_found=err_found)

    return ret

def get_human_friendly_time_duration_str(dt, fmt='.2f'):
    """"""

    template = f'{{val:{fmt}}} [{{unit}}]'

    if dt < 60:
        return template.format(val=dt, unit='s')
    elif dt < 60 * 60:
        return template.format(val=dt/60, unit='min')
    elif dt < 60 * 60 * 24:
        return template.format(val=dt/60/60, unit='hr')
    else:
        return template.format(val=dt/60/60/24, unit='day')

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

    duration_in_sec = days_in_secs + datetime.timedelta(
        hours=d.tm_hour, minutes=d.tm_min, seconds=d.tm_sec).total_seconds()

    return duration_in_sec

#----------------------------------------------------------------------
def get_cluster_time_limits():
    """"""

    if False:
        p = Popen('sinfo', stdout=PIPE, stderr=PIPE, encoding='utf-8')

        out , err = p.communicate()

        if err:
            print('\n* ERROR:')
            print(err)
            raise RuntimeError()

        lines = out.split('\n')

        all_u_partition_names, uinds = np.unique(
            [L.split()[0].replace('*', '') for L in lines[1:] if len(L.split()) != 0],
            return_index=True)

        all_timelimit_strs = np.array(
            [L.split()[2] for L in lines[1:] if len(L.split()) != 0])

        all_u_timelimit_strs = all_timelimit_strs[uinds]

        all_u_timelimit_secs = []
        for s in all_u_timelimit_strs:

            if s == 'infinite':
                s = '1000-00:00:00' # set the limit to 1000 days (or any other arbitrarily large number)

            s_list = s.split(':')
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
            all_u_timelimit_secs.append(
                days_in_secs + datetime.timedelta(
                    hours=d.tm_hour, minutes=d.tm_min, seconds=d.tm_sec).total_seconds())

        timelimit_d = {}
        for partition, sec in zip(all_u_partition_names, all_u_timelimit_secs):
            timelimit_d[partition] = sec

    else:
        if not SLURM_PARTITIONS:
            SLURM_PARTITIONS.update(_get_slurm_partition_info())

        timelimit_d = {}
        for partition, info_d in SLURM_PARTITIONS.items():
            time_limit_str = info_d['MaxTime']
            if time_limit_str == 'UNLIMITED':
                time_limit_str = '1000-00:00:00' # set the limit to 1000 days (or any other arbitrarily large number)
            timelimit_d[partition] = \
                _convert_slurm_time_duration_str_to_seconds(time_limit_str)

    return timelimit_d

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


def write_geneticOptimizer_dot_local(remote_opts=None):
    """"""

    filtered_remote_opts = {}

    if remote_opts:
        for k, v in remote_opts.items():
            if k in ['output', 'job_name']:
                print(
                    (f'* WARNING: The value of "remote_opts" for key "{k}" is '
                     f'automatically set, so your specified option value will '
                     f'be ignored.'))
            else:
                filtered_remote_opts[k] = v

    slurm_opts = extract_slurm_opts(filtered_remote_opts)

    contents = '''#!/bin/sh
# \
exec oagtclsh "$0" "$@"

# These two procedures may need to be customized to your installation.
# If so, then this file (i.e,. your modified version of it) needs to
# be in the directory from which you run geneticOptimizer.

# This proc is used if `programName` != ""
# NOT IMPLEMENTED YET
proc SubmitJob {args} {
    set code ""
    set input ""
    APSStrictParseArguments {code input}
    global env
    eval file delete -force $input.log [file rootname $input].done
    set tmpFile [file rootname $input].csh
    set fd [open $tmpFile w]
    puts $fd "#!/bin/csh "
    puts $fd "ml elegant-latest"
    puts $fd "unset savehist"
    puts $fd "echo running $code $input on [exec uname -a]"
    puts $fd "cd [pwd]"
    puts $fd "./$code $input >& $input.log"
    close $fd

    catch {exec sbatch -o [pwd] -J [file root [file tail $input]] $tmpFile } result

    return "$result"
}

# This proc is used if `runScript` != ""
proc SubmitRunScript {args} {
    set script ""
    set tagList ""
    set valueList ""
    set rootname ""
    APSStrictParseArguments {script rootname tagList valueList}
    global env

    # If your input SDDS file is named like "optim1.sdds", then $rootname would be
    # like "optim1-000001", "optim1-000002", etc.

    # $script = `runScript`

    eval file delete -force $rootname.log $rootname.done $rootname.run
    #set tmpFile [file rootname $rootname].csh
    set tmpFile [file rootname $rootname].bash
    APSAddToTempFileList $tmpFile
    set fd [open $tmpFile w]
    #puts $fd "#!/bin/csh "
    #puts $fd "unset savehist"
    #puts $fd "ml elegant-latest"
    puts $fd "echo Using python executable = [exec which python]"
    puts $fd "echo running $script on [exec uname -a]"
    puts $fd "echo running $script $rootname $tagList $valueList on [exec uname -a]"
    puts $fd "cd [pwd]"
    #puts $fd "./$script -rootname $rootname -tagList '$tagList' -valueList '$valueList' >& $rootname.log"
    puts $fd "python $script -rootname $rootname -tagList '$tagList' -valueList '$valueList' >& $rootname.log"
    close $fd
'''

    # Note that this section is separated because the original curly brackets
    # had to be converted to double-curly brackets, in order to allow insertion
    # of Python string formatting for "slurm_switches"
    contents += '''
    #catch {{exec cat $tmpFile | qsub -V -o [pwd] -j y -N [file root [file tail $rootname]] }} result
    #catch {{exec sbatch -o [pwd]/$rootname.slog -J [file root [file tail $rootname]] $tmpFile }} result

    #catch {{exec srun -o [pwd]/$rootname.slog -J [file root [file tail $rootname]] --ntasks=1 bash $tmpFile & }} result
    #catch {{exec srun -o [pwd]/$rootname.slog -J [file root [file tail $rootname]] {slurm_switches} bash $tmpFile & }} result
    catch {{exec srun -o [pwd]/$rootname.slog -J [file root [file tail $rootname]] --ntasks=2 {slurm_switches} bash $tmpFile & }} result
'''.format(slurm_switches=' '.join([v for v in slurm_opts.values()]))

    contents += '''

    return "$result"
}

proc UpdateJobsRunning {} {
    global rootname jobsRunning jobsStarted jobsToProc inputFileExtension jobsProc jobsCurrent pulse
    #set jobsCurrent [llength [glob -nocomplain $rootname-??????.csh]]
    set jobsCurrent [llength [glob -nocomplain $rootname-??????.bash]]
    set jobsDone [llength [glob -nocomplain $rootname-??????.done]]
    set jobsProc [llength [glob -nocomplain $rootname-??????.proc]]
    set jobsToProc [expr $jobsDone-$jobsProc]
    set jobsRunning [expr $jobsCurrent-$jobsDone]
    set message "[clock format [clock seconds]]: Jobs: current=$jobsCurrent, done=$jobsDone, proc'd=$jobsProc, toProc=$jobsToProc, running=$jobsRunning"
    puts -nonewline stderr $message
    for {set i 0} {$i<[string length $message]} {incr i} {
        puts -nonewline stderr "\b"
    }
}
'''

    with open('geneticOptimizer.local', 'w') as f:
        f.write(contents)

def run_mpi_python(remote_opts, module_name, func_name, param_list, args,
                   paths_to_prepend=None, err_log_check=None,
                   ret_slurm_info=False):
    """
    Example:
        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
    """

    d = dict(module_name=module_name, func_name=func_name,
             param_list=param_list, args=args)

    job_name = remote_opts.get('job_name', 'job')

    input_filepath, output_filepath, mpi_sub_sh_filepath = gen_mpi_submit_script(
        job_name=job_name,
        partition=remote_opts.get('partition', 'normal'),
        ntasks=remote_opts.get('ntasks', 50), x11=remote_opts.get('x11', False),
        spread_job=remote_opts.get('spread_job', False),
        timelimit_str=remote_opts.get('time', None),
        exclude=remote_opts.get('exclude', None),
        nodelist=remote_opts.get('nodelist', None),
    )

    d['output_filepath'] = output_filepath

    with open(input_filepath, 'wb') as f:

        if paths_to_prepend is None:
            paths_to_prepend = []
        dill.dump(paths_to_prepend, f, protocol=-1)

        dill.dump(d, f, protocol=-1)

    # Add re-try functionality in case of "sbatch: error: Slurm controller not responding, sleeping and retrying."
    err_counter = 0
    while True:
        p = Popen('sbatch {0}'.format(mpi_sub_sh_filepath),
                  stdout=PIPE, stderr=PIPE, shell=True)
        out, err = p.communicate()
        out = out.decode()
        err = err.decode()
        print(out)
        if err:
            err_counter += 1

            if err_counter >= 10:
                print(err)
                raise RuntimeError('Encountered error during main job submission')
            else:
                print(err)
                sys.stdout.flush()
                time.sleep(30.0)
                continue
        else:
            break

    job_ID_str = out.replace('Submitted batch job', '').strip()

    _min_err_log_check = _get_min_err_log_check()
    if err_log_check is None:
        err_log_check = _min_err_log_check
    else:
        for _func in _min_err_log_check['funcs']:
            if _func not in err_log_check['funcs']:
                err_log_check['funcs'].append(_func)

    err_log_check['job_name'] = job_name

    info = wait_for_completion(
        job_ID_str, remote_opts.get('status_check_interval', 3.0),
        err_log_check=err_log_check)

    if not info['err_found']:
        results = util.load_pgz_file(output_filepath)
    else:
        results = info['err_log']

    if remote_opts.get('del_input_file', True):
        try: os.remove(input_filepath)
        except: pass
    if remote_opts.get('del_sub_sh_file', True):
        try: os.remove(mpi_sub_sh_filepath)
        except: pass
    if remote_opts.get('del_output_file', True):
        try: os.remove(output_filepath)
        except: pass
    if remote_opts.get('del_job_out_file', True):
        try: os.remove(f'{job_name}.{job_ID_str}.out')
        except: pass
    if remote_opts.get('del_job_err_file', True):
        try: os.remove(f'{job_name}.{job_ID_str}.err')
        except: pass

    if not ret_slurm_info:
        return results
    else:
        return results, info

def srun_python_func(
    remote_opts, module_name, func_name, func_args, func_kwargs, paths_to_prepend=None):
    """
    Example:
        module_name = 'pyelegant.nonlin'
        func_name = '_calc_chrom_track_get_tbt'
    """

    d = dict(module_name=module_name, func_name=func_name,
             func_args=func_args, func_kwargs=func_kwargs)

    tmp = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False,
                                      prefix='tmpInput_', suffix='.pgz')
    input_filepath = os.path.abspath(tmp.name)
    output_filepath = os.path.abspath(tmp.name.replace('tmpInput_', 'tmpOutput_'))
    tmp.close()

    d['output_filepath'] = output_filepath

    job_name = remote_opts.get('job_name', 'job')
    partition = remote_opts.get('partition', 'normal')
    cpus_per_task = remote_opts.get('cpus_per_task', 2)
    # ^ Don't use "-c 1". If you do, it will still uses 2 cores, but runs
    # 2 process of the same simultaneously (you can tell from the print
    # statements.) With "-c 2", there will be only one process but can
    # utilize 2 cores.
    x11 = ('' if not remote_opts.get('x11', False) else 'export MPLBACKEND="agg"')
    spread_job = ('' if not remote_opts.get('spread_job', False) else '--spread-job')
    nodelist = ('' if 'nodelist' not in remote_opts
                else f'--nodelist={",".join(remote_opts["nodelist"])}')
    exclude = ('' if 'exclude' not in remote_opts
               else f'--exclude={",".join(remote_opts["exclude"])}')
    timelimit_str = remote_opts.get('time', None)
    if timelimit_str is not None:
        timelimit_str = f'--time={timelimit_str}'
    else:
        timelimit_str = ''

    with open(input_filepath, 'wb') as f:

        if paths_to_prepend is None:
            paths_to_prepend = []
        dill.dump(paths_to_prepend, f, protocol=-1)

        dill.dump(d, f, protocol=-1)

    main_script_path = __file__[:-3]+'_srun_py_func_script.py'
    cmd = (f'srun -c {cpus_per_task:d} -J {job_name} -p {partition} {x11} {timelimit_str} '
           f'{nodelist} {exclude} python {main_script_path} {input_filepath}')
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    print(out)
    if err:
        print(err)

    results = util.load_pgz_file(output_filepath)

    if remote_opts.get('del_input_file', True):
        try: os.remove(input_filepath)
        except: pass
    if remote_opts.get('del_output_file', True):
        try: os.remove(output_filepath)
        except: pass

    return results

#----------------------------------------------------------------------
def convertLocalTimeStrToSecFromUTCEpoch(
    time_str, frac_sec=False, time_format=None):
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
        t += float(_frac_str) / (10**len(_frac_str))
        return t

#----------------------------------------------------------------------
def get_constrained_timelimit_str(abs_timelimit, partition):
    """
    "abs_timelimit" must be in the format of '%Y-%m-%dT%H-%M-%S'.
    """

    dt_till_timelimit = convertLocalTimeStrToSecFromUTCEpoch(
        abs_timelimit, time_format='%Y-%m-%dT%H-%M-%S') - time.time()

    def_timelimit = get_cluster_time_limits()[partition]

    if dt_till_timelimit > def_timelimit:
        timelimit_str = None
    elif dt_till_timelimit < 0.0:
        raise RuntimeError('It is already past specified absolute time limit.')
    else:
        sec = datetime.timedelta(seconds=dt_till_timelimit)
        dobj = datetime.datetime(1,1,1) + sec
        if dobj.day - 1 != 0:
            timelimit_str = '{:d}-'.format(dobj.day - 1)
        else:
            timelimit_str = ''
        timelimit_str += '{:02d}:{:02d}:{:02d}'.format(
            dobj.hour, dobj.minute, dobj.second)

    return timelimit_str

#----------------------------------------------------------------------
def gen_mpi_submit_script(
    job_name='job', partition='normal', ntasks=10, x11=False, spread_job=False,
    timelimit_str=None, exclude=None, nodelist=None):
    """"""

    tmp = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False,
                                      prefix='tmpInput_', suffix='.pgz')
    input_filepath = os.path.abspath(tmp.name)
    output_filepath = os.path.abspath(tmp.name.replace('tmpInput_', 'tmpOutput_'))
    mpi_sub_sh_filepath = os.path.abspath(
        tmp.name.replace('tmpInput_', 'tmpMpiSub_').replace('.pgz', '.sh'))

    # CRITICAL: The line "#!/bin/bash" must come on the first line, not the second or later.
    contents_template = '''#!/bin/bash
#
#SBATCH --job-name={job_name}
#
#SBATCH --error={job_name}.%J.err
#SBATCH --output={job_name}.%J.out

#SBATCH --partition={partition}

{slurm_timelimit_str}

#SBATCH --ntasks={ntasks:d}

# #SBATCH --nodelist=apcpu-001,apcpu-002
# SBATCH --nodelist=apcpu-003,apcpu-004

{nodelist}
{exclude}
{spread_job}

{x11}
srun {mpi_compiler_opt} python -m mpi4py.futures {main_script_path} _mpi_starmap {input_filepath}
    '''

    try:
        abs_timelimit = SLURM_ABS_TIME_LIMIT[partition]
    except:
        _init_SLURM_ABS_TIME_LIMIT()
        abs_timelimit = SLURM_ABS_TIME_LIMIT[partition]

    if timelimit_str is None:
        if abs_timelimit is None:
            slurm_timelimit_str = ''
        else:
            timelimit = get_constrained_timelimit_str(abs_timelimit, partition)
            if timelimit is None:
                slurm_timelimit_str = ''
            else:
                slurm_timelimit_str = '#SBATCH --time={}'.format(timelimit)
    else:
        slurm_timelimit_str = '#SBATCH --time={}'.format(timelimit_str)

    _make_sure_slurm_excl_nodes_initialized()

    if exclude is None:
        exclude = SLURM_EXCL_NODES
    _exclude = ('' if exclude == [] else '#SBATCH --exclude={}'.format(
        ','.join(exclude)))

    if nodelist is None:
        _nodelist = ''
    else:
        _nodelist = '#SBATCH --nodelist={}'.format(','.join(nodelist))

    contents = contents_template.format(
        mpi_compiler_opt=MPI_COMPILER_OPT_STR,
        main_script_path=__file__[:-3]+'_mpi_script.py', input_filepath=input_filepath,
        job_name=job_name, partition=partition, ntasks=ntasks,
        slurm_timelimit_str=slurm_timelimit_str,
        x11=('' if not x11 else 'export MPLBACKEND="agg"'),
        exclude=_exclude, nodelist=_nodelist,
        spread_job=('' if not spread_job else '#SBATCH --spread-job')
    )

    Path(mpi_sub_sh_filepath).write_text(contents)

    return input_filepath, output_filepath, mpi_sub_sh_filepath

#----------------------------------------------------------------------
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
        elif stat.st_size < int(1024**2):
            size = stat.st_size / 1024
            size_str = f'{size:>3.0f}K'
        elif stat.st_size < int(1024**3):
            size = stat.st_size / 1024 / 1024
            size_str = f'{size:>3.0f}M'
        elif stat.st_size < int(1024**4):
            size = stat.st_size / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}G'
        elif stat.st_size < int(1024**5):
            size = stat.st_size / 1024 / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}T'
        else:
            size = stat.st_size / 1024 / 1024 / 1024 / 1024
            size_str = f'{size:>3.0f}T'

        time_str = datetime.datetime.fromtimestamp(
            stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        print(f'{f.owner()} {f.group()} {size_str} {time_str} {f.name}')

#----------------------------------------------------------------------
def tmp_glob(node_name, pattern, sort_order='mtime'):
    """"""

    tmp = tempfile.NamedTemporaryFile(dir=Path.cwd(), delete=False,
                                      prefix='tmpGlob_', suffix='.py')
    temp_py = Path(tmp.name)
    tmp.close()

    import inspect

    func_lines = inspect.getsource(_tmp_glob).split('\n')
    for i, line in enumerate(func_lines):
        if line.strip().startswith(('import ', 'from ')):
            func_body_start_index = i
            break

    py_contents = ['if __name__ == "__main__":',
                   f'    pattern = "{pattern}"',
                   f'    sort_order = "{sort_order}"',
                   ] + func_lines[func_body_start_index:]

    temp_py.write_text('\n'.join(py_contents))

    cmd = f'srun --nodelist={node_name} --partition=debug python {str(temp_py)}'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out.strip())

    if err.strip():
        print('\n*** stderr ***')
        print(err.strip())

    # Remove temp Python file
    temp_py.unlink()

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
def tmp_rm(node_name, pattern):
    """
    Remove files/directories in /tmp on specified node.
    """

    tmp = tempfile.NamedTemporaryFile(dir=Path.cwd(), delete=False,
                                      prefix='tmpRm_', suffix='.py')
    temp_py = Path(tmp.name)
    tmp.close()

    import inspect

    func_lines = inspect.getsource(_tmp_rm).split('\n')
    for i, line in enumerate(func_lines):
        if line.strip().startswith(('import ', 'from ')):
            func_body_start_index = i
            break

    py_contents = ['if __name__ == "__main__":',
                   f'    pattern = "{pattern}"',
                   ] + func_lines[func_body_start_index:]

    temp_py.write_text('\n'.join(py_contents))

    cmd = f'srun --nodelist={node_name} --partition=debug python {str(temp_py)}'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out.strip())

    if err.strip():
        print('\n*** stderr ***')
        print(err.strip())

    # Remove temp Python file
    temp_py.unlink()

def bash_copy(src, dst):
    """
    The purpose of this module is to replace the standard Python module "shutil"
    in the event that shutil.copy() results in an empty copied file, which was found
    to occur on cpu-* machines on 09/14/2020.
    """

    cmd = f'cp {src} {dst}'

    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    print(out)
    if err:
        print('# stderr:')
        print(err)

def bash_move(src, dst):
    """
    The purpose of this module is to replace the standard Python module "shutil"
    in the event that shutil.copy() results in an empty copied file, which was found
    to occur on cpu-* machines on 09/14/2020.
    """

    cmd = f'mv {src} {dst}'

    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()
    print(out)
    if err:
        print('# stderr:')
        print(err)

def starmap_async(remote_opts, module_name, func_name, func_args_iterable,
                  paths_to_prepend=None, err_log_check=None):
    """"""

    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    if 'abort_filepath' in remote_opts:
        abort_info = dict(filepath=remote_opts['abort_filepath'],
                          ref_timestamp=time.time())
    else:
        abort_info = None

    slurm_opts = extract_slurm_opts(remote_opts)

    if 'job_name' not in slurm_opts:
        print('* Using `sbatch` requires "job_name" option to be specified. Using default.')
        slurm_opts['job_name'] = '--job-name={}'.format(
            DEFAULT_REMOTE_OPTS['job_name'])

    # Make sure output/error log filenames conform to expectations
    job_name = slurm_opts['job_name'].split('=')[-1]
    slurm_opts['output'] = '--output={}.%J.out'.format(job_name)
    slurm_opts['error'] = '--error={}.%J.err'.format(job_name)

    job_info_list = []
    for func_args in func_args_iterable:

        d = dict(module_name=module_name, func_name=func_name,
                 func_args=func_args, func_kwargs={})

        tmp = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False,
                                          prefix='tmpInput_', suffix='.pgz')
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
            dir=os.getcwd(), delete=False, prefix='tmpSbatch_', suffix='.sh')
        sbatch_sh_filepath = os.path.abspath(tmp.name)

        main_script_path = __file__[:-3]+'_srun_py_func_script.py'
        srun_cmd = f'srun python {main_script_path} {input_filepath}'

        write_sbatch_shell_file(
            sbatch_sh_filepath, slurm_opts, srun_cmd, nMaxTry=10, sleep=10.0)

        (job_ID_str, slurm_out_filepath, slurm_err_filepath, _
         ) = _sbatch(sbatch_sh_filepath, job_name,
                     exit_right_after_submission=True)

        job_info = dict(
            input_filepath=input_filepath, output_filepath=output_filepath,
            sbatch_sh_filepath=sbatch_sh_filepath, job_ID_str=job_ID_str,
            slurm_out_filepath=slurm_out_filepath,
            slurm_err_filepath=slurm_err_filepath)
        job_info_list.append(job_info)

    status_check_interval = 5.0 #10.0

    _min_err_log_check = _get_min_err_log_check()
    if err_log_check is None:
        err_log_check = _min_err_log_check
    else:
        for _func in _min_err_log_check['funcs']:
            if _func not in err_log_check['funcs']:
                err_log_check['funcs'].append(_func)

    err_log_check['job_name'] = job_name

    results_list = []
    for job_d in job_info_list:

        if (abort_info is not None) and util.is_file_updated(
            abort_info['filepath'], abort_info['ref_timestamp']):
            print('\n\n*** Immediate abort requested. Aborting now.')
            raise RuntimeError('Abort requested.')

        sbatch_info = wait_for_completion(
            job_d['job_ID_str'], status_check_interval, err_log_check=err_log_check)

        if not sbatch_info['err_found']:
            results = util.load_pgz_file(job_d['output_filepath'])
        else:
            results = sbatch_info['err_log']
            return results

        results_list.append(results)

        if not remote_opts.get('diag_mode', False):
            if remote_opts.get('del_input_file', True):
                try: os.remove(job_d['input_filepath'])
                except: pass
            if remote_opts.get('del_output_file', True):
                try: os.remove(job_d['output_filepath'])
                except: pass
            if remote_opts.get('del_sub_sh_file', True):
                try: os.remove(job_d['sbatch_sh_filepath'])
                except: pass
            if remote_opts.get('del_job_out_file', True):
                try: os.remove(job_d['slurm_out_filepath'])
                except: pass
            if remote_opts.get('del_job_err_file', True):
                try: os.remove(job_d['slurm_err_filepath'])
                except: pass

    return results_list

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
