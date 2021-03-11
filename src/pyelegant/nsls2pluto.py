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

import numpy as np
import dill
from ruamel import yaml

from . import facility_name
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
        use_sbatch=False,
        exit_right_after_sbatch=False,
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
    'w': 'nodelist',
    'x': 'exclude',
}


def extract_slurm_opts(remote_opts):
    """"""

    slurm_opts = {}

    _make_sure_slurm_excl_nodes_initialized()

    if SLURM_EXCL_NODES != []:
        slurm_opts['exclude'] = f'--exclude={",".join(SLURM_EXCL_NODES)}'

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

    need_mail_user = False

    for k, v in remote_opts.items():

        # Retrieve the full option name, if it's a shortcut name or alternative name
        if k in SRUN_OPTION_MAP:
            k = SRUN_OPTION_MAP[k]

        if k in ('output', 'error', 'partition', 'time'):

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
                if not all([_type in valid_types for _type in v]):
                    raise ValueError(f'Invalid "mail-type" input: {valid_types}')
                slurm_opts[k] = f'--mail-type={",".join(v)}'

                need_mail_user = True

        elif k in ('nodelist', 'exclude'):
            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, ','.join(v))

        elif k in ('sbatch', 'pelegant', 'diag_mode', 'abort_filepath'):
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
):
    """"""

    nMaxSbatch = 3
    for iSbatch in range(nMaxSbatch):
        p = Popen(
            ['sbatch', sbatch_sh_filepath], stdout=PIPE, stderr=PIPE, encoding='utf-8'
        )
        out, err = p.communicate()
        if print_stdout:
            print('\n' + out.strip())
        if err and print_stderr:
            print('\n*** stderr ***')
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

    if print_stdout:
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

    slurm_out_filepath = '{job_name}.{job_ID_str}.out'.format(
        job_name=job_name, job_ID_str=job_ID_str
    )
    slurm_err_filepath = '{job_name}.{job_ID_str}.err'.format(
        job_name=job_name, job_ID_str=job_ID_str
    )

    return job_ID_str, slurm_out_filepath, slurm_err_filepath, sbatch_info


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
                DEFAULT_REMOTE_OPTS['job-name']
            )

        # Make sure output/error log filenames conform to expectations
        job_name = slurm_opts['job-name'].split('=')[-1]
        slurm_opts['output'] = '--output={}.%J.out'.format(job_name)
        slurm_opts['error'] = '--error={}.%J.err'.format(job_name)

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
                print_stdout=print_stdout,
                print_stderr=print_stderr,
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
