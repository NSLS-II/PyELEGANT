from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

import sys
import os
import time
import datetime
from copy import deepcopy
from subprocess import Popen, PIPE
import tempfile

try:
    from mpi4py import MPI
except:
    pass

#import gzip
#try:
    #from six.moves import cPickle as pickle
#except:
    #print('Package "six" could not be found.')
    #import cPickle as pickle


from . import util

MODULE_LOAD_CMD_STR = 'elegant-latest'

nMaxTry = 3
for iTry in range(nMaxTry):
    try:
        # Load "Elegant" module (for Lmod v.8.1)
        sys.path.insert(0, os.path.join(os.environ['MODULESHOME'], 'init'))
        from env_modules_python import module
        #module('load', 'gcc', 'mpich', 'elegant')
        module('load', *MODULE_LOAD_CMD_STR.split())
        print('Elegant module loaded successfully')
        break
    except:
        if iTry != nMaxTry - 1:
            time.sleep(5.0)
        else:
            raise RuntimeError('# Loading module "elegant" failed.')

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
    use_sbatch=False, exit_right_after_sbatch=False, pelegant=False,
    # -------------
    # SLURM options
    job_name='job', output='job.%J.out', error='job.%J.err',
    partition='normal', ntasks=2, time=None, nodelist=None, exclude=None,
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

def extract_slurm_opts(remote_opts):
    """"""

    slurm_opts = {}

    need_mail_user = False

    for k, v in remote_opts.items():

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

        elif k in ('job_name', 'mail_user'):

            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k.replace('_', '-'), v)

        elif k in ('mail_type_begin', 'mail_type_end'):

            if (v is None) or (not v):
                slurm_opts[k] = ''
            else:
                _type = k.split('_')[-1]
                slurm_opts[k] = f'--mail-type={_type}'

                need_mail_user = True

        elif k in ('nodelist', 'exclude'):
            if v is None:
                slurm_opts[k] = ''
            else:
                slurm_opts[k] = '--{}={}'.format(k, ','.join(v))

        elif k in ('use_sbatch', 'pelegant',
                   'sbatch_err_check_tree',):
            pass
        else:
            raise ValueError(f'Unknown slurm option keyword: {k}')

    if need_mail_user:
        if len([s for s in list(slurm_opts) if s == 'mail_user']) != 1:
            raise ValueError('"mail_user" option must be specified when '
                             '"mail_type_begin" or "mail_type_end" is True')

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
    remote_opts, ele_filepath, macros=None, print_cmd=False, print_stdout=True,
    print_stderr=True, output_filepaths=None):
    """"""

    if remote_opts is None:
        remote_opts = deepcopy(DEFAULT_REMOTE_OPTS)

    slurm_opts = extract_slurm_opts(remote_opts)

    output = None

    if remote_opts['use_sbatch']:

        if 'job_name' not in slurm_opts:
            print('* Using `sbatch` requires "job_name" option to be specified. Using default.')
            slurm_opts['job_name'] = '--job-name={}'.format(
                DEFAULT_REMOTE_OPTS['job_name'])

        # Make sure output/error log filenames conform to expectations
        job_name = slurm_opts['job_name'].split('=')[-1]
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

            exit_right_after_sbatch = (
                remote_opts['exit_right_after_sbatch']
                if 'exit_right_after_sbatch' in remote_opts else False)

            job_ID_str, slurm_out_filepath, slurm_err_filepath = _sbatch(
                sbatch_sh_filepath, job_name,
                exit_right_after_submission=exit_right_after_sbatch)

            if exit_right_after_sbatch:
                output = dict(
                    job_ID_str=job_ID_str, slurm_out_filepath=slurm_out_filepath,
                    slurm_err_filepath=slurm_err_filepath)
                return output

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

        try:
            os.remove(sbatch_sh_filepath)
        except IOError:
            print('* Failed to delete temporary sbatch shell file "{}"'.format(
                sbatch_sh_filepath))

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

        if print_cmd:
            print('$ ' + ' '.join(cmd_list))

        p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        out, err = out.decode('utf-8'), err.decode('utf-8')

        if out and print_stdout:
            print(out)

        if err and print_stderr:
            print('ERROR:')
            print(err)

    return output

def _sbatch(sbatch_sh_filepath, job_name, exit_right_after_submission=False):
    """"""

    mpi_rank_header = get_mpi_rank_header()

    nMaxSbatch = 3
    for iSbatch in range(nMaxSbatch):
        p = Popen(['sbatch', sbatch_sh_filepath], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        out = out.decode('utf-8')
        err = err.decode('utf-8')
        print(mpi_rank_header + out)
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

    print('* {}'.format(mpi_rank_header))

    sys.stdout.flush()

    if not exit_right_after_submission:

        status_check_interval = 5.0 #10.0

        err_log_check = dict(
            interval=60.0, func=check_unable_to_open_mode_w_File_exists,
            job_name=job_name)

        used_nodes = wait_for_completion(
            job_ID_str, status_check_interval, err_log_check=err_log_check)
        #print('Used Nodes: {0}'.format(used_nodes))

    slurm_out_filepath = '{job_name}.{job_ID_str}.out'.format(
        job_name=job_name, job_ID_str=job_ID_str)
    slurm_err_filepath = '{job_name}.{job_ID_str}.err'.format(
        job_name=job_name, job_ID_str=job_ID_str)

    return job_ID_str, slurm_out_filepath, slurm_err_filepath

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
        mpi_rank = MPI.COMM_WORLD.Get_rank() + 1
        mpi_rank_header = 'MPI Rank #{:d}: '.format(mpi_rank)
    except:
        mpi_rank = 0 # MPI NOT being used
        mpi_rank_header = ''

    return mpi_rank_header

def check_unable_to_open_mode_w_File_exists(err_filepath):
    """"""

    if os.path.exists(err_filepath):
        with open(err_filepath, 'r') as f:
            contents = f.read()

        if ('unable to open' in contents) and ('in mode w: File exists' in contents):
            abort = True

            job_ID_str = err_filepath.split('.')[-2]

            print('\n##### Error: unable to open in mode w: File exists #####')
            print('Cancelling the Pelegant job {}'.format(job_ID_str))

            err_counter = 0
            while True:
                p = Popen('scancel {}'.format(job_ID_str),
                          stdout=PIPE, stderr=PIPE, shell=True)
                out, err = p.communicate()
                out = out.decode('utf-8')
                err = err.decode('utf-8')
                print(out)
                if err:
                    err_counter += 1

                    if err_counter >= 10:
                        print(err)
                        raise RuntimeError('Encountered error during cancellation of failed Pelegant run')
                    else:
                        print(err)
                        sys.stdout.flush()
                        time.sleep(30.0)
                        continue
                else:
                    break

        else:
            abort = False
    else:
        abort = False

    return abort

def wait_for_completion(
    job_ID_str, status_check_interval, timelimit_action=None, err_log_check=None):
    """"""

    t0 = t_err_log = time.time()

    if timelimit_action is not None:
        assert callable(timelimit_action['abort_func'])
        timelimit_sec = \
            get_cluster_time_limits()[timelimit_action['partition_name']] \
            - timelimit_action['margin_sec']
    else:
        timelimit_sec = None

    err_counter = 0

    while True:
        p = Popen('squeue -o "%.12i %R"',
                  stdout=PIPE, stderr=PIPE, shell=True)
        out, err = p.communicate()
        out = out.decode('utf-8')
        err = err.decode('utf-8')

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

        job_ID_res_list = [
            L.split() for L in out.split('\n')[1:] if L.strip() != '']
        if job_ID_res_list == []: # Job is finished
            break
        else:
            job_ID_list, resource_list = zip(*job_ID_res_list)

        if job_ID_str not in job_ID_list: # Job is finished
            break
        else:
            used_nodes = resource_list[job_ID_list.index(job_ID_str)]

            if timelimit_sec is not None:

                dt = time.time() - t0

                if dt >= timelimit_sec:

                    print('Time limit of selected partition is nearing, '
                          'so performing a graceful abort task now.')
                    timelimit_action['abort_func'](
                        *timelimit_action.get('abort_func_args', ()))

                    break

            if err_log_check is not None:

                dt = time.time() - t_err_log

                if dt >= err_log_check['interval']:
                    if err_log_check['func']('{}.{}.err'.format(
                        err_log_check['job_name'], job_ID_str)):
                        break

                    t_err_log = time.time()

            time.sleep(status_check_interval)

    print('Total Elapsed [s]: {0:.1f}'.format(time.time() - t0))

#----------------------------------------------------------------------
def get_cluster_time_limits():
    """"""

    p = Popen('sinfo', stdout=PIPE, stderr=PIPE)

    out , err = p.communicate()
    out = out.decode('utf-8')
    err = err.decode('utf-8')

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

    return timelimit_d
