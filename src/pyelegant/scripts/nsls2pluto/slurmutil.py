import shlex
from subprocess import Popen, PIPE
import re
import argparse
import getpass
import time
from datetime import datetime

from .. import facility_name
from ..nsls2pluto import get_n_free_cores, sendRunCompleteMail

if facility_name == 'nsls2apcluster':
    #                 (combined_node_name) (_template)   (node_num_range)
    GROUPED_NODES = [('apcpu-[001-005]', 'apcpu-{:03d}', range(1, 5+1)), 
                     ('cpu-[019-026]', 'cpu-{:03d}', range(19, 26+1)),
                     ('cpu-[002-005],[007-015]', 'cpu-{:03d}',
                      list(range(2, 5+1)) + list(range(7, 15+1))),
                     ]
    
elif facility_name == 'nsls2pluto':
    GROUPED_NODES = [('gpu[001-012]', 'gpu{:03d}', range(1, 12+1)), 
                     ('hpc[001-014]', 'hpc{:03d}', range(1, 14+1)),
                     ('master[1-2]', 'master{:d}', range(1, 2+1)), 
                     ('submit[1-2]', 'submit{:d}', range(1, 2+1)),
                     ]
else:
    raise ValueError(f'Invalid "facility_name": {facility_name}')

def chained_Popen(cmd_list):
    """"""

    if len(cmd_list) == 1:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    else:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        for cmd in cmd_list[1:-1]:
            p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE)
        cmd = cmd_list[-1]
        p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    out, err = p.communicate()

    return out, err, p.returncode

def print_queue():
    """"""

    cmd = 'squeue -o "%.9i %.9P %.18j %.8u %.2t %.10M %.10L %.6D %.4C %R"'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    print(out)

def print_load():
    """"""

    cmd = 'scontrol show node'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    out, err = p.communicate()

    parsed = re.findall(
        'NodeName=([\w\d\-]+)\s+[\w=\s]+CPUAlloc=(\d+)\s+CPUTot=(\d+)\s+CPULoad=([\d\.N/A]+)',
        out)

    nMaxNodeNameLen = max(
        [len(s) for s in 
         list(list(zip(*parsed))[0]) + 
         [v[0] for v in GROUPED_NODES]])

    node_name = 'Node-Name'
    print(f'#{node_name:>{nMaxNodeNameLen:d}s} :: Alloc / Tot :: CPU Load')
    for node_name, nAlloc, nTot, cpu_load in parsed:
        print(f'{node_name:>{nMaxNodeNameLen+1:d}s} :: {nAlloc:>5s} / {nTot:>3s} :: {cpu_load:>7s}')
    print('###################################################')
    for combined_node_name, _template, node_num_range in GROUPED_NODES:
        _nAlloc = _nTot = _cpu_load = 0
        node_list = [_template.format(i) for i in node_num_range]
        for node_name, nAlloc, nTot, cpu_load in parsed:
            if node_name in node_list:
                _nAlloc += int(nAlloc)
                _nTot += int(nTot)
                if cpu_load != 'N/A':
                    _cpu_load += float(cpu_load)
                else:
                    _cpu_load += float('nan')
        nAlloc = '{:d}'.format(_nAlloc)
        nTot = '{:d}'.format(_nTot)
        cpu_load = '{:.2f}'.format(_cpu_load)
        node_name = combined_node_name
        print(f'{node_name:>{nMaxNodeNameLen+1:d}s} :: {nAlloc:>5s} / {nTot:>3s} :: {cpu_load:>7s}')

def scancel_by_regex_jobname():
    """
    """

    parser = argparse.ArgumentParser(
        prog='pyele_slurm_scancel_regex_job_name',
        description='SLURM scancel command enhanced with regular experssion pattern matching')
    parser.add_argument('job_name_pattern', type=str,
                        help='act only on jobs whose names match this regex pattern')
    parser.add_argument(
        '-d', '--dryrun', default=False, action='store_true',
        help='Print which JOB IDs will be terminated without actually doing so.')

    args = parser.parse_args()

    queue_cmd = f'squeue --user={getpass.getuser()} -o "%.9i %.18j"'

    cmd_list = [queue_cmd,
                f'grep {args.job_name_pattern}']

    result, err, returncode = chained_Popen(cmd_list)

    if args.dryrun:
        if result.strip() == '':
            print('No job name match found. No jobs will be terminated.')
        else:
            print('Only the following jobs will be terminated:')
            print('(JOBID, NAME)')
            print(result)

    else:
        job_ID_list = [L.split()[0].strip() for L in result.splitlines()]

        cmd = 'scancel ' + ' '.join(job_ID_list)
        print(f'Executing "$ {cmd}"')
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        print(out)
        if err:
            print('** stderr **')
            print(err)


def notify_on_num_free_cores_change():

    parser = argparse.ArgumentParser(
        prog='pyele_slurm_nfree_notify',
        description='Send emails upon change in the number of available cores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('email_address', type=str, help='Full email address')
    parser.add_argument('-p', '--partition', type=str, default='normal',
                        help='Partition name to monitor')
    parser.add_argument('-c', '--check-interval', type=int, default=60,
                        help='Interval in seconds between each check')
    parser.add_argument('-n', '--n-min-free', type=int, default=0,
                        help='Min. number of free cores for sending emails')
    parser.add_argument('-i', '--sleep-hr-ini', type=int, default=0,
                        help='Starting hour for dormant period')
    parser.add_argument('-f', '--sleep-hr-fin', type=int, default=7,
                        help='Ending hour for dormant period')    
    
    args = parser.parse_args()    
    
    _notify_on_num_free_cores_change(
        args.email_address, 
        partition=args.partition, 
        check_interval=args.check_interval, 
        n_min_free=args.n_min_free,
        sleep_hr_ini=args.sleep_hr_ini, 
        sleep_hr_fin=args.sleep_hr_fin)
    
def _notify_on_num_free_cores_change(
    email_address, partition='normal', check_interval=60.0, n_min_free=0,
    sleep_hr_ini=0, sleep_hr_fin=7):
    """
    check_interval [s]
    
    n_min_free (int): Only sends emails when # of free cores >= this integer.
    
    sleep_hr_ini, sleep_hr_fin (int): 0-23
      During the hours between "sleep_hr_ini" and "sleep_hr_fin", no checking is
      performed. For example, if you specify sleep_hr_ini=0, sleep_hr_fin=7,
      no checking is performed between midnight and 7 am.
    """
    
    assert sleep_hr_ini in range(24)
    assert sleep_hr_fin in range(24)

    print(f'When # of free cores change on the partition "{partition}", '
          f'emails will be sent to "{email_address}".')

    n_free_prev = get_n_free_cores(partition=partition)
    timestamp = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
    print(f'{timestamp}: n_free = {n_free_prev}')
    
    notified_ts = None
            
    while True:
        n_free_now = get_n_free_cores(partition=partition)

        d = datetime.now().astimezone()
        timestamp = d.strftime('%Y-%m-%dT%H:%M:%S%z')

        if (n_free_now != n_free_prev) and (n_free_now >= n_min_free):
            
            if notified_ts is None:
                send_email = True
            else:
                # If the last notification email was sent less than 1 hour ago,
                # do not notify to reduce the number of emails.
                if (d - notified_ts).total_seconds() >= 60 * 60:
                    send_email = True
                else:
                    send_email = False
            
            if send_email:
                email_content = timestamp
                sendRunCompleteMail(f'"{partition}": # of free cores = {n_free_now}', email_content)
                n_free_prev = n_free_now
                notified_ts = d
        
        dormant = False
        
        if sleep_hr_ini == sleep_hr_fin:
            # No dormant time
            actual_interval = check_interval
        elif sleep_hr_ini < sleep_hr_fin:
            if sleep_hr_ini <= d.hour < sleep_hr_fin:
                # Go dormant
                wakeup_d = datetime.now().astimezone()
                wakeup_d = wakeup_d.replace(hour=sleep_hr_fin, minute=0, second=0)
                actual_interval = (wakeup_d - d).total_seconds()
                dormant = True
            else:
                actual_interval = check_interval
        else: # if sleep_hr_ini > sleep_hr_fin
            if d.hour >= sleep_hr_ini:
                # Go dormant
                wakeup_d = datetime.now().astimezone()
                wakeup_d = wakeup_d.replace(day=d.day+1, hour=sleep_hr_fin, minute=0, second=0)
                actual_interval = (wakeup_d - d).total_seconds()
                dormant = True                
            elif d.hour < sleep_hr_fin:
                # Go dormant
                wakeup_d = datetime.now().astimezone()
                wakeup_d = wakeup_d.replace(hour=sleep_hr_fin, minute=0, second=0)
                actual_interval = (wakeup_d - d).total_seconds()
                dormant = True
            else:
                actual_interval = check_interval
                
        if dormant:
            wakeup_timestamp = wakeup_d.strftime('%Y-%m-%dT%H:%M:%S%z')
            print(
                f'* Going dormant @ {timestamp} until {wakeup_timestamp}. '
                f'Next check in {actual_interval:.1f} [s]')
                    
        #print(actual_interval)
        
        time.sleep(actual_interval)
        if dormant:
            timestamp = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
            print(f'** Came out of dormancy @ {timestamp}.')