import shlex
from subprocess import Popen, PIPE
import re
import argparse
import getpass

from .. import facility_name

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

