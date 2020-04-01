import shlex
from subprocess import Popen, PIPE
import re

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
        'NodeName=([\w\d\-]+)\s+[\w=\s]+CPUAlloc=(\d+)\s+CPUTot=(\d+)\s+CPULoad=([\d\.]+)',
        out)

    nMaxNodeNameLen = max(
        [len(s) for s in list(list(zip(*parsed))[0]) +
         ['apcpu-[001-005]', 'cpu-[019-026]',
          'cpu-[002-005],[007-015]']])

    node_name = 'Node-Name'
    print(f'#{node_name:>{nMaxNodeNameLen:d}s} :: Alloc / Tot :: CPU Load')
    for node_name, nAlloc, nTot, cpu_load in parsed:
        print(f'{node_name:>{nMaxNodeNameLen+1:d}s} :: {nAlloc:>5s} / {nTot:>3s} :: {cpu_load:>7s}')
    print('###################################################')
    for combined_node_name, _template, node_num_range in (
        ['apcpu-[001-005]', 'apcpu-{:03d}', range(1, 5+1)],
        ['cpu-[019-026]', 'cpu-{:03d}', range(19, 26+1)],
        ['cpu-[002-005],[007-015]', 'cpu-{:03d}',
         list(range(2, 5+1)) + list(range(7, 15+1))],
        ):

        _nAlloc = _nTot = _cpu_load = 0
        node_list = [_template.format(i) for i in node_num_range]
        for node_name, nAlloc, nTot, cpu_load in parsed:
            if node_name in node_list:
                _nAlloc += int(nAlloc)
                _nTot += int(nTot)
                _cpu_load += float(cpu_load)
        nAlloc = '{:d}'.format(_nAlloc)
        nTot = '{:d}'.format(_nTot)
        cpu_load = '{:.2f}'.format(_cpu_load)
        node_name = combined_node_name
        print(f'{node_name:>{nMaxNodeNameLen+1:d}s} :: {nAlloc:>5s} / {nTot:>3s} :: {cpu_load:>7s}')
