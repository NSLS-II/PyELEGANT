# Python script created by Y. Hidaka (01/24/2020) based on:
#
# Script computeLifetime
# Used to compute the touschek lifetime from the local momentum aperture
# including a cap on the maximum momentum aperture.

# Test command:
# $ cd /tmp/yhidaka
# $ python computeLifetime.py -rootname optim1-000033 -current 100 -bunches 24 -coupling 0.01 -deltaLimit 2.35

import sys
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import shlex
import numpy as np

import pyelegant as pe # will do "$ module load elegant-latest"

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

def main(rootname, current, bunches, coupling_str, deltaLimit_str):
    """"""

    #debug = True
    debug = False

    if not Path(f'{rootname}.mmap').exists():
        sys.stderr.write(f'not found {rootname}.mmap\n')
        sys.exit(1)
    if not Path(f'{rootname}.twi').exists():
        sys.stderr.write(f'not found {rootname}.twi\n')
        sys.exit(1)

    # Figure out how many sectors are covered by the LMA data
    circumference = 1104.0
    sectors = 40
    lsector = circumference / sectors

    cmd_list = [
        f'sddsprocess {rootname}.mmap -pipe=out -process=s,max,sMax',
        'sdds2stream -pipe -parameter=sMax',
    ]
    result, err, returncode = chained_Popen(cmd_list)
    if debug:
        print('Step #1')
        print(returncode)
        print(result)
        if err:
            print('*** stderr ***')
            print(err)
    sMax = float(result)

    msectors = int(sMax / lsector + 0.5)

    if msectors < 40:
        cmd_list = [
            f'sddsprocess {rootname}.twi -pipe=out -process=s,max,sMax',
            f'sdds2stream -pipe -parameter=sMax',
        ]
        result, err, returncode = chained_Popen(cmd_list)
        if debug:
            print('Step #2')
            print(returncode)
            print(result)
            if err:
                print('*** stderr ***')
                print(err)
        sMax = float(result)
        number = int(sectors / msectors + 0.5)
        dup_filenames = ' '.join([f'{rootname}.mmap'] * number)
        cmd_list = [
            f'sddscombine {dup_filenames} -pipe=out',
            f'sddsprocess -pipe "-redefine=col,s,s i_page 1 - {sMax:.16g} {sectors} / * {msectors} * +,units=m"',
            f'sddscombine -pipe -merge',
            f'sddsprocess -pipe=in {rootname}.mmapxt -filter=col,s,0,{sMax:.16g}',
        ]
        if debug: print(cmd_list)
        result, err, returncode = chained_Popen(cmd_list)
        if debug:
            print('Step #3')
            print(returncode)
            print(result)
            if err:
                print('*** stderr ***')
                print(err)
        # $rootname.mmapxt now contains the LMA for the full ring, created by
        # replicating the data for {msectors} sectors a sufficient number of times
        mmapFile = f'{rootname}.mmapxt'

    else:
        # {rootname}.mmap is for the whole ring already
        mmapFile = f'{rootname}.mmap'

    # Compute bunch length using experimental curve for length (in mm) vs
    # current (in mA) for APS
    bunchCurrent = current / bunches
    length = 25.1 * (
        bunchCurrent**(0.1484+0.0346*np.log10(bunchCurrent)) ) * 0.29979

    charge = bunchCurrent * (1104.0 / 2.9979e8) * 1e6

    cmd_list = [f'sdds2stream -parameter=ex0 {rootname}.twi']
    result, err, returncode = chained_Popen(cmd_list)
    if debug:
        print('Step: extract ex0')
        print(returncode)
        print(result)
        if err:
            print('*** stderr ***')
            print(err)
    ex0 = float(result)

    cmd_list = [f'sdds2stream -parameter=Sdelta0 {rootname}.twi']
    result, err, returncode = chained_Popen(cmd_list)
    if debug:
        print('Step: extract Sdelta0')
        print(returncode)
        print(result)
        if err:
            print('*** stderr ***')
            print(err)
    Sdelta0 = float(result)

    cmd_list = [(
        f'touschekLifetime {rootname}.ltime -twiss={rootname}.twi '
        f'-aperture={mmapFile} -coupling={coupling_str} '
        f'-emitxInput={ex0:.16g} -deltaInput={Sdelta0:.16g} '
        f'-charge={charge:.16g} -length={length:.16g} '
        f'-deltaLimit={deltaLimit_str} -ignoreMismatch')]
    result, err, returncode = chained_Popen(cmd_list)
    if debug:
        print('Step: touschekLifetime')
        print(returncode)
        print(result)
        if err:
            print('*** stderr ***')
            print(err)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-rootname', help='Root name, e.g., "optim1-000012"')
    parser.add_argument('-current')
    parser.add_argument('-bunches')
    parser.add_argument('-coupling')
    parser.add_argument('-deltaLimit')

    args = parser.parse_args()
    #print(args)

    rootname = args.rootname
    current = float(args.current)
    bunches = int(args.bunches)
    coupling_str = args.coupling
    deltaLimit_str = args.deltaLimit

    main(rootname, current, bunches, coupling_str, deltaLimit_str)