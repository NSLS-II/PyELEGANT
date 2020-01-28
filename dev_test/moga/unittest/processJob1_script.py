# Python script created by Y. Hidaka (01/24/2020) based on:
#
# Script processJob1
# Purpose: post-process jobs for genetic optimization of dynamic aperture and Touschek lifetime
# M. Borland, ANL, 6/2010

# Test command:
# $ cd /tmp/yhidaka
# $ python processJob1_script.py -rootname optim1-000033 -valueList "9.4929257171811816 -21.196030780539502 -9.7547463771023981 -21.496656110051283 9.5644793440344582 36.233123781164522 19.316520298455014" -tagList "S1AS1 S1AS2 S1BS3 S1BS2 S1BS1 nuxTarget nuyTarget" -oldDir /GPFS/APC/yhidaka/git_repos/nsls2cb/20200123_opt_CLS/moga -xchrom 2.0 -ychrom 2.0


import sys
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import shlex
import time
import shutil

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

def failAndExit(fileobj, msg, runID, rootname):
    """"""

    f = fileobj

    f.write(f'\nError processing data: argv =\n')
    f.write(str(sys.argv) + '\n')
    f.write(f'runID={runID}, rootname={rootname}\n')
    f.write(msg + '\n')
    f.write('Files for this rootname:\n')
    for fp in Path.cwd().glob(f'{rootname}.*'):
        f.write(str(fp)+'\n')

    f.close()

    sys.exit(1)

def waitForFiles(rootname, extensionList):
    """
    wait for the files to all exist
    can be helpful if the file server has problems
    """

    #print(f'cwd = {str(Path.cwd())}')

    tries = 5
    for _ in range(tries):
        ok = True
        notFoundList = []
        for ext in extensionList:
            p = Popen(
                ['sddscheck', f'{rootname}.{ext}'], stdout=PIPE, stderr=PIPE,
                encoding='utf-8')
            sddscheck_out, err = p.communicate()
            if sddscheck_out.strip() != 'ok':
                notFoundList.append(f'{rootname}.{ext} ({sddscheck_out})')
                ok = False

        if ok:
            break
        else:
            time.sleep(5.0)

    return notFoundList

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-rootname', help='Root name, e.g., "optim1-000012"')
    parser.add_argument(
        '-tagList', help='List of parameter names, e.g., "S1 S2 nuxTarget nuyTarget"')
    parser.add_argument(
        '-valueList', help='List of parameter names, e.g., "0.12 -1.5 36.203 19.321"')
    parser.add_argument('-oldDir')
    parser.add_argument('-xchrom')
    parser.add_argument('-ychrom')

    args = parser.parse_args()
    #print(args)

    rootname = args.rootname
    oldDir = args.oldDir
    xchrom_str = args.xchrom
    ychrom_str = args.ychrom

    if ',' in args.tagList:
        sep = ','
    else:
        sep = ' '
    tagList = [v.strip() for v in args.tagList.split(sep) if v.strip() != '']

    if ',' in args.valueList:
        sep = ','
    else:
        sep = ' '
    valueList = [v.strip() for v in args.valueList.split(sep) if v.strip() != '']
    # ^ This list of values should NOT be a list of floats. Rather, the list
    #   should be a list of strings that represent the float values.

    if len(tagList) != len(valueList):
        raise ValueError((f'Size of "tagList" ({len(tagList)}) must match with '
                          f'size of "valueList" ({len(valueList)})'))

    #print(tagList)
    #print(valueList)

    # Check semaphores to see if processing has already started or been done
    if Path(f'{rootname}.procStart').exists():
        print(f'{rootname}.procStart already exists. So, not starting another job to process.')
        sys.exit(0)
    if Path(f'{rootname}.proc').exists():
        print(f'{rootname}.proc already exists. So, not starting another job to process.')
        sys.exit(0)

    Path(f'{rootname}.procStart').write_text('')

    plog_filepath = f'{oldDir}/{rootname}.plog'
    plog = open(plog_filepath, 'w')
    plog.write(pe.util.get_current_local_time_str() + '\n')

    # Extract the run ID from the rootname
    runID = rootname.split('-')[-1]

    extensionList = ['inp', 'aper', 'mmap', 'twi']
    notFoundFileList = waitForFiles(rootname, extensionList)
    if notFoundFileList:
        msg = 'Unabled to find needed files: {}'.format(', '.join(notFoundFileList))
        failAndExit(plog, msg, runID, rootname)

    plog.write(f'{pe.util.get_current_local_time_str()}: files found\n')

    plog.write(f'runID: {runID}\n')

    # Compute the maximum stable momentum deviation by looking at
    # tunes vs momentum.  This is simply to avoid major resonance
    # crossings that might be stable (e.g., due to large tune shift
    # with amplitude), but that make me nervous.

    deltaLimit = 2.35e-2
    if Path(f'{rootname}.w1').exists() and Path(f'{rootname}.fin').exists():
        cmd = f'sddscollapse {rootname}.fin {rootname}.finc'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()

        # First, look for cases where the particle is lost and find the minimum
        # |delta|
        cmd_list = [
            (f'sddsprocess {rootname}.finc -pipe=out -nowarning '
             '"-define=col,deltaAbs,MALIN.DP abs" '
             '-filter=column,Transmission,1,1,! '
             '-process=deltaAbs,min,%sMin'),
            'sdds2stream -pipe -parameter=deltaAbsMin',
        ]
        result, err, returncode = chained_Popen(cmd_list)
        if returncode:
            failAndExit(plog, f'delta limit 1: {result}', runID, rootname)


        plog.write(f'Delta limit 1 : {result}\n')
        if (not result.endswith('e+308')) and (float(result) < deltaLimit):
            deltaLimit = float(result)

        # Second, look for undefined tunes
        cmd_list = [
            (f'sddsnaff {rootname}.w1 -pipe=out '
             '-column=Pass -pair=Cx,Cxp -pair=Cy,Cyp -terminate=frequencies=1 '),
            'sddscombine -pipe -merge',
            f'sddsxref -pipe {rootname}.finc -take=MALIN.DP',
            ('sddsprocess -pipe "-define=column,deltaAbs,MALIN.DP abs" '
             '-filter=col,deltaAbs,1e-3,1'),
            'sddssort -pipe -column=deltaAbs',
            (f'sddsprocess -nowarning -pipe=in {rootname}.naff '
             f'-process=C*Frequency,first,%s0'),
        ]
        result, err, returncode = chained_Popen(cmd_list)
        if returncode:
            failAndExit(plog, f'delta limit 2: {result}', runID, rootname)

        cmd_list = [
            (f'sddsprocess -nowarning {rootname}.naff -pipe=out '
             '-filter=col,CxFrequency,-1,-1,CyFrequency,-1,-1,| '
             '-process=deltaAbs,min,%sMin'),
            'sdds2stream -pipe -parameter=deltaAbsMin',
        ]
        result, err, returncode = chained_Popen(cmd_list)
        if returncode:
            failAndExit(plog, f'delta limit 3: {result}', runID, rootname)

        plog.write(f'Delta limit 2: {result}\n')
        if (not result.endswith('e+308')) and (float(result) < deltaLimit):
            deltaLimit = float(result)

        # Look for integer or half-integer crossings
        for plane in ['x', 'y']:
            cmd_list = [
                (f'sddsprocess {rootname}.naff -pipe=out -nowarning '
                 f'"-define=column,badTune,C{plane}Frequency 2 * int C{plane}Frequency0 2 * int - abs" '
                 '-filter=col,badTune,1,1 -process=deltaAbs,min,%sMin'),
                'sdds2stream -pipe -parameter=deltaAbsMin',
            ]
            result, err, returncode = chained_Popen(cmd_list)
            if returncode:
                failAndExit(plog, f'delta limit 4 ({plane}): {result}', runID, rootname)

            plog.write(f'Delta limit 3 ({plane}): {result}\n')
            if (not result.endswith('e+308')) and (float(result) < deltaLimit):
                deltaLimit = float(result)
        plog.write(f'New delta limit: {deltaLimit:.16g}\n')

    # Compute the Touschek lifetime for 100 mA in 24 bunches.
    # The local momentum aperture is capped at the value we just computed usig the -deltaLimit option
    if True:
        cmd = (
            f'bash computeLifetime -rootname {rootname} -current 100 -bunches 24 '
            f'-coupling 0.01 -deltaLimit {1e2 * deltaLimit:.16g}')
    else:
        cmd = (
            f'python computeLifetime.py -rootname {rootname} -current 100 -bunches 24 '
            f'-coupling 0.01 -deltaLimit {1e2 * deltaLimit:.16g}')
    #print(cmd)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    result, err = p.communicate()
    if p.returncode:
        msg = f'{result}\n*** stderr ***\n{err}'
        failAndExit(plog, msg, runID, rootname)
    plog.write('Lifetime done\n')

    # Postprocess to find the DA area and create the penalty-function columns
    # ChromPenalty, DAPenalty, and LTPenalty for chromaticity, DA, and lifetime

    # The processing of the .aper file uses the clipped DA boundary (yClipped
    # and xClipped).  The xClipped data is further limited to <7mm, because at
    # APS we inject on the negative side and don't care about positive DA beyond
    # that.

    cmd_list = [
        (f'sddsprocess {rootname}.mmap -pipe=out '
         '"-define=column,deltaLimit,deltaPositive deltaNegative abs 2 minn" '
         '-process=deltaLimit,min,%sMin'),
        f'sddscollapse -pipe=in {rootname}.mmapc',
    ]
    result, err, returncode = chained_Popen(cmd_list)
    if returncode:
        msg = f'Processing Step #1: {result}\n*** stderr ***\n{err}'
        failAndExit(plog, msg, runID, rootname)
    print('Processing Step #1 completed')
    #
    cmd = f'sddscollapse {rootname}.twi {rootname}.twic'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    result, err = p.communicate()
    if p.returncode:
        msg = f'Processing Step #2: {result}\n*** stderr ***\n{err}'
        failAndExit(plog, msg, runID, rootname)
    print('Processing Step #2 completed')
    #
    cmd = f'sddscollapse {rootname}.ltime {rootname}.ltimec'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    result, err = p.communicate()
    if p.returncode:
        msg = f'Processing Step #3: {result}\n*** stderr ***\n{err}'
        failAndExit(plog, msg, runID, rootname)
    print('Processing Step #3 completed')
    #
    cmd_list = [
        (f'sddsprocess {rootname}.aper -pipe=out '
         f'"-define=column,xClipped1,xClipped 0.007 > ? 0.007 : xClipped $ ,units=m" '
         f'-process=yClipped,integral,Area1,functionOf=xClipped1'),
        'sddscollapse -pipe',
        f'sddsxref {rootname}.ltimec -pipe -take=*',
        f'sddsxref {rootname}.inp -pipe -take=*',
        f'sddsxref {rootname}.mmapc -pipe -take=* -nowarning',
        f'sddsxref {rootname}.twic -pipe -take=*',
        (f'sddsprocess -pipe=in {rootname}.proc1 '
         f'"-redefine=column,deltaLimitLT,{deltaLimit:.16g}" '
         f'"-reprint=column,runName,{rootname}" '
         f'"-redefine=column,runID,{runID},type=long" '
         f'"-redefine=column,Time,{time.time():.0f},units=s,type=long" '
         f'"-redefine=column,ChromPenalty,dnux/dp {xchrom_str} .01 sene  dnuy/dp {ychrom_str} .01 sene +" '
         f'"-redefine=column,DAPenalty,Area1 chs" '
         f'"-redefine=column,LTPenalty,tLifetime chs"'
         ),
    ]
    result, err, returncode = chained_Popen(cmd_list)
    if returncode:
        msg = f'Processing Step #4: {result}\n*** stderr ***\n{err}'
        failAndExit(plog, msg, runID, rootname)
    print('Processing Step #4 completed')


    plog.write('Processing done\n')

    try:
        shutil.move(f'{rootname}.proc1', f'{rootname}.proc')
    except:
        plog.write(f'Failed to move {rootname}.proc1 into {rootname}.proc')

    plog.close()


