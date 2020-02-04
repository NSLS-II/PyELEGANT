# Test command:
# $ python runJob1.py -rootname optim1-000033 -tagList 'S1AS1 S1AS2 S1BS3 S1BS2 S1BS1 nuxTarget nuyTarget' -valueList '9.4929257171811816 -21.196030780539502 -9.7547463771023981 -21.496656110051283 9.5644793440344582 36.233123781164522 19.316520298455014' >& optim1-000033.log

import sys
import os
import argparse
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
import shlex
import tempfile

import pyelegant as pe

def sendRunCompleteMail(successful):
    """"""

    import getpass

    username = getpass.getuser()

    if successful:
        msg = 'Python run completed SUCCESSFULLY'
    else:
        msg = 'Python run FAILED'

    p = Popen(["mail", f"{username}@bnl.gov", "-s", msg], stdin=PIPE)
    p.communicate(msg.encode())

if __name__ == '__main__':

    #notify_via_email = True
    notify_via_email = False

    #production = True
    production = False

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('-rootname', help='Root name, e.g., "optim1-000012"')
        parser.add_argument(
            '-tagList', help='List of parameter names, e.g., "S1 S2 nuxTarget nuyTarget"')
        parser.add_argument(
            '-valueList', help='List of parameter names, e.g., "0.12 -1.5 36.203 19.321"')

        args = parser.parse_args()
        #print(args)

        rootname = args.rootname

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

        ksixTarget = +2.0
        ksiyTarget = +2.0

        if production:
            n_turns = 400
            n_lines = 21
            mom_aper_elem_name_pattern = 'S*'
            off_mom_clo_nIndexes = 101
        else:
            n_turns = 10
            n_lines = 3
            mom_aper_elem_name_pattern = 'S1'
            off_mom_clo_nIndexes = 11

        # Prepare macro option for passing to elegant
        macros = dict(
            rootname=rootname, xchrom=f'{ksixTarget:.1f}', ychrom=f'{ksiyTarget:.1f}',
            turns=f'{n_turns:d}', n_lines=f'{n_lines:d}',
            mom_aper_elem_name_pattern=mom_aper_elem_name_pattern,
            off_mom_clo_nIndexes=f'{off_mom_clo_nIndexes:d}')
        # ^ All the values of "macros" dict must be of type "str", NOT "float".
        for tag, val in zip(tagList, valueList):
            macros[tag] = val

        # Open log file
        print(f'{rootname} {macros}')
        sys.stdout.flush()

        # Use TMPDIR if defined, otherwise make a subdirectory
        if ('TMPDIR' in os.environ) and Path(os.environ['TMPDIR']).exists():
            #tmpdir = Path(os.environ['TMPDIR'])
            tmpdir_obj = tempfile.TemporaryDirectory(
                prefix='tmpRunJob_', dir=Path(os.environ['TMPDIR']))
        else:
            #tmpdir = Path(rootname).mkdir(exists_ok=True)
            tmpdir_obj = tempfile.TemporaryDirectory(
                prefix='tmpRunJob_', dir=Path(rootname))
        tmpdir = Path(tmpdir_obj.name)

        # Copy all input files to the temporary directory
        print('Copying files: ', end='')
        sys.stdout.flush()
        for fname in [
            'matchTemplate.ele', f'{rootname}.inp', 'computeLifetime.py',
            'evalTemplate.ele', 'processJob1.py',
            'sample.lte']:
            shutil.copy(fname, tmpdir.joinpath(fname))
        print('done')
        sys.stdout.flush()

        oldDir = os.getcwd()

        os.chdir(tmpdir)

        cmd = 'hostname'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()
        print(f'\nHostname = {out.strip()}\n')

        # Perform linear matching
        print('linear matching: ', end='')
        pe.run('matchTemplate.ele', macros=macros, print_stdout=True, print_stderr=True)
        sys.stdout.flush()

        # Perform chromaticity correction, DA, LMA
        print('running elegant for chromaticity, DA, LMA')
        print(f'Check {oldDir}/{rootname}-main.log for status')
        main_log_filepath = f'{oldDir}/{rootname}-main.log'
        print(f'tracking: ', end='')
        pe.run('evalTemplate.ele', macros=macros, print_cmd=False,
               print_stdout=False, print_stderr=True,
               tee_to=main_log_filepath, tee_stderr=True)
        sys.stdout.flush()

        if Path(f'{rootname}.done0').exists():
            run_success = True

            proc_failed = False

            if False:
                cmd = 'which python'
                p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
                out, err = p.communicate()
                print(f'Python executable used for processJob1.py: {out.strip()}')

            if False:
                # The following Popen  results in a hard crash due to MPI
                # re-initialization when launched from an MPI process. In this
                # case you must use the python import.
                cmd = (
                    f'python processJob1_script.py -rootname {rootname} '
                    f'-valueList "{args.valueList}" -tagList "{args.tagList}" -oldDir {oldDir} '
                    f'-xchrom {macros["xchrom"]} -ychrom {macros["ychrom"]}')
                print(cmd)
                print(shlex.split(cmd))
                #p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
                p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8')
                out, err = p.communicate()

                if not Path(f'{rootname}.proc').exists():
                    print('processing failed:')
                    out, err = out.strip(), err.strip()
                    print(out)
                    if err:
                        print('\n** stderr **')
                        print(err)
                    sys.stdout.flush()

                    errLog_folderpath = Path(oldDir).joinpath('errorLog')
                    os.makedirs(errLog_folderpath, exist_ok=True)
                    for fp in Path.cwd().glob(f'{rootname}*'):
                        try:
                            shutil.copy(fp, errLog_folderpath.joinpath(fp.name))
                        except:
                            print(f'Failed to copy {str(fp.resolve())}')

                    proc_failed = True
            else:
                import processJob1
                xchrom_str = macros["xchrom"]
                ychrom_str = macros["ychrom"]
                try:
                    processJob1.main(rootname, oldDir, xchrom_str, ychrom_str,
                                     tagList, valueList)
                    err = None
                except Exception as err:
                    proc_failed = True

                if proc_failed or (not Path(f'{rootname}.proc').exists()):

                    proc_failed = True

                    print('processing failed:')
                    if err:
                        print(err.args)
                        sys.stdout.flush()

                    errLog_folderpath = Path(oldDir).joinpath('errorLog')
                    os.makedirs(errLog_folderpath, exist_ok=True)
                    for fp in Path.cwd().glob(f'{rootname}*'):
                        try:
                            shutil.copy(fp, errLog_folderpath.joinpath(fp.name))
                        except:
                            print(f'Failed to copy {str(fp.resolve())}')

        else:
            run_success = False
            proc_failed = True

        for fp in Path.cwd().glob(f'{rootname}*'):
            print(fp.name)

        # Copy files back to the main directory
        for ext in ['.twi', '.proc', '.param', '.aper', '.mmap', '.w1', '.naff', '.fin']:
            try:
                shutil.copy(f'{rootname}{ext}', f'{oldDir}/{rootname}{ext}')
                print(f'Copied {rootname}{ext} to {oldDir}/{rootname}{ext}')
            except Exception as e:
                print(f'** Problem copying {rootname}{ext} to {oldDir}/{rootname}{ext}: {e.args}')

        # Create semaphore to tell the optimizer that this run is done
        Path(f'{oldDir}/{rootname}.done').write_text('')

        tmpdir_obj.cleanup()

        if notify_via_email:
            successful = run_success and (not proc_failed)
            sendRunCompleteMail(successful)

    except:

        if notify_via_email:
            successful = False
            sendRunCompleteMail(successful)

        raise
