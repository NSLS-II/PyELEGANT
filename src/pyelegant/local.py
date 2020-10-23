import os
from subprocess import Popen, PIPE, STDOUT

from . import std_print_enabled

def enable_stdout():
    """"""

    std_print_enabled['out'] = True

def disable_stdout():
    """"""

    std_print_enabled['out'] = False

def enable_stderr():
    """"""

    std_print_enabled['err'] = True

def disable_stderr():
    """"""

    std_print_enabled['err'] = False

def run(ele_filepath, macros=None, print_cmd=False,
        print_stdout=True, print_stderr=True, tee_to=None, tee_stderr=True):
    """"""

    cmd_list = ['elegant', ele_filepath]

    if macros is not None:
        macro_str_list = []
        for k, v in macros.items():
            macro_str_list.append('='.join([k, v]))
        cmd_list.append('-macro=' + ','.join(macro_str_list))

    if tee_to is None:
        if print_cmd:
            print('$ ' + ' '.join(cmd_list))
        p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, env=os.environ)
    else:
        if tee_stderr:
            p1 = Popen(cmd_list, stdout=PIPE, stderr=STDOUT, env=os.environ)
        else:
            p1 = Popen(cmd_list, stdout=PIPE, stderr=PIPE, env=os.environ)

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

        p = Popen(cmd_list_2, stdin=p1.stdout, stdout=PIPE, stderr=PIPE,
                  env=os.environ)
    out, err = p.communicate()
    out, err = out.decode('utf-8'), err.decode('utf-8')

    if out and print_stdout:
        print(out)

    if err and print_stderr:
        print('ERROR:')
        print(err)
