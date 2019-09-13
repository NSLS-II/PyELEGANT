from subprocess import Popen, PIPE

def run(ele_filepath, macros=None, print_cmd=False, print_stdout=True, print_stderr=True):
    """"""

    cmd_list = ['elegant', ele_filepath]

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
