# Test command:
#   $ srun -n 1 python test_mpi_re_init_crash.py
#
# If I had "$ module load elegant-latest", which load MPI-related stuff, then
# I get the following command output due to apparent re-initialization attempt
# of MPI:
'''
[cli_0]: write_line error; fd=11 buf=:cmd=init pmi_version=1 pmi_subversion=1
:
system msg for write_line failure : Bad file descriptor
[cli_0]: Unable to write to PMI_fd
[cli_0]: write_line error; fd=11 buf=:cmd=get_appnum
:
system msg for write_line failure : Bad file descriptor
Fatal error in PMPI_Init_thread: Other MPI error, error stack:
MPIR_Init_thread(572):
MPID_Init(175).......: channel initialization failed
MPID_Init(463).......: PMI_Get_appnum returned -1
[cli_0]: write_line error; fd=11 buf=:cmd=abort exitcode=1094415
:
system msg for write_line failure : Bad file descriptor
'''
# This does not happen, if I do not have the "$ module load" command executed.
# Even if the "$ module load" is executed, if I issue
#   $ srun --mpi=pmix -n 1 python test_mpi_re_init_crash.py
# Then no hard crash occurs.

from subprocess import Popen, PIPE
import shlex

if __name__ == '__main__':

    if True:
        cmd = 'python -c "import pyelegant as pe"'
        #cmd = 'python -c "from mpi4py import MPI"'

        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    else:

        cmd = 'python import_mpi.py'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')

    out, err = p.communicate()

    print(out.strip())
    if err.strip():
        print('*** stderr ***')
        print(err.strip())

