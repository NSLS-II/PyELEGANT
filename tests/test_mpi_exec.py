from pathlib import Path
import shutil

import pyelegant as pe

def multiply(v, multiplier):
    
    return v * multiplier

if __name__ == '__main__':
    
    remote_opts = dict(
        job_name='test', ntasks=2, partition='debug', qos='long',
        time='30:00'
    )
    
    exec_d = pe.remote.launch_mpi_python_executor(
        remote_opts, paths=[Path.cwd()], err_log_check=None)

    module_name = 'test_mpi_exec'
    func_name = 'multiply'
    param_list = list(range(5))
    args = (2,)
    
    try:
        results, dt = pe.remote.submit_job_to_mpi_executor(
            exec_d, module_name, func_name, param_list,
            args, check_interval=1.0, print_stdout=True, print_stderr=True,
        )
    except:        
        # Without the following line, the temp directory will be deleted
        # on program exit, which makes debugging difficult.
        shutil.copytree(exec_d['tmp_dir'].name, exec_d['tmp_dir'].name + '.debug')
        raise
    
    print(results)
    print(dt)
    
    pe.remote.stop_mpi_executor(exec_d, del_log_files=True)

