import concurrent
from functools import partial
import gzip
import importlib
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import traceback

import dill
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from pyelegant.nsls2pluto import file_exists_w_nfs_cache_flushing

# To allow class methods usable in mpi4py
MPI.pickle.__init__(dill.dumps, dill.loads)  # <= This works for mpi4py v.3.0


def _wrapper(func, param_list, *args):
    """"""

    try:
        # Put your body of code you are trying to debug here
        return func(param_list, *args)

    except Exception as err:

        tmp = tempfile.NamedTemporaryFile(
            dir=Path.cwd(), delete=False, prefix="error.", suffix=".txt"
        )
        Path(tmp.name).write_text(traceback.format_exc())
        tmp.close()

        raise


def _start_mpi_python_executor_loop(
    tmp_dir, paths_to_prepend, dt_timeout, check_interval=5.0
):

    tStart = time.time()  # Must be wall time, i.e., don't use time.perf_counter() here

    if paths_to_prepend is not None:

        # pathlib.Path object not acceptable for sys.path and the keyword arg
        # "path" for MPIPoolExecutor(). Must be strings.
        path_list = [str(_path) for _path in paths_to_prepend]

        executor = MPIPoolExecutor(path=path_list)

        for path_str in path_list:
            sys.path.insert(0, path_str)
    else:
        executor = MPIPoolExecutor()

    tmp_dirpath = Path(tmp_dir.name)

    stop_requested = False
    job_counter = 0

    while not stop_requested:

        for fp in tmp_dirpath.glob("*"):

            # Stop the executor if a stop request is detected
            if fp.name == "stop_requested":

                stop_requested = True
                stopped_fp = Path(fp.parent.joinpath("stopped"))

                try:
                    fp.unlink()
                except:
                    pass

                break

            elif fp.name.endswith(f"_{job_counter:d}.ready"):
                job_counter += 1
                # Proceed to run the requested job
            else:
                continue

            t0_setup = time.perf_counter()

            if not fp.is_absolute():
                fp = fp.resolve()

            full_filepath_str = str(fp)

            prefix = full_filepath_str[: -len(".ready")]

            input_filepath = Path(f"{prefix}_input.dill")

            with open(input_filepath, "rb") as f:
                input_d = dill.load(f)

            output_filepath = input_d["output_filepath"]
            module_name = input_d["module_name"]
            func_name = input_d["func_name"]
            param_list = input_d["param_list"]
            args = input_d["args"]

            # mod = importlib.import_module('pyelegant.nonlin')
            mod = importlib.import_module(module_name)
            # func = getattr(mod, '_calc_chrom_track_get_tbt')
            func = getattr(mod, func_name)

            dt_so_far = time.time() - tStart
            actual_dt_timeout = dt_timeout - dt_so_far

            futures = executor.starmap(
                partial(_wrapper, func),
                [tuple([param] + list(args)) for param in param_list],
                timeout=actual_dt_timeout,
            )

            t0_run = time.perf_counter()
            try:
                results = list(futures)
            except concurrent.futures._base.TimeoutError:
                print(
                    "* Running a task, but job timeout is imminent. Shutting down MPI executor now..."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                print("* Shutdown complete.")
                raise
            t_end = time.perf_counter()

            dt = dict(setup=t0_run - t0_setup, run=t_end - t0_run)

            with gzip.GzipFile(output_filepath, "wb") as f:
                pickle.dump([results, dt], f)

            # Signal the main script that writing output file is complete and
            # ready to be read by the main script.
            output_ready_filepath = output_filepath.parent.joinpath(
                f"{output_filepath.name}.done"
            )
            output_ready_filepath.write_text("")
            file_exists_w_nfs_cache_flushing(output_ready_filepath)

        else:  # If stop_requested == False

            dt_so_far = time.time() - tStart
            if dt_so_far + check_interval > dt_timeout:
                print(
                    "* Waiting for a task, but job timeout is imminent. Shutting down MPI executor now."
                )
                stop_requested = True
                stopped_fp = Path(fp.parent.joinpath("stopped"))
            else:
                time.sleep(check_interval)

    print("A request to stop MPI executor detected. Shutting down...")
    executor.shutdown(wait=False, cancel_futures=True)
    print("* Shutdown complete.")

    # Signal the main script that the executor has been stopped.
    stopped_fp.write_text("")
    file_exists_w_nfs_cache_flushing(stopped_fp)
    print("Shutdown complete.")


if __name__ == "__main__":

    if len(sys.argv) == 3:

        paths_filepath = Path(sys.argv[1])
        dt_timeout = float(sys.argv[2])  # [sec]

        tmp_dir = paths_filepath.parent

        with open(paths_filepath, "rb") as f:
            paths_to_prepend = dill.load(f)

        _start_mpi_python_executor_loop(tmp_dir, paths_to_prepend, dt_timeout)

    else:
        raise RuntimeError
