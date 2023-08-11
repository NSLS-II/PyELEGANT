import importlib

from .. import remote

slurmutil = importlib.import_module(
    ".slurmutil", f"pyelegant.scripts.{remote.REMOTE_NAME}"
)

slurm_print_queue = slurmutil.print_queue
slurm_print_load = slurmutil.print_load
slurm_scancel_by_regex_jobname = slurmutil.scancel_by_regex_jobname
slurm_notify_on_num_free_cores_change = slurmutil.notify_on_num_free_cores_change
