from importlib import import_module

from .. import remote

cluster_status = import_module(
    ".main", f"pyelegant.guis.{remote.REMOTE_NAME}.cluster_status"
)
genreport_wizard = import_module(
    ".main", f"pyelegant.guis.{remote.REMOTE_NAME}.genreport_wizard"
)

gui_slurm_main = cluster_status.main
gui_report_wiz_main = genreport_wizard.main
