from copy import deepcopy
import datetime
import errno
from functools import partial
import getpass
import hashlib
import os
from pathlib import Path
import pty
import re
from select import select
import shlex
import shutil
from subprocess import PIPE, Popen
import sys
import tempfile
import time
import traceback

from qtpy import QtCore, QtGui, QtWidgets, uic

Qt = QtCore.Qt

import matplotlib.pyplot as plt
import numpy as np
from ruamel import yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import SingleQuotedScalarString

import pyelegant as pe
from pyelegant.scripts.common import genreport

TEST_MODE = False


def get_slurm_partitions():
    cmd = "scontrol show partition"
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
    out, err = p.communicate()

    parsed = {}
    for k, v in re.findall("([\w\d]+)=([^\s]+)", out):
        if k == "PartitionName":
            d = parsed[v] = {}
        else:
            d[k] = v

    return parsed


def get_slurm_qos_info():

    p = Popen(
        shlex.split("sacctmgr show qos -P"), stdout=PIPE, stderr=PIPE, encoding="utf-8"
    )
    out, err = p.communicate()

    lines = [line.strip() for line in out.split("\n") if line.strip() != ""]

    header = lines[0].split("|")
    ncol = len(header)
    qos_d = {}
    for L in lines[1:]:
        vals = L.split("|")
        assert len(vals) == ncol
        name = vals[0]
        qos_d[name] = {}
        for k, v in zip(header[1:], vals[1:]):
            qos_d[name][k] = v

    return qos_d


def get_slurm_allowed_qos_list():

    username = getpass.getuser()

    if pe.remote.REMOTE_NAME == "nsls2apcluster":
        allowed_qos_list = ["default"]

    elif pe.remote.REMOTE_NAME == "nsls2pluto":
        cmd = f"sacctmgr show assoc user={username} format=qos -P --noheader"
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
        out, err = p.communicate()
        allowed_qos_list = out.strip().split(",")

        allowed_qos_list = ["default"] + sorted(allowed_qos_list)
    else:
        raise ValueError(f"Invalid REMOTE_NAME: {pe.remote.REMOTE_NAME}")

    return allowed_qos_list


def _convert_slurm_time_duration_str_to_seconds(slurm_time_duration_str):
    """"""

    s_list = slurm_time_duration_str.split(":")
    if len(s_list) == 1:
        s_list = ["00", "00"] + s_list
    elif len(s_list) == 2:
        s_list = ["00"] + s_list
    elif (len(s_list) >= 4) or (len(s_list) == 0):
        raise RuntimeError("Unexpected number of splits")

    if "-" in s_list[0]:
        days_str, hrs_str = s_list[0].split("-")
        s_list[0] = hrs_str

        days_in_secs = int(days_str) * 60.0 * 60.0 * 24.0
    else:
        days_in_secs = 0.0

    d = time.strptime(":".join(s_list), "%H:%M:%S")

    duration_in_sec = (
        days_in_secs
        + datetime.timedelta(
            hours=d.tm_hour, minutes=d.tm_min, seconds=d.tm_sec
        ).total_seconds()
    )

    return duration_in_sec


def get_slurm_time_limits():

    part_d = get_slurm_partitions()

    partition_default_qos_list = [_pd["QoS"] for _pd in part_d.values()]

    qos_d = get_slurm_qos_info()

    allowed_qos_list = get_slurm_allowed_qos_list()

    max_time_limits = {partition_name: {} for partition_name in list(part_d)}

    for part_name, _pd in part_d.items():
        default_qos_name = _pd["QoS"]

        if default_qos_name == "N/A":
            max_wall = part_d[part_name]["MaxTime"]
        else:
            max_wall = qos_d[default_qos_name]["MaxWall"]
            if max_wall.strip() == "":
                # Use default qos for the partition
                max_wall = qos_d[_pd["QoS"]]["MaxWall"]

        # print(f'{part_name} + (Default QoS): {max_wall}')
        max_time_limits[part_name]["default"] = max_wall

        for qos_name, d in qos_d.items():
            if qos_name in partition_default_qos_list:
                continue

            max_wall = qos_d[qos_name]["MaxWall"]
            if max_wall.strip() == "":
                if _pd["QoS"] in qos_d:
                    # Use default qos for the partition
                    max_wall = qos_d[_pd["QoS"]]["MaxWall"]
                else:
                    max_wall = part_d[part_name]["MaxTime"]

            if qos_name in allowed_qos_list:
                # print(f'{part_name} + ({qos_name}): {max_wall}')
                max_time_limits[part_name][qos_name] = max_wall

    for part_name, d in max_time_limits.items():
        for qos_name, max_time_str in d.items():
            if max_time_str.strip() == "":
                max_time_limits[part_name][qos_name] = "UNLIMITED"

    default_time_limits = {partition_name: {} for partition_name in list(part_d)}
    for part_name, d in part_d.items():
        def_time_str = d["DefaultTime"]
        for qos_name, max_time_str in max_time_limits[part_name].items():
            if def_time_str.upper() == "NONE":
                default_time_limits[part_name][qos_name] = max_time_str
            else:
                def_time = _convert_slurm_time_duration_str_to_seconds(def_time_str)
                if max_time_str == "UNLIMITED":
                    default_time_limits[part_name][qos_name] = def_time_str
                elif def_time > _convert_slurm_time_duration_str_to_seconds(
                    max_time_str
                ):
                    default_time_limits[part_name][qos_name] = "INVALID"
                else:
                    default_time_limits[part_name][qos_name] = def_time_str

    return max_time_limits, default_time_limits


SLURM_MAX_TIME_LIMITS, SLURM_DEF_TIME_LIMITS = get_slurm_time_limits()
QOS_ENABLED = set(sum([list(_d) for _d in SLURM_MAX_TIME_LIMITS.values()], [])) != set(
    ["default"]
)


def _strip_unquote(s):
    """"""

    s = s.strip()

    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    else:
        return s


def _check_if_yaml_writable(yaml_object):
    """"""

    yml = yaml.YAML()
    yml.preserve_quotes = True
    yml.width = 70
    yml.boolean_representation = ["False", "True"]
    yml.dump(yaml_object, sys.stdout)


def _test_write_yaml_config(yaml_object, config_filepath):
    """"""

    yml = yaml.YAML()
    yml.preserve_quotes = True
    yml.width = 70
    yml.boolean_representation = ["False", "True"]
    with open(config_filepath, "w") as f:
        yml.dump(yaml_object, f)


def yaml_append_map(*args, **kwargs):
    """"""

    genreport._yaml_append_map(*args, **kwargs)


def duplicate_yaml_conf(orig_conf):
    """"""

    # Here, you cannot use "deepcopy()" and "pe.util.deepcopy_dict(orig_conf)".
    # Instead, you must dump to a temp file and load it back in order
    # to retain the anchors/references.

    _temp_yaml_file = tempfile.NamedTemporaryFile(suffix=".yaml", dir=None, delete=True)

    yml = yaml.YAML()
    yml.preserve_quotes = True
    yml.width = 70
    yml.boolean_representation = ["False", "True"]

    with open(_temp_yaml_file.name, "w") as f:
        yml.dump(orig_conf, f)

    dup_conf = yml.load(Path(_temp_yaml_file.name).read_text())

    _temp_yaml_file.close()

    return dup_conf


def get_yaml_repr_str(yaml_commented_obj):
    """"""

    _temp_yaml_file = tempfile.NamedTemporaryFile(suffix=".yaml", dir=None, delete=True)

    yml = yaml.YAML()
    yml.dump(yaml_commented_obj, _temp_yaml_file)

    repr = Path(_temp_yaml_file.name).read_text().splitlines()[0]

    _temp_yaml_file.close()

    return repr


def convert_to_ScalarFloat(float_val):
    """
    For ScalarFloat(), you cannot simply pass a float object, as it requires
    many extra arguments to be still YAML dumpable. So, here we use the YAML
    loader to correctly convert a Python float.
    """

    float_str = f"{float_val:.16g}"
    if "." not in float_str:
        float_str = f"%#.{len(float_str)+1}g" % float_val

    return yaml.YAML().load(float_str)


def get_all_anchor_names(root_commented_obj):
    """"""

    d = root_commented_obj

    anchor_names = []

    if isinstance(d, dict):
        for k, v in d.items():
            if hasattr(v, "anchor") and (v.anchor.value is not None):
                anchor_names.append(v.anchor.value)
            anchor_names.extend(get_all_anchor_names(v))
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            if hasattr(item, "anchor") and (item.anchor.value is not None):
                anchor_names.append(item.anchor.value)
            anchor_names.extend(get_all_anchor_names(item))

    return np.unique(anchor_names).tolist()


def recurse_anchor_find_replace(root_commented_obj, find_name, replace_name):
    """"""

    d = root_commented_obj

    if isinstance(d, dict):
        for k, v in d.items():
            if hasattr(v, "anchor") and (v.anchor.value == find_name):
                v.anchor.value = replace_name
            recurse_anchor_find_replace(v, find_name, replace_name)
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            if hasattr(item, "anchor") and (item.anchor.value == find_name):
                item.anchor.value = replace_name
            recurse_anchor_find_replace(item, find_name, replace_name)


def update_aliased_scalar(data, parent_obj, key_or_index, val):
    """
    Based on an answer on "https://stackoverflow.com/questions/55716068/how-to-change-an-anchored-scalar-in-a-sequence-without-destroying-the-anchor-in"

    Critical bug fix from "https://stackoverflow.com/questions/58118589/ruamel-yaml-recurse-function-unexpectedly-updates-every-instance-of-non-string"
    """

    def recurse(d, parent, key_index, ref, nv):
        if isinstance(d, dict):
            for i, k in [(idx, key) for idx, key in enumerate(d.keys()) if key is ref]:
                d.insert(i, nv, d.pop(k))
            for k, v in d.non_merged_items():
                if v is ref:
                    if hasattr(v, "anchor") or (d is parent and k == key_index):
                        d[k] = nv
                else:
                    recurse(v, parent, key_index, ref, nv)
        elif isinstance(d, list):
            for idx, item in enumerate(d):
                if item is ref:
                    d[idx] = nv
                else:
                    recurse(item, parent, key_index, ref, nv)

    if isinstance(parent_obj, dict):
        key = key_or_index
        if key in parent_obj:
            obj = parent_obj[key]
        else:
            parent_obj[key] = val
            return
    elif isinstance(parent_obj, list):
        index = key_or_index
        obj = parent_obj[index]
    else:
        raise ValueError(f'Unexpected type for "parent_obj": {type(parent_obj)}')

    if hasattr(obj, "anchor"):
        if isinstance(obj, ScalarFloat):
            if isinstance(val, ScalarFloat):  # This condition must come before
                # the condition check for "isinstance(val, float)", as
                # ScalarFloat() is also an instance of "float"
                new_obj = duplicate_yaml_conf(val)
            elif isinstance(val, float):
                new_obj = convert_to_ScalarFloat(val)
            elif isinstance(val, CommentedSeq):
                new_obj = val
            elif isinstance(val, list):
                new_obj = CommentedSeq(val)
            else:
                raise ValueError
            new_obj.yaml_set_anchor(obj.anchor.value)
        else:
            if isinstance(obj, str) and isinstance(val, list):
                if isinstance(val, CommentedSeq):
                    new_obj = val
                else:
                    new_obj = CommentedSeq(val)
                new_obj.yaml_set_anchor(obj.anchor.value)
            else:
                new_obj = type(obj)(val, anchor=obj.anchor.value)
        recurse(data, parent_obj, key_or_index, obj, new_obj)
    elif obj is None:
        recurse(data, parent_obj, key_or_index, obj, val)
    else:
        if isinstance(val, ScalarFloat):  # This condition must come before
            # the condition check for "isinstance(val, float)", as
            # ScalarFloat() is also an instance of "float"
            new_obj = duplicate_yaml_conf(val)
        elif isinstance(val, float):
            new_obj = convert_to_ScalarFloat(val)
        else:
            new_obj = type(obj)(val)
        recurse(data, parent_obj, key_or_index, obj, new_obj)

    if (hasattr(obj, "fa") and obj.fa.flow_style()) or (
        hasattr(val, "fa") and val.fa.flow_style()
    ):
        parent_obj[key_or_index].fa.set_flow_style()


class ListModelStdLogger(QtCore.QAbstractListModel):
    """"""

    def __init__(self, data, view):
        """Constructor"""

        super().__init__()

        self._data = data
        self._view = view

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def write(self, new_str):
        """"""

        lines = new_str.splitlines()

        self.beginInsertRows(
            QtCore.QModelIndex(), len(self._data), len(self._data) + len(lines) - 1
        )
        self._data.extend(lines)
        self.endInsertRows()

    def flush(self):
        """"""

        self._view.scrollToBottom()

        QtWidgets.QApplication.processEvents()

        ## Trigger refresh
        # self.layoutChanged.emit()

    def write_from_scratch(self, whole_str):
        """"""

        self.beginResetModel()
        self._data.clear()
        self.endResetModel()

        lines = whole_str.splitlines()

        self.beginInsertRows(QtCore.QModelIndex(), 0, len(lines) - 1)
        self._data.extend(lines)
        self.endInsertRows()

        self.flush()


def realtime_updated_Popen(
    cmd, view_stdout=None, view_stderr=None, robust_tail=True, cwd=None
):
    """
    If "robust_tail" is False, occasionally, you may see the tail part of the
    output texts missing. Setting "robust_tail" True resolves this issue.

    Based on an answer posted at

    https://stackoverflow.com/questions/31926470/run-command-and-get-its-stdout-stderr-separately-in-near-real-time-like-in-a-te
    """

    stdout = ""
    stderr = ""
    masters, slaves = zip(pty.openpty(), pty.openpty())
    if not robust_tail:
        p = Popen(
            shlex.split(cmd),
            stdin=slaves[0],
            stdout=slaves[0],
            stderr=slaves[1],
            cwd=cwd,
        )
    else:
        stdout_tmp_file = tempfile.NamedTemporaryFile(
            suffix=".log", dir=None, delete=True
        )
        stderr_tmp_file = tempfile.NamedTemporaryFile(
            suffix=".log", dir=None, delete=True
        )
        stdout_filename = stdout_tmp_file.name
        stderr_filename = stderr_tmp_file.name
        new_cmd = (
            f"(({cmd}) | tee {stdout_filename}) 3>&1 1>&2 2>&3 | "
            f"tee {stderr_filename}"
        )
        p = Popen(
            new_cmd,
            stdin=slaves[0],
            stdout=slaves[0],
            stderr=slaves[1],
            cwd=cwd,
            shell=True,
        )
    for fd in slaves:
        os.close(fd)

    # readable = { masters[0]: sys.stdout, masters[1]: sys.stderr }
    if not robust_tail:
        readable = {
            masters[0]: (sys.stdout if view_stdout is None else view_stdout.model()),
            masters[1]: (sys.stderr if view_stderr is None else view_stderr.model()),
        }
    else:
        readable = {
            masters[0]: (sys.stderr if view_stderr is None else view_stderr.model()),
            masters[1]: (sys.stdout if view_stdout is None else view_stdout.model()),
        }

    try:
        if False:
            print(" ######### REAL-TIME ######### ")

        while readable:
            # t0 = time.time()
            for fd in select(readable, [], [])[0]:
                try:
                    data = os.read(fd, 1024)
                except OSError as e:
                    if e.errno != errno.EIO:
                        raise
                    del readable[fd]
                    data = b""
                finally:
                    if not data:
                        del readable[fd]
                    else:
                        if fd == masters[0]:
                            stdout += data.decode("utf-8")
                        else:
                            stderr += data.decode("utf-8")
                        readable[fd].write(data.decode("utf-8"))
                        readable[fd].flush()
                        # print(time.time()-t0)
    except:
        pass

    finally:
        p.wait()
        for fd in masters:
            os.close(fd)

        if not robust_tail:
            if view_stdout:
                view_stdout.model().write_from_scratch(stdout)
            if view_stderr:
                view_stderr.model().write_from_scratch(stderr)
        else:
            if view_stdout:
                view_stdout.model().write_from_scratch(
                    Path(stdout_filename).read_text()
                )
            if view_stderr:
                view_stderr.model().write_from_scratch(
                    Path(stderr_filename).read_text()
                )

            stdout_tmp_file.close()
            stderr_tmp_file.close()

        if False:
            print("")
            print(" ########## RESULTS ########## ")
            print("STDOUT:")
            print(stdout)
            print("STDERR:")
            print(stderr)


def convert_multiline_yaml_str_to_oneline_str(multiline_yaml_str):
    """
    Allow multi-line definition for a long LTE filepath in YAML
    """

    return "".join([_s.strip() for _s in multiline_yaml_str.splitlines()])


def showInvalidPageInputDialog(text, informative_text):
    """"""

    QMessageBox = QtWidgets.QMessageBox

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.setWindowTitle("Invalid Input")
    msg.setStyleSheet("QIcon{max-width: 100px;}")
    msg.setStyleSheet("QLabel{min-width: 300px;}")
    # msg.setStyleSheet("QLabel{min-width:500 px; font-size: 24px;} QPushButton{ width:250px; font-size: 18px; }")
    msg.exec_()


def getFileDialogFilterStr(extension_dict):
    """
    Example:

    extension_dict = {
        'YAML Files': ['*.yaml', '*.yml'],
        'All Files': ['*'],
    }

    will return

    'YAML Files (*.yaml *.yml);;All Files (*)'
    """

    filter_str_list = []
    for file_type_description, extension_list in extension_dict.items():
        extension_str = " ".join(extension_list)
        filter_str_list.append(f"{file_type_description} ({extension_str})")

    filter_str = ";;".join(filter_str_list)

    return filter_str


def getFileDialogInitDir(lineEdit_text):
    """"""

    cur_filepath = lineEdit_text.strip()
    cur_file = Path(cur_filepath)
    try:
        if cur_file.exists():
            directory = str(cur_file.parent)
        else:
            directory = ""
    except PermissionError:
        print(f'You do not have permission to the folder "{cur_file.parent}".')
        directory = ""

    return directory


def openFileNameDialog(widget, caption="", directory="", filter_str=""):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    # directory = ''
    # filter_str = 'All Files (*);;Python Files (*.py)'

    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(
        widget, caption, directory, filter_str, options=options
    )
    if fileName:
        # print(fileName)
        return fileName


def openFileNamesDialog(widget, caption="", directory="", filter_str=""):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(
        widget, caption, directory, filter_str, options=options
    )
    if files:
        # print(files)
        return files


def saveFileDialog(widget, caption="", directory="", filter_str=""):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    options = QFileDialog.Options()
    # options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getSaveFileName(
        widget, caption, directory, filter_str, options=options
    )
    if fileName:
        # print(fileName)
        return fileName


def openDirNameDialog(widget, caption="", directory=""):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.DirectoryOnly)
    dialog.setOption(QFileDialog.ShowDirsOnly, False)
    dialog.setOption(QFileDialog.DontUseNativeDialog, False)
    dialog.setWindowTitle(caption)
    dialog.setDirectory(QtCore.QDir(directory))
    accepted = dialog.exec()
    if accepted:
        dir_path = dialog.directory().path()
    else:
        dir_path = QtCore.QDir(directory).absolutePath()

    return dir_path


def generate_report(config_filename, view_stdout=None, view_stderr=None, cwd=None):
    """"""

    cmd = f"pyele_report {config_filename}"

    model_stdout = view_stdout.model()
    model_stderr = view_stderr.model()

    model_stdout.beginResetModel()
    model_stdout._data.clear()
    model_stdout.endResetModel()

    model_stderr.beginResetModel()
    model_stderr._data.clear()
    model_stderr.beginResetModel()

    QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

    QtWidgets.QApplication.processEvents()

    # model_stdout.layoutChanged.emit()
    # model_stderr.layoutChanged.emit()

    # view_stdout.update()
    # view_stderr.update()

    realtime_updated_Popen(
        cmd, view_stdout=view_stdout, view_stderr=view_stderr, cwd=cwd
    )

    QtWidgets.QApplication.restoreOverrideCursor()


def open_pdf_report(pdf_filepath):
    """"""

    cmd = f"evince {pdf_filepath}"
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding="utf-8")
    # out, err = p.communicate()

    # print(out)
    # if err:
    # print('\n### stderr ###')
    # print(err)


class PageStandard(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._pageInitialized = False

        self._registeredFields = []
        self.conf = None

        self._next_id = None

    def safeFindChild(self, qt_class, obj_name):
        """
        Same as self.findChild(), except that if a child object cannot be found,
        this function will throw an error.
        """

        w = self.findChild(qt_class, obj_name)

        if w is None:
            raise AttributeError(
                (
                    f'There is no object of type "{qt_class}" '
                    f'with object name "{obj_name}"'
                )
            )

        return w

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def setNextId(self, next_id):
        """"""

        self._next_id = next_id

    def nextId(self):
        """"""

        if self._next_id is None:
            return super().nextId()
        else:
            return self._next_id

    def cleanupPage(self):
        """"""

        self.initializePage()


class PageGenReport(PageStandard):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._connections_established = False

        self.all_calc_types = [
            "xy_aper",
            "fmap_xy",
            "fmap_px",
            "cmap_xy",
            "cmap_px",
            "tswa",
            "nonlin_chrom",
            "mom_aper",
        ]

        YAML_OBJ = yaml.YAML()

        self.converters = {
            "spin": dict(set=lambda v: v, get=lambda v: v),
            "edit_float": dict(
                # set = lambda v: f'{v:.6g}',
                set=lambda v: get_yaml_repr_str(v),
                # get = lambda v: float(v)),
                get=lambda v: YAML_OBJ.load(
                    (v if ("." in v) or ("e" in v) or ("E" in v) else v + ".0")
                ),
            ),
            "edit_%": dict(
                set=lambda v: f"{v * 1e2:.6g}",
                # get = lambda v: float(v) * 1e-2,
                get=lambda v: YAML_OBJ.load(v + "e-2" if float(v) != 0.0 else "0.0"),
            ),
            "edit_str": dict(set=lambda v: str(v), get=lambda v: v),
            "edit_str_None": dict(
                set=lambda v: str(v) if v is not None else "",
                get=lambda v: v if v.strip() != "" else None,
            ),
            "check": dict(set=lambda v: v, get=lambda v: v),
            "combo": dict(set=lambda v: v, get=lambda v: v),
            "edit_list_int": dict(
                set=lambda v_list: ", ".join([f"{v:d}" for v in v_list]),
                get=lambda v: [int(s) for s in v.split(",")],
            ),
            "edit_list_float": dict(
                set=lambda v_list: ", ".join([f"{v:.6g}" for v in v_list]),
                get=lambda v: [float(s.strip()) for s in v.split(",")],
            ),
            "combo_special": dict(
                set=self._set_combo_special, get=self._get_combo_special
            ),
            "edit_special": dict(
                set=self._set_edit_special, get=self._get_edit_special
            ),
        }

    @staticmethod
    def _get_edit_special(name, v):
        """"""

        if name == "coupling_list":
            return [s.strip() for s in v.split(",")]
        elif name == "rf_voltages":
            # return [float(s.strip()) * 1e6 for s in v.split(',')]

            valid_example_str = (
                'Valid examples: "1.5, 2, 2.5" for single or any multiple beam '
                'energies; "[1, 1.5], [1.5, 2, 2.5]" for 2 beam energies.'
            )

            if ("[" in v) or ("]" in v):
                if ("[" in v) and ("]" in v):
                    try:
                        list_strs = re.findall("\[[^\]]+\]", v)
                        list_strs = [s[1:-1] for s in list_strs]  # remove "[" & "]"
                        rf_voltages = []
                        for list_s in list_strs:
                            rf_voltages.append(
                                [
                                    yaml.YAML().load(s.strip() + "e6")
                                    for s in list_s.split(",")
                                    if s.strip() != ""
                                ]
                            )
                        return rf_voltages
                    except:
                        text = f"Invalid input string for RF voltages: {v}"
                        info_text = valid_example_str
                        showInvalidPageInputDialog(text, info_text)
                        return
                else:
                    text = f"Invalid input string for RF voltages: {v}"
                    info_text = "Unmatched square brackets.\n\n" + valid_example_str
                    showInvalidPageInputDialog(text, info_text)
                    return
            else:
                try:
                    return [
                        yaml.YAML().load(s.strip() + "e6")
                        for s in v.split(",")
                        if s.strip() != ""
                    ]
                except:
                    text = f"Invalid input string for RF voltages: {v}"
                    info_text = valid_example_str
                    showInvalidPageInputDialog(text, info_text)
                    return
        else:
            raise ValueError()

    @staticmethod
    def _set_edit_special(name, v):
        """"""

        if name == "coupling_list":
            v_list = v
            return ", ".join([v.strip() for v in v_list])
        elif name == "rf_voltages":
            try:
                iter(v[0])
            except TypeError:
                v_list = v
                return ", ".join([f"{v/1e6:.6g}" for v in v_list])
            else:
                v_LoL = v
                list_strs = []
                for v_list in v_LoL:
                    list_s = "[{}]".format(", ".join([f"{v/1e6:.6g}" for v in v_list]))
                    list_strs.append(list_s)
                return ", ".join(list_strs)
        else:
            raise ValueError()

    @staticmethod
    def _get_combo_special(name, v):
        """"""

        if name == "max_mom_aper":
            if v == "None":
                return None
            elif v == "auto":
                return v
            else:
                return float(v)
        else:
            raise ValueError()

    @staticmethod
    def _set_combo_special(name, v):
        """"""

        if name == "max_mom_aper":
            if v in (None, "None", "none"):
                return "None"
            elif v in ("Auto", "auto"):
                return v
            else:
                return f"{v:.6g}"
        else:
            raise ValueError()

    def establish_connections(self):
        """"""

        if self._connections_established:
            return

        # Establish connections
        view_out = self.findChildren(
            QtWidgets.QListView, QtCore.QRegExp("listView_stdout_.+")
        )[0]
        view_err = self.findChildren(
            QtWidgets.QListView, QtCore.QRegExp("listView_stderr_.+")
        )[0]
        view_out.setModel(ListModelStdLogger([], view_out))
        view_err.setModel(ListModelStdLogger([], view_err))

        gen_buttons = self.findChildren(
            QtWidgets.QPushButton, QtCore.QRegExp("pushButton_gen_.+")
        )
        if len(gen_buttons) == 1:
            b = gen_buttons[0]
            b.clicked.connect(partial(self.generate_report, view_out, view_err, None))
        elif len(gen_buttons) == 2:
            for b in gen_buttons:
                _args = [view_out, view_err]
                if "_recalc_" in b.objectName():
                    _args += ["recalc"]
                elif "_replot_" in b.objectName():
                    _args += ["replot"]
                else:
                    RuntimeError()
                b.clicked.connect(partial(self.generate_report, *_args))
        else:
            raise RuntimeError()

        b = self.findChildren(
            QtWidgets.QPushButton, QtCore.QRegExp("pushButton_open_pdf_.+")
        )[0]
        b.clicked.connect(self.open_pdf_report)

        buttons = self.findChildren(
            QtWidgets.QPushButton, QtCore.QRegExp("pushButton_global_opts_.+")
        )
        if len(buttons) != 0:
            b = buttons[0]
            b.clicked.connect(self.open_global_opts_dialog)

        combos = self.findChildren(
            QtWidgets.QComboBox, QtCore.QRegExp("comboBox_partition_.+")
        )
        for w in combos:
            w.currentTextChanged.connect(self._update_qos_combo_items)

            part_combo_name = w.objectName()
            w_qos = self._qos_comboBoxes[part_combo_name]
            w_qos.currentTextChanged.connect(self._update_time_limit_tooltip)

        self._connections_established = True

    def _setup_partition_qos_objects(self):
        """"""

        combos = self.findChildren(
            QtWidgets.QComboBox, QtCore.QRegExp("comboBox_partition_.+")
        )
        self._qos_comboBoxes = {}
        self._time_lineEdits = {}
        for w in combos:
            part_combo_name = w.objectName()

            qos_combo = self.findChild(
                QtWidgets.QComboBox, part_combo_name.replace("_partition_", "_qos_")
            )
            assert qos_combo is not None
            self._qos_comboBoxes[part_combo_name] = qos_combo

            time_edit = self.findChild(
                QtWidgets.QLineEdit,
                part_combo_name.replace("comboBox_partition_", "lineEdit_time_"),
            )
            assert time_edit is not None
            self._time_lineEdits[part_combo_name] = time_edit

    def _update_qos_combo_items(self, current_partition, sender=None):
        """"""

        if sender is None:
            sender = self.sender()
        sender_name = sender.objectName()
        qos_combo = self._qos_comboBoxes[sender_name]

        current_qos = qos_combo.currentText()

        new_qos_names = list(SLURM_MAX_TIME_LIMITS[current_partition])

        qos_combo.clear()
        qos_combo.insertItems(0, new_qos_names)

        if current_qos in new_qos_names:
            i = new_qos_names.index(current_qos)
        else:
            i = 0

        qos_combo.setCurrentIndex(i)

    def _update_time_limit_tooltip(self, current_qos, sender=None):
        """"""

        if sender is None:
            sender = self.sender()

        for partition_combo_name, qos_combo in self._qos_comboBoxes.items():
            if qos_combo is sender:
                w = self.findChild(QtWidgets.QComboBox, partition_combo_name)
                current_partition = w.currentText()

                time_edit = self._time_lineEdits[partition_combo_name]

                break
        else:
            raise RuntimeError("Current partition name could not be found")

        try:
            max_time_limit = SLURM_MAX_TIME_LIMITS[current_partition][current_qos]
        except:
            return

        def_time_limit = SLURM_DEF_TIME_LIMITS[current_partition][current_qos]
        if def_time_limit == "INVALID":
            def_time_sentence = (
                'You MUST specify "time" (less than max allowed) for this case, '
                "as the default time limit exceeds the max allowed limit."
            )
        else:
            def_time_sentence = (
                'If "time" is not specified, the time limit will be:'
                f"<br><br>{def_time_limit}"
            )

        time_edit.setToolTip(
            (
                '<html><head/><body><p>days-hours:minutes:seconds ("days" and "hours" '
                'are optional. For example, use "5:00" if you want 5 minutes.)<br>'
                "Leave empty if you want to run up to the max time limit for the "
                f'selected partition "{current_partition}" & qos "{current_qos}":<br><br>'
                f"{max_time_limit}<br><br>"
                f"{def_time_sentence}</p></body></html>"
            )
        )
        # ^ "<br>" are line breaks for rich texts. Here, a rich text is used to
        #   automatically wrap to multi-lines.

    def generate_report(self, view_stdout=None, view_stderr=None, recalc_replot=None):
        """"""

        config_filepath = self.wizardObj.config_filepath

        config_file = Path(config_filepath)
        config_filename = config_file.name
        config_filedirpath = str(config_file.parent)

        try:
            if recalc_replot is None:
                mod_conf = self.modify_conf(self.conf)
            else:
                mod_conf = self.modify_conf(self.conf, recalc_replot=recalc_replot)
        except:
            traceback.print_exc()
            return

        yml = yaml.YAML()
        yml.preserve_quotes = True
        yml.width = 70
        yml.boolean_representation = ["False", "True"]
        with open(config_filepath, "w") as f:
            yml.dump(mod_conf, f)

        generate_report(
            config_filename,
            view_stdout=view_stdout,
            view_stderr=view_stderr,
            cwd=config_filedirpath,
        )

    def open_pdf_report(self):
        """"""

        pdf_filepath = self.wizardObj.pdf_filepath

        open_pdf_report(pdf_filepath)

    def open_global_opts_dialog(self):
        """"""

        mod_conf = duplicate_yaml_conf(self.conf)

        dialog = GlobalOptionsDialog(self, mod_conf)
        dialog.exec()

        if dialog.result() == dialog.Accepted:
            self.wizardObj.update_conf_on_all_pages(mod_conf)
            self.wizardObj.update_common_remote_opts()

    def modify_conf(self, orig_conf, recalc_replot=None):
        """"""
        return duplicate_yaml_conf(orig_conf)


class GlobalOptionsDialog(QtWidgets.QDialog):
    """"""

    def __init__(self, wizard_obj, conf, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)
        ui_file = os.path.join(os.path.dirname(__file__), "global_options.ui")
        uic.loadUi(ui_file, self)

        self.conf = conf

        self.upload_options()

        self.accepted.connect(self.download_options)

    def upload_options(self):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        w = self.findChild(check, "checkBox_pyelegant_stdout")
        w.setChecked(self.conf["enable_pyelegant_stdout"])

        try:
            common_remote_opts = self.conf["nonlin"]["common_remote_opts"]
        except:
            common_remote_opts = {}

        for suffix in ["nodelist", "exclude"]:
            if suffix in common_remote_opts:
                w = self.findChild(edit, f"lineEdit_{suffix}")
                try:
                    w.setText(",".join(common_remote_opts[suffix]))
                except:
                    w.setText("")

    def download_options(self):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        w = self.findChild(check, "checkBox_pyelegant_stdout")
        self.conf["enable_pyelegant_stdout"] = w.isChecked()

        try:
            common_remote_opts = self.conf["nonlin"]["common_remote_opts"]
        except:
            self.conf["nonlin"]["common_remote_opts"] = {}
            common_remote_opts = self.conf["nonlin"]["common_remote_opts"]

        for suffix in ["nodelist", "exclude"]:
            w = self.findChild(edit, f"lineEdit_{suffix}")
            text = w.text().strip()
            if text != "":
                common_remote_opts[suffix] = [s.strip() for s in text.split(",")]
            else:
                if suffix in common_remote_opts:
                    del common_remote_opts[suffix]


class PageNonlinCalcTest(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self.calc_type = None
        self.test_list = None
        self.prod_list = None
        self.setter_getter = None

    def register_test_prod_option_widgets(self):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        self._partition_fieldnames_comboBoxes = {}
        self._qos_fieldnames_comboBoxes = {}

        for mode, k_wtype_list in [
            ("test", self.test_list),
            ("production", self.prod_list),
        ]:

            short_mode = mode[:4]

            for k, wtype in k_wtype_list:
                w_suffix = f"{k}_{self.calc_type}_{short_mode}"
                f_suffix = f"{k}_{self.calc_type}_{mode}"
                if wtype == spin:
                    w = self.safeFindChild(spin, f"spinBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"spin_{f_suffix}", w)
                elif wtype == edit:
                    w = self.safeFindChild(edit, f"lineEdit_{w_suffix}")
                    self.registerFieldOnFirstShow(f"edit_{f_suffix}", w)
                elif wtype == check:
                    w = self.safeFindChild(check, f"checkBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"check_{f_suffix}", w)
                elif wtype == combo:
                    w = self.safeFindChild(combo, f"comboBox_{w_suffix}")
                    self.registerFieldOnFirstShow(
                        f"combo_{f_suffix}", w, property="currentText"
                    )

                    if k == "partition":
                        w.insertItems(0, list(SLURM_MAX_TIME_LIMITS))
                        self._partition_fieldnames_comboBoxes[f"combo_{f_suffix}"] = w
                    elif k == "qos":
                        self._qos_fieldnames_comboBoxes[f"combo_{f_suffix}"] = w
                        if QOS_ENABLED:
                            w.setEnabled(True)
                else:
                    raise ValueError()

    def set_test_prod_option_fields(self):
        """"""

        try:
            common_remote_opts = self.conf["nonlin"]["common_remote_opts"]
        except:
            self.conf["nonlin"]["common_remote_opts"] = {}
            common_remote_opts = self.conf["nonlin"]["common_remote_opts"]

        for mode in ["test", "production"]:
            try:
                opts = self.conf["nonlin"]["calc_opts"][self.calc_type][mode]
            except:
                opts = None
            if opts is not None:
                for k, v in opts.items():
                    if not isinstance(v, dict):
                        if k not in self.setter_getter[mode]:
                            continue
                        conv_type = self.setter_getter[mode][k]
                        conv = self.converters[conv_type]["set"]
                        wtype = conv_type.split("_")[0]
                        self.setField(f"{wtype}_{k}_{self.calc_type}_{mode}", conv(v))
                    else:
                        if k == "remote_opts":
                            if "partition" not in v:
                                if "partition" in common_remote_opts:
                                    v["partition"] = str(
                                        common_remote_opts["partition"]
                                    )
                            # Make sure "partition" comes before "qos". Otherwise,
                            # their comboBox initializations will not work as expected.
                            key_list = sorted(list(v))
                            #
                            for k2 in key_list:
                                v2 = v[k2]
                                # concat_k = f'{k}___{k2}'
                                # if concat_k not in self.setter_getter[mode]:
                                # continue
                                if k2 not in self.setter_getter[mode]:
                                    continue
                                # conv_type = self.setter_getter[mode][concat_k]
                                conv_type = self.setter_getter[mode][k2]
                                conv = self.converters[conv_type]["set"]
                                wtype = conv_type.split("_")[0]
                                # self.setField(
                                # f'{wtype}_{concat_k}_{self.calc_type}_{mode}',
                                # conv(v2))
                                fieldname = f"{wtype}_{k2}_{self.calc_type}_{mode}"
                                self.setField(fieldname, conv(v2))
                        else:
                            for k2, v2 in v.items():
                                concat_k = f"{k}___{k2}"
                                if concat_k not in self.setter_getter[mode]:
                                    continue
                                conv_type = self.setter_getter[mode][concat_k]
                                conv = self.converters[conv_type]["set"]
                                wtype = conv_type.split("_")[0]
                                self.setField(
                                    f"{wtype}_{concat_k}_{self.calc_type}_{mode}",
                                    conv(v2),
                                )

            # Set up the QoS comboBox items based on the
            # selected "partition". This needs to be done
            # manually at this point, as self._update_qos_combo_items()
            # is not yet connected to the "partition" change
            for fieldname, sender in self._partition_fieldnames_comboBoxes.items():
                self._update_qos_combo_items(self.field(fieldname), sender=sender)

            # Update the tooltip for the time lineEdit item
            # based on the selected "partition"/"qos". This
            # needs to be done manually at this point, as
            # self._update_time_limit_tooltip()
            # is not yet connected to the "qos" change
            for fieldname, sender in self._qos_fieldnames_comboBoxes.items():
                self._update_time_limit_tooltip(self.field(fieldname), sender=sender)

    def validatePage(self):
        """"""

        # Ask if the user did check/update the info on "Production" tab
        QMessageBox = QtWidgets.QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText('Have you checked/updated the options in "Production" tab?')
        msg.setInformativeText(
            (
                'The option values in "Production" tab will override those in '
                '"Test" tab when you run in "Production" mode later.'
            )
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setWindowTitle('Confirm "Production" Options')
        msg.setStyleSheet("QIcon{max-width: 100px;}")
        msg.setStyleSheet("QLabel{min-width: 300px;}")
        reply = msg.exec_()
        if reply == QMessageBox.No:
            return False

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        w = self.findChildren(QtWidgets.QTabWidget, QtCore.QRegExp("tabWidget_std_.+"))[
            0
        ]
        w.setCurrentIndex(0)  # show "stdout" tab before report generation starts

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        ncf = mod_conf["nonlin"]

        for _sel_calc_type in self.all_calc_types:
            if _sel_calc_type == self.calc_type:
                ncf["include"][_sel_calc_type] = True
                ncf["recalc"][_sel_calc_type] = True
            else:
                ncf["include"][_sel_calc_type] = False
                ncf["recalc"][_sel_calc_type] = False

        if ncf["use_beamline"] is not mod_conf["use_beamline_ring"]:
            ncf["use_beamline"] = mod_conf["use_beamline_ring"]

        ncf["selected_calc_opt_names"][self.calc_type] = "test"

        common_remote_opts = self.wizardObj.common_remote_opts

        new_calc_opts = CommentedMap({})
        for mode in ["test", "production"]:
            calc_opts = ncf["calc_opts"][self.calc_type][mode]

            new_calc_opts[mode] = CommentedMap({})
            for k, conv_type in self.setter_getter[mode].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{k}_{self.calc_type}_{mode}"))
                if k in ("partition", "qos", "ntasks", "time"):
                    if k in common_remote_opts:
                        indiv_remote_opts = calc_opts.get("remote_opts", {})
                        if common_remote_opts[k] == v:
                            if k in indiv_remote_opts:
                                try:
                                    del new_calc_opts[mode]["remote_opts"][k]
                                except KeyError:
                                    pass
                            else:
                                continue
                        else:
                            if "remote_opts" not in new_calc_opts[mode]:
                                yaml_append_map(
                                    new_calc_opts[mode], "remote_opts", CommentedMap({})
                                )
                            if (k == "qos") and (v in ("", "default")):
                                if k in new_calc_opts[mode]["remote_opts"]:
                                    del new_calc_opts[mode]["remote_opts"][k]
                            else:
                                yaml_append_map(
                                    new_calc_opts[mode]["remote_opts"], k, v
                                )
                    else:
                        if "remote_opts" not in new_calc_opts[mode]:
                            yaml_append_map(
                                new_calc_opts[mode], "remote_opts", CommentedMap({})
                            )
                        if (k == "qos") and (v in ("", "default")):
                            if k in new_calc_opts[mode]["remote_opts"]:
                                del new_calc_opts[mode]["remote_opts"][k]
                        else:
                            yaml_append_map(new_calc_opts[mode]["remote_opts"], k, v)
                elif "___" in k:
                    k1, k2 = k.split("___")
                    if k1 not in new_calc_opts[mode]:
                        yaml_append_map(new_calc_opts[mode], k1, CommentedMap({}))
                    yaml_append_map(new_calc_opts[mode][k1], k2, v)
                else:
                    yaml_append_map(new_calc_opts[mode], k, v)

        mode = "test"
        calc_opts = new_calc_opts[mode]
        new_anchor_name = f"{self.calc_type}_{mode}"
        calc_opts.yaml_set_anchor(new_anchor_name)
        recurse_anchor_find_replace(
            ncf["calc_opts"], new_anchor_name, f"{new_anchor_name}_2"
        )
        if self.calc_type.startswith("cmap_"):
            fmap_calc_type = self.calc_type.replace("cmap_", "fmap_")
            fmap_test_calc_opts = ncf["calc_opts"][fmap_calc_type]["test"]
            if fmap_test_calc_opts.anchor.value is None:
                fmap_test_calc_opts.yaml_set_anchor(f"{fmap_calc_type}_test")
            calc_opts.add_yaml_merge([(0, fmap_test_calc_opts)])
        del ncf["calc_opts"][self.calc_type][mode]
        yaml_append_map(ncf["calc_opts"][self.calc_type], mode, calc_opts)
        test_calc_opts = calc_opts

        mode = "production"
        calc_opts = new_calc_opts[mode]
        new_anchor_name = f"{self.calc_type}_{mode}"
        calc_opts.yaml_set_anchor(new_anchor_name)
        recurse_anchor_find_replace(
            ncf["calc_opts"], new_anchor_name, f"{new_anchor_name}_2"
        )
        if self.calc_type.startswith("cmap_"):
            fmap_calc_type = self.calc_type.replace("cmap_", "fmap_")
            fmap_prod_calc_opts = ncf["calc_opts"][fmap_calc_type]["production"]
            if fmap_prod_calc_opts.anchor.value is None:
                fmap_prod_calc_opts.yaml_set_anchor(f"{fmap_calc_type}_production")
            calc_opts.add_yaml_merge([(0, fmap_prod_calc_opts)])
        else:
            calc_opts.add_yaml_merge([(0, test_calc_opts)])
        del ncf["calc_opts"][self.calc_type][mode]
        yaml_append_map(ncf["calc_opts"][self.calc_type], mode, calc_opts)

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageNonlinCalcPlot(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self.calc_type = None
        self.calc_list = None
        self.plot_list = None
        self.setter_getter = None

    def register_calc_plot_option_widgets(self):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        self._partition_fieldnames_comboBoxes = {}
        self._qos_fieldnames_comboBoxes = {}

        for mode, k_wtype_list in [("calc", self.calc_list), ("plot", self.plot_list)]:

            for k, wtype in k_wtype_list:
                w_suffix = f_suffix = f"{k}_{self.calc_type}_{mode}"
                if wtype == spin:
                    w = self.safeFindChild(spin, f"spinBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"spin_{f_suffix}", w)
                elif wtype == edit:
                    w = self.safeFindChild(edit, f"lineEdit_{w_suffix}")
                    self.registerFieldOnFirstShow(f"edit_{f_suffix}", w)
                elif wtype == check:
                    w = self.safeFindChild(check, f"checkBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"check_{f_suffix}", w)
                elif wtype == combo:
                    w = self.safeFindChild(combo, f"comboBox_{w_suffix}")
                    self.registerFieldOnFirstShow(
                        f"combo_{f_suffix}", w, property="currentText"
                    )

                    if k == "partition":
                        w.insertItems(0, list(SLURM_MAX_TIME_LIMITS))
                        self._partition_fieldnames_comboBoxes[f"combo_{f_suffix}"] = w
                    elif k == "qos":
                        self._qos_fieldnames_comboBoxes[f"combo_{f_suffix}"] = w
                        if QOS_ENABLED:
                            w.setEnabled(True)
                else:
                    raise ValueError()

    def set_calc_plot_option_fields(self):
        """"""

        assert self.calc_type in ("tswa", "nonlin_chrom")

        ncf = self.conf["nonlin"]

        try:
            common_remote_opts = ncf["common_remote_opts"]
        except:
            ncf["common_remote_opts"] = {}
            common_remote_opts = ncf["common_remote_opts"]

        for mode in ["calc", "plot"]:
            try:
                if mode == "calc":
                    opts = ncf["calc_opts"][self.calc_type]["production"]
                elif mode == "plot":
                    opts = ncf[f"{self.calc_type}_plot_opts"]
                else:
                    raise ValueError
            except:
                opts = None
            if opts is not None:
                for k, v in opts.items():
                    if not isinstance(v, dict):
                        if k not in self.setter_getter[mode]:
                            continue
                        conv_type = self.setter_getter[mode][k]
                        conv = self.converters[conv_type]["set"]
                        wtype = conv_type.split("_")[0]
                        self.setField(f"{wtype}_{k}_{self.calc_type}_{mode}", conv(v))
                    else:
                        if k == "remote_opts":
                            if "partition" not in v:
                                if "partition" in common_remote_opts:
                                    v["partition"] = str(
                                        common_remote_opts["partition"]
                                    )
                            # Make sure "partition" comes before "qos". Otherwise,
                            # their comboBox initializations will not work as expected.
                            key_list = sorted(list(v))
                            for k2 in key_list:
                                v2 = v[k2]
                                if k2 not in self.setter_getter[mode]:
                                    continue
                                conv_type = self.setter_getter[mode][k2]
                                conv = self.converters[conv_type]["set"]
                                wtype = conv_type.split("_")[0]
                                fieldname = f"{wtype}_{k2}_{self.calc_type}_{mode}"
                                self.setField(fieldname, conv(v2))
                        elif k == "fft_plot_opts":
                            pass  # TODO
                        else:
                            raise NotImplementedError

            if mode == "calc":
                # Set up the QoS comboBox items based on the
                # selected "partition". This needs to be done
                # manually at this point, as self._update_qos_combo_items()
                # is not yet connected to the "partition" change
                for fieldname, sender in self._partition_fieldnames_comboBoxes.items():
                    self._update_qos_combo_items(self.field(fieldname), sender=sender)

                # Update the tooltip for the time lineEdit item
                # based on the selected "partition"/"qos". This
                # needs to be done manually at this point, as
                # self._update_time_limit_tooltip()
                # is not yet connected to the "qos" change
                for fieldname, sender in self._qos_fieldnames_comboBoxes.items():
                    self._update_time_limit_tooltip(
                        self.field(fieldname), sender=sender
                    )

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf, recalc_replot="no_recalc_replot")

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf, recalc_replot=None):
        """"""

        if recalc_replot is None:
            recalc_replot = "recalc"
        else:
            assert recalc_replot in ("recalc", "replot", "no_recalc_replot")

        w = self.findChildren(QtWidgets.QTabWidget, QtCore.QRegExp("tabWidget_std_.+"))[
            0
        ]
        w.setCurrentIndex(0)  # show "stdout" tab before report generation starts

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        ncf = mod_conf["nonlin"]

        for _sel_calc_type in self.all_calc_types:
            if _sel_calc_type == self.calc_type:
                ncf["include"][_sel_calc_type] = True
                if recalc_replot == "recalc":
                    ncf["recalc"][_sel_calc_type] = True
                    ncf["replot"][_sel_calc_type] = True
                elif recalc_replot == "replot":
                    ncf["recalc"][_sel_calc_type] = False
                    ncf["replot"][_sel_calc_type] = True
                elif recalc_replot == "no_recalc_replot":
                    ncf["recalc"][_sel_calc_type] = False
                    ncf["replot"][_sel_calc_type] = False
                else:
                    raise ValueError("This can never be reached")
            else:
                ncf["include"][_sel_calc_type] = False
                ncf["recalc"][_sel_calc_type] = False
                ncf["replot"][_sel_calc_type] = False

        if ncf["use_beamline"] is not mod_conf["use_beamline_ring"]:
            ncf["use_beamline"] = mod_conf["use_beamline_ring"]

        ncf["selected_calc_opt_names"][self.calc_type] = "production"

        common_remote_opts = self.wizardObj.common_remote_opts

        if False:
            del ncf["calc_opts"][self.calc_type]["production"]

            mode = "calc"

            new_calc_opts = {}
            for k, conv_type in self.setter_getter[mode].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{k}_{self.calc_type}_{mode}"))
                if k in ("partition", "ntasks", "time"):
                    if k in common_remote_opts:
                        if common_remote_opts[k] == v:
                            continue
                        else:
                            if "remote_opts" not in new_calc_opts:
                                new_calc_opts["remote_opts"] = {}
                            new_calc_opts["remote_opts"][k] = v
                    else:
                        if "remote_opts" not in new_calc_opts:
                            new_calc_opts["remote_opts"] = {}
                        new_calc_opts["remote_opts"][k] = v
                else:
                    new_calc_opts[k] = v

            yaml_append_map(
                ncf["calc_opts"][self.calc_type],
                "production",
                CommentedMap(new_calc_opts),
            )
        else:
            mode = "calc"

            try:
                calc_opts = ncf["calc_opts"][self.calc_type]["production"]
            except:
                if "calc_opts" not in ncf:
                    m1 = CommentedMap({})
                    m2 = CommentedMap({"production": m1})
                    m3 = CommentedMap({self.calc_type: m2})
                    yaml_append_map(ncf, "calc_opts", m3)
                elif self.calc_type not in ncf["calc_opts"]:
                    m1 = CommentedMap({})
                    m2 = CommentedMap({"production": m1})
                    yaml_append_map(ncf["calc_opts"], self.calc_type, m2)
                elif "production" not in ncf["calc_opts"][self.calc_type]:
                    m1 = CommentedMap({})
                    yaml_append_map(ncf["calc_opts"][self.calc_type], "production", m1)
                else:
                    raise ValueError

            prop_names_from_wizard = list(self.setter_getter[mode])
            for k in list(calc_opts):
                if k != "remote_opts":
                    if k not in prop_names_from_wizard:
                        del calc_opts[k]
            if "remote_opts" in calc_opts:
                for k in list(calc_opts["remote_opts"]):
                    if k not in prop_names_from_wizard:
                        del calc_opts[k]
            for k, conv_type in self.setter_getter[mode].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{k}_{self.calc_type}_{mode}"))
                if k in ("partition", "qos", "ntasks", "time"):
                    if k in common_remote_opts:
                        indiv_remote_opts = calc_opts.get("remote_opts", {})
                        if common_remote_opts[k] == v:
                            if k in indiv_remote_opts:
                                try:
                                    del calc_opts["remote_opts"][k]
                                except KeyError:
                                    pass
                            else:
                                continue
                        else:
                            if "remote_opts" not in calc_opts:
                                # calc_opts['remote_opts'] = CommentedMap({})
                                yaml_append_map(
                                    calc_opts, "remote_opts", CommentedMap({})
                                )
                            # f(calc_opts['remote_opts'], k, v)
                            if (k == "qos") and (v in ("", "default")):
                                if k in calc_opts["remote_opts"]:
                                    del calc_opts["remote_opts"][k]
                            else:
                                yaml_append_map(calc_opts["remote_opts"], k, v)
                    else:
                        if "remote_opts" not in calc_opts:
                            # calc_opts['remote_opts'] = CommentedMap({})
                            yaml_append_map(calc_opts, "remote_opts", CommentedMap({}))
                        # f(calc_opts['remote_opts'], k, v)
                        if (k == "qos") and (v in ("", "default")):
                            if k in calc_opts["remote_opts"]:
                                del calc_opts["remote_opts"][k]
                        else:
                            yaml_append_map(calc_opts["remote_opts"], k, v)
                else:
                    f(calc_opts, k, v)

        if False:
            del ncf[f"{self.calc_type}_plot_opts"]

            mode = "plot"

            new_plot_opts = {}
            for k, conv_type in self.setter_getter[mode].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{k}_{self.calc_type}_{mode}"))
                new_plot_opts[k] = v

                if k.startswith("footprint_nux"):
                    if "footprint_nuxlim" not in new_plot_opts:
                        new_plot_opts["footprint_nuxlim"] = [np.nan, np.nan]
                    if k == "footprint_nux_min":
                        new_plot_opts["footprint_nuxlim"][0] = v
                    elif k == "footprint_nux_max":
                        new_plot_opts["footprint_nuxlim"][1] = v
                    else:
                        raise ValueError
                    del new_plot_opts[k]
                elif k.startswith("footprint_nuy"):
                    if "footprint_nuylim" not in new_plot_opts:
                        new_plot_opts["footprint_nuylim"] = [np.nan, np.nan]
                    if k == "footprint_nuy_min":
                        new_plot_opts["footprint_nuylim"][0] = v
                    elif k == "footprint_nuy_max":
                        new_plot_opts["footprint_nuylim"][1] = v
                    else:
                        raise ValueError
                    del new_plot_opts[k]
                elif k.startswith("fit_delta_"):
                    if "fit_deltalim" not in new_plot_opts:
                        new_plot_opts["fit_deltalim"] = [np.nan, np.nan]
                    if k == "fit_delta_min":
                        new_plot_opts["fit_deltalim"][0] = v
                    elif k == "fit_delta_max":
                        new_plot_opts["fit_deltalim"][1] = v
                    del new_plot_opts[k]
            for plane in ["x", "y"]:
                k = f"footprint_nu{plane}lim"
                if k in new_plot_opts:
                    seq = CommentedSeq(new_plot_opts[k])
                    seq.fa.set_flow_style()
                    new_plot_opts[k] = seq
            k = "fit_deltalim"
            if k in new_plot_opts:
                seq = CommentedSeq(new_plot_opts[k])
                seq.fa.set_flow_style()
                new_plot_opts[k] = seq

            yaml_append_map(
                ncf, f"{self.calc_type}_plot_opts", CommentedMap(new_plot_opts)
            )
        else:
            mode = "plot"

            if f"{self.calc_type}_plot_opts" in ncf:
                plot_opts = ncf[f"{self.calc_type}_plot_opts"]
            else:
                plot_opts = CommentedMap({})
                yaml_append_map(ncf, f"{self.calc_type}_plot_opts", plot_opts)

            prop_names_from_wizard = list(self.setter_getter[mode])
            for k in list(plot_opts):
                if k not in prop_names_from_wizard:
                    if k == "fft_plot_opts":
                        print(
                            'WARNING: TODO: "fft_plot_opts" needs to be handled properly by this wizard.'
                        )
                    elif k == "plot_fft":
                        print(
                            'WARNING: TODO: "plot_fft" needs to be handled properly by this wizard.'
                        )
                    else:
                        del plot_opts[k]

            for k, conv_type in self.setter_getter[mode].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{k}_{self.calc_type}_{mode}"))

                if k.startswith("footprint_nux"):
                    if "footprint_nuxlim" not in plot_opts:
                        seq = CommentedSeq([np.nan, np.nan])
                        seq.fa.set_flow_style()
                        yaml_append_map(plot_opts, "footprint_nuxlim", seq)

                    if k == "footprint_nux_min":
                        # f(plot_opts['footprint_nuxlim'], 0, v)
                        plot_opts["footprint_nuxlim"][0] = v
                    elif k == "footprint_nux_max":
                        # f(plot_opts['footprint_nuxlim'], 1, v)
                        plot_opts["footprint_nuxlim"][1] = v
                    else:
                        raise ValueError
                elif k.startswith("footprint_nuy"):
                    if "footprint_nuylim" not in plot_opts:
                        seq = CommentedSeq([np.nan, np.nan])
                        seq.fa.set_flow_style()
                        yaml_append_map(plot_opts, "footprint_nuylim", seq)

                    if k == "footprint_nuy_min":
                        # f(plot_opts['footprint_nuylim'], 0, v)
                        plot_opts["footprint_nuylim"][0] = v
                    elif k == "footprint_nuy_max":
                        # f(plot_opts['footprint_nuylim'], 1, v)
                        plot_opts["footprint_nuylim"][1] = v
                    else:
                        raise ValueError
                elif k.startswith("fit_delta_"):
                    if "fit_deltalim" not in plot_opts:
                        seq = CommentedSeq([np.nan, np.nan])
                        seq.fa.set_flow_style()
                        yaml_append_map(plot_opts, "fit_deltalim", seq)

                    if k == "fit_delta_min":
                        # f(plot_opts['fit_deltalim'], 0, v)
                        plot_opts["fit_deltalim"][0] = v
                    elif k == "fit_delta_max":
                        # f(plot_opts['fit_deltalim'], 1, v)
                        plot_opts["fit_deltalim"][1] = v
                    else:
                        raise ValueError
                else:
                    f(plot_opts, k, v)

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageLoadSeedConfig(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._pageInitialized = False

        self._registeredFields = []

        self._next_id = None

    def setNextId(self, next_id):
        """"""

        self._next_id = next_id

    def nextId(self):
        """"""

        if self._next_id is None:
            return super().nextId()
        else:
            return self._next_id

    def cleanupPage(self):
        """"""

        self.initializePage()

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            return

        self.wizardObj = self.wizard()

        self.edit_obj = self.findChild(
            QtWidgets.QLineEdit, "lineEdit_seed_config_filepath"
        )

        self.check_create_new_report = self.findChild(
            QtWidgets.QCheckBox, "checkBox_create_new_report"
        )

        # Establish connections

        b = self.findChild(QtWidgets.QPushButton, "pushButton_browse")
        b.clicked.connect(self.browse_yaml_config_file)

        self.edit_obj.setText(self.wizardObj._settings["seed_config_filepath"])

        self._pageInitialized = True

    def browse_yaml_config_file(self):
        """"""

        caption = "Select YAML config file to load"
        extension_dict = {"YAML Files": ["*.yaml", "*.yml"], "All Files": ["*"]}
        filter_str = getFileDialogFilterStr(extension_dict)
        directory = getFileDialogInitDir(self.edit_obj.text())
        filepath = openFileNameDialog(
            self, caption=caption, directory=directory, filter_str=filter_str
        )

        if filepath:
            self.edit_obj.setText(filepath)

    def validatePage(self):
        """"""

        self.wizardObj.new_report = self.check_create_new_report.isChecked()

        seed_config_filepath = self.edit_obj.text().strip()

        if seed_config_filepath == "":

            if not self.wizardObj.new_report:
                text = "Need existing YAML config file path"
                info_text = (
                    "When not creating a new report, you must specify the path "
                    "to an existing config file you want to resume/modify."
                )
                showInvalidPageInputDialog(text, info_text)
                return False

            genreport.Report_NSLS2U_Default(
                self.wizardObj.config_filepath, example_args=["full", None]
            )

            seed_config_filepath = self.wizardObj.config_filepath

        try:
            if not Path(seed_config_filepath).exists():
                text = "Invalid file path"
                info_text = (
                    f'Specified config file "{seed_config_filepath}" '
                    f"does not exist!"
                )
                showInvalidPageInputDialog(text, info_text)

                return False
        except PermissionError:
            text = "Permission error"
            info_text = (
                f'Specified config file "{seed_config_filepath}" ' f"is not accessible!"
            )
            showInvalidPageInputDialog(text, info_text)

            return False

        self.wizardObj._settings["seed_config_filepath"] = seed_config_filepath

        try:
            yml = yaml.YAML()
            yml.preserve_quotes = True
            user_conf = yml.load(Path(seed_config_filepath).read_text())
            self.wizardObj.conf = user_conf

        except:
            text = "Invalid YAML file"
            info_text = (
                f'Specified config file "{seed_config_filepath}" does not '
                f"appear to be a valid YAML file!"
            )
            showInvalidPageInputDialog(text, info_text)

            return False

        report_class = genreport.Report_NSLS2U_Default
        latest_config_ver = report_class.get_latest_config_version_str()
        while self.wizardObj.conf["report_version"] != latest_config_ver:
            self.wizardObj.conf = report_class.upgrade_config(self.wizardObj.conf)

        self.wizardObj.update_conf_on_all_pages(self.wizardObj.conf)

        self.wizardObj.update_common_remote_opts()

        should_skip = not self.wizardObj.new_report
        if should_skip:
            config_file = Path(seed_config_filepath)

            config_filename = config_file.name
            if config_filename.endswith((".yaml", ".yml")):
                rootname = ".".join(config_filename.split(".")[:-1])
            else:
                text = "Invalid YAML file extension"
                info_text = (
                    f'Specified config file "{seed_config_filepath}" does not '
                    f'have the extensions ".yaml" or ".yml"!'
                )
                showInvalidPageInputDialog(text, info_text)
                return False
            report_foldername = f"report_{rootname}"
            report_folder = config_file.parent.joinpath(report_foldername)
            pdf_filename = f"{rootname}_report.pdf"

            self.wizardObj.config_filepath = str(config_file.resolve())
            self.wizardObj.pdf_filepath = str(report_folder.joinpath(pdf_filename))
        #
        self.wizardObj.skip_new_report_setup_page(should_skip)

        return True


class PageNewSetup(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._pageInitialized = False

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            return

        self.wizardObj = self.wizard()

        # Register fields

        self.registerFieldOnFirstShow(
            "edit_rootname*", self.findChild(QtWidgets.QLineEdit, "lineEdit_rootname")
        )

        self.registerFieldOnFirstShow(
            "edit_new_config_folder*",
            self.findChild(QtWidgets.QLineEdit, "lineEdit_new_config_folder"),
        )

        for obj_name in [
            "textEdit_full_new_config",
            "textEdit_full_report_folder",
            "textEdit_full_pdf_path",
            "textEdit_full_xlsx_path",
        ]:
            w = self.findChild(QtWidgets.QTextEdit, obj_name)
            w.setAutoFillBackground(False)
            self.registerFieldOnFirstShow(obj_name, w, property="plainText")

        # Set fields

        if TEST_MODE:
            self.setField("edit_rootname", "CBA_0001")

            self.setField(
                "edit_new_config_folder",
                "/GPFS/APC/yhidaka/git_repos/pyelegant/src/pyelegant/guis/nsls2apcluster/genreport_wizard",
            )

        # Establish connections

        self.edit_obj = self.findChild(
            QtWidgets.QLineEdit, "lineEdit_new_config_folder"
        )
        self.edit_obj.textChanged.connect(self.update_paths_on_folder)
        self.edit_obj.textChanged.connect(self.completeChanged)

        b = self.findChild(QtWidgets.QPushButton, "pushButton_browse_new_config_folder")
        b.clicked.connect(self.browse_new_config_folder)

        e = self.findChild(QtWidgets.QLineEdit, "lineEdit_rootname")
        e.textChanged.connect(self.update_paths_on_rootname)
        e.textChanged.connect(self.completeChanged)

        self.edit_obj.setText(self.wizardObj._settings["config_folderpath"])
        self.update_paths_on_folder(self.edit_obj.text())

        self._pageInitialized = True

    def isComplete(self):
        """"""

        if self.field("edit_rootname").strip() == "":
            return False

        config_folderpath = self.field("edit_new_config_folder").strip()
        if config_folderpath == "":
            return False

        try:
            if not Path(config_folderpath).exists():
                return False
        except PermissionError:
            print(f'You do not have permission to "{Path(config_folderpath)}"')
            return False

        return True

    def update_paths_on_folder(self, new_config_folderpath):
        """"""

        new_config_folder = Path(new_config_folderpath)
        try:
            if not new_config_folder.exists():
                return
        except PermissionError:
            print(f'You do not have permission to "{new_config_folder}"')
            return

        rootname = self.field("edit_rootname").strip()
        if rootname == "":
            return

        self.update_paths(new_config_folder, rootname)

    def update_paths_on_rootname(self, new_rootname):
        """"""

        if new_rootname == "":
            return

        new_config_folderpath = self.field("edit_new_config_folder")
        new_config_folder = Path(new_config_folderpath)
        try:
            if not new_config_folder.exists():
                return
        except PermissionError:
            print(f'You do not have permission to "{new_config_folder}"')
            return

        self.update_paths(new_config_folder, new_rootname)

    def update_paths(self, new_config_folder, rootname):
        """"""

        new_config_filename = f"{rootname}.yaml"
        new_report_foldername = f"report_{rootname}"
        pdf_filename = f"{rootname}_report.pdf"
        xlsx_filename = f"{rootname}_report.xlsx"

        report_folderpath = new_config_folder.joinpath(new_report_foldername)

        self.wizardObj.config_filepath = str(
            new_config_folder.joinpath(new_config_filename)
        )
        self.wizardObj.pdf_filepath = str(
            Path(report_folderpath).joinpath(pdf_filename)
        )

        self.setField(
            "textEdit_full_new_config", "=> {}".format(self.wizardObj.config_filepath)
        )
        self.setField("textEdit_full_report_folder", f"=> {report_folderpath}")
        self.setField(
            "textEdit_full_pdf_path", "=> {}".format(self.wizardObj.pdf_filepath)
        )
        self.setField(
            "textEdit_full_xlsx_path",
            "=> {}".format(Path(report_folderpath).joinpath(xlsx_filename)),
        )

    def browse_new_config_folder(self):
        """"""

        caption = "Select a folder where new config/data/report files will be generated"
        directory = getFileDialogInitDir(self.edit_obj.text())
        folderpath = openDirNameDialog(self, caption=caption, directory=directory)

        if folderpath:
            self.edit_obj.setText(folderpath)

    def validatePage(self):
        """"""

        new_config_folderpath = self.edit_obj.text()
        new_config_folder = Path(new_config_folderpath)
        try:
            if not new_config_folder.exists():
                return False
            else:
                self.wizardObj._settings["config_folderpath"] = new_config_folderpath
        except PermissionError:
            print(f'You do not have permission to "{new_config_folder}"')
            return False

        rootname = self.field("edit_rootname")

        self.update_paths(new_config_folder, rootname)

        return True


def adjust_input_LTE_kickmap_filepaths(
    orig_LTE_filepath, altered_LTE_filepath, alter_elements
):
    """"""

    use_elegant_alter_elements = False

    if use_elegant_alter_elements:

        pe.eleutil.save_lattice_after_alter_elements(
            orig_LTE_filepath, altered_LTE_filepath, alter_elements
        )

    else:
        LTE_contents = Path(orig_LTE_filepath).read_text()

        def sub_kickmap_filepath(new_filepath, matchobj):
            s0, s1, s2, s3, s4 = matchobj.groups()
            new_s = f'{s0}{s1}{s2}{s3}"{new_filepath}"'
            return new_s

        for ae in alter_elements:
            name = ae["name"]
            pattern = f'({name})(\s*:\s*UKICKMAP\s*,)([^:]*)(INPUT_FILE\s*=\s*)"(.+)"'
            LTE_contents = re.sub(
                pattern,
                partial(sub_kickmap_filepath, ae["string_value"]),
                LTE_contents,
                flags=re.IGNORECASE,
            )

        if False:
            for ae in alter_elements:
                name = ae["name"]
                pattern = f'({name})\s*:\s*UKICKMAP\s*,([^:]*)INPUT_FILE\s*=\s*"(.+)"'
                out = re.findall(pattern, LTE_contents, flags=re.IGNORECASE)
                print(out)

        Path(altered_LTE_filepath).write_text(LTE_contents)


class PageLTE(PageStandard):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Register fields

        for k in [
            "report_author",
            "orig_LTE_path*",
            "LTE_authors*",
            "parent_LTE_hash",
            "ref_FLR_path*",
            "E_GeV*",
            "use_beamline_cell*",
            "use_beamline_ring*",
        ]:
            k_wo_star = k[:-1] if k.endswith("*") else k
            self.registerFieldOnFirstShow(
                f"edit_{k}",
                self.safeFindChild(QtWidgets.QLineEdit, f"lineEdit_{k_wo_star}"),
            )
        self.registerFieldOnFirstShow(
            "spin_harmonic_number*",
            self.safeFindChild(QtWidgets.QSpinBox, "spinBox_harmonic_number"),
        )
        self.registerFieldOnFirstShow(
            "date_LTE_received*",
            self.safeFindChild(QtWidgets.QDateEdit, "dateEdit_LTE_received"),
        )
        self.registerFieldOnFirstShow(
            "check_ring_is_simple_mult_cells",
            self.safeFindChild(
                QtWidgets.QCheckBox, "checkBox_ring_is_simple_mult_cells"
            ),
        )
        self.registerFieldOnFirstShow(
            "check_pyele_stdout",
            self.safeFindChild(QtWidgets.QCheckBox, "checkBox_pyelegant_stdout"),
        )

        # Set fields
        self._update_fields()

        # Establish connections

        b = self.safeFindChild(QtWidgets.QPushButton, "pushButton_browse_LTE")
        b.clicked.connect(self.browse_LTE_file)

        b = self.safeFindChild(QtWidgets.QPushButton, "pushButton_browse_FLR")
        b.clicked.connect(self.browse_FLR_file)

        for k in [
            "orig_LTE_path*",
            "LTE_authors*",
            "ref_FLR_path*",
            "E_GeV*",
            "use_beamline_cell*",
            "use_beamline_ring*",
        ]:
            k_wo_star = k[:-1] if k.endswith("*") else k
            w = self.safeFindChild(QtWidgets.QLineEdit, f"lineEdit_{k_wo_star}")
            w.textChanged.connect(self.completeChanged)
        #
        w = self.safeFindChild(QtWidgets.QSpinBox, "spinBox_harmonic_number")
        w.valueChanged.connect(self.completeChanged)
        # w.textChanged.connect(self.completeChanged) # Only available from Qt 5.14

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        self.setField(
            "edit_report_author", str(self.conf.get("report_author", "").strip())
        )

        self.setField(
            "edit_orig_LTE_path",
            convert_multiline_yaml_str_to_oneline_str(
                self.conf.get("orig_LTE_filepath", "")
            ),
        )

        lattice_authors = self.conf.get("lattice_authors", "")
        if isinstance(lattice_authors, str):
            # Convet to Python str from YAML's commented string object
            lattice_authors = str(lattice_authors.strip())
        elif isinstance(lattice_authors, list):
            lattice_authors = ", ".join([str(v).strip() for v in lattice_authors])
        else:
            raise ValueError
        self.setField("edit_LTE_authors", lattice_authors)

        self.setField(
            "edit_parent_LTE_hash", str(self.conf.get("parent_LTE_hash", "").strip())
        )

        date = self.conf.get("lattice_received_date", None)
        if date is None:
            date = QtCore.QDateTime().currentDateTime()
        else:
            month, day, year = [int(s) for s in date.split("/")]
            date = QtCore.QDateTime(datetime.date(year, month, day))
        self.setField("date_LTE_received", date)

        try:
            ref_FLR_filepath = self.conf["lattice_props"]["req_props"][
                "floor_comparison"
            ]["ref_flr_filepath"]
        except:
            ref_FLR_filepath = ""
        self.setField(
            "edit_ref_FLR_path",
            convert_multiline_yaml_str_to_oneline_str(ref_FLR_filepath),
        )

        E_MeV = self.conf.get("E_MeV", 3e3)
        if not isinstance(E_MeV, list):
            E_MeV_list = [E_MeV]
        else:
            E_MeV_list = E_MeV
        E_GeV_str = ", ".join([f"{E_MeV / 1e3:.3g}" for E_MeV in E_MeV_list])
        self.setField("edit_E_GeV", E_GeV_str)

        h = self.conf.get("harmonic_number", 1320)
        assert isinstance(h, int)
        self.setField("spin_harmonic_number", h)

        self.setField(
            "edit_use_beamline_cell",
            str(self.conf.get("use_beamline_cell", "").strip()),
        )

        self.setField(
            "edit_use_beamline_ring",
            str(self.conf.get("use_beamline_ring", "").strip()),
        )

        self.setField(
            "check_ring_is_simple_mult_cells",
            self.conf.get("ring_is_a_simple_multiple_of_cells", True),
        )

        self.setField(
            "check_pyele_stdout", self.conf.get("enable_pyelegant_stdout", False)
        )

    def browse_LTE_file(self):
        """"""

        caption = "Select LTE file to be characterized"
        filter_str = "LTE Files (*.lte);;All Files (*)"
        extension_dict = {"LTE Files": ["*.lte"], "All Files": ["*"]}
        filter_str = getFileDialogFilterStr(extension_dict)
        directory = getFileDialogInitDir(self.field("edit_orig_LTE_path"))
        filepath = openFileNameDialog(
            self, caption=caption, directory=directory, filter_str=filter_str
        )

        if filepath:
            self.setField("edit_orig_LTE_path", filepath)

    def browse_FLR_file(self):
        """"""

        caption = "Select reference FLR (floor) file to be compared against"
        filter_str = "FLR Files (*.flr);;All Files (*)"
        extension_dict = {"FLR Files": ["*.flr"], "All Files": ["*"]}
        filter_str = getFileDialogFilterStr(extension_dict)
        directory = getFileDialogInitDir(self.field("edit_ref_FLR_path"))
        filepath = openFileNameDialog(
            self, caption=caption, directory=directory, filter_str=filter_str
        )

        if filepath:
            self.setField("edit_ref_FLR_path", filepath)

    def validatePage(self):
        """"""

        orig_LTE_filepath = self.field("edit_orig_LTE_path").strip()

        if not orig_LTE_filepath.endswith(".lte"):
            text = "Invalid file extension"
            info_text = 'LTE file name must end with ".lte".'
            showInvalidPageInputDialog(text, info_text)
            return False

        orig_LTE_Path = Path(orig_LTE_filepath)
        try:
            if not orig_LTE_Path.exists():
                text = "Invalid file path"
                info_text = f'Specified LTE file "{orig_LTE_Path}" does not exist!'
                showInvalidPageInputDialog(text, info_text)
                return False
        except PermissionError:
            text = "Permission error"
            info_text = f'Specified LTE file "{orig_LTE_Path}" is not accessible!'
            showInvalidPageInputDialog(text, info_text)
            return False

        ref_FLR_filepath = self.field("edit_ref_FLR_path").strip()

        if not ref_FLR_filepath.endswith(".flr"):
            text = "Invalid file extension"
            info_text = 'FLR file name must end with ".flr".'
            showInvalidPageInputDialog(text, info_text)
            return False

        ref_FLR_Path = Path(ref_FLR_filepath)
        try:
            if not ref_FLR_Path.exists():
                text = "Invalid file path"
                info_text = f'Specified FLR file "{ref_FLR_Path}" does not exist!'
                showInvalidPageInputDialog(text, info_text)
                return False
        except PermissionError:
            text = "Permission error"
            info_text = f'Specified FLR file "{ref_FLR_Path}" is not accessible!'
            showInvalidPageInputDialog(text, info_text)
            return False

        use_beamline_cell = self.field("edit_use_beamline_cell").strip()
        use_beamline_ring = self.field("edit_use_beamline_ring").strip()

        ring_is_simple_mult_cells = self.field("check_ring_is_simple_mult_cells")

        report_folder = Path(self.wizardObj.pdf_filepath).parent

        kickmap_filepaths = {"raw": {}, "abs": {}}
        # ^ This dict will contain the raw/absolute paths to the kickmap files,
        #   if kickmap elements exist.

        try:
            LTE = pe.ltemanager.Lattice(LTE_filepath=orig_LTE_filepath)
        except:
            text = "Invalid LTE file"
            info_text = f'Specified LTE file "{orig_LTE_Path}" failed to be parsed!'
            showInvalidPageInputDialog(text, info_text)
            traceback.print_exc()
            return False

        try:
            LTE = pe.ltemanager.Lattice(
                LTE_filepath=orig_LTE_filepath, used_beamline_name=use_beamline_ring
            )
            LTE_ring = LTE
        except:
            text = 'Invalid beamline name for "Ring Beamline Name"'
            info_text = f'Specified name "{use_beamline_ring}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            # traceback.print_exc()
            return False

        new_km_fps = LTE.get_kickmap_filepaths()
        for typ in list(new_km_fps):
            kickmap_filepaths[typ].update(new_km_fps[typ])

        try:
            LTE = pe.ltemanager.Lattice(
                LTE_filepath=orig_LTE_filepath, used_beamline_name=use_beamline_cell
            )
            LTE_cell = LTE
        except:
            text = 'Invalid beamline name for "Super-Period Beamline Name"'
            info_text = f'Specified name "{use_beamline_cell}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            # traceback.print_exc()
            return False

        new_km_fps = LTE.get_kickmap_filepaths()
        for typ in list(new_km_fps):
            kickmap_filepaths[typ].update(new_km_fps[typ])

        self.wizardObj.LTE = LTE

        all_beamline_defs = LTE.get_all_beamline_defs(LTE.cleaned_LTE_text)
        all_beamline_names = [v[0] for v in all_beamline_defs]

        for spec_name, label in [
            (use_beamline_cell, "Super-Period"),
            (use_beamline_ring, "Ring"),
        ]:
            if spec_name not in all_beamline_names:
                text = f'Invalid name for "{label}" beamline: "{spec_name}"'
                info_text = "Available beamline names: {}".format(
                    ", ".join(all_beamline_names)
                )
                showInvalidPageInputDialog(text, info_text)
                return False

        if ring_is_simple_mult_cells:  # Must check if the ring beamline is indeed
            # a simple multiple of thr cell beamlines. If not, advise the user
            # to correct the LTE file.
            est_ring_mult = int(
                np.floor(
                    len(LTE_ring.flat_used_elem_names)
                    / len(LTE_cell.flat_used_elem_names)
                )
            )
            ext_flat_used_elem_names = LTE_cell.flat_used_elem_names * est_ring_mult

            err_text = (
                f'"{use_beamline_cell}" * {est_ring_mult} != "{use_beamline_ring}"'
            )
            err_info_text = (
                f'Ring Beamline "{use_beamline_ring}" is NOT a simple '
                f'multiple of Super-Period Beamline "{use_beamline_cell}". \nTo fix '
                f"this problem, you may want to define the Ring Beamline as \n"
                f"    `{use_beamline_ring}: LINE=({est_ring_mult}*{use_beamline_cell})`"
            )
            if len(ext_flat_used_elem_names) != len(LTE_ring.flat_used_elem_names):
                showInvalidPageInputDialog(err_text, err_info_text)
                return False
            else:
                if np.all(
                    np.array(ext_flat_used_elem_names)
                    == np.array(LTE_ring.flat_used_elem_names)
                ):
                    pass
                else:
                    showInvalidPageInputDialog(err_text, err_info_text)
                    return False

        mod_conf = duplicate_yaml_conf(self.conf)

        f = partial(update_aliased_scalar, mod_conf)
        #
        f(mod_conf, "report_author", self.field("edit_report_author").strip())
        #
        f(mod_conf, "orig_LTE_filepath", orig_LTE_filepath)
        #
        input_LTE_filepath = self.wizardObj.pdf_filepath[: -len("_report.pdf")] + ".lte"
        input_LTE_Path = Path(input_LTE_filepath)
        input_LTE_Path.parent.mkdir(parents=True, exist_ok=True)
        #
        # (SEGMENT#1 Start) The following code segment must come after the
        # "mkdir" function call right above. Otherwise, shutil.copy() could fail
        # in this segment.
        #
        # Since the input LTE file will reside in the report folder, and if the
        # kickmap files are specified with relative paths, the kickmap files
        # will need to be copied into the report folder.
        alter_elements = []
        for name, _fp in kickmap_filepaths["abs"].items():
            abs_kickmap_f = Path(_fp)
            try:
                if not abs_kickmap_f.exists():
                    text = f'Non-existing kickmap file for Element "{name}"'
                    info_text = f'Specified file "{_fp}" does not exist!'
                    showInvalidPageInputDialog(text, info_text)
                    # traceback.print_exc()
                    return False
            except PermissionError:
                text = "Permission error"
                info_text = f'Specified file "{_fp}" is not accessible!'
                showInvalidPageInputDialog(text, info_text)
                # traceback.print_exc()
                return False

            dst = report_folder.joinpath(abs_kickmap_f.name).resolve()

            alter_elements.append(
                dict(name=name, item="INPUT_FILE", string_value=str(dst))
            )

            try:
                if not dst.exists():
                    print(
                        (
                            f'Kickmap element "{name}": Copying file "{abs_kickmap_f}" '
                            f"into {dst}."
                        )
                    )
                    shutil.copy(str(abs_kickmap_f), str(dst))
            except PermissionError:
                text = "Permission error"
                info_text = (
                    f'Kickmap element "{name}": Kickmap file copy destination '
                    f'"{dst}" is not accessible!'
                )
                showInvalidPageInputDialog(text, info_text)
                # traceback.print_exc()
                return False
        # (SEGMENT#1 End)
        #
        try:
            input_LTE_Path.exists()
        except PermissionError:
            text = "Permission error"
            info_text = (
                f'Specified input LTE file "{input_LTE_Path}" is not accessible!'
            )
            showInvalidPageInputDialog(text, info_text)
            # traceback.print_exc()
            return False
        #
        if not input_LTE_Path.exists():
            if alter_elements:
                pe.eleutil.save_lattice_after_alter_elements(
                    orig_LTE_filepath, input_LTE_filepath, alter_elements
                )
            else:
                shutil.copy(orig_LTE_filepath, input_LTE_filepath)
            f(mod_conf["input_LTE"], "regenerate_zeroSexts", True)
        else:
            sha = hashlib.sha1()
            sha.update(input_LTE_Path.read_text().encode("utf-8"))
            existing_SHA1 = sha.hexdigest()

            if alter_elements:

                altered_LTE_filepath = input_LTE_filepath + ".tmp"

                adjust_input_LTE_kickmap_filepaths(
                    orig_LTE_filepath, altered_LTE_filepath, alter_elements
                )

                sha = hashlib.sha1()
                sha.update(Path(altered_LTE_filepath).read_text().encode("utf-8"))
                altered_SHA1 = sha.hexdigest()

                if altered_SHA1 != existing_SHA1:
                    shutil.move(altered_LTE_filepath, input_LTE_filepath)
                    f(mod_conf["input_LTE"], "regenerate_zeroSexts", True)
                else:
                    f(mod_conf["input_LTE"], "regenerate_zeroSexts", False)

            else:

                sha = hashlib.sha1()
                sha.update(orig_LTE_Path.read_text().encode("utf-8"))
                orig_SHA1 = sha.hexdigest()

                if orig_SHA1 != existing_SHA1:
                    shutil.copy(orig_LTE_filepath, input_LTE_filepath)
                    f(mod_conf["input_LTE"], "regenerate_zeroSexts", True)
                else:
                    f(mod_conf["input_LTE"], "regenerate_zeroSexts", False)

        f(mod_conf["input_LTE"], "filepath", input_LTE_filepath)
        #
        f(
            mod_conf["input_LTE"],
            "parent_LTE_hash",
            self.field("edit_parent_LTE_hash").strip(),
        )
        #
        lattice_authors = self.field("edit_LTE_authors").strip()
        seq = CommentedSeq(
            [
                yaml.scalarstring.SingleQuotedScalarString(v)
                for v in lattice_authors.split(",")
            ]
        )
        seq.fa.set_flow_style()
        f(mod_conf, "lattice_authors", seq)
        #
        f(
            mod_conf,
            "lattice_received_date",
            self.field("date_LTE_received").toString("MM/dd/yyyy"),
        )
        #
        f(
            mod_conf["lattice_props"]["req_props"]["floor_comparison"],
            "ref_flr_filepath",
            ref_FLR_filepath,
        )
        #
        E_GeV = self.field("edit_E_GeV")
        try:
            E_MeV = float(E_GeV) * 1e3
            E_MeV_list = [E_MeV]
        except ValueError:
            E_MeV_list = [float(v) * 1e3 for v in E_GeV.split(",")]
        seq = CommentedSeq(E_MeV_list)
        seq.fa.set_flow_style()
        f(mod_conf, "E_MeV", seq)
        if False:
            _check_if_yaml_writable(mod_conf)
        #
        harmonic_number = self.field("spin_harmonic_number")
        try:
            assert harmonic_number > 1
        except:
            traceback.print_exc()
            text = "Invalid value"
            info_text = "Harmonic number must be a positive integer!"
            showInvalidPageInputDialog(text, info_text)
            return False
        f(mod_conf, "harmonic_number", harmonic_number)
        #
        f(mod_conf, "use_beamline_cell", use_beamline_cell)
        f(mod_conf, "use_beamline_ring", use_beamline_ring)
        #
        f(mod_conf, "ring_is_a_simple_multiple_of_cells", ring_is_simple_mult_cells)
        #
        f(mod_conf, "enable_pyelegant_stdout", self.field("check_pyele_stdout"))

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        flat_used_elem_names = LTE.flat_used_elem_names
        all_elem_names = [name for name, _, _ in LTE.elem_defs]
        spos_list = []
        elem_type_list = []
        occur_list = []
        occur_dict = {}
        cur_spos = 0.0
        for elem_name in flat_used_elem_names:
            matched_index = all_elem_names.index(elem_name)
            _, elem_type, prop_str = LTE.elem_defs[matched_index]

            elem_type_list.append(elem_type)

            prop = LTE.parse_elem_properties(prop_str)
            if "L" in prop:
                L = prop["L"]
            else:
                L = 0.0
            cur_spos += L
            spos_list.append(cur_spos)

            if elem_name in occur_dict:
                occur_dict[elem_name] += 1
            else:
                occur_dict[elem_name] = 1
            occur_list.append(occur_dict[elem_name])
        data = list(zip(spos_list, flat_used_elem_names, elem_type_list, occur_list))
        self.wizardObj.model_elem_list = TableModelElemList(data)

        return True


class TableModelElemList(QtCore.QAbstractTableModel):
    """"""

    def __init__(self, data):
        """Constructor"""

        super().__init__()

        self._data = data

        self._headers = ["s [m]", "Name", "Type", "ElemOccurrence"]
        self._row_numbers = range(len(data))

        self._spos = np.array([v[0] for v in data])
        self._flat_elem_names = np.array([v[1] for v in data])

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._headers[section])

            if orientation == Qt.Vertical:
                return str(self._row_numbers[section])

    def validateStraightCenterElements(self, M_LS_name, M_SS_name):
        """"""

        matched_inds = np.where(self._flat_elem_names == M_LS_name)[0]

        if len(matched_inds) != 2:
            return "not 2 M_LS"

        ds_thresh = 1e-6

        if np.abs(self._spos[matched_inds[0]] - 0.0) > ds_thresh:
            return "wrong M_LS#1 spos"

        s_max = np.max(self._spos)

        if np.abs(self._spos[matched_inds[1]] - s_max) > ds_thresh:
            return "wrong M_LS#2 spos"

        matched_inds = np.where(self._flat_elem_names == M_SS_name)[0]

        if np.abs(self._spos[matched_inds[0]] - s_max / 2) > ds_thresh:
            return "wrong M_SS#1 spos"


class PageStraightCenters(PageStandard):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.safeFindChild(QtWidgets.QTableView, "tableView_elem_list")
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_1st_center_name")
        self.registerFieldOnFirstShow("edit_LS_center_name*", edit)

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_SS_1st_center_name")
        self.registerFieldOnFirstShow("edit_SS_center_name*", edit)

        # Set fields
        self._update_fields()

        # Establish connections

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_1st_center_name")
        edit.textChanged.connect(self.synchronize_LS_elem_names)

        b = self.safeFindChild(QtWidgets.QPushButton, "pushButton_reload_LTE")
        b.clicked.connect(self.update_LTE_table)

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_1st_center_name")
        edit.textChanged.connect(self.completeChanged)
        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_SS_1st_center_name")
        edit.textChanged.connect(self.completeChanged)

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            beta = self.conf["lattice_props"]["req_props"]["beta"]
            LS_elem_name = str(beta["LS"]["name"])
        except:
            if TEST_MODE:
                LS_elem_name = "M_LS"
            else:
                LS_elem_name = ""
        self.setField("edit_LS_center_name", LS_elem_name)
        self.synchronize_LS_elem_names(LS_elem_name)

        try:
            SS_elem_name = str(beta["SS"]["name"])
            self.setField("edit_SS_center_name", SS_elem_name)
        except:
            if TEST_MODE:
                self.setField("edit_SS_center_name", "M_SS")
            else:
                self.setField("edit_SS_center_name", "")

    def isComplete(self):
        """"""

        if self.field("edit_LS_center_name").strip() == "":
            return False

        if self.field("edit_SS_center_name").strip() == "":
            return False

        return True

    def synchronize_LS_elem_names(self, new_text):
        """"""

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_2nd_center_name")
        edit.setText(self.field("edit_LS_center_name"))

    def update_LTE_table(self):
        """"""

        self.wizardObj.page_links["LTE"].validatePage()
        tableView = self.safeFindChild(QtWidgets.QTableView, "tableView_elem_list")
        tableView.setModel(self.wizardObj.model_elem_list)

    def validatePage(self):
        """"""

        M_LS_name = self.field("edit_LS_center_name").strip()
        M_SS_name = self.field("edit_SS_center_name").strip()

        for elem_name in [M_LS_name, M_SS_name]:
            if elem_name not in self.wizardObj.LTE.flat_used_elem_names:
                text = f'Invalid element name "{elem_name}"'
                info_text = (
                    f'Element "{elem_name}" does not exist in the specfied LTE '
                    f"file."
                )
                showInvalidPageInputDialog(text, info_text)
                return False

        flag = self.wizardObj.model_elem_list.validateStraightCenterElements(
            M_LS_name, M_SS_name
        )
        if flag is None:
            pass
        elif flag == "not 2 M_LS":
            text = f'Invalid number of elements named "{M_LS_name}"'
            info_text = (
                f'There must be only 2 elements named "{M_LS_name}" in the '
                f"one super-period beamline."
            )
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == "wrong M_LS#1 spos":
            text = f"Invalid s-pos of 1st LS center"
            info_text = f'First LS center element "{M_LS_name}" must have s-pos = 0.'
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == "wrong M_LS#2 spos":
            text = f"Invalid s-pos of 2nd LS center"
            info_text = (
                f'2nd LS center element "{M_LS_name}" must be at the end of the'
                f"one super-period beamline."
            )
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == "wrong M_SS#1 spos":
            text = f"Invalid s-pos of 1st SS center"
            info_text = (
                f'1st SS center element "{M_SS_name}" must be at the middle of '
                f"the one super-period beamline."
            )
            showInvalidPageInputDialog(text, info_text)
            return False
        else:
            raise ValueError()

        mod_conf = duplicate_yaml_conf(self.conf)

        f = partial(update_aliased_scalar, mod_conf)
        #
        d = mod_conf["lattice_props"]["req_props"]
        #
        f(d["beta"]["LS"], "name", M_LS_name)
        f(d["beta"]["LS"], "occur", 0)
        #
        f(d["beta"]["SS"], "name", M_SS_name)
        f(d["beta"]["SS"], "occur", 0)
        #
        f(d["floor_comparison"]["LS"]["ref_elem"], "name", "MID")
        f(d["floor_comparison"]["LS"]["ref_elem"], "occur", 1)
        #
        f(d["floor_comparison"]["LS"]["cur_elem"], "name", M_LS_name)
        f(d["floor_comparison"]["LS"]["cur_elem"], "occur", 1)
        #
        f(d["floor_comparison"]["SS"]["ref_elem"], "name", "MID")
        f(d["floor_comparison"]["SS"]["ref_elem"], "occur", 0)
        #
        f(d["floor_comparison"]["SS"]["cur_elem"], "name", M_SS_name)
        f(d["floor_comparison"]["SS"]["cur_elem"], "occur", 0)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True


class PagePhaseAdv(PageStandard):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.safeFindChild(
            QtWidgets.QTableView, "tableView_elem_list_phase_adv"
        )
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        obj = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_disp_bump_marker_name")
        self.registerFieldOnFirstShow("edit_disp_bump_marker_name", obj)

        obj = self.safeFindChild(QtWidgets.QComboBox, "comboBox_n_disp_bumps")
        self.registerFieldOnFirstShow("combo_n_disp_bumps", obj, property="currentText")

        # Set fields
        self._update_fields()

        # Establish connections

        b = self.safeFindChild(QtWidgets.QPushButton, "pushButton_reload_LTE_phase_adv")
        b.clicked.connect(self.update_LTE_table)

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            phase_adv = self.conf["lattice_props"]["opt_props"]["phase_adv"]
        except:
            if TEST_MODE:
                elem_name = "MDISP"
            else:
                elem_name = ""
            self.setField("edit_disp_bump_marker_name", elem_name)
            self.setField("combo_n_disp_bumps", "2")

            return

        try:
            if "MDISP across LS" in phase_adv:
                elem_name = str(phase_adv["MDISP across LS"]["elem2"]["name"])
                n_disp_bumps = 2
            else:
                elem_name = str(phase_adv["MDISP 0&1"]["elem1"]["name"])
                n_disp_bumps = 4
            self.setField("edit_disp_bump_marker_name", elem_name)
            self.setField("combo_n_disp_bumps", f"{n_disp_bumps:d}")
        except:
            if TEST_MODE:
                elem_name = "MDISP"
            else:
                elem_name = ""
            self.setField("edit_disp_bump_marker_name", elem_name)
            self.setField("combo_n_disp_bumps", "2")

    def update_LTE_table(self):
        """"""

        self.wizardObj.page_links["LTE"].validatePage()
        self.wizardObj.page_links["straight_centers"].validatePage()
        tableView = self.safeFindChild(
            QtWidgets.QTableView, "tableView_elem_list_phase_adv"
        )
        tableView.setModel(self.wizardObj.model_elem_list)

    def validatePage(self):
        """"""

        flat_used_elem_names = self.wizardObj.LTE.flat_used_elem_names

        marker_name = self.field("edit_disp_bump_marker_name").strip()
        n_disp_bumps = int(self.field("combo_n_disp_bumps"))

        mod_conf = duplicate_yaml_conf(self.conf)

        if marker_name == "":
            d = mod_conf["lattice_props"]
            if "opt_props" in d:
                # This wizard only allows "phase_adv" as part of "opt_props". If the
                # other types of "opt_props" exist (for example, from the default
                # full example config), then those are deleted.
                del d["opt_props"]

            self.wizardObj.update_conf_on_all_pages(mod_conf)

            return True

        if marker_name not in flat_used_elem_names:
            text = "Invalid element name"
            info_text = f'Specified element "{marker_name}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            return False

        actual_n = flat_used_elem_names.count(marker_name)
        if actual_n != n_disp_bumps:
            text = "Mismatch in number of dispersion bump markers"
            info_text = (
                f"There are {actual_n:d} instances of the specified element "
                f'"{marker_name}", while you expect {n_disp_bumps:d}!'
            )
            showInvalidPageInputDialog(text, info_text)
            return False

        # This wizard only allows "phase_adv" as part of "opt_props". If the
        # other types of "opt_props" exist (for example, from the default
        # full example config), then those are deleted.
        if "opt_props" in mod_conf["lattice_props"]:
            for k in list(mod_conf["lattice_props"]["opt_props"]):
                if k != "phase_adv":
                    del mod_conf["lattice_props"]["opt_props"][k]
        else:
            mod_conf["lattice_props"]["opt_props"] = CommentedMap(
                {"phase_adv": CommentedMap({})}
            )

        d = mod_conf["lattice_props"]["opt_props"]["phase_adv"]
        #
        f = partial(update_aliased_scalar, mod_conf)

        sqss = SingleQuotedScalarString

        if n_disp_bumps == 2:

            for _k in list(d):
                if _k not in ["MDISP across LS", "MDISP across SS"]:
                    del d[_k]

            if "MDISP across LS" not in d:
                d["MDISP across LS"] = CommentedMap({})
            d2 = d["MDISP across LS"]
            f(
                d2,
                "pdf_label",
                (
                    r"Phase Advance btw. Disp. Bumps\n across LS "
                    r"$(\Delta\nu_x, \Delta\nu_y)$"
                ),
            )
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Horizontal Phase Advance btw. Disp. Bumps across LS "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("x"),
                ]
            )
            seq.fa.set_flow_style()
            if "xlsx_label" not in d2:
                d2["xlsx_label"] = CommentedMap({})
            f(d2["xlsx_label"], "x", seq)
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Vertical Phase Advance btw. Disp. Bumps across LS "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("y"),
                ]
            )
            seq.fa.set_flow_style()
            f(d2["xlsx_label"], "y", seq)
            if "elem1" not in d2:
                d2["elem1"] = CommentedMap({})
            f(
                d2["elem1"],
                "name",
                mod_conf["lattice_props"]["req_props"]["beta"]["LS"]["name"],
            )
            f(d2["elem1"], "occur", 0)
            if "elem2" not in d2:
                d2["elem2"] = CommentedMap({})
            f(d2["elem2"], "name", marker_name)
            f(d2["elem2"], "occur", 0)
            #
            if "MDISP across SS" not in d:
                d["MDISP across SS"] = CommentedMap({})
            d2 = d["MDISP across SS"]
            f(
                d2,
                "pdf_label",
                (
                    r"Phase Advance btw. Disp. Bumps\n across SS "
                    r"$(\Delta\nu_x, \Delta\nu_y)$"
                ),
            )
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Horizontal Phase Advance btw. Disp. Bumps across SS "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("x"),
                ]
            )
            seq.fa.set_flow_style()
            if "xlsx_label" not in d2:
                d2["xlsx_label"] = CommentedMap({})
            f(d2["xlsx_label"], "x", seq)
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Vertical Phase Advance btw. Disp. Bumps across SS "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("y"),
                ]
            )
            seq.fa.set_flow_style()
            f(d2["xlsx_label"], "y", seq)
            if "elem1" not in d2:
                d2["elem1"] = CommentedMap({})
            f(d2["elem1"], "name", marker_name)
            f(d2["elem1"], "occur", 0)
            if "elem2" not in d2:
                d2["elem2"] = CommentedMap({})
            f(d2["elem2"], "name", marker_name)
            f(d2["elem2"], "occur", 1)

        elif n_disp_bumps == 4:

            for _k in list(d):
                if _k not in ["MDISP 0&1"]:
                    del d[_k]

            if "MDISP 0&1" not in d:
                d["MDISP 0&1"] = CommentedMap({})
            d2 = d["MDISP 0&1"]
            f(
                d2,
                "pdf_label",
                (
                    r"Phase Advance btw. Dispersion Bumps\n "
                    r"$(\Delta\nu_x, \Delta\nu_y)$"
                ),
            )
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Horizontal Phase Advance btw. Disp. Bumps "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("x"),
                ]
            )
            seq.fa.set_flow_style()
            if "xlsx_label" not in d2:
                d2["xlsx_label"] = CommentedMap({})
            f(d2["xlsx_label"], "x", seq)
            seq = CommentedSeq(
                [
                    "normal",
                    sqss("Vertical Phase Advance btw. Disp. Bumps "),
                    "italic_greek",
                    sqss("Delta"),
                    "italic_greek",
                    sqss("nu"),
                    "italic_sub",
                    sqss("y"),
                ]
            )
            seq.fa.set_flow_style()
            f(d2["xlsx_label"], "y", seq)
            if "elem1" not in d2:
                d2["elem1"] = CommentedMap({})
            f(d2["elem1"], "name", marker_name)
            f(d2["elem1"], "occur", 0)
            if "elem2" not in d2:
                d2["elem2"] = CommentedMap({})
            f(d2["elem2"], "name", marker_name)
            f(d2["elem2"], "occur", 1)

        else:
            raise ValueError()

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True


class PageStraightDrifts(PageStandard):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.safeFindChild(
            QtWidgets.QTableView, "tableView_elem_list_straight_drifts"
        )
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_drift_names")
        self.registerFieldOnFirstShow("edit_half_LS_drifts*", edit)

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_SS_drift_names")
        self.registerFieldOnFirstShow("edit_half_SS_drifts*", edit)

        # Set fields
        self._update_fields()

        # Establish connections

        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_LS_drift_names")
        edit.textChanged.connect(self.completeChanged)
        edit = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_SS_drift_names")
        edit.textChanged.connect(self.completeChanged)

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            length = self.conf["lattice_props"]["req_props"]["length"]
            self.setField(
                "edit_half_LS_drifts",
                ", ".join([_strip_unquote(s) for s in length["LS"]["name_list"]]),
            )
            self.setField(
                "edit_half_SS_drifts",
                ", ".join([_strip_unquote(s) for s in length["SS"]["name_list"]]),
            )
        except:
            if TEST_MODE:
                self.setField("edit_half_LS_drifts", "DH5")
                self.setField("edit_half_SS_drifts", "DL5")
            else:
                self.setField("edit_half_LS_drifts", "")
                self.setField("edit_half_SS_drifts", "")

    def validatePage(self):
        """"""

        flat_used_elem_names = np.array(self.wizardObj.LTE.flat_used_elem_names)

        half_LS_drift_name_list = [
            token.strip()
            for token in self.field("edit_half_LS_drifts").split(",")
            if token.strip()
        ]
        half_SS_drift_name_list = [
            token.strip()
            for token in self.field("edit_half_SS_drifts").split(",")
            if token.strip()
        ]

        for drift_names, LS_or_SS in [
            (half_LS_drift_name_list, "LS"),
            (half_SS_drift_name_list, "SS"),
        ]:

            for name in drift_names:
                if name not in flat_used_elem_names:
                    text = "Invalid element name"
                    info_text = (
                        f'Specified element "{name}" as part of {LS_or_SS} '
                        f"drift does not exist!"
                    )
                    showInvalidPageInputDialog(text, info_text)
                    return False

            match_found = False
            for starting_ind in np.where(flat_used_elem_names == drift_names[0])[0]:

                for i, next_name in enumerate(drift_names[1:]):
                    if flat_used_elem_names[starting_ind + i + 1] != next_name:
                        break
                else:
                    match_found = True

                if match_found:
                    break
            else:
                text = f"No matching element name list for {LS_or_SS} drift"
                info_text = (
                    f"Specified consecutive element name list for {LS_or_SS} "
                    f"drift does not exist!"
                )
                showInvalidPageInputDialog(text, info_text)
                return False

        mod_conf = duplicate_yaml_conf(self.conf)

        d = mod_conf["lattice_props"]["req_props"]
        f = partial(update_aliased_scalar, mod_conf)
        #
        f(
            d["length"]["LS"],
            "name_list",
            [
                name if ":" not in name else f'"{name}"'
                for name in half_LS_drift_name_list
            ],
        )
        # ^ Need to add quotes to avoid YAML parsing error for those elements
        #   whose names contain ":".
        f(d["length"]["LS"], "multiplier", 2.0)
        #
        f(
            d["length"]["SS"],
            "name_list",
            [
                name if ":" not in name else f'"{name}"'
                for name in half_SS_drift_name_list
            ],
        )
        # ^ Need to add quotes to avoid YAML parsing error for those elements
        #   whose names contain ":".
        f(d["length"]["SS"], "multiplier", 2.0)

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True


class PageGenReportTest1(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            return

        self.wizardObj = self.wizard()

        self.establish_connections()

        self._pageInitialized = True

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = True
        for k, v in mod_conf["nonlin"]["include"].items():
            mod_conf["nonlin"]["include"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        if "pdf_table_order" in mod_conf["lattice_props"]:
            del mod_conf["lattice_props"]["pdf_table_order"]
        if "xlsx_table_order" in mod_conf["lattice_props"]:
            del mod_conf["lattice_props"]["xlsx_table_order"]

        d = mod_conf["lattice_props"]
        if "opt_props" in d:
            if "phase_adv" in d["opt_props"]:
                if "MDISP 0&1" in d["opt_props"]["phase_adv"]:
                    Ls = [
                        CommentedSeq(["opt_props", "phase_adv", "MDISP 0&1"]),
                    ]
                    for L in Ls:
                        L.fa.set_flow_style()
                    f(d, "append_opt_props_to_pdf_table", Ls)

                    Ls = [
                        CommentedSeq(["opt_props", "phase_adv", "MDISP 0&1", "x"]),
                        CommentedSeq(["opt_props", "phase_adv", "MDISP 0&1", "y"]),
                    ]
                    for L in Ls:
                        L.fa.set_flow_style()
                    f(d, "append_opt_props_to_xlsx_table", Ls)
                else:
                    Ls = [
                        CommentedSeq(["opt_props", "phase_adv", "MDISP across LS"]),
                        CommentedSeq(["opt_props", "phase_adv", "MDISP across SS"]),
                    ]
                    for L in Ls:
                        L.fa.set_flow_style()
                    f(d, "append_opt_props_to_pdf_table", Ls)

                    Ls = [
                        CommentedSeq(
                            ["opt_props", "phase_adv", "MDISP across LS", "x"]
                        ),
                        CommentedSeq(
                            ["opt_props", "phase_adv", "MDISP across LS", "y"]
                        ),
                        CommentedSeq(
                            ["opt_props", "phase_adv", "MDISP across SS", "x"]
                        ),
                        CommentedSeq(
                            ["opt_props", "phase_adv", "MDISP across SS", "y"]
                        ),
                    ]
                    for L in Ls:
                        L.fa.set_flow_style()
                    f(d, "append_opt_props_to_xlsx_table", Ls)
            else:
                if "append_opt_props_to_pdf_table" in d:
                    del d["append_opt_props_to_pdf_table"]
                if "append_opt_props_to_xlsx_table" in d:
                    del d["append_opt_props_to_xlsx_table"]
        else:
            if "append_opt_props_to_pdf_table" in d:
                del d["append_opt_props_to_pdf_table"]
            if "append_opt_props_to_xlsx_table" in d:
                del d["append_opt_props_to_xlsx_table"]

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageTwissPlots(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        w = self.safeFindChild(QtWidgets.QSpinBox, "spinBox_element_divisions")
        self.registerFieldOnFirstShow("spin_element_divisions", w)

        w = self.safeFindChild(QtWidgets.QSpinBox, "spinBox_font_size")
        self.registerFieldOnFirstShow("spin_font_size", w)

        w = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_extra_dy_frac")
        self.registerFieldOnFirstShow("edit_extra_dy_frac", w)

        w = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_full_r_margin")
        self.registerFieldOnFirstShow("edit_full_r_margin", w)

        for sec in ["sec1", "sec2", "sec3"]:
            for suffix in ["smin", "smax", "r_margin"]:
                w = self.safeFindChild(QtWidgets.QLineEdit, f"lineEdit_{sec}_{suffix}")
                self.registerFieldOnFirstShow(f"edit_{sec}_{suffix}", w)

        # Set fields
        self._update_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            twiss_calc_opts = self.conf["lattice_props"]["twiss_calc_opts"]
            element_divisions = twiss_calc_opts["one_period"]["element_divisions"]
        except:
            element_divisions = None

        if element_divisions:
            self.setField("spin_element_divisions", element_divisions)

        try:
            twiss_plot_opts = self.conf["lattice_props"]["twiss_plot_opts"][
                "one_period"
            ]
        except:
            twiss_plot_opts = None

        if twiss_plot_opts is None:
            return

        try:
            self.setField(
                "edit_full_r_margin",
                "{:.2g}".format(twiss_plot_opts[0]["right_margin_adj"]),
            )
        except:
            pass

        for iSec, sec in enumerate(["sec1", "sec2", "sec3"]):
            for suffix in ["smin", "smax", "r_margin"]:
                try:
                    _d = twiss_plot_opts[iSec + 1]
                    if suffix == "r_margin":
                        value = _d["right_margin_adj"]
                        value = "{:.3g}".format(value)
                    elif suffix == "smin":
                        value = "{:.3g}".format(_d["slim"][0])
                    elif suffix == "smax":
                        value = "{:.3g}".format(_d["slim"][1])

                    self.setField(f"edit_{sec}_{suffix}", value)
                except:
                    pass

        try:
            disp_elem_names = twiss_plot_opts[1]["disp_elem_names"]
            try:
                self.setField("spin_font_size", disp_elem_names["font_size"])
            except:
                pass
            try:
                self.setField(
                    "edit_extra_dy_frac",
                    "{:.3g}".format(disp_elem_names["extra_dy_frac"]),
                )
            except:
                pass
        except:
            pass

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = True
        for k, v in mod_conf["nonlin"]["include"].items():
            mod_conf["nonlin"]["include"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        mod_conf["lattice_props"]["twiss_calc_opts"]["one_period"][
            "element_divisions"
        ] = self.field("spin_element_divisions")

        plot_opts = mod_conf["lattice_props"]["twiss_plot_opts"]

        m = CommentedMap(
            {
                "bends": True,
                "quads": True,
                "sexts": True,
                "octs": True,
                "font_size": self.field("spin_font_size"),
                "extra_dy_frac": float(self.field("edit_extra_dy_frac")),
            }
        )
        m.fa.set_flow_style()
        m.yaml_set_anchor("disp_elem_names")
        disp_elem_names = m

        m_list = []
        m = CommentedMap({"right_margin_adj": float(self.field("edit_full_r_margin"))})
        m.fa.set_flow_style()
        m_list.append(m)

        smins, smaxs = {}, {}
        for sec in ["sec1", "sec2", "sec3"]:
            smins[sec] = float(self.field(f"edit_{sec}_smin"))
            smaxs[sec] = float(self.field(f"edit_{sec}_smax"))

            sq = CommentedSeq([smins[sec], smaxs[sec]])
            sq.fa.set_flow_style()

            m = CommentedMap(
                {
                    "right_margin_adj": float(self.field(f"edit_{sec}_r_margin")),
                    "slim": sq,
                    "disp_elem_names": disp_elem_names,
                }
            )

            m_list.append(m)

        f(plot_opts, "one_period", CommentedSeq(m_list))
        f(plot_opts, "ring_natural", [])
        f(plot_opts, "ring", [])

        # _check_if_yaml_writable(plot_opts)

        plot_captions = mod_conf["lattice_props"]["twiss_plot_captions"]

        for i, sec in enumerate(list(smins)):
            _smin, _smax = smins[sec], smaxs[sec]
            f(
                plot_captions["one_period"],
                i + 1,
                SingleQuotedScalarString(
                    rf"Twiss functions $({_smin:.6g} \le s \le {_smax:.6g})$."
                ),
            )
        f(plot_captions, "ring_natural", [])
        f(plot_captions, "ring", [])

        # _check_if_yaml_writable(plot_captions)

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageParagraphs(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Adjust initial splitter ratio
        w = self.safeFindChild(QtWidgets.QSplitter, "splitter_paragraphs")
        # w.setSizes([10, 2])
        w.setSizes([200, 90])

        # Hook up models to views

        # Register fields

        w = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_keywords")
        self.registerFieldOnFirstShow("edit_keywords*", w)

        w = self.safeFindChild(
            QtWidgets.QPlainTextEdit, "plainTextEdit_lattice_description"
        )
        self.registerFieldOnFirstShow("edit_lattice_description*", w, "plainText")

        w = self.safeFindChild(
            QtWidgets.QPlainTextEdit, "plainTextEdit_lattice_properties"
        )
        self.registerFieldOnFirstShow("edit_lattice_properties", w, "plainText")

        # Set fields
        self._update_fields()

        # Establish connections

        self.establish_connections()

        w = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_keywords")
        w.textChanged.connect(self.completeChanged)

        w = self.safeFindChild(
            QtWidgets.QPlainTextEdit, "plainTextEdit_lattice_description"
        )
        w.textChanged.connect(self.completeChanged)

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            text_list = self.conf["lattice_keywords"]
        except:
            text_list = None
        if text_list is not None:
            self.setField("edit_keywords", ", ".join(text_list))

        try:
            text_list = self.conf["report_paragraphs"]["lattice_description"]
        except:
            text_list = None
        if text_list is not None:
            self.setField("edit_lattice_description", "\n".join(text_list))

        try:
            text_list = self.conf["report_paragraphs"]["lattice_properties"]
        except:
            text_list = None
        if text_list is not None:
            self.setField("edit_lattice_properties", "\n".join(text_list))

    def isComplete(self):
        """"""

        if self.field("edit_keywords").strip() == "":
            return False

        if self.field("edit_lattice_description").strip() == "":
            return False

        return True

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = duplicate_yaml_conf(orig_conf)

        # f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False
        for k, v in mod_conf["nonlin"]["include"].items():
            mod_conf["nonlin"]["include"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        keywords = CommentedSeq(
            [s.strip() for s in self.field("edit_keywords").split(",")]
        )
        keywords.fa.set_flow_style()
        mod_conf["lattice_keywords"] = keywords

        mod_conf["report_paragraphs"]["lattice_description"] = self.field(
            "edit_lattice_description"
        ).splitlines()
        mod_conf["report_paragraphs"]["lattice_properties"] = self.field(
            "edit_lattice_properties"
        ).splitlines()

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageNKicks(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        for k in ["CSBEND", "KQUAD", "KSEXT", "KOCT"]:
            w = self.safeFindChild(QtWidgets.QSpinBox, f"spinBox_N_KICKS_{k}")
            self.registerFieldOnFirstShow(f"spin_{k}", w)

        # Set fields
        self._update_fields()

        # Establish connections

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        try:
            N_KICKS = self.conf["nonlin"]["N_KICKS"]
        except:
            N_KICKS = None
        if N_KICKS is not None:
            for k, v in N_KICKS.items():
                self.setField(f"spin_{k}", v)

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = duplicate_yaml_conf(orig_conf)

        # f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False
        for k, v in mod_conf["nonlin"]["include"].items():
            mod_conf["nonlin"]["include"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        for k in ["CSBEND", "KQUAD", "KSEXT", "KOCT"]:
            mod_conf["nonlin"]["N_KICKS"][k] = self.field(f"spin_{k}")

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageXYAperTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("abs_xmax", edit),
            ("abs_ymax", edit),
            ("ini_ndiv", spin),
            ("n_lines", spin),
            ("neg_y_search", check),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                abs_xmax="edit_float",
                abs_ymax="edit_float",
                ini_ndiv="spin",
                n_lines="spin",
                neg_y_search="check",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "xy_aper"
        self.register_test_prod_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageFmapXYTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("xmin", edit),
            ("xmax", edit),
            ("ymin", edit),
            ("ymax", edit),
            ("nx", spin),
            ("ny", spin),
            ("x_offset", edit),
            ("y_offset", edit),
            ("delta_offset", edit),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("nx", spin),
            ("ny", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                xmin="edit_float",
                xmax="edit_float",
                ymin="edit_float",
                ymax="edit_float",
                nx="spin",
                ny="spin",
                x_offset="edit_float",
                y_offset="edit_float",
                delta_offset="edit_%",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                nx="spin",
                ny="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "fmap_xy"
        self.register_test_prod_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageFmapPXTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("xmin", edit),
            ("xmax", edit),
            ("delta_min", edit),
            ("delta_max", edit),
            ("nx", spin),
            ("ndelta", spin),
            ("x_offset", edit),
            ("y_offset", edit),
            ("delta_offset", edit),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("nx", spin),
            ("ndelta", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                xmin="edit_float",
                xmax="edit_float",
                delta_min="edit_%",
                delta_max="edit_%",
                nx="spin",
                ndelta="spin",
                x_offset="edit_float",
                y_offset="edit_float",
                delta_offset="edit_%",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                nx="spin",
                ndelta="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "fmap_px"
        self.register_test_prod_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageCmapXYTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "cmap_xy"
        self.register_test_prod_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageCmapPXTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "cmap_px"
        self.register_test_prod_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageMomAperTest(PageNonlinCalcTest):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_test_prod_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.test_list = [
            ("n_turns", spin),
            ("x_initial", edit),
            ("y_initial", edit),
            ("delta_negative_start", edit),
            ("delta_negative_limit", edit),
            ("delta_positive_start", edit),
            ("delta_positive_limit", edit),
            ("init_delta_step_size", edit),
            ("include_name_pattern", edit),
            ("steps_back", spin),
            ("splits", spin),
            ("split_step_divisor", spin),
            ("forbid_resonance_crossing", check),
            ("soft_failure", check),
            ("rf_cavity___on", check),
            ("radiation_on", check),
            # ('rf_cavity___auto_voltage_from_nonlin_chrom', combo),
            # ('rf_cavity___manual', edit),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.prod_list = [
            ("n_turns", spin),
            ("init_delta_step_size", edit),
            ("include_name_pattern", edit),
            ("steps_back", spin),
            ("splits", spin),
            ("split_step_divisor", spin),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.setter_getter = {
            "test": dict(
                n_turns="spin",
                x_initial="edit_float",
                y_initial="edit_float",
                delta_negative_start="edit_%",
                delta_negative_limit="edit_%",
                delta_positive_start="edit_%",
                delta_positive_limit="edit_%",
                init_delta_step_size="edit_%",
                include_name_pattern="edit_str",
                steps_back="spin",
                splits="spin",
                split_step_divisor="spin",
                forbid_resonance_crossing="check",
                soft_failure="check",
                rf_cavity___on="check",
                radiation_on="check",
                # rf_cavity___auto_voltage_from_nonlin_chrom='combo',
                # rf_cavity___manual='edit_float',
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "production": dict(
                n_turns="spin",
                init_delta_step_size="edit_%",
                include_name_pattern="edit_str",
                steps_back="spin",
                splits="spin",
                split_step_divisor="spin",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
        }

        self.calc_type = "mom_aper"
        self.register_test_prod_option_widgets()
        #
        # Register special widgets
        w = self.safeFindChild(
            combo, "comboBox_rf_cavity___auto_voltage_from_nonlin_chrom_mom_aper_test"
        )
        self.registerFieldOnFirstShow("combo_auto_voltage", w)
        w = self.safeFindChild(edit, "lineEdit_manual_rf_mom_aper_test")
        self.registerFieldOnFirstShow("edit_manual_voltage", w)

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_test_prod_option_fields().

        # Set fields
        self.set_test_prod_option_fields()
        #
        # Set special fields
        try:
            rf_cav_opts = self.conf["nonlin"]["calc_opts"]["mom_aper"]["test"][
                "rf_cavity"
            ]
        except:
            rf_cav_opts = None
        if rf_cav_opts is not None:
            w_c = self.safeFindChild(
                combo,
                "comboBox_rf_cavity___auto_voltage_from_nonlin_chrom_mom_aper_test",
            )
            w_e = self.safeFindChild(edit, "lineEdit_manual_rf_mom_aper_test")
            man_rf_volt = rf_cav_opts.get("rf_volt", None)
            man_rf_percent = rf_cav_opts.get("rf_bucket_percent", None)
            auto_opt = rf_cav_opts.get(
                "auto_voltage_from_nonlin_chrom", "resonance_crossing"
            )
            if man_rf_volt is not None:
                w_e.setText(self.converters["edit_float"]["set"](man_rf_volt))
                index = w_c.findText("manual [V]")
                assert index != -1
                w_c.setCurrentIndex(index)
                w_e.setEnabled(True)
            elif man_rf_percent is not None:
                w_e.setText(self.converters["edit_float"]["set"](man_rf_percent))
                index = w_c.findText("manual [%]")
                assert index != -1
                w_c.setCurrentIndex(index)
                w_e.setEnabled(True)
            else:
                index = w_c.findText(auto_opt)
                if index != -1:
                    w_c.setCurrentIndex(index)
                    w_e.setEnabled(False)
                else:
                    text = "Invalid rf_cavity option"
                    info_text = (
                        f"nonlin/calc_opts/mom_aper/test/rf_cavity: {str(rf_cav_opts)}"
                    )
                    showInvalidPageInputDialog(text, info_text)
                    return
        else:
            index = w_c.findText("resonance_crossing")
            assert index != -1
            w_c.setCurrentIndex(index)
            w_e.setEnabled(False)

        # Establish connections
        self.establish_connections()

        w = self.safeFindChild(
            combo, "comboBox_rf_cavity___auto_voltage_from_nonlin_chrom_mom_aper_test"
        )
        w.currentTextChanged.connect(self.updateManualRFLineEditState)

        self._pageInitialized = True

    def updateManualRFLineEditState(self, new_text):
        """"""

        w = self.safeFindChild(QtWidgets.QLineEdit, "lineEdit_manual_rf_mom_aper_test")

        if new_text in ("resonance_crossing", "undefined_tunes", "scan_range"):
            w.setEnabled(False)
        elif new_text in ("manual [V]", "manual [%]"):
            w.setEnabled(True)
        else:
            raise ValueError(f'Unexpected text value: "{new_text}"')

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = super().modify_conf(orig_conf)

        # Handle special fields

        edit, combo = (QtWidgets.QLineEdit, QtWidgets.QComboBox)
        w_c = self.safeFindChild(
            combo, "comboBox_rf_cavity___auto_voltage_from_nonlin_chrom_mom_aper_test"
        )
        w_e = self.safeFindChild(edit, "lineEdit_manual_rf_mom_aper_test")

        rf_cav_opts = mod_conf["nonlin"]["calc_opts"]["mom_aper"]["test"]["rf_cavity"]

        text = w_c.currentText()
        if text in ("resonance_crossing", "undefined_tunes", "scan_range"):
            yaml_append_map(rf_cav_opts, "auto_voltage_from_nonlin_chrom", text)
            for k in ["rf_volt", "rf_bucket_percent"]:
                if k in rf_cav_opts:
                    del rf_cav_opts[k]
        elif text == "manual [V]":
            v = self.converters["edit_float"]["get"](w_e.text())
            yaml_append_map(rf_cav_opts, "rf_volt", v)
            for k in ["rf_bucket_percent", "auto_voltage_from_nonlin_chrom"]:
                if k in rf_cav_opts:
                    del rf_cav_opts[k]
        elif text == "manual [%]":
            v = self.converters["edit_float"]["get"](w_e.text())
            yaml_append_map(rf_cav_opts, "rf_bucket_percent", v)
            for k in ["rf_volt", "auto_voltage_from_nonlin_chrom"]:
                if k in rf_cav_opts:
                    del rf_cav_opts[k]
        else:
            raise ValueError(f'Unexpected text value: "{text}"')

        return mod_conf


class PageTswa(PageNonlinCalcPlot):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_calc_plot_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.calc_list = [
            ("n_turns", spin),
            ("abs_xmax", edit),
            ("abs_ymax", edit),
            ("nx", spin),
            ("ny", spin),
            ("x_offset", edit),
            ("y_offset", edit),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.plot_list = [
            ("fit_xmin", edit),
            ("fit_ymin", edit),
            ("fit_xmax", edit),
            ("fit_ymax", edit),
            ("footprint_nux_min", edit),
            ("footprint_nux_max", edit),
            ("footprint_nuy_min", edit),
            ("footprint_nuy_max", edit),
        ]
        self.setter_getter = {
            "calc": dict(
                n_turns="spin",
                abs_xmax="edit_float",
                abs_ymax="edit_float",
                nx="spin",
                ny="spin",
                x_offset="edit_float",
                y_offset="edit_float",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "plot": dict(
                fit_xmin="edit_float",
                fit_ymin="edit_float",
                fit_xmax="edit_float",
                fit_ymax="edit_float",
                footprint_nux_min="edit_float",
                footprint_nux_max="edit_float",
                footprint_nuy_min="edit_float",
                footprint_nuy_max="edit_float",
            ),
        }

        self.calc_type = "tswa"
        self.register_calc_plot_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_calc_plot_option_fields().

        # Set fields
        self.set_calc_plot_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageNonlinChrom(PageNonlinCalcPlot):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self.set_calc_plot_option_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.calc_list = [
            ("n_turns", spin),
            ("delta_min", edit),
            ("delta_max", edit),
            ("ndelta", spin),
            ("x_offset", edit),
            ("y_offset", edit),
            ("delta_offset", edit),
            ("save_fft", check),
            ("partition", combo),
            ("qos", combo),
            ("ntasks", spin),
            ("time", edit),
        ]
        self.plot_list = [
            ("max_chrom_order", spin),
            ("plot_fft", check),
            ("fit_delta_min", edit),
            ("fit_delta_max", edit),
            ("footprint_nux_min", edit),
            ("footprint_nux_max", edit),
            ("footprint_nuy_min", edit),
            ("footprint_nuy_max", edit),
        ]
        self.setter_getter = {
            "calc": dict(
                n_turns="spin",
                delta_min="edit_%",
                delta_max="edit_%",
                ndelta="spin",
                x_offset="edit_float",
                y_offset="edit_float",
                delta_offset="edit_%",
                save_fft="check",
                partition="combo",
                qos="combo",
                ntasks="spin",
                time="edit_str_None",
            ),
            "plot": dict(
                max_chrom_order="spin",
                plot_fft="check",
                fit_delta_min="edit_%",
                fit_delta_max="edit_%",
                footprint_nux_min="edit_float",
                footprint_nux_max="edit_float",
                footprint_nuy_min="edit_float",
                footprint_nuy_max="edit_float",
            ),
        }

        self.calc_type = "nonlin_chrom"
        self.register_calc_plot_option_widgets()

        self._setup_partition_qos_objects()
        # ^ This must come before self.set_calc_plot_option_fields().

        # Set fields
        self.set_calc_plot_option_fields()

        # Establish connections
        self.establish_connections()

        self._pageInitialized = True


class PageNonlinProduction(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        self._checkboxes = {}
        for opt_type in ["include", "recalc", "replot"]:
            for calc_type in ["all"] + self.all_calc_types:
                w = self.safeFindChild(check, f"checkBox_nonlin_{opt_type}_{calc_type}")
                self._checkboxes[f"{opt_type}:{calc_type}"] = w
                self.registerFieldOnFirstShow(f"check_{opt_type}_{calc_type}", w)

        self.setter_getter = {
            "cmap_xy": dict(cmin="spin", cmax="spin"),
            "cmap_px": dict(cmin="spin", cmax="spin"),
        }

        for calc_type, d in self.setter_getter.items():
            for prop_name, wtype in d.items():
                suffix = f"{calc_type}_{prop_name}"
                if wtype == "spin":
                    w = self.safeFindChild(spin, f"spinBox_{suffix}")
                    self.registerFieldOnFirstShow(f"{wtype}_{suffix}", w)
                else:
                    raise ValueError()

        # Set fields
        self._update_fields()

        # Establish connections

        self.establish_connections()  # "Run Production" & "Open PDF" buttons

        for opt_type in ["include", "recalc", "replot"]:
            obj = self._checkboxes[f"{opt_type}:all"]
            obj.clicked.connect(
                partial(self.change_all_nonlin_calc_type_checkbox_states, opt_type)
            )

        for calc_type in self.all_calc_types:
            obj = self._checkboxes[f"include:{calc_type}"]
            obj.clicked.connect(partial(self.uncheck_all, "include"))
            # obj.stateChanged.connect(partial(
            # self.change_state_recalc_replot_checkboxes, calc_type))

            obj = self._checkboxes[f"recalc:{calc_type}"]
            obj.clicked.connect(partial(self.uncheck_all, "recalc"))
            # obj.stateChanged.connect(partial(
            # self.change_state_replot_checkboxes, calc_type))

            obj = self._checkboxes[f"replot:{calc_type}"]
            obj.clicked.connect(partial(self.uncheck_all, "replot"))

        self._pageInitialized = True

    def _update_fields(self):
        """"""

        ncf = self.conf["nonlin"]

        for opt_type in ["include", "recalc", "replot"]:
            all_true = True
            for calc_type in self.all_calc_types:
                v = ncf[opt_type][calc_type]
                self.setField(f"check_{opt_type}_{calc_type}", v)
                if not v:
                    all_true = False
            self._checkboxes[f"{opt_type}:all"].setChecked(all_true)

        for calc_type in ["cmap_xy", "cmap_px"]:
            if f"{calc_type}_plot_opts" in ncf:
                opts = ncf[f"{calc_type}_plot_opts"]
                for k, v in opts.items():
                    conv_type = self.setter_getter[calc_type][k]
                    conv = self.converters[conv_type]["set"]
                    wtype = conv_type.split("_")[0]
                    self.setField(f"{wtype}_{calc_type}_{k}", conv(v))

    def change_all_nonlin_calc_type_checkbox_states(self, opt_type, checked):
        """"""

        for calc_type in self.all_calc_types:
            obj = self._checkboxes[f"{opt_type}:{calc_type}"]
            obj.setChecked(checked)

    def uncheck_all(self, opt_type, checked):
        """"""
        if not checked:
            self._checkboxes[f"{opt_type}:all"].setChecked(False)

    def change_state_recalc_replot_checkboxes(self, calc_type, include_checked):
        """"""

        obj = self._checkboxes[f"recalc:{calc_type}"]
        obj.setEnabled(include_checked)
        obj = self._checkboxes[f"replot:{calc_type}"]
        obj.setEnabled(include_checked)

    def change_state_replot_checkboxes(self, calc_type, recalc_checked):
        """"""

        obj = self._checkboxes[f"replot:{calc_type}"]
        obj.setEnabled(recalc_checked)

    def validatePage(self):
        """"""

        mod_conf = self.modify_conf(self.conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        w = self.findChildren(QtWidgets.QTabWidget, QtCore.QRegExp("tabWidget_std_.+"))[
            0
        ]
        w.setCurrentIndex(0)  # show "stdout" tab before report generation starts

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False
        for k in ["rf", "lifetime"]:
            if k in mod_conf:
                mod_conf[k]["include"] = False

        ncf = mod_conf["nonlin"]

        for opt_type in ["include", "recalc", "replot"]:
            for calc_type in self.all_calc_types:
                v = self.field(f"check_{opt_type}_{calc_type}")
                ncf[opt_type][calc_type] = v

        # TODO
        if "driving_terms" in ncf["include"]:
            ncf["include"]["driving_terms"] = True

        if ncf["use_beamline"] is not mod_conf["use_beamline_ring"]:
            ncf["use_beamline"] = mod_conf["use_beamline_ring"]

        for calc_type in self.all_calc_types:
            ncf["selected_calc_opt_names"][calc_type] = "production"

        for calc_type in ["cmap_xy", "cmap_px"]:
            for k, conv_type in self.setter_getter[calc_type].items():
                conv = self.converters[conv_type]["get"]
                wtype = conv_type.split("_")[0]
                v = conv(self.field(f"{wtype}_{calc_type}_{k}"))

                f(ncf[f"{calc_type}_plot_opts"], k, v)

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class PageRfTau(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

    def initializePage(self):
        """"""

        if self._pageInitialized:
            self._update_fields()
            return

        self.wizardObj = self.wizard()

        # Hook up models to views

        # Register fields

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )
        self.rf_list = [
            ("rf_include", check),
            ("rf_recalc", check),
            # ('harmonic_number', spin),
            ("rf_voltages", edit),
            ("bucket_height_min", edit),
            ("bucket_height_max", edit),
            ("v_scan_npts", spin),
            ("ntasks_rf_tau", spin),
        ]
        self.tau_calc_list = [
            ("tau_calc_include", check),
            ("tau_recalc", check),
            ("num_filled_bunches", spin),
            ("max_mom_aper", combo),
            ("coupling_list", edit),
        ]
        self.tau_plot_list = [
            ("tau_plot_include", check),
            ("tau_replot", check),
        ]

        self.setter_getter = {
            "rf": dict(
                rf_include="check",
                rf_recalc="check",  # harmonic_number='spin',
                rf_voltages="edit_special",
                bucket_height_min="edit_float",
                bucket_height_max="edit_float",
                v_scan_npts="spin",
                ntasks_rf_tau="spin",
            ),
            "tau_calc": dict(
                tau_calc_include="check",
                tau_recalc="check",
                num_filled_bunches="spin",
                max_mom_aper="combo_special",
                coupling_list="edit_special",
            ),
            "tau_plot": dict(
                tau_plot_include="check",
                tau_replot="check",
            ),
        }

        for k_wtype_list in [self.rf_list, self.tau_calc_list, self.tau_plot_list]:
            for k, wtype in k_wtype_list:
                w_suffix = f_suffix = k
                if wtype == spin:
                    w = self.safeFindChild(spin, f"spinBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"spin_{f_suffix}", w)
                elif wtype == edit:
                    w = self.safeFindChild(edit, f"lineEdit_{w_suffix}")
                    self.registerFieldOnFirstShow(f"edit_{f_suffix}", w)
                elif wtype == check:
                    w = self.safeFindChild(check, f"checkBox_{w_suffix}")
                    self.registerFieldOnFirstShow(f"check_{f_suffix}", w)
                elif wtype == combo:
                    w = self.safeFindChild(combo, f"comboBox_{w_suffix}")
                    self.registerFieldOnFirstShow(
                        f"combo_{f_suffix}", w, property="currentText"
                    )
                else:
                    raise ValueError()

        self.tau_special_data = dict(
            total_beam_current=[],
            loss_plots_set={"E_MeV": [], "rf_V": [], "coupling": []},
        )
        for k in list(self.tau_special_data):
            w = self.safeFindChild(edit, f"lineEdit_{k}")
            self.registerFieldOnFirstShow(f"edit_{k}", w)
        self.tau_calc_special_widgets = [
            self.safeFindChild(edit, "lineEdit_total_beam_current"),
            self.safeFindChild(
                QtWidgets.QPushButton, "pushButton_edit_total_beam_current"
            ),
        ]
        self.tau_plot_special_widgets = [
            self.safeFindChild(edit, "lineEdit_loss_plots_set"),
            self.safeFindChild(QtWidgets.QPushButton, "pushButton_edit_loss_plots_set"),
        ]

        # Set fields
        self._update_fields()

        # Establish connections

        self.establish_connections()  # "Run Production" & "Open PDF" buttons

        w = self.safeFindChild(check, "checkBox_rf_include")
        w.stateChanged.connect(partial(self.enable_opts, self.rf_list))

        w = self.safeFindChild(check, "checkBox_tau_calc_include")
        w.stateChanged.connect(partial(self.enable_opts, self.tau_calc_list))
        w.stateChanged.connect(
            partial(self.enable_special_opts, self.tau_calc_special_widgets)
        )

        w = self.safeFindChild(check, "checkBox_tau_plot_include")
        w.stateChanged.connect(partial(self.enable_opts, self.tau_plot_list))
        w.stateChanged.connect(
            partial(self.enable_special_opts, self.tau_plot_special_widgets)
        )

        w = self.safeFindChild(combo, "comboBox_max_mom_aper")
        w.currentTextChanged.connect(self.setEditable_comboBox_max_mom_aper)
        self.setEditable_comboBox_max_mom_aper(w.currentText())

        w = self.safeFindChild(
            QtWidgets.QPushButton, f"pushButton_edit_total_beam_current"
        )
        w.clicked.connect(self.edit_total_beam_current)

        w = self.safeFindChild(QtWidgets.QPushButton, f"pushButton_edit_loss_plots_set")
        w.clicked.connect(self.edit_loss_plots_set)

        w = self.safeFindChild(QtWidgets.QPushButton, f"pushButton_pre_scan_V_tau")
        w.clicked.connect(self.pre_scan_V_tau)

        self._pageInitialized = True

    def pre_scan_V_tau(self):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        w = self.findChild(edit, "lineEdit_bucket_height_min")
        min_height_percent = float(w.text().strip())

        w = self.findChild(edit, "lineEdit_bucket_height_max")
        max_height_percent = float(w.text().strip())

        w = self.findChild(spin, "spinBox_v_scan_npts")
        v_scan_npts = w.value()

        min_rf_V_step = 0.01e6

        if min_height_percent <= 0.0:
            text = "Invalid min RF bucket height"
            informative_text = "Min RF bucket height must be positive."
            showInvalidPageInputDialog(text, informative_text)
            return

        if min_height_percent == max_height_percent:
            text = "Invalid min/max RF bucket heights"
            informative_text = "Min and max RF bucket height must be different."
            showInvalidPageInputDialog(text, informative_text)
            return
        elif min_height_percent > max_height_percent:
            text = "Invalid min/max RF bucket heights"
            informative_text = (
                "Min RF bucket height cannot be smaller than max RF bucket height."
            )
            showInvalidPageInputDialog(text, informative_text)
            return

        w = self.findChild(spin, "spinBox_ntasks_rf_tau")
        ntasks = w.value()

        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)

        QtWidgets.QApplication.processEvents()

        yml = yaml.YAML()
        yml.preserve_quotes = True
        user_conf = yml.load(Path(self.wizardObj.config_filepath).read_text())

        report_obj = genreport.Report_NSLS2U_Default(
            self.wizardObj.config_filepath, user_conf=user_conf, build=False
        )

        rf_volt_ranges = report_obj.calc_rf_volt_range_from_bucket_height_range(
            min_height_percent, max_height_percent, min_rf_V_step
        )

        E_MeV_list = report_obj._get_E_MeV_list()

        calc_opts = report_obj.conf["lifetime"]["calc_opts"]
        #
        total_beam_current_mA = calc_opts["total_beam_current_mA"]
        if isinstance(total_beam_current_mA, list):
            total_beam_current_mA_list = total_beam_current_mA
            if len(total_beam_current_mA_list) != len(E_MeV_list):
                total_beam_current_mA_list = self.tau_special_data["total_beam_current"]
                if len(total_beam_current_mA_list) != len(E_MeV_list):
                    print(
                        (
                            "Error: Number of total beam current values must be "
                            "the same as that of energy values."
                        )
                    )
                    QtWidgets.QApplication.restoreOverrideCursor()
                    return
        else:
            total_beam_current_mA_list = [total_beam_current_mA] * len(E_MeV_list)
        #
        num_filled_bunches = calc_opts["num_filled_bunches"]
        raw_coupling_specs = calc_opts["coupling"]
        #
        raw_max_mom_aper_percent = calc_opts["max_mom_aper_percent"]
        if raw_max_mom_aper_percent in (None, "None", "none"):
            max_mom_aper_percent = None
        elif raw_max_mom_aper_percent in ("Auto", "auto"):
            print(
                (
                    "NotImplementedError: Automatically determine deltaLimit from "
                    "nonlin_chrom integer-crossing & nan"
                )
            )
            QtWidgets.QApplication.restoreOverrideCursor()
            return
        else:
            max_mom_aper_percent = float(raw_max_mom_aper_percent)

        req_d = report_obj.req_data_for_calc["lifetime_props"]
        #
        T_rev_s = req_d["T_rev_s"]  # [s]
        n_periods_in_ring = req_d["n_periods_in_ring"]
        circumf = req_d["circumf"]
        #
        # equilibrium emittance [m-rad]
        eps_0_list = [req_d["eps_x_pm"] * 1e-12]
        for _d in req_d["extra_Es"]:
            eps_0_list.append(_d["eps_x_pm"] * 1e-12)

        req_d = report_obj.req_data_for_calc["rf_dep_props"]
        #
        alphac = req_d["alphac"]  # momentum compaction
        U0_ev_list = [req_d["U0_eV"]]  # energy loss per turn [eV]
        sigma_delta_list = [req_d["sigma_delta_percent"] * 1e-2]  # energy spread [frac]
        for _d in req_d["extra_Es"]:
            U0_ev_list.append(_d["U0_eV"])
            sigma_delta_list.append(_d["sigma_delta_percent"] * 1e-2)

        h = report_obj.conf["harmonic_number"]

        try:
            mmap_pgz_filepath = report_obj.get_nonlin_data_filepaths()["mom_aper"]
            assert os.path.exists(mmap_pgz_filepath)
            d = pe.util.load_pgz_file(mmap_pgz_filepath)
            mmap_d = d["data"]["mmap"]
        except PermissionError:
            print(f'** Permission Error: File "{mmap_pgz_filepath}" is not accessible')
        except:
            print(traceback.format_exc())

            print(
                (
                    "To compute beam lifetime, you must first "
                    "compute momentum aperture."
                )
            )

            QtWidgets.QApplication.restoreOverrideCursor()
            return

        # Since the momentum aperture data only extend to one super period,
        # the data must be duplicated to cover the whole ring. Also, create
        # an SDDS file from the .pgz/.hdf5 file, which can be directly fed
        # into the ELEGANT's lifetime calculation function.
        mmap_sdds_filepath_cell = mmap_pgz_filepath[:-4] + ".mmap"
        mmap_sdds_filepath_ring = mmap_pgz_filepath[:-4] + ".mmapxt"
        pe.sdds.dicts2sdds(
            mmap_sdds_filepath_cell,
            params=mmap_d["scalars"],
            columns=mmap_d["arrays"],
            outputMode="binary",
        )
        # Based on computeLifetime.py used with MOGA (geneopt.py)
        if report_obj.is_ring_a_multiple_of_superperiods():
            dup_filenames = " ".join([mmap_sdds_filepath_cell] * n_periods_in_ring)
            msectors = 1
            cmd_list = [
                f"sddscombine {dup_filenames} -pipe=out",
                f'sddsprocess -pipe "-redefine=col,s,s i_page 1 - {circumf:.16g} {n_periods_in_ring} / * {msectors} * +,units=m"',
                f"sddscombine -pipe -merge",
                f"sddsprocess -pipe=in {mmap_sdds_filepath_ring} -filter=col,s,0,{circumf:.16g}",
            ]
            if False:
                print(cmd_list)
            result, err, returncode = pe.util.chained_Popen(cmd_list)
        else:
            # "mmap_sdds_filepath_cell" actually covers the whole ring,
            # even though the name name implies only one super-period.
            # So, there is no need for extending it.
            shutil.copy(mmap_sdds_filepath_cell, mmap_sdds_filepath_ring)

        LTE_filepath = report_obj.input_LTE_filepath
        use_beamline_ring = report_obj.conf["use_beamline_ring"]

        try:
            common_remote_opts = {}
            common_remote_opts.update(self.conf["nonlin"]["common_remote_opts"])

            LoLs = report_obj.scan_V_tau(
                rf_volt_ranges,
                v_scan_npts,
                ntasks,
                E_MeV_list,
                eps_0_list,
                total_beam_current_mA_list,
                U0_ev_list,
                sigma_delta_list,
                T_rev_s,
                num_filled_bunches,
                raw_coupling_specs,
                alphac,
                circumf,
                LTE_filepath,
                h,
                mmap_sdds_filepath_ring,
                max_mom_aper_percent,
                use_beamline_ring,
                remote_opts=common_remote_opts,
            )
        except:
            print(traceback.format_exc())
            QtWidgets.QApplication.restoreOverrideCursor()
            return
        rf_Vs_LoL = LoLs["rf_V"]
        bucket_heights_percent_LoL = LoLs["bucket_height_percent"]
        taus_LoL = LoLs["tau"]
        coupling_percent_str_LoL = LoLs["coupling_percent_str"]
        eps_y_str_LoL = LoLs["eps_y_str"]

        figs, axs = [], []
        for _ in raw_coupling_specs:
            fig1, ax1 = plt.subplots()
            plt.xlabel(r"$\mathrm{RF\, Voltage\, [MV]}$", size=18)
            fig2, ax2 = plt.subplots()
            plt.xlabel(r"$\mathrm{RF\, Bucket\, Height\, [\%]}$", size=18)
            figs.append({"MV": fig1, "%": fig2})
            axs.append({"MV": ax1, "%": ax2})

        for iEnergy, (
            E_MeV,
            rf_Vs_list,
            bucket_heights_percent_list,
            taus_list,
            coupling_percent_str_list,
            eps_y_str_list,
        ) in enumerate(
            zip(
                E_MeV_list,
                rf_Vs_LoL,
                bucket_heights_percent_LoL,
                taus_LoL,
                coupling_percent_str_LoL,
                eps_y_str_LoL,
            )
        ):

            for iCoup, (
                rf_Vs,
                bucket_heights_percent,
                tau_hrs,
                coupling_percent_str,
                eps_y_str,
            ) in enumerate(
                zip(
                    rf_Vs_list,
                    bucket_heights_percent_list,
                    taus_list,
                    coupling_percent_str_list,
                    eps_y_str_list,
                )
            ):

                ax1, ax2 = axs[iCoup]["MV"], axs[iCoup]["%"]
                label = (
                    f"{E_MeV/1e3:.1f} GeV; "
                    f"({coupling_percent_str}/{eps_y_str}) Coup."
                )
                label = (
                    r"$\mathrm{" + label.replace("%", r"\%").replace(" ", r"\, ") + "}$"
                )
                ax1.plot(rf_Vs / 1e6, tau_hrs, ".-", label=label)
                ax2.plot(bucket_heights_percent, tau_hrs, ".-", label=label)
        for ax_d in axs:
            for ax in [ax_d["MV"], ax_d["%"]]:
                plt.sca(ax)
                plt.ylabel(r"$\mathrm{{Beam\, Lifetime\, [hr]}}$", size=18)
                plt.legend(loc="best")
                plt.tight_layout()

        plt.show()

        QtWidgets.QApplication.restoreOverrideCursor()

    def edit_total_beam_current(self):
        """"""

        data = []
        for i, E_MeV in enumerate(self.E_MeV_list):
            try:
                mA = self.tau_special_data["total_beam_current"][i]
            except:
                mA = None
            data.append([E_MeV / 1e3, mA])

        dialog = TotalBeamCurrentEditor(data)
        dialog.exec()

        if dialog.result() == QtWidgets.QDialog.Accepted:
            total_beam_current_mA_list = [a[1] for a in data]
            self.update_total_beam_current_view(total_beam_current_mA_list)

    def update_total_beam_current_view(self, total_beam_current_mA_list):
        """"""

        self.tau_special_data["total_beam_current"] = total_beam_current_mA_list

        self.setField(
            "edit_total_beam_current",
            ", ".join(
                [
                    f"{mA:.1f} ({E_MeV/1e3:.2g} GeV)"
                    for E_MeV, mA in zip(self.E_MeV_list, total_beam_current_mA_list)
                ]
            ),
        )

    def edit_loss_plots_set(self):
        """"""

        loss_plots_set = self.tau_special_data["loss_plots_set"]
        assert (
            len(loss_plots_set["E_MeV"])
            == len(loss_plots_set["rf_V"])
            == len(loss_plots_set["coupling"])
        )
        data = {
            "loss_plots_set": list(
                zip(
                    loss_plots_set["E_MeV"],
                    loss_plots_set["rf_V"],
                    loss_plots_set["coupling"],
                )
            )
        }

        data["E_GeV"] = np.array(self.E_MeV_list) / 1e3

        rf_volts = self._get_rf_volts()
        if rf_volts is None:
            return
        try:
            data["rf_MV"] = np.array(rf_volts) / 1e6
        except TypeError:
            data["rf_MV"] = np.array([np.array(_v) for _v in rf_volts]) / 1e6

        coupling_list = self._get_coupling_list()
        data["coupling"] = coupling_list

        dialog = LossPlotsSetEditor(data)
        dialog.exec()

        if dialog.result() == QtWidgets.QDialog.Accepted:
            loss_plots_indexes = {"E_MeV": [], "rf_V": [], "coupling": []}
            for iGeV, iVolt, iCoup in data["loss_plots_set"]:
                loss_plots_indexes["E_MeV"].append(iGeV)
                loss_plots_indexes["rf_V"].append(iVolt)
                loss_plots_indexes["coupling"].append(iCoup)
            self.update_loss_plots_set_view(loss_plots_indexes, rf_volts, coupling_list)

    def _get_rf_volts(self):
        """"""
        conv_type = self.setter_getter["rf"]["rf_voltages"]
        conv = partial(self.converters[conv_type]["get"], "rf_voltages")
        rf_volts = conv(self.field("edit_rf_voltages"))

        return rf_volts

    def _get_coupling_list(self):
        """"""

        conv_type = self.setter_getter["tau_calc"]["coupling_list"]
        conv = partial(self.converters[conv_type]["get"], "coupling_list")
        coupling_list = conv(self.field("edit_coupling_list"))

        return coupling_list

    def update_loss_plots_set_view(self, loss_plots_indexes, rf_volts, coupling_list):
        """"""

        if rf_volts is None:
            return

        self.tau_special_data["loss_plots_set"] = loss_plots_indexes

        try:
            try:
                iter(rf_volts[0])
            except TypeError:
                s = ", ".join(
                    [
                        (
                            f"({self.E_MeV_list[iGeV]/1e3:.2g}GeV, "
                            f"{rf_volts[iVolt]/1e6:.2g}MV, {coupling_list[iCoup]})"
                        )
                        for iGeV, iVolt, iCoup in zip(
                            loss_plots_indexes["E_MeV"],
                            loss_plots_indexes["rf_V"],
                            loss_plots_indexes["coupling"],
                        )
                    ]
                )
            else:
                s = ", ".join(
                    [
                        (
                            f"({self.E_MeV_list[iGeV]/1e3:.2g}GeV, "
                            f"{rf_volts[iGeV][iVolt]/1e6:.2g}MV, {coupling_list[iCoup]})"
                        )
                        for iGeV, iVolt, iCoup in zip(
                            loss_plots_indexes["E_MeV"],
                            loss_plots_indexes["rf_V"],
                            loss_plots_indexes["coupling"],
                        )
                    ]
                )
        except:
            s = ""
            for k, v in self.tau_special_data["loss_plots_set"].items():
                v.clear()

        self.setField("edit_loss_plots_set", s)

    def _update_fields(self):
        """"""

        if "rf" in self.conf:
            rf = self.conf["rf"]
            #
            if rf.get("include", True):
                self.setField("check_rf_include", True)
                self.enable_opts(self.rf_list, True)
            else:
                self.setField("check_rf_include", False)
                self.enable_opts(self.rf_list, False)
            #
            conv_type = self.setter_getter["rf"]["rf_recalc"]
            conv = self.converters[conv_type]["set"]
            self.setField("check_rf_recalc", conv(rf.get("recalc", False)))
            #
            if "calc_opts" in rf:
                calc_opts = rf["calc_opts"]
                #
                # conv_type = self.setter_getter['rf']['harmonic_number']
                # conv = self.converters[conv_type]['set']
                # self.setField('spin_harmonic_number',
                # conv(calc_opts.get('harmonic_number', 1320)))
                #
                conv_type = self.setter_getter["rf"]["rf_voltages"]
                conv = partial(self.converters[conv_type]["set"], "rf_voltages")
                self.setField("edit_rf_voltages", conv(calc_opts.get("rf_V", [3e6])))
        else:
            self.setField("check_rf_include", False)
            self.enable_opts(self.rf_list, False)

        if "lifetime" not in self.conf:

            self.setField("check_tau_calc_include", False)
            self.enable_opts(self.tau_calc_list, False)
            self.enable_special_opts(self.tau_calc_special_widgets, False)

            self.setField("check_tau_plot_include", False)
            self.enable_opts(self.tau_plot_list, False)
            self.enable_special_opts(self.tau_plot_special_widgets, False)

        else:

            lifetime = self.conf["lifetime"]

            if lifetime.get("include", True):
                self.setField("check_tau_calc_include", True)
                self.enable_opts(self.tau_calc_list, True)
                self.enable_special_opts(self.tau_calc_special_widgets, True)

                if "plot_opts" in lifetime:
                    self.setField("check_tau_plot_include", True)
                    self.enable_opts(self.tau_plot_list, True)
                    self.enable_special_opts(self.tau_plot_special_widgets, True)
                else:
                    self.setField("check_tau_plot_include", False)
                    self.enable_opts(self.tau_plot_list, False)
                    self.enable_special_opts(self.tau_plot_special_widgets, False)
            else:
                self.setField("check_tau_calc_include", False)
                self.enable_opts(self.tau_calc_list, False)
                self.enable_special_opts(self.tau_calc_special_widgets, False)

                self.setField("check_tau_plot_include", False)
                self.enable_opts(self.tau_plot_list, False)
                self.enable_special_opts(self.tau_plot_special_widgets, False)

            conv_type = self.setter_getter["tau_calc"]["tau_recalc"]
            conv = self.converters[conv_type]["set"]
            self.setField("check_tau_recalc", conv(lifetime.get("recalc", False)))

            conv_type = self.setter_getter["tau_plot"]["tau_replot"]
            conv = self.converters[conv_type]["set"]
            self.setField("check_tau_replot", conv(lifetime.get("replot", False)))

            E_MeV_list = self.conf["E_MeV"]
            if not isinstance(E_MeV_list, list):
                E_MeV_list = [E_MeV_list]
            n_E_MeV = len(E_MeV_list)
            #
            self.E_MeV_list = E_MeV_list

            if "calc_opts" in lifetime:
                calc_opts = lifetime["calc_opts"]

                total_beam_current_mA_list = calc_opts.get("total_beam_current_mA", 5e2)
                if not isinstance(total_beam_current_mA_list, list):
                    total_beam_current_mA_list = [total_beam_current_mA_list] * n_E_MeV
                if len(total_beam_current_mA_list) == n_E_MeV:
                    self.update_total_beam_current_view(total_beam_current_mA_list)

                conv_type = self.setter_getter["tau_calc"]["num_filled_bunches"]
                conv = self.converters[conv_type]["set"]
                self.setField(
                    "spin_num_filled_bunches",
                    conv(calc_opts.get("num_filled_bunches", 1200)),
                )
                #
                conv_type = self.setter_getter["tau_calc"]["max_mom_aper"]
                conv = partial(self.converters[conv_type]["set"], "max_mom_aper")
                self.setField(
                    "combo_max_mom_aper",
                    conv(calc_opts.get("max_mom_aper_percent", None)),
                )
                #
                conv_type = self.setter_getter["tau_calc"]["coupling_list"]
                conv = partial(self.converters[conv_type]["set"], "coupling_list")
                self.setField(
                    "edit_coupling_list",
                    conv(calc_opts.get("coupling", ["8pm", "100%"])),
                )

                if "V_scan" not in calc_opts:
                    scan_opts = dict(
                        min_bucket_height_percent=1.0,
                        max_bucket_height_percent=6.0,
                        nVolts=51,
                        ntasks=50,
                    )
                else:
                    scan_opts = calc_opts["V_scan"]
                #
                conv_type = self.setter_getter["rf"]["bucket_height_min"]
                conv = self.converters[conv_type]["set"]
                self.setField(
                    "edit_bucket_height_min",
                    conv(scan_opts.get("min_bucket_height_percent", 1.0)),
                )
                #
                conv_type = self.setter_getter["rf"]["bucket_height_max"]
                conv = self.converters[conv_type]["set"]
                self.setField(
                    "edit_bucket_height_max",
                    conv(scan_opts.get("max_bucket_height_percent", 6.0)),
                )
                #
                conv_type = self.setter_getter["rf"]["v_scan_npts"]
                conv = self.converters[conv_type]["set"]
                self.setField("spin_v_scan_npts", conv(scan_opts.get("nVolts", 51)))
                #
                conv_type = self.setter_getter["rf"]["ntasks_rf_tau"]
                conv = self.converters[conv_type]["set"]
                self.setField("spin_ntasks_rf_tau", conv(scan_opts.get("ntasks", 50)))

            if "plot_opts" in lifetime:
                plot_opts = lifetime["plot_opts"]
                loss_plots_indexes = plot_opts.get(
                    "loss_plots_indexes", dict(E_MeV=[], rf_V=[], coupling=[])
                )
                self.update_loss_plots_set_view(
                    loss_plots_indexes, self._get_rf_volts(), self._get_coupling_list()
                )

    def setEditable_comboBox_max_mom_aper(self, new_text):
        """"""

        w = self.safeFindChild(QtWidgets.QComboBox, "comboBox_max_mom_aper")

        if new_text != "None":
            w.setEditable(True)
            validator = QtGui.QDoubleValidator(-100, 100, 6)
            validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
            w.setValidator(validator)

            w.installEventFilter(self)
        else:
            w.setEditable(False)

    def comboBox_max_mom_aper_focusOut_handler(self, comboBox_max_mom_aper):
        """"""

        w = comboBox_max_mom_aper
        validator = w.validator()

        if validator is None:
            return

        new_text = w.currentText()

        custom_val_index = 1

        if validator.validate(new_text, 0)[0] == QtGui.QValidator.Acceptable:
            if w.findText(new_text) != -1:
                return

            w.insertItem(custom_val_index, new_text)

            while w.count() >= 3:
                w.removeItem(custom_val_index + 1)
        else:
            w.setCurrentIndex(custom_val_index)

    def eventFilter(self, widget, event):

        # FocusOut event
        if event.type() == QtCore.QEvent.FocusOut:

            if widget == self.safeFindChild(
                QtWidgets.QComboBox, "comboBox_max_mom_aper"
            ):

                self.comboBox_max_mom_aper_focusOut_handler(widget)

        return super().eventFilter(widget, event)

    def enable_special_opts(self, widget_list, enabled):
        """"""

        for w in widget_list:
            w.setEnabled(enabled)

    def enable_opts(self, k_wtype_list, enabled):
        """"""

        spin, edit, check, combo = (
            QtWidgets.QSpinBox,
            QtWidgets.QLineEdit,
            QtWidgets.QCheckBox,
            QtWidgets.QComboBox,
        )

        for k, wtype in k_wtype_list:

            if k.endswith("_include"):
                continue

            w_suffix = k
            if wtype == spin:
                w = self.safeFindChild(spin, f"spinBox_{w_suffix}")
            elif wtype == edit:
                w = self.safeFindChild(edit, f"lineEdit_{w_suffix}")
            elif wtype == check:
                w = self.safeFindChild(check, f"checkBox_{w_suffix}")
            elif wtype == combo:
                w = self.safeFindChild(combo, f"comboBox_{w_suffix}")
            else:
                raise ValueError()

            w.setEnabled(enabled)

    def validatePage(self):
        """"""

        try:
            mod_conf = self.modify_conf(self.conf)
        except AssertionError:
            return False

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return True

    def modify_conf(self, orig_conf):
        """"""

        w = self.findChildren(QtWidgets.QTabWidget, QtCore.QRegExp("tabWidget_std_.+"))[
            0
        ]
        w.setCurrentIndex(0)  # show "stdout" tab before report generation starts

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        for k in ["recalc", "replot"]:
            mod_conf["lattice_props"][k] = False

        ncf = mod_conf["nonlin"]
        for opt_type in ["recalc", "replot"]:
            for calc_type in self.all_calc_types:
                ncf[opt_type][calc_type] = False

        conv_type = self.setter_getter["rf"]["rf_include"]
        conv = self.converters[conv_type]["get"]
        if conv(self.field("check_rf_include")):
            if "rf" not in mod_conf:
                yaml_append_map(mod_conf, "rf", CommentedMap({}))
            #
            rf = mod_conf["rf"]
            rf["include"] = True
            #
            conv_type = self.setter_getter["rf"]["rf_recalc"]
            conv = self.converters[conv_type]["get"]
            rf["recalc"] = conv(self.field("check_rf_recalc"))
            #
            if "calc_opts" not in rf:
                yaml_append_map(rf, "calc_opts", CommentedMap({}))
            #
            calc_opts = rf["calc_opts"]
            #
            # conv_type = self.setter_getter['rf']['harmonic_number']
            # conv = self.converters[conv_type]['get']
            # mod_conf['harmonic_number'] = conv(self.field('spin_harmonic_number'))
            #
            conv_type = self.setter_getter["rf"]["rf_voltages"]
            conv = partial(self.converters[conv_type]["get"], "rf_voltages")
            v = conv(self.field("edit_rf_voltages"))
            if v is None:
                raise AssertionError("Invalid RF voltages")
            seq = CommentedSeq(v)
            seq.fa.set_flow_style()
            calc_opts["rf_V"] = seq
        else:
            if "rf" in mod_conf:
                mod_conf["rf"]["include"] = False

        conv_type = self.setter_getter["tau_calc"]["tau_calc_include"]
        conv = self.converters[conv_type]["get"]
        if conv(self.field("check_tau_calc_include")):
            if "lifetime" not in mod_conf:
                yaml_append_map(mod_conf, "lifetime", CommentedMap({}))
            #
            lifetime = mod_conf["lifetime"]
            lifetime["include"] = True
            #
            conv_type = self.setter_getter["tau_calc"]["tau_recalc"]
            conv = self.converters[conv_type]["get"]
            lifetime["recalc"] = conv(self.field("check_tau_recalc"))
            #
            if "calc_opts" not in lifetime:
                yaml_append_map(lifetime, "calc_opts", CommentedMap({}))
            #
            calc_opts = lifetime["calc_opts"]
            #
            n_E_MeV = len(self.E_MeV_list)
            if len(self.tau_special_data["total_beam_current"]) != n_E_MeV:
                raise AssertionError(
                    (
                        f"Since you specified {n_E_MeV} energy values, you must "
                        f"also specify the same number of total beam current "
                        f"values for lifetime computation."
                    )
                )
            seq = CommentedSeq(self.tau_special_data["total_beam_current"])
            seq.fa.set_flow_style()
            calc_opts["total_beam_current_mA"] = seq
            #
            conv_type = self.setter_getter["tau_calc"]["num_filled_bunches"]
            conv = self.converters[conv_type]["get"]
            calc_opts["num_filled_bunches"] = conv(
                self.field("spin_num_filled_bunches")
            )
            #
            conv_type = self.setter_getter["tau_calc"]["max_mom_aper"]
            conv = partial(self.converters[conv_type]["get"], "max_mom_aper")
            v = conv(self.field("combo_max_mom_aper"))
            if v not in (None, "auto") and not isinstance(v, float):
                text = 'Invalid input for "max_mom_aper"'
                info_text = f'"max_mom_aper" must be None, "auto", or a float'
                showInvalidPageInputDialog(text, info_text)
                raise AssertionError
            calc_opts["max_mom_aper_percent"] = v
            #
            conv_type = self.setter_getter["tau_calc"]["coupling_list"]
            conv = partial(self.converters[conv_type]["get"], "coupling_list")
            v = conv(self.field("edit_coupling_list"))

            def _coupling_list_error_msg():
                text = 'Invalid input for "Coupling List"'
                info_text = (
                    f"Must be a comma-separated list of strings that end with"
                    f'"pm" or "%", preceded with a number.'
                )
                showInvalidPageInputDialog(text, info_text)

            for val in v:
                if val.endswith("pm"):
                    try:
                        float(val[:-2])
                    except:
                        _coupling_list_error_msg()
                        raise AssertionError
                elif val.endswith("%"):
                    try:
                        float(val[:-1])
                    except:
                        _coupling_list_error_msg()
                        raise AssertionError
                else:
                    _coupling_list_error_msg()
                    raise AssertionError
            seq = CommentedSeq(v)
            seq.fa.set_flow_style()
            calc_opts["coupling"] = seq

            # Modify conf['lifetime']['calc_opts']['V_scan']
            if "V_scan" not in calc_opts:
                yaml_append_map(calc_opts, "V_scan", CommentedMap({}))
            #
            scan_opts = calc_opts["V_scan"]
            #
            conv_type = self.setter_getter["rf"]["bucket_height_min"]
            conv = self.converters[conv_type]["get"]
            scan_opts["min_bucket_height_percent"] = conv(
                self.field("edit_bucket_height_min")
            )
            #
            conv_type = self.setter_getter["rf"]["bucket_height_max"]
            conv = self.converters[conv_type]["get"]
            scan_opts["max_bucket_height_percent"] = conv(
                self.field("edit_bucket_height_max")
            )
            #
            conv_type = self.setter_getter["rf"]["v_scan_npts"]
            conv = self.converters[conv_type]["get"]
            scan_opts["nVolts"] = conv(self.field("spin_v_scan_npts"))
            #
            conv_type = self.setter_getter["rf"]["ntasks_rf_tau"]
            conv = self.converters[conv_type]["get"]
            scan_opts["ntasks"] = conv(self.field("spin_ntasks_rf_tau"))

            conv_type = self.setter_getter["tau_plot"]["tau_plot_include"]
            conv = self.converters[conv_type]["get"]
            if conv(self.field("check_tau_plot_include")):
                conv_type = self.setter_getter["tau_plot"]["tau_replot"]
                conv = self.converters[conv_type]["get"]
                lifetime["replot"] = conv(self.field("check_tau_replot"))
                #
                if "plot_opts" not in lifetime:
                    loss_plots_indexes = CommentedMap({})
                    for _k in ["E_MeV", "rf_V", "coupling"]:
                        seq = CommentedSeq([])
                        seq.fa.set_flow_style()
                        yaml_append_map(loss_plots_indexes, _k, seq)
                    yaml_append_map(
                        lifetime,
                        "plot_opts",
                        CommentedMap({"loss_plots_indexes": loss_plots_indexes}),
                    )
                #
                plot_opts = lifetime["plot_opts"]
                if "loss_plots_indexes" not in plot_opts:
                    loss_plots_indexes = CommentedMap({})
                    for _k in ["E_MeV", "rf_V", "coupling"]:
                        seq = CommentedSeq([])
                        seq.fa.set_flow_style()
                        yaml_append_map(loss_plots_indexes, _k, seq)
                    yaml_append_map(plot_opts, "loss_plots_indexes", loss_plots_indexes)
                #
                loss_plots_indexes = plot_opts["loss_plots_indexes"]
                #
                for _k, _v in self.tau_special_data["loss_plots_set"].items():
                    loss_plots_indexes[_k].clear()
                    loss_plots_indexes[_k].extend(_v)
            else:
                if "plot_opts" in lifetime:
                    del lifetime["plot_opts"]

        else:
            if "lifetime" in mod_conf:
                mod_conf["lifetime"]["include"] = False

        # _check_if_yaml_writable(mod_conf)

        self.wizardObj.update_conf_on_all_pages(mod_conf)

        return mod_conf


class TotalBeamCurrentEditor(QtWidgets.QDialog):
    """"""

    def __init__(self, data, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)
        ui_file = os.path.join(
            os.path.dirname(__file__), "total_beam_current_editor.ui"
        )
        uic.loadUi(ui_file, self)

        self.data = data

        QTableWidgetItem = QtWidgets.QTableWidgetItem

        t = self.tableWidget

        t.setRowCount(len(data))
        t.setColumnCount(2)
        self.iGeV, self.iCurrent = 0, 1
        t.setHorizontalHeaderItem(self.iGeV, QTableWidgetItem("E [GeV]"))
        t.setHorizontalHeaderItem(
            self.iCurrent, QTableWidgetItem("Total Beam Current [mA]")
        )
        for iRow, (E_GeV, I_total_mA) in enumerate(data):
            item = QTableWidgetItem(f"{E_GeV:.2g}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            t.setItem(iRow, self.iGeV, item)
            if I_total_mA is None:
                text = ""
            else:
                text = f"{I_total_mA:.0f}"
            t.setItem(iRow, self.iCurrent, QTableWidgetItem(text))

        self.tableWidget.resizeColumnsToContents()

        self.accepted.connect(self.apply_changes)

    def apply_changes(self):
        """"""

        t = self.tableWidget

        for iRow in range(t.rowCount()):
            self.data[iRow][self.iCurrent] = float(t.item(iRow, self.iCurrent).text())


class LossPlotsSetEditor(QtWidgets.QDialog):
    """"""

    def __init__(self, data, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)
        ui_file = os.path.join(
            os.path.dirname(__file__), "total_beam_current_editor.ui"
        )
        uic.loadUi(ui_file, self)

        self.setWindowTitle("Index Set for Loss Plots")
        self.label.setText("Check sets for which you want loss plots:")

        self.data = data

        loss_plots_set = data["loss_plots_set"]
        E_GeV_list = data["E_GeV"]
        rf_MV_list = data["rf_MV"]
        coupling_list = data["coupling"]

        QTableWidgetItem = QtWidgets.QTableWidgetItem

        t = self.tableWidget

        t.setColumnCount(4)
        self.iSel, self.iGeV, self.iVolt, self.iCoup = 0, 1, 2, 3
        t.setHorizontalHeaderItem(self.iSel, QTableWidgetItem("Plot"))
        t.setHorizontalHeaderItem(self.iGeV, QTableWidgetItem("E [GeV]"))
        t.setHorizontalHeaderItem(self.iVolt, QTableWidgetItem("V_RF [MV]"))
        t.setHorizontalHeaderItem(self.iCoup, QTableWidgetItem("Couplng"))
        #
        check_all_item = QTableWidgetItem("All")
        check_all_item.setFlags(check_all_item.flags() | Qt.ItemIsUserCheckable)
        #
        try:
            iter(rf_MV_list[0])
        except TypeError:
            self.is_rf_MV_list_LoL = False
        else:
            self.is_rf_MV_list_LoL = True
        #
        if self.is_rf_MV_list_LoL:
            t.setRowCount(sum([len(_v) for _v in rf_MV_list]) * len(coupling_list) + 1)
        else:
            t.setRowCount(len(E_GeV_list) * len(rf_MV_list) * len(coupling_list) + 1)
        iRow = 0
        t.setItem(iRow, self.iSel, check_all_item)
        iRow += 1
        all_checked = True
        for iGeV, E_GeV in enumerate(E_GeV_list):
            if self.is_rf_MV_list_LoL:
                rf_MV_1d = rf_MV_list[iGeV]
            else:
                rf_MV_1d = rf_MV_list
            #
            for iVolt, MV in enumerate(rf_MV_1d):
                for iCoup, coupling in enumerate(coupling_list):
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    if (iGeV, iVolt, iCoup) in loss_plots_set:
                        item.setCheckState(Qt.Checked)
                    else:
                        item.setCheckState(Qt.Unchecked)
                        all_checked = False
                    t.setItem(iRow, self.iSel, item)

                    item = QTableWidgetItem(f"{E_GeV:.2g}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    t.setItem(iRow, self.iGeV, item)

                    item = QTableWidgetItem(f"{MV:.3g}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    t.setItem(iRow, self.iVolt, item)

                    item = QTableWidgetItem(f"{coupling}")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    t.setItem(iRow, self.iCoup, item)

                    iRow += 1

        if all_checked:
            check_all_item.setCheckState(Qt.Checked)
        else:
            check_all_item.setCheckState(Qt.Unchecked)

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.accepted.connect(self.apply_changes)

        self.tableWidget.itemChanged.connect(partial(self.check_all, check_all_item))

    def check_all(self, check_all_item, changed_item):
        """"""

        if changed_item is check_all_item:
            new_state = check_all_item.checkState()
            for iRow in range(1, self.tableWidget.rowCount()):
                self.tableWidget.item(iRow, self.iSel).setCheckState(new_state)

    def apply_changes(self):
        """"""

        data = self.data

        E_GeV_list = data["E_GeV"]
        rf_MV_list = data["rf_MV"]
        coupling_list = data["coupling"]

        t = self.tableWidget

        data["loss_plots_set"].clear()

        iRow = 1
        for iGeV, E_GeV in enumerate(E_GeV_list):
            if self.is_rf_MV_list_LoL:
                rf_MV_1d = rf_MV_list[iGeV]
            else:
                rf_MV_1d = rf_MV_list
            #
            for iVolt, MV in enumerate(rf_MV_1d):
                for iCoup, coupling in enumerate(coupling_list):

                    if t.item(iRow, self.iSel).checkState() == Qt.Checked:
                        data["loss_plots_set"].append((iGeV, iVolt, iCoup))

                    iRow += 1
