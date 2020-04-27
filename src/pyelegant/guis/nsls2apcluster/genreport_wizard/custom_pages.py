import sys
import os
import traceback
from functools import partial
from pathlib import Path
import datetime
import time
import shutil
import tempfile
import hashlib
import shlex
import pty
from select import select
import errno
from subprocess import Popen, PIPE
from copy import deepcopy

from qtpy import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt

import numpy as np
from ruamel import yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

import pyelegant as pe
from pyelegant.scripts import genreport

def _check_if_yaml_writable(yaml_object):
    """"""

    yml = yaml.YAML()
    yml.preserve_quotes = True
    yml.width = 70
    yml.boolean_representation = ['False', 'True']
    yml.dump(yaml_object, sys.stdout)

def duplicate_yaml_conf(orig_conf):
    """"""

    # Here, you cannot use "deepcopy()" and "pe.util.deepcopy_dict(orig_conf)".
    # Instead, you must dump to a temp file and load it back in order
    # to retain the anchors/references.

    _temp_yaml_file = tempfile.NamedTemporaryFile(
        suffix='.yaml', dir=None, delete=True)

    yml = yaml.YAML()
    yml.preserve_quotes = True
    yml.width = 70
    yml.boolean_representation = ['False', 'True']

    with open(_temp_yaml_file.name, 'w') as f:
        yml.dump(orig_conf, f)

    dup_conf = yml.load(Path(_temp_yaml_file.name).read_text())

    _temp_yaml_file.close()

    return dup_conf

def update_aliased_scalar(data, parent_obj, key_or_index, val, **kwargs):
    """
    Based on an answer on "https://stackoverflow.com/questions/55716068/how-to-change-an-anchored-scalar-in-a-sequence-without-destroying-the-anchor-in"
    """

    def recurse(d, ref, nv):
        if isinstance(d, dict):
            for i, k in [(idx, key) for idx, key in enumerate(d.keys()) if key is ref]:
                d.insert(i, nv, d.pop(k))
            for k, v in d.non_merged_items():
                if v is ref:
                    d[k] = nv
                else:
                    recurse(v, ref, nv)
        elif isinstance(d, list):
            for idx, item in enumerate(d):
                if item is ref:
                    d[idx] = nv
                else:
                    recurse(item, ref, nv)

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

    if hasattr(obj, 'anchor'):
        recurse(data, obj, type(obj)(val, anchor=obj.anchor.value, **kwargs))
    else:
        recurse(data, obj, type(obj)(val, **kwargs))

    if hasattr(obj, 'fa') and obj.fa.flow_style():
        parent_obj[key].fa.set_flow_style()

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

        self.beginInsertRows(QtCore.QModelIndex(), len(self._data),
                             len(self._data)+len(lines)-1)
        self._data.extend(lines)
        self.endInsertRows()

    def flush(self):
        """"""

        self._view.scrollToBottom()

        QtWidgets.QApplication.processEvents()

        ## Trigger refresh
        #self.layoutChanged.emit()

    def write_from_scratch(self, whole_str):
        """"""

        self.beginResetModel()
        self._data.clear()
        self.endResetModel()

        lines = whole_str.splitlines()

        self.beginInsertRows(QtCore.QModelIndex(), 0, len(lines)-1)
        self._data.extend(lines)
        self.endInsertRows()

        self.flush()

def realtime_updated_Popen(
    cmd, view_stdout=None, view_stderr=None, robust_tail=True):
    """
    If "robust_tail" is False, occasionally, you may see the tail part of the
    output texts missing. Setting "robust_tail" True resolves this issue.

    Based on an answer posted at

    https://stackoverflow.com/questions/31926470/run-command-and-get-its-stdout-stderr-separately-in-near-real-time-like-in-a-te
    """

    stdout = ''
    stderr = ''
    masters, slaves = zip(pty.openpty(), pty.openpty())
    if not robust_tail:
        p = Popen(shlex.split(cmd), stdin=slaves[0], stdout=slaves[0],
                  stderr=slaves[1])
    else:
        stdout_tmp_file = tempfile.NamedTemporaryFile(
            suffix='.log', dir=None, delete=True)
        stderr_tmp_file = tempfile.NamedTemporaryFile(
            suffix='.log', dir=None, delete=True)
        stdout_filename = stdout_tmp_file.name
        stderr_filename = stderr_tmp_file.name
        new_cmd = (
            f'(({cmd}) | tee {stdout_filename}) 3>&1 1>&2 2>&3 | '
            f'tee {stderr_filename}')
        p = Popen(new_cmd, stdin=slaves[0], stdout=slaves[0],
                  stderr=slaves[1], shell=True)
    for fd in slaves: os.close(fd)

    #readable = { masters[0]: sys.stdout, masters[1]: sys.stderr }
    if not robust_tail:
        readable = {
            masters[0]: (sys.stdout if view_stdout is None
                         else view_stdout.model()),
            masters[1]: (sys.stderr if view_stderr is None
                         else view_stderr.model()) }
    else:
        readable = {
            masters[0]: (sys.stderr if view_stderr is None
                         else view_stderr.model()),
            masters[1]: (sys.stdout if view_stdout is None
                         else view_stdout.model()) }

    try:
        if False:
            print(' ######### REAL-TIME ######### ')

        while readable:
            #t0 = time.time()
            for fd in select(readable, [], [])[0]:
                try: data = os.read(fd, 1024)
                except OSError as e:
                    if e.errno != errno.EIO: raise
                    del readable[fd]
                    data = b''
                finally:
                    if not data: del readable[fd]
                    else:
                        if fd == masters[0]: stdout += data.decode('utf-8')
                        else: stderr += data.decode('utf-8')
                        readable[fd].write(data.decode('utf-8'))
                        readable[fd].flush()
                        #print(time.time()-t0)
    except:
        pass

    finally:
        p.wait()
        for fd in masters: os.close(fd)

        if not robust_tail:
            if view_stdout:
                view_stdout.model().write_from_scratch(stdout)
            if view_stderr:
                view_stderr.model().write_from_scratch(stderr)
        else:
            if view_stdout:
                view_stdout.model().write_from_scratch(
                    Path(stdout_filename).read_text())
            if view_stderr:
                view_stderr.model().write_from_scratch(
                    Path(stderr_filename).read_text())

            stdout_tmp_file.close()
            stderr_tmp_file.close()

        if False:
            print('')
            print(' ########## RESULTS ########## ')
            print('STDOUT:')
            print(stdout)
            print('STDERR:')
            print(stderr)

def convert_multiline_yaml_str_to_oneline_str(multiline_yaml_str):
    """
    Allow multi-line definition for a long LTE filepath in YAML
    """

    return ''.join([_s.strip() for _s in multiline_yaml_str.splitlines()])

def showInvalidPageInputDialog(text, informative_text):
    """"""

    QMessageBox = QtWidgets.QMessageBox

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.setWindowTitle('Invalid Input')
    msg.setStyleSheet("QIcon{max-width: 100px;}")
    msg.setStyleSheet("QLabel{min-width: 300px;}")
    #msg.setStyleSheet("QLabel{min-width:500 px; font-size: 24px;} QPushButton{ width:250px; font-size: 18px; }")
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
        extension_str = ' '.join(extension_list)
        filter_str_list.append(f'{file_type_description} ({extension_str})')

    filter_str = ';;'.join(filter_str_list)

    return filter_str

def getFileDialogInitDir(lineEdit_text):
    """"""

    cur_filepath = lineEdit_text.strip()
    cur_file = Path(cur_filepath)
    if cur_file.exists():
        directory = str(cur_file.parent)
    else:
        directory = ''

    return directory

def openFileNameDialog(widget, caption='', directory='', filter_str=''):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    #directory = ''
    #filter_str = 'All Files (*);;Python Files (*.py)'

    options = QFileDialog.Options()
    #options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(
        widget, caption, directory, filter_str, options=options)
    if fileName:
        #print(fileName)
        return fileName

def openFileNamesDialog(widget, caption='', directory='', filter_str=''):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    options = QFileDialog.Options()
    #options |= QFileDialog.DontUseNativeDialog
    files, _ = QFileDialog.getOpenFileNames(
        widget, caption, directory, filter_str, options=options)
    if files:
        #print(files)
        return files

def saveFileDialog(widget, caption='', directory='', filter_str=''):
    """"""

    QFileDialog = QtWidgets.QFileDialog

    options = QFileDialog.Options()
    #options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getSaveFileName(
        widget, caption, directory, filter_str, options=options)
    if fileName:
        #print(fileName)
        return fileName

def openDirNameDialog(widget, caption='', directory=''):
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

def generate_report(
    config_filename, view_stdout=None, view_stderr=None):
    """"""

    cmd = f'pyele_report {config_filename}'

    model_stdout = view_stdout.model()
    model_stderr = view_stderr.model()

    model_stdout.beginResetModel()
    model_stdout._data.clear()
    model_stdout.endResetModel()

    model_stderr.beginResetModel()
    model_stderr._data.clear()
    model_stderr.beginResetModel()

    QtWidgets.QApplication.processEvents()

    #model_stdout.layoutChanged.emit()
    #model_stderr.layoutChanged.emit()

    #view_stdout.update()
    #view_stderr.update()

    realtime_updated_Popen(
        cmd, view_stdout=view_stdout, view_stderr=view_stderr)

def open_pdf_report(pdf_filepath):
    """"""

    cmd = f'evince {pdf_filepath}'
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
    #out, err = p.communicate()

    #print(out)
    #if err:
        #print('\n### stderr ###')
        #print(err)

class PageGenReport(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()
        config_filepath = self.wizardObj.config_filepath
        config_filename = Path(config_filepath).name
        pdf_filepath = self.wizardObj.pdf_filepath

        mod_conf = self.modify_conf(self.wizardObj.conf)

        yml = yaml.YAML()
        yml.preserve_quotes = True
        yml.width = 70
        yml.boolean_representation = ['False', 'True']
        with open(config_filepath, 'w') as f:
            yml.dump(mod_conf, f)

        # Establish connections
        if False:
            tb_out = self.findChildren(QtWidgets.QTextBrowser,
                                       QtCore.QRegExp('textBrowser_stdout_.+'))[0]
            tb_err = self.findChildren(QtWidgets.QTextBrowser,
                                       QtCore.QRegExp('textBrowser_stderr_.+'))[0]
        elif False:
            tb_out = self.findChildren(QtWidgets.QPlainTextEdit,
                                       QtCore.QRegExp('plainTextEdit_stdout_.+'))[0]
            tb_err = self.findChildren(QtWidgets.QPlainTextEdit,
                                       QtCore.QRegExp('plainTextEdit_stderr_.+'))[0]
        else:
            view_out = self.findChildren(QtWidgets.QListView,
                                         QtCore.QRegExp('listView_stdout_.+'))[0]
            view_err = self.findChildren(QtWidgets.QListView,
                                         QtCore.QRegExp('listView_stderr_.+'))[0]
            view_out.setModel(ListModelStdLogger([], view_out))
            view_err.setModel(ListModelStdLogger([], view_err))
        b = self.findChildren(QtWidgets.QPushButton,
                              QtCore.QRegExp('pushButton_gen_.+'))[0]
        b.clicked.connect(partial(
            generate_report, config_filename, view_out, view_err))
        b = self.findChildren(QtWidgets.QPushButton,
                              QtCore.QRegExp('pushButton_open_pdf_.+'))[0]
        b.clicked.connect(partial(open_pdf_report, pdf_filepath))

    def modify_conf(self, orig_conf):
        """"""
        return deepcopy(orig_conf)

class PageGenReportTest1(PageGenReport):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def modify_conf(self, orig_conf):
        """"""

        mod_conf = duplicate_yaml_conf(orig_conf)

        f = partial(update_aliased_scalar, mod_conf)

        f(mod_conf['lattice_props'], 'recalc', True)

        if 'pdf_table_order' in mod_conf['lattice_props']:
            del mod_conf['lattice_props']['pdf_table_order']
        if 'xlsx_table_order' in mod_conf['lattice_props']:
            del mod_conf['lattice_props']['xlsx_table_order']

        d = mod_conf['lattice_props']
        if 'opt_props' in d:
            if 'phase_adv' in d['opt_props']:
                if 'MDISP 0&1' in d['opt_props']['phase_adv']:
                    Ls = [
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP 0&1']),
                    ]
                    for L in Ls: L.fa.set_flow_style()
                    f(d, 'append_opt_props_to_pdf_table', Ls)

                    Ls = [
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP 0&1', 'x']),
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP 0&1', 'y']),
                    ]
                    for L in Ls: L.fa.set_flow_style()
                    f(d, 'append_opt_props_to_xlsx_table', Ls)
                else:
                    Ls = [
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across LS']),
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across SS']),
                    ]
                    for L in Ls: L.fa.set_flow_style()
                    f(d, 'append_opt_props_to_pdf_table', Ls)

                    Ls = [
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across LS', 'x']),
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across LS', 'y']),
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across SS', 'x']),
                        CommentedSeq(['opt_props', 'phase_adv', 'MDISP across SS', 'y']),
                    ]
                    for L in Ls: L.fa.set_flow_style()
                    f(d, 'append_opt_props_to_xlsx_table', Ls)

        for k, v in mod_conf['nonlin']['include'].items():
            f(mod_conf['nonlin']['include'], k, False)

        for k in ['rf_dep_calc_opts', 'lifetime_calc_opts']:
            if k in mod_conf:
                del mod_conf[k]

        return mod_conf


class PageNewSetup(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        # Register fields

        self.registerFieldOnFirstShow(
            #'edit_rootname*',
            'edit_rootname', # TO-BE-DELETED
            self.findChild(QtWidgets.QLineEdit, 'lineEdit_rootname'))
        # TO-BE-DELETED
        self.setField('edit_rootname', 'CBA_0001')

        self.registerFieldOnFirstShow(
            #'edit_new_config_folder*',
            'edit_new_config_folder', # TO-BE-DELETED
            self.findChild(QtWidgets.QLineEdit, 'lineEdit_new_config_folder'))
        # TO-BE-DELETED
        self.setField('edit_new_config_folder',
            '/GPFS/APC/yhidaka/git_repos/pyelegant/src/pyelegant/guis/nsls2apcluster/genreport_wizard')

        for obj_name in [
            'label_full_new_config', 'label_full_report_folder',
            'label_full_pdf_path', 'label_full_xlsx_path',]:
            self.registerFieldOnFirstShow(
                obj_name,
                self.findChild(QtWidgets.QLabel, obj_name), property='text')

        # Set fields

        # Establish connections

        self.edit_obj = self.findChild(
            QtWidgets.QLineEdit, 'lineEdit_new_config_folder')
        self.edit_obj.textChanged.connect(self.update_paths_on_folder)

        b = self.findChild(QtWidgets.QPushButton,
                           'pushButton_browse_new_config_folder')
        b.clicked.connect(self.browse_new_config_folder)

        e = self.findChild(QtWidgets.QLineEdit, 'lineEdit_rootname')
        e.textChanged.connect(self.update_paths_on_rootname)

    def update_paths_on_folder(self, new_config_folderpath):
        """"""

        new_config_folder = Path(new_config_folderpath)
        if not new_config_folder.exists():
            return

        rootname = self.field('edit_rootname').strip()
        if rootname == '':
            return

        self.update_paths(new_config_folder, rootname)

    def update_paths_on_rootname(self, new_rootname):
        """"""

        if new_rootname == '':
            return

        new_config_folderpath = self.field('edit_new_config_folder')
        new_config_folder = Path(new_config_folderpath)
        if not new_config_folder.exists():
            return

        self.update_paths(new_config_folder, rootname)

    def update_paths(self, new_config_folder, rootname):
        """"""

        new_config_filename = f'{rootname}.yaml'
        new_report_foldername = f'report_{rootname}'
        pdf_filename = f'{rootname}_report.pdf'
        xlsx_filename = f'{rootname}_report.xlsx'

        report_folderpath = new_config_folder.joinpath(new_report_foldername)

        self.wizardObj.config_filepath = str(new_config_folder.joinpath(
            new_config_filename))
        self.wizardObj.pdf_filepath = str(Path(report_folderpath).joinpath(
            pdf_filename))

        self.setField(
            'label_full_new_config', '=> {}'.format(
                self.wizardObj.config_filepath))
        self.setField(
            'label_full_report_folder', f'=> {report_folderpath}')
        self.setField(
            'label_full_pdf_path', '=> {}'.format(self.wizardObj.pdf_filepath))
        self.setField(
            'label_full_xlsx_path', '=> {}'.format(
                Path(report_folderpath).joinpath(xlsx_filename)))

    def browse_new_config_folder(self):
        """"""

        caption = 'Select a folder where new config/data/report files will be generated'
        directory = getFileDialogInitDir(self.edit_obj.text())
        folderpath = openDirNameDialog(
            self, caption=caption, directory=directory)

        if folderpath:
            self.edit_obj.setText(folderpath)

    def validatePage(self):
        """"""

        new_config_folderpath = self.edit_obj.text()
        new_config_folder = Path(new_config_folderpath)
        if not new_config_folder.exists():
            return False

        rootname = self.field('edit_rootname')

        self.update_paths(new_config_folder, rootname)

        return True

class PageLoadSeedConfig(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        self.edit_obj = self.findChild(
            QtWidgets.QLineEdit, 'lineEdit_seed_config_filepath')

        # TO-BE-DELETED
        self.edit_obj.setText(
            '/GPFS/APC/yhidaka/git_repos/nsls2cb/lat_reports/20200422_VS_DCBA25pm.yaml')

        # Establish connections

        b = self.findChild(QtWidgets.QPushButton, 'pushButton_browse')
        b.clicked.connect(self.browse_yaml_config_file)

    def browse_yaml_config_file(self):
        """"""

        caption = 'Select YAML config file to load'
        extension_dict = {'YAML Files': ['*.yaml', '*.yml'],
                          'All Files': ['*']}
        filter_str = getFileDialogFilterStr(extension_dict)
        directory = getFileDialogInitDir(self.edit_obj.text())
        filepath = openFileNameDialog(
            self, caption=caption, directory=directory, filter_str=filter_str)

        if filepath:
            self.edit_obj.setText(filepath)

    def validatePage(self):
        """"""

        seed_config_filepath = self.edit_obj.text().strip()

        if seed_config_filepath == '':
            genreport.Report_NSLS2U_Default(
                self.wizardObj.config_filepath, example_args=['full', None])

            seed_config_filepath = self.wizardObj.config_filepath

        if not Path(seed_config_filepath).exists():
            text = 'Invalid file path'
            info_text = (
                f'Specified config file "{seed_config_filepath}" '
                f'does not exist!')
            showInvalidPageInputDialog(text, info_text)

            return False

        try:
            yml = yaml.YAML()
            yml.preserve_quotes = True
            user_conf = yml.load(Path(seed_config_filepath).read_text())
            self.wizardObj.conf = user_conf

        except:
            text = 'Invalid YAML file'
            info_text = (
                f'Specified config file "{seed_config_filepath}" does not '
                f'appear to be a valid YAML file!')
            showInvalidPageInputDialog(text, info_text)

            return False

        return True

class PageLTE(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        user_conf = self.wizardObj.conf

        # Register fields

        for k in [
            'report_author', 'orig_LTE_path*', 'LTE_authors*', 'parent_LTE_hash',
            'E_GeV*', 'use_beamline_cell*', 'use_beamline_ring*',]:
            k_wo_star = (k[:-1] if k.endswith('*') else k)
            self.registerFieldOnFirstShow(
                f'edit_{k}',
                self.findChild(QtWidgets.QLineEdit, f'lineEdit_{k_wo_star}'))
        self.registerFieldOnFirstShow(
            'date_LTE_received*',
            self.findChild(QtWidgets.QDateEdit, 'dateEdit_LTE_received'))
        self.registerFieldOnFirstShow(
            'check_pyele_stdout',
            self.findChild(QtWidgets.QCheckBox, 'checkBox_pyelegant_stdout'))

        # Set fields

        self.setField('edit_report_author',
                      str(user_conf.get('report_author', '').strip()))

        self.setField(
            'edit_orig_LTE_path',
            convert_multiline_yaml_str_to_oneline_str(
                user_conf.get('orig_LTE_filepath', '')))

        self.setField('edit_LTE_authors',
                      str(user_conf.get('lattice_author', '').strip()))

        self.setField('edit_parent_LTE_hash',
                      str(user_conf.get('parent_LTE_hash', '').strip()))

        date = user_conf.get('lattice_received_date', None)
        if date is None:
            date = QtCore.QDateTime().currentDateTime()
        else:
            month, day, year = [int(s) for s in date.split('/')]
            date = QtCore.QDateTime(datetime.date(year, month, day))
        self.setField('date_LTE_received', date)

        E_GeV = user_conf.get('E_MeV', 3e3) / 1e3
        self.setField('edit_E_GeV', f'{E_GeV:.3g}')

        self.setField('edit_use_beamline_cell',
                      str(user_conf.get('use_beamline_cell', '').strip()))

        self.setField('edit_use_beamline_ring',
                      str(user_conf.get('use_beamline_ring', '').strip()))

        self.setField('check_pyele_stdout',
                      user_conf.get('enable_pyelegant_stdout', False))

        # Establish connections

        b = self.findChild(QtWidgets.QPushButton, 'pushButton_browse_LTE')
        b.clicked.connect(self.browse_LTE_file)

    def browse_LTE_file(self):
        """"""

        caption = 'Select LTE file to be characterized'
        filter_str = 'LTE Files (*.lte);;All Files (*)'
        extension_dict = {'LTE Files': ['*.lte'],
                          'All Files': ['*']}
        filter_str = getFileDialogFilterStr(extension_dict)
        directory = getFileDialogInitDir(self.field('edit_orig_LTE_path'))
        filepath = openFileNameDialog(
            self, caption=caption, directory=directory, filter_str=filter_str)

        if filepath:
            self.setField('edit_orig_LTE_path', filepath)

    def validatePage(self):
        """"""

        orig_LTE_filepath = self.field('edit_orig_LTE_path').strip()

        if not orig_LTE_filepath.endswith('.lte'):
            text = 'Invalid file extension'
            info_text = 'LTE file name must end with ".lte".'
            showInvalidPageInputDialog(text, info_text)
            return False

        orig_LTE_Path = Path(orig_LTE_filepath)
        if not orig_LTE_Path.exists():
            text = 'Invalid file path'
            info_text = f'Specified LTE file "{orig_LTE_Path}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            return False

        use_beamline_cell = self.field('edit_use_beamline_cell').strip()
        use_beamline_ring = self.field('edit_use_beamline_ring').strip()

        try:
            LTE = pe.ltemanager.Lattice(LTE_filepath=orig_LTE_filepath)
        except:
            text = 'Invalid LTE file'
            info_text = f'Specified LTE file "{orig_LTE_Path}" failed to be parsed!'
            showInvalidPageInputDialog(text, info_text)
            traceback.print_exc()
            return False

        try:
            LTE = pe.ltemanager.Lattice(LTE_filepath=orig_LTE_filepath,
                                        used_beamline_name=use_beamline_ring)
        except:
            text = 'Invalid beamline name for "Ring Beamline Name"'
            info_text = f'Specified name "{use_beamline_ring}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            #traceback.print_exc()
            return False

        try:
            LTE = pe.ltemanager.Lattice(LTE_filepath=orig_LTE_filepath,
                                        used_beamline_name=use_beamline_cell)
        except:
            text = 'Invalid beamline name for "Super-Period Beamline Name"'
            info_text = f'Specified name "{use_beamline_cell}" does not exist!'
            showInvalidPageInputDialog(text, info_text)
            #traceback.print_exc()
            return False

        self.wizardObj.LTE = LTE

        all_beamline_defs = LTE.get_all_beamline_defs(LTE.cleaned_LTE_text)
        all_beamline_names = [v[0] for v in all_beamline_defs]

        for spec_name, label in [(use_beamline_cell, 'Super-Period'),
                                 (use_beamline_ring, 'Ring')]:
            if spec_name not in all_beamline_names:
                text = f'Invalid name for "{label}" beamline: "{spec_name}"'
                info_text = 'Available beamline names: {}'.format(
                    ', '.join(all_beamline_names))
                showInvalidPageInputDialog(text, info_text)
                return False

        conf = self.wizardObj.conf
        f = partial(update_aliased_scalar, conf)
        #
        f(conf, 'report_author', self.field('edit_report_author').strip())
        #
        f(conf, 'orig_LTE_filepath', orig_LTE_filepath)
        #
        input_LTE_filepath = \
            self.wizardObj.pdf_filepath[:-len('_report.pdf')] + '.lte'
        input_LTE_Path = Path(input_LTE_filepath)
        input_LTE_Path.parent.mkdir(parents=True, exist_ok=True)
        if not input_LTE_Path.exists():
            shutil.copy(orig_LTE_filepath, input_LTE_filepath)
            f(conf['input_LTE'], 'regenerate_zeroSexts', True)
        else:
            sha = hashlib.sha1()
            sha.update(input_LTE_Path.read_text().encode('utf-8'))
            existing_SHA1 = sha.hexdigest()

            sha = hashlib.sha1()
            sha.update(orig_LTE_Path.read_text().encode('utf-8'))
            orig_SHA1 = sha.hexdigest()

            if orig_SHA1 != existing_SHA1:
                shutil.copy(orig_LTE_filepath, input_LTE_filepath)
                f(conf['input_LTE'], 'regenerate_zeroSexts', True)
            else:
                f(conf['input_LTE'], 'regenerate_zeroSexts', False)
        f(conf['input_LTE'], 'filepath', input_LTE_filepath)
        #
        f(conf['input_LTE'], 'parent_LTE_hash',
          self.field('edit_parent_LTE_hash').strip())
        #
        f(conf, 'lattice_author', self.field('edit_LTE_authors').strip())
        #
        f(conf, 'lattice_received_date',
          self.field('date_LTE_received').toString('MM/dd/yyyy'))
        #
        #f(conf, 'E_MeV', float(self.field('edit_E_GeV')) * 1e3)
        f(conf, 'E_MeV', float(self.field('edit_E_GeV')) * 1e3,
          prec=conf['E_MeV']._prec, width=conf['E_MeV']._width)
        # ^ For some reason, "prec=-1" and "width=1" (not the default values
        #   None for these properties) are required to be able to still
        #   write "conf" into a YAML file.
        if False:
            _check_if_yaml_writable(conf)
        #
        f(conf, 'use_beamline_cell', use_beamline_cell)
        f(conf, 'use_beamline_ring', use_beamline_ring)
        #
        f(conf, 'enable_pyelegant_stdout', self.field('check_pyele_stdout'))


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
            if 'L' in prop:
                L = prop['L']
            else:
                L = 0.0
            cur_spos += L
            spos_list.append(cur_spos)

            if elem_name in occur_dict:
                occur_dict[elem_name] += 1
            else:
                occur_dict[elem_name] = 1
            occur_list.append(occur_dict[elem_name])
        data = list(zip(
            spos_list, flat_used_elem_names, elem_type_list, occur_list))
        self.wizardObj.model_elem_list = TableModelElemList(data)

        return True

class TableModelElemList(QtCore.QAbstractTableModel):
    """"""

    def __init__(self, data):
        """Constructor"""

        super().__init__()

        self._data = data

        self._headers = ['s [m]', 'Name', 'Type', 'ElemOccurrence']
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
            return 'not 2 M_LS'

        ds_thresh = 1e-6

        if np.abs(self._spos[matched_inds[0]] - 0.0) > ds_thresh:
            return 'wrong M_LS#1 spos'

        s_max = np.max(self._spos)

        if np.abs(self._spos[matched_inds[1]] - s_max) > ds_thresh:
            return 'wrong M_LS#2 spos'

        matched_inds = np.where(self._flat_elem_names == M_SS_name)[0]

        if np.abs(self._spos[matched_inds[0]] - s_max / 2) > ds_thresh:
            return 'wrong M_SS#1 spos'

class PageStraightCenters(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.findChild(QtWidgets.QTableView, 'tableView_elem_list')
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_LS_1st_center_name')
        #self.registerFieldOnFirstShow('edit_LS_center_name*', edit)
        # TO-BE-DELETED
        self.registerFieldOnFirstShow('edit_LS_center_name', edit)
        # TO-BE-DELETED
        self.setField('edit_LS_center_name', 'M_LS')

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_SS_1st_center_name')
        #self.registerFieldOnFirstShow('edit_SS_center_name*', edit)
        # TO-BE-DELETED
        self.registerFieldOnFirstShow('edit_SS_center_name', edit)
        self.setField('edit_SS_center_name', 'M_SS')

        # Set fields

        # Establish connections

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_LS_1st_center_name')
        edit.textChanged.connect(self.synchronize_LS_elem_names)

        b = self.findChild(QtWidgets.QPushButton, 'pushButton_reload_LTE')
        b.clicked.connect(self.update_LTE_table)

    def synchronize_LS_elem_names(self, new_text):
        """"""

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_LS_2nd_center_name')
        edit.setText(self.field('edit_LS_center_name'))

    def update_LTE_table(self):
        """"""

        self.wizardObj.page_links['LTE'].validatePage()
        tableView = self.findChild(QtWidgets.QTableView, 'tableView_elem_list')
        tableView.setModel(self.wizardObj.model_elem_list)

    def validatePage(self):
        """"""

        M_LS_name = self.field('edit_LS_center_name').strip()
        M_SS_name = self.field('edit_SS_center_name').strip()

        for elem_name in [M_LS_name, M_SS_name]:
            if elem_name not in self.wizardObj.LTE.flat_used_elem_names:
                text = f'Invalid element name "{elem_name}"'
                info_text = (
                    f'Element "{elem_name}" does not exist in the specfied LTE '
                    f'file.')
                showInvalidPageInputDialog(text, info_text)
                return False

        flag = self.wizardObj.model_elem_list.validateStraightCenterElements(
            M_LS_name, M_SS_name)
        if flag is None:
            pass
        elif flag == 'not 2 M_LS':
            text = f'Invalid number of elements named "{M_LS_name}"'
            info_text = (
                f'There must be only 2 elements named "{M_LS_name}" in the '
                f'one super-period beamline.')
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == 'wrong M_LS#1 spos':
            text = f'Invalid s-pos of 1st LS center'
            info_text = (
                f'First LS center element "{M_LS_name}" must have s-pos = 0.')
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == 'wrong M_LS#2 spos':
            text = f'Invalid s-pos of 2nd LS center'
            info_text = (
                f'2nd LS center element "{M_LS_name}" must be at the end of the'
                f'one super-period beamline.')
            showInvalidPageInputDialog(text, info_text)
            return False
        elif flag == 'wrong M_SS#1 spos':
            text = f'Invalid s-pos of 1st SS center'
            info_text = (
                f'1st SS center element "{M_SS_name}" must be at the middle of '
                f'the one super-period beamline.')
            showInvalidPageInputDialog(text, info_text)
            return False
        else:
            raise ValueError()

        conf = self.wizardObj.conf
        f = partial(update_aliased_scalar, conf)
        #
        d = conf['lattice_props']['req_props']
        #
        f(d['beta']['LS'], 'name', M_LS_name)
        f(d['beta']['LS'], 'occur', 0)
        #
        f(d['beta']['SS'], 'name', M_SS_name)
        f(d['beta']['SS'], 'occur', 0)
        #
        d['floor_comparison']['ref_flr_filepath'] = \
            '/GPFS/APC/yhidaka/common/nsls2.flr'
        #
        f(d['floor_comparison']['LS']['ref_elem'], 'name', 'MID')
        f(d['floor_comparison']['LS']['ref_elem'], 'occur', 1)
        #
        f(d['floor_comparison']['LS']['cur_elem'], 'name', M_LS_name)
        f(d['floor_comparison']['LS']['cur_elem'], 'occur', 1)
        #
        f(d['floor_comparison']['SS']['ref_elem'], 'name', 'MID')
        f(d['floor_comparison']['SS']['ref_elem'], 'occur', 0)
        #
        f(d['floor_comparison']['SS']['cur_elem'], 'name', M_SS_name)
        f(d['floor_comparison']['SS']['cur_elem'], 'occur', 0)

        return True

class PagePhaseAdv(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.findChild(QtWidgets.QTableView,
                                   'tableView_elem_list_phase_adv')
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        obj = self.findChild(QtWidgets.QLineEdit,
                             'lineEdit_disp_bump_marker_name')
        self.registerFieldOnFirstShow('edit_disp_bump_marker_name', obj)
        # TO-BE-DELETED
        self.setField('edit_disp_bump_marker_name', 'MDISP')

        obj = self.findChild(QtWidgets.QComboBox, 'comboBox_n_disp_bumps')
        self.registerFieldOnFirstShow('combo_n_disp_bumps', obj, property='currentText')

        # Set fields

        # Establish connections

        b = self.findChild(QtWidgets.QPushButton,
                           'pushButton_reload_LTE_phase_adv')
        b.clicked.connect(self.update_LTE_table)

    def update_LTE_table(self):
        """"""

        self.wizardObj.page_links['LTE'].validatePage()
        self.wizardObj.page_links['straight_centers'].validatePage()
        tableView = self.findChild(QtWidgets.QTableView,
                                   'tableView_elem_list_phase_adv')
        tableView.setModel(self.wizardObj.model_elem_list)

    def validatePage(self):
        """"""

        flat_used_elem_names = self.wizardObj.LTE.flat_used_elem_names

        marker_name = self.field('edit_disp_bump_marker_name').strip()
        n_disp_bumps = int(self.field('combo_n_disp_bumps'))

        if marker_name == '':
            conf = self.wizardObj.conf
            d = conf['lattice_props']
            if 'opt_props' in d:
                if 'phase_adv' in d['opt_props']:
                    del d['opt_props']['phase_adv']

            return True

        if marker_name not in flat_used_elem_names:
            text = 'Invalid element name'
            info_text = (
                f'Specified element "{marker_name}" does not exist!')
            showInvalidPageInputDialog(text, info_text)
            return False

        actual_n = flat_used_elem_names.count(marker_name)
        if actual_n != n_disp_bumps:
            text = 'Mismatch in number of dispersion bump markers'
            info_text = (
                f'There are {actual_n:d} instances of the specified element '
                f'"{marker_name}", while you expect {n_disp_bumps:d}!')
            showInvalidPageInputDialog(text, info_text)
            return False

        conf = self.wizardObj.conf
        d = conf['lattice_props']['opt_props']['phase_adv']
        #
        f = partial(update_aliased_scalar, conf)

        if n_disp_bumps == 2:
            d2 = d['MDISP across LS']
            f(d2, 'pdf_label', (
                r'Phase Advance btw. Disp. Bumps across LS '
                r'$(\Delta\nu_x, \Delta\nu_y)$'))
            f(d2['xlsx_label'], 'x', CommentedSeq([
                "normal", 'Horizontal Phase Advance btw. Disp. Bumps across LS ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'x']))
            f(d2['xlsx_label'], 'y', CommentedSeq([
                "normal", 'Vertical Phase Advance btw. Disp. Bumps across LS ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'y']))
            f(d2['elem1'], 'name',
              conf['lattice_props']['req_props']['beta']['LS']['name'])
            f(d2['elem1'], 'occur', 0)
            f(d2['elem2'], 'name', marker_name)
            f(d2['elem2'], 'occur', 0)
            #
            d2 = d['MDISP across SS']
            f(d2, 'pdf_label', (
                r'Phase Advance btw. Disp. Bumps across SS '
                r'$(\Delta\nu_x, \Delta\nu_y)$'))
            f(d2['xlsx_label'], 'x', CommentedSeq([
                "normal", 'Horizontal Phase Advance btw. Disp. Bumps across SS ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'x']))
            f(d2['xlsx_label'], 'y', CommentedSeq([
                "normal", 'Vertical Phase Advance btw. Disp. Bumps across SS ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'y']))
            f(d2['elem1'], 'name', marker_name)
            f(d2['elem1'], 'occur', 0)
            f(d2['elem2'], 'name', marker_name)
            f(d2['elem2'], 'occur', 1)

        elif n_disp_bumps == 4:
            d2 = d['MDISP 0&1']
            f(d2, 'pdf_label', (
                r'Phase Advance btw. Dispersion Bumps '
                r'$(\Delta\nu_x, \Delta\nu_y)$'))
            f(d2['xlsx_label'], 'x', CommentedSeq([
                "normal", 'Horizontal Phase Advance btw. Disp. Bumps ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'x']))
            f(d2['xlsx_label'], 'y', CommentedSeq([
                "normal", 'Vertical Phase Advance btw. Disp. Bumps ',
                "italic_greek", 'Delta', "italic_greek", 'nu', "italic_sub", 'y']))
            f(d2['elem1'], 'name', marker_name)
            f(d2['elem1'], 'occur', 0)
            f(d2['elem2'], 'name', marker_name)
            f(d2['elem2'], 'occur', 1)

        else:
            raise ValueError()

        return True

class PageStraightDrifts(QtWidgets.QWizardPage):
    """"""

    def __init__(self, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)

        self._registeredFields = []

    def registerFieldOnFirstShow(self, name, widget, *args, **kwargs):
        """"""

        if name not in self._registeredFields:
            self.registerField(name, widget, *args, **kwargs)
            self._registeredFields.append(name)

    def initializePage(self):
        """"""

        self.wizardObj = self.wizard()

        # Hook up models to views

        tableView = self.findChild(QtWidgets.QTableView,
                                   'tableView_elem_list_straight_drifts')
        tableView.setModel(self.wizardObj.model_elem_list)

        # Register fields

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_LS_drift_names')
        #self.registerFieldOnFirstShow('edit_half_LS_drifts*', edit)
        # TO-BE-DELETED
        self.registerFieldOnFirstShow('edit_half_LS_drifts', edit)
        # TO-BE-DELETED
        self.setField('edit_half_LS_drifts', 'DH5')

        edit = self.findChild(QtWidgets.QLineEdit, 'lineEdit_SS_drift_names')
        #self.registerFieldOnFirstShow('edit_half_SS_drifts*', edit)
        # TO-BE-DELETED
        self.registerFieldOnFirstShow('edit_half_SS_drifts', edit)
        # TO-BE-DELETED
        self.setField('edit_half_SS_drifts', 'DL5')

        # Set fields

        # Establish connections

    def validatePage(self):
        """"""

        flat_used_elem_names = np.array(self.wizardObj.LTE.flat_used_elem_names)

        half_LS_drift_name_list = [
            token.strip() for token in
            self.field('edit_half_LS_drifts').split(',') if token.strip()]
        half_SS_drift_name_list = [
            token.strip() for token in
            self.field('edit_half_SS_drifts').split(',') if token.strip()]

        for drift_names, LS_or_SS in [(half_LS_drift_name_list, 'LS'),
                                      (half_SS_drift_name_list, 'SS')]:

            for name in drift_names:
                if name not in flat_used_elem_names:
                    text = 'Invalid element name'
                    info_text = (
                        f'Specified element "{name}" as part of {LS_or_SS} '
                        f'drift does not exist!')
                    showInvalidPageInputDialog(text, info_text)
                    return False

            match_found = False
            for starting_ind in np.where(
                flat_used_elem_names == drift_names[0])[0]:

                for i, next_name in enumerate(drift_names[1:]):
                    if flat_used_elem_names[starting_ind + i + 1] != next_name:
                        break
                else:
                    match_found = True

                if match_found:
                    break
            else:
                text = f'No matching element name list for {LS_or_SS} drift'
                info_text = (
                    f'Specified consecutive element name list for {LS_or_SS} '
                    f'drift does not exist!')
                showInvalidPageInputDialog(text, info_text)
                return False

        conf = self.wizardObj.conf
        d = conf['lattice_props']['req_props']
        f = partial(update_aliased_scalar, conf)
        #
        f(d['length']['LS'], 'name_list', half_LS_drift_name_list)
        f(d['length']['LS'], 'multiplier', 2.0)
        #
        f(d['length']['SS'], 'name_list', half_SS_drift_name_list)
        f(d['length']['SS'], 'multiplier', 2.0)

        #_check_if_yaml_writable(conf)

        return True

