import os
import sys
from subprocess import Popen, PIPE
import shlex
import re
import getpass
import time
import datetime

import numpy as np

from qtpy import QtCore, QtGui, QtWidgets
Qt = QtCore.Qt
from qtpy import uic

def chained_Popen(cmd_list):
    """"""

    if len(cmd_list) == 1:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    else:
        cmd = cmd_list[0]
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        for cmd in cmd_list[1:-1]:
            p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE)
        cmd = cmd_list[-1]
        p = Popen(shlex.split(cmd), stdin=p.stdout, stdout=PIPE, stderr=PIPE,
                  encoding='utf-8')

    out, err = p.communicate()

    return out, err, p.returncode

class TableModelQueue(QtCore.QAbstractTableModel):
    """"""

    def __init__(self, squeue_output_format, header_list,
                 time_duration_column_inds):
        """Constructor"""

        super().__init__()

        self._data = [[]]

        self._headers = header_list
        self._row_numbers = []

        self._time_duration_column_inds = time_duration_column_inds

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list

            val = self._data[index.row()][index.column()]

            if index.column() not in self._time_duration_column_inds:
                return val
            else:
                if val:
                    return convert_slurm_time_duration_seconds_to_str(val)
                else:
                    return val

        elif role == Qt.UserRole: # Used for sorting
            val = self._data[index.row()][index.column()]
            return val

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
                try:
                    return str(self._row_numbers[section])
                except:
                    return '0'

    def get_hearders(self):
        """"""

        return self._headers

    def update_data(self, table):
        """"""

        self.beginResetModel()

        self._data.clear()

        if table != []:
            self._data.extend(table)
        else:
            self._data.append([None] * len(self.get_hearders()))

        self.endResetModel()

        self._update_row_numbers()

    def _update_row_numbers(self):
        """"""

        self._row_numbers = range(len(self._data))

class ClusterStatusWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        ui_file = os.path.join(os.path.dirname(__file__), 'cluster_status.ui')
        uic.loadUi(ui_file, self)

        self.partition_info = None

        q_output_format_delimiter = '#' # Cannot use "|" here, as this will
        # conflict with piping in custom commands.
        self.q_output_format_delimiter = q_output_format_delimiter

        q_output_format_list = [
            '%A', '%i','%P', '%j','%u','%t','%M','%L','%D','%C','%R', '%S',
            '%V', '%Q']
        self.q_output_format = q_output_format_delimiter.join(q_output_format_list)

        self.time_duration_column_inds = [
            i for i, _format in enumerate(q_output_format_list)
            if _format in ('%M', '%L')]

        header, _ = self.squeue('-u nonexistent')
        header = [s.strip() for s in header.split(q_output_format_delimiter)]
        self.model_q = TableModelQueue(self.q_output_format, header,
                                       self.time_duration_column_inds)

        self.proxy_model_q = QtCore.QSortFilterProxyModel()
        self.proxy_model_q.setSourceModel(self.model_q)
        self.proxy_model_q.setSortRole(Qt.UserRole)

        self.tableView_q.setModel(self.proxy_model_q)
        self.tableView_q.setSortingEnabled(True)
        self.tableView_q.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)

        self.tableWidget_load.verticalHeader().setVisible(False)
        self.tableWidget_load.resizeColumnsToContents()

        self.q_cmd_extra_args = {}
        for i in range(self.comboBox_q_cmd.count()):
            k = self.comboBox_q_cmd.itemText(i)
            self.q_cmd_extra_args[k] = ''

        x0, y0 = 100, 300
        width, height = 1200, 700
        ini_width_load_table = 500
        self.setGeometry(x0, y0, width, height)

        # Adjust initial splitter ratio
        self.splitter.setSizes([ini_width_load_table,
                                width - ini_width_load_table])

        # Change the initial selection for "scancel" type
        self.comboBox_scancel_type.setCurrentText('Only Selected')

        self.update_edit_q_cmd_suppl('All')

        self.pushButton_update_q.clicked.connect(self.update_q_table)
        self.pushButton_update_load.clicked.connect(self.update_load_table)
        self.pushButton_scancel.clicked.connect(self.scancel)

        self.lineEdit_q_cmd_suppl.textEdited.connect(self.update_q_cmd_extra_args)
        self.comboBox_q_cmd.currentTextChanged.connect(self.update_edit_q_cmd_suppl)

    def scancel(self):
        """"""

        keys = ['JOBID', 'PARTITION', 'NAME', 'USER', 'ST', 'TIME',
                'TIME_LEFT', 'CPUS']
        col_inds = {}
        for k in keys:
            col_inds[k] = self.model_q._headers.index(k)

        username = getpass.getuser()

        cancel_all_user_jobs = False

        cancel_type = self.comboBox_scancel_type.currentText()
        if cancel_type == 'Only Selected':
            sel_model = self.tableView_q.selectionModel()
            sel_rows = sel_model.selectedRows()
            job_ID_list = [
                self.proxy_model_q.index(index.row(), col_inds['JOBID']).data()
                for index in sel_rows if
                self.proxy_model_q.index(index.row(), col_inds['USER']).data()
                == username]
            njobs_yours = len(job_ID_list)

            informative_text = (f'Your {njobs_yours:d} selected jobs.')
            no_job_text = 'No jobs of yours have been selected.'

        elif cancel_type == 'All Shown Below': # all jobs shown in the table
            njobs = self.model_q.rowCount(0)
            job_ID_list = [
                self.proxy_model_q.index(i, col_inds['JOBID']).data()
                for i in range(njobs) if
                self.proxy_model_q.index(i, col_inds['USER']).data() == username]
            njobs_yours = len(job_ID_list)

            informative_text = (
                f'All of your {njobs_yours:d} jobs currently shown in the table.')
            no_job_text = 'No jobs of yours exist in the table.'

        elif cancel_type == 'All of My Jobs':
            cancel_all_user_jobs = True
        else:
            raise ValueError()

        QMessageBox = QtWidgets.QMessageBox

        if cancel_all_user_jobs:
            if self.checkBox_scancel_confirm.isChecked():

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setText('Do you want to cancel all of your jobs?')
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                msg.setWindowTitle('Confirm "scancel"')
                msg.setStyleSheet("QIcon{max-width: 100px;}")
                msg.setStyleSheet("QLabel{min-width: 300px;}")
                reply = msg.exec_()

                if reply == QMessageBox.No:
                    return

            cmd = f'scancel -u {username}'
        else:
            if njobs_yours == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText('No jobs to cancel.')
                msg.setInformativeText(no_job_text)
                msg.setWindowTitle('No jobs')
                msg.setStyleSheet("QIcon{max-width: 100px;}")
                msg.setStyleSheet("QLabel{min-width: 300px;}")
                msg.exec_()
                return

            if self.checkBox_scancel_confirm.isChecked():

                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setText('Do you want to cancel the follwoing jobs?')
                msg.setInformativeText(informative_text)
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                msg.setWindowTitle('Confirm "scancel"')
                msg.setStyleSheet("QIcon{max-width: 100px;}")
                msg.setStyleSheet("QLabel{min-width: 300px;}")
                reply = msg.exec_()

                if reply == QMessageBox.No:
                    return

            cmd = 'scancel ' + ' '.join(job_ID_list)

        #print(f'Executing "$ {cmd}"')
        self.statusbar.showMessage(f'Executing "$ {cmd}"')
        if True:
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
            out, err = p.communicate()
            print(out)
            if err:
                print('** stderr **')
                print(err)

    def update_edit_q_cmd_suppl(self, cmd_type):
        """"""

        self.lineEdit_q_cmd_suppl.setText(self.q_cmd_extra_args[cmd_type])

        if cmd_type in ('All', 'me'):
            self.lineEdit_q_cmd_suppl.setEnabled(False)
        else:
            self.lineEdit_q_cmd_suppl.setEnabled(True)

    def update_q_cmd_extra_args(self, new_text):
        """"""

        cmd_type = self.comboBox_q_cmd.currentText()

        self.q_cmd_extra_args[cmd_type] = new_text

    def update_partition_info(self):
        """"""

        cmd = 'scontrol show partition'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()

        parsed = {}
        for k, v in re.findall('([\w\d]+)=([^\s]+)', out):
            if k == 'PartitionName':
                d = parsed[v] = {}
            else:
                d[k] = v

        self.partition_info = parsed

    def _expand_node_range_pattern(self, index_str):
        """
        Examples for "index_str":
            '[019-025]'
        """

        tokens = index_str[1:-1].split(',')
        pat_list = []
        for tok in tokens:
            if '-' in tok:
                iStart, iEnd = tok.split('-')
                pat_list.extend([f'{i:03d}' for i in range(
                    int(iStart), int(iEnd)+1)])
            else:
                pat_list.append(tok)
        pat = '|'.join(pat_list)

        return pat

    def update_load_table(self):
        """"""

        QTableWidgetItem = QtWidgets.QTableWidgetItem

        t = self.tableWidget_load
        t.setHorizontalHeaderLabels([
            'Partitions', 'Nodes', '# of Allocated / Total Cores (%)',
            'CPU Load\n(cores)', 'Free\n(cores)', 'Free &\nSuspendable\n(cores)'])

        self.update_partition_info()
        #
        grouped_partition_names = {}
        for p in list(self.partition_info):
            preempt_mode = self.partition_info[p]['PreemptMode']
            nodes_str = self.partition_info[p]['Nodes']
            nodes_tuple = tuple(re.findall('\w+\-[\d\-\[\],]+(?<!,)', nodes_str))
            #print((p, preempt_mode, nodes_str, nodes_tuple))
            k = (nodes_tuple, preempt_mode)
            if k not in grouped_partition_names:
                grouped_partition_names[k] = [p]
            else:
                grouped_partition_names[k].append(p)


        nMaxNodeIndex = 100
        group_summary = []
        preempted_partitions = []
        for (nodes_tuple, preempt_mode), partition_names in \
            grouped_partition_names.items():

            if preempt_mode != 'OFF':
                preempted_partitions.extend(partition_names)

            d = dict(
                partition_names=partition_names, nodes_tuple=nodes_tuple,
                node_list=[],
            )
            for nodes_str in nodes_tuple:
                prefix = nodes_str.split('-')[0]
                index_str = nodes_str[len(prefix)+1:]
                #print((prefix, index_str))
                if ',' in index_str:
                    assert index_str.startswith('[') and index_str.endswith(']')
                    pat = self._expand_node_range_pattern(index_str)
                elif index_str.startswith('[') and index_str.endswith(']'):
                    pat = self._expand_node_range_pattern(index_str)
                else:
                    pat = index_str
                matched_indexes = re.findall(pat, ','.join(
                    [f'{i:03d}' for i in range(nMaxNodeIndex)]))
                #print(matched_indexes)
                node_list = [f'{prefix}-{s}' for s in matched_indexes]
                #print(nodes_str)
                #print(prefix, node_list)
                d['node_list'].extend(node_list)
            group_summary.append(d)

        cmd = 'scontrol show node'
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
        out, err = p.communicate()

        parsed = re.findall(
            'NodeName=([\w\d\-]+)\s+[\w=\s]+CPUAlloc=(\d+)\s+CPUTot=(\d+)\s+CPULoad=([\d\.N/A]+)',
            out)

        temp_tables = []
        for pname in preempted_partitions:
            cmd = f'squeue --noheader -p {pname} -o "%t#%C#%R"'
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
            out, err = p.communicate()
            if out.strip() != '':
                table = np.array([line.split('#') for line in out.splitlines()])
                temp_tables.append(table)
        if temp_tables:
            combined_tables = np.vstack(temp_tables)
            running = combined_tables[:, 0] == 'R'
            running_cpus = combined_tables[running, 1].astype(int)
            running_nodes = combined_tables[running, 2]
            suspendables = {}
            for node_name in np.unique(running_nodes):
                suspendables[node_name] = np.sum(
                    running_cpus[running_nodes == node_name])
        else:
            suspendables = {}

        for d in group_summary:
            _nAlloc = _nTot = _nSuspendable = _cpu_load = 0
            node_list = d['node_list']
            for node_name, nAlloc, nTot, cpu_load in parsed:
                if node_name in node_list:
                    _nAlloc += int(nAlloc)
                    _nTot += int(nTot)
                    if node_name in suspendables:
                        _nSuspendable += suspendables[node_name]
                    if cpu_load != 'N/A':
                        _cpu_load += float(cpu_load)
                    else:
                        _cpu_load += float('nan')
            d['n_alloc'] = _nAlloc
            d['n_tot'] = _nTot
            d['load'] = _cpu_load
            d['n_free'] = _nTot - _nAlloc
            d['n_suspendable'] = _nSuspendable

        t.setRowCount(len(group_summary) + 1 + len(parsed))


        nonsusp_color = 'red'
        susp_color = 'green'

        progbar_style_sheet_templates = {'0': '', '1': '', 'else': ''}
        for k in list(progbar_style_sheet_templates):
            progbar_style_sheet_templates[k] = '''
        QProgressBar{{
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center
        }}
        '''
        k = '0'
        progbar_style_sheet_templates[k] += '''
        QProgressBar::chunk {{
            background-color: {susp_color};

            margin: 0px;
        }}'''
        progbar_style_sheet_templates[k] = \
            progbar_style_sheet_templates[k].format(susp_color=susp_color)
        k = '1'
        progbar_style_sheet_templates[k] += '''
        QProgressBar::chunk {{
            background-color: {nonsusp_color};

            margin: 0px;
        }}'''
        progbar_style_sheet_templates[k] = \
            progbar_style_sheet_templates[k].format(nonsusp_color=nonsusp_color)
        k = 'else'
        progbar_style_sheet_templates[k] += '''
        QProgressBar::chunk {{{{
            background-color:
            qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {nonsusp_color}, stop: {{pbar_frac1:.5f}} {nonsusp_color},
            stop: {{pbar_frac2:.5f}} {susp_color}, stop: 1 {susp_color});

            margin: 0px;
        }}}}'''.format(nonsusp_color=nonsusp_color, susp_color=susp_color)

        for iRow, d in enumerate(group_summary):

            iCol = 0

            t.setItem(iRow, iCol,
                      QTableWidgetItem('\n'.join(d['partition_names'])))
            iCol += 1

            nMaxNodeListLen = 16
            nodes_list = []
            for nodes_str in d['nodes_tuple']:
                if len(nodes_str) <= nMaxNodeListLen:
                    nodes_list.append(nodes_str)
                else:
                    indent = ''
                    s = ''
                    for tok in nodes_str.split(','):
                        if s == '':
                            s = tok
                        elif len(f'{s},{tok}') > nMaxNodeListLen:
                            nodes_list.append(indent + s + ',')
                            indent = ' ' * 2
                            s = tok
                        else:
                            s = f'{s},{tok}'
                    if s:
                        nodes_list.append(indent + s)
            t.setItem(iRow, iCol, QTableWidgetItem(
                '\n'.join(nodes_list)))
            iCol += 1

            self.tableWidget_load.resizeRowToContents(iRow)

            prog = QtWidgets.QProgressBar()
            prog.setMaximum(d['n_tot'])
            prog.setMinimum(0)
            prog.setValue(d['n_alloc'])
            prog.setFormat('%v / %m (%p%)')

            if d["n_alloc"] != 0:
                pbar_frac = (d["n_alloc"] - d['n_suspendable']) / d["n_alloc"]
            else:
                pbar_frac = 1.0
            if pbar_frac == 0.0:
                sheet = progbar_style_sheet_templates['0']
            elif pbar_frac == 1.0:
                sheet = progbar_style_sheet_templates['1']
            else:
                sheet = progbar_style_sheet_templates['else'].format(
                    pbar_frac1=pbar_frac, pbar_frac2=min([pbar_frac+1e-5, 1.0]))
            prog.setStyleSheet(sheet)

            t.setCellWidget(iRow, iCol, prog)
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(f'{d["load"]:.2f}'))
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(f'{d["n_free"]:d}'))
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(
                f'{d["n_free"] + d["n_suspendable"]:d}'))
            iCol += 1

        row_offset = len(group_summary)

        # Change divider row's background color
        for iCol in range(t.columnCount()):
            divider_item = QTableWidgetItem('')
            divider_item.setBackground(Qt.black)
            t.setItem(row_offset, iCol, divider_item)
        t.setRowHeight(row_offset, t.rowHeight(row_offset) // 4)

        row_offset += 1

        for iRow, (node_name, n_alloc_str, n_tot_str, load_val_str
                   ) in enumerate(parsed):

            iRow += row_offset
            iCol = 1

            t.setItem(iRow, iCol, QTableWidgetItem(node_name))
            iCol += 1

            n_alloc = int(n_alloc_str)
            n_tot = int(n_tot_str)
            n_free = n_tot - n_alloc
            if node_name in suspendables:
                n_suspendable = suspendables[node_name]
            else:
                n_suspendable = 0

            prog = QtWidgets.QProgressBar()
            prog.setMaximum(n_tot)
            prog.setMinimum(0)
            prog.setValue(n_alloc)
            prog.setFormat('%v / %m (%p%)')

            if n_alloc != 0:
                pbar_frac = (n_alloc - n_suspendable) / n_alloc
            else:
                pbar_frac = 1.0
            if pbar_frac == 0.0:
                sheet = progbar_style_sheet_templates['0']
            elif pbar_frac == 1.0:
                sheet = progbar_style_sheet_templates['1']
            else:
                sheet = progbar_style_sheet_templates['else'].format(
                    pbar_frac1=pbar_frac, pbar_frac2=min([pbar_frac+1e-5, 1.0]))
            prog.setStyleSheet(sheet)

            t.setCellWidget(iRow, iCol, prog)
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(load_val_str))
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(f'{n_free:d}'))
            iCol += 1

            t.setItem(iRow, iCol, QTableWidgetItem(f'{n_free + n_suspendable:d}'))
            iCol += 1

        self.tableWidget_load.resizeColumnsToContents()

    def squeue(self, extra_arg_str=''):
        """"""

        cmd = f'squeue -o "{self.q_output_format}" {extra_arg_str}'
        self.statusbar.showMessage(f'Last "squeue" command: $ {cmd}')

        if '|' not in cmd:
            p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE, encoding='utf-8')
            out, err = p.communicate()
        else:
            cmd_list = cmd.split('|')
            out, err, returncode = chained_Popen(cmd_list)

        return out, err

    def update_q_table(self):
        """"""

        extra_arg_list = ['--noheader']

        cmd_type = self.comboBox_q_cmd.currentText()
        if cmd_type == 'All':
            pass
        elif cmd_type == 'me':
            extra_arg_list.append(f'-u {getpass.getuser()}')
        elif cmd_type == 'grep':
            extra = self.q_cmd_extra_args[cmd_type]
            extra_arg_list.append(f'| grep {extra}')
        elif cmd_type == 'grep exclude':
            extra = self.q_cmd_extra_args[cmd_type]
            extra_arg_list.append(f'| grep -v {extra}')
        elif cmd_type == '$ squeue':
            extra = self.q_cmd_extra_args[cmd_type]
            extra_tokens = extra.split()
            if '--noheader' in extra_tokens:
                extra_tokens.remove('--noheader')
            elif '-h' in extra_tokens:
                extra_tokens.remove('-h')
            extra_arg_list += extra_tokens
        else:
            raise ValueError()

        out, err = self.squeue(' '.join(extra_arg_list))
        table = [line.split(self.q_output_format_delimiter)
                 for line in out.splitlines()]

        for row in table:
            for iCol in self.time_duration_column_inds:
                row[iCol] = convert_slurm_time_duration_str_to_seconds(row[iCol])

        self.model_q.update_data(table)

        self.tableView_q.resizeColumnsToContents()

def convert_slurm_time_duration_str_to_seconds(slurm_time_duration_str):
    """"""

    s_list = slurm_time_duration_str.split(':')
    if len(s_list) == 1:
        s_list = ['00', '00'] + s_list
    elif len(s_list) == 2:
        s_list = ['00'] + s_list
    elif (len(s_list) >= 4) or (len(s_list) == 0):
        raise RuntimeError('Unexpected number of splits')

    if '-' in s_list[0]:
        days_str, hrs_str = s_list[0].split('-')
        s_list[0] = hrs_str

        days_in_secs = int(days_str) * 60.0 * 60.0 * 24.0
    else:
        days_in_secs = 0.0

    d = time.strptime(':'.join(s_list), '%H:%M:%S')

    duration_in_sec = days_in_secs + datetime.timedelta(
        hours=d.tm_hour, minutes=d.tm_min, seconds=d.tm_sec).total_seconds()

    print(slurm_time_duration_str, duration_in_sec)

    return duration_in_sec

def convert_slurm_time_duration_seconds_to_str(slurm_time_duration_sec):
    """"""

    sec = datetime.timedelta(seconds=slurm_time_duration_sec)
    dobj = datetime.datetime(1,1,1) + sec
    if dobj.day - 1 != 0:
        str_duration = '{:d}-'.format(dobj.day - 1)
    else:
        str_duration = ''
    str_duration += '{:02d}:{:02d}:{:02d}'.format(
        dobj.hour, dobj.minute, dobj.second)

    return str_duration

def main():
    """"""
    app = QtWidgets.QApplication(sys.argv)
    window = ClusterStatusWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()