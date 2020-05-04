import os
import sys
from subprocess import Popen, PIPE
import shlex
import re
import getpass

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

    def __init__(self, squeue_output_format, header_list):
        """Constructor"""

        super().__init__()

        self._data = [[]]

        self._headers = header_list
        #self._row_numbers = range(len(data))

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

            #if orientation == Qt.Vertical:
                #return str(self._row_numbers[section])

    def get_hearders(self):
        """"""

        return self._headers

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
            '%A', '%i','%P', '%j','%u','%t','%M','%L','%D','%C','%R', '%Q']
        self.q_output_format = q_output_format_delimiter.join(q_output_format_list)

        header, _ = self.squeue('-u nonexistent')
        header = [s.strip() for s in header.split(q_output_format_delimiter)]
        self.model_q = TableModelQueue(self.q_output_format, header)

        self.proxy_model_q = QtCore.QSortFilterProxyModel()
        self.proxy_model_q.setSourceModel(self.model_q)

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

        #progbar_color = 'lightblue'
        #progbar_color = 'green'
        progbar_color = 'red'

        progbar_style_sheet = f"""
        QProgressBar{{
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center
        }}

        QProgressBar::chunk {{
            background-color: {progbar_color};
            width: 10px;
            margin: 0px;
        }}
        """

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
            prog.setStyleSheet(progbar_style_sheet)
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
            prog.setStyleSheet(progbar_style_sheet)
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

        m = self.model_q

        m.beginResetModel()
        m._data.clear()
        if table != []:
            m._data.extend(table)
        else:
            m._data.append([None] * len(m.get_hearders()))
        m.endResetModel()

        self.tableView_q.resizeColumnsToContents()

def main():
    """"""
    app = QtWidgets.QApplication(sys.argv)
    window = ClusterStatusWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()