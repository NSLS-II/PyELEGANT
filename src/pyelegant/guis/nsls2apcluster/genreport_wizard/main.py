import os
import sys

from ruamel.yaml.comments import CommentedMap

from qtpy import QtCore, QtGui, QtWidgets
from qtpy import uic

class SkipPageListModel(QtCore.QAbstractListModel):
    """"""

    def __init__(self, data, view):
        """Constructor"""

        super().__init__()

        self._data = data
        self._view = view

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            return self._data[index.row()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

class SkipPageSelector(QtWidgets.QDialog):
    """"""

    def __init__(self, wizard_obj, output_dict, *args, **kwargs):
        """Constructor"""

        super().__init__(*args, **kwargs)
        ui_file = os.path.join(os.path.dirname(__file__), 'skip_page.ui')
        uic.loadUi(ui_file, self)

        w = wizard_obj
        if isinstance(w, ReportWizard):
            pass

        if False: # only allows forward jumping
            cur_page_suffix = w.currentPage().objectName()[len('wizardPage_'):]
            self.avail_page_suffixes = w.page_name_list[
                w.page_name_list.index(cur_page_suffix)+1:]
        else: # allows both forward & backward jumping
            self.avail_page_suffixes = w.page_name_list[:]

        data = [w.page_links[suffix].title()
                for suffix in self.avail_page_suffixes]

        model = SkipPageListModel(data, self.listView)
        self.listView.setModel(model)

        self.accepted.connect(self.process_selection)

        self.output_dict = output_dict

    def process_selection(self):
        """"""

        sel_model = self.listView.selectionModel()
        sel_rows = sel_model.selectedRows()
        if sel_rows:
            sel_row_index = sel_rows[0].row()
            self.output_dict['sel_suffix'] = self.avail_page_suffixes[sel_row_index]
        else:
            self.output_dict.clear()

class ReportWizard(QtWidgets.QWizard):

    def __init__(self):
        super().__init__()
        ui_file = os.path.join(os.path.dirname(__file__), 'wizard.ui')
        uic.loadUi(ui_file, self)

        self.config_filepath = None
        self.pdf_filepath = None
        self.conf = None
        self.LTE = None
        self.model_elem_list = None
        self.common_remote_opts = {}
        self.new_report = True

        self.page_name_list = [
            'LTE', 'straight_centers', 'phase_adv', 'straight_length',
            'test1', 'twiss_plots', 'paragraphs', 'N_KICKS', 'xy_aper_test',
            'fmap_xy_test', 'fmap_px_test', 'cmap_xy_test', 'cmap_px_test',
            'mom_aper_test', 'tswa', 'nonlin_chrom', 'nonlin_prod', 'rf_tau']

        self.page_links = {}
        for k in self.page_name_list:
            self.page_links[k] = self.findChild(QtWidgets.QWizardPage,
                                                f'wizardPage_{k}')

        self.page_indexes = {}
        for i in self.pageIds():
            obj_name = self.page(i).objectName()
            if obj_name.startswith('wizardPage_'):
                suffix = obj_name[len('wizardPage_'):]
                self.page_indexes[suffix] = i

        self.setButtonText(QtWidgets.QWizard.CustomButton1, 'Jump to...')
        self.showSkipButton(False)
        self.customButtonClicked.connect(self.skip_to)

        self.settings = QtCore.QSettings('nsls2', 'report_wizard')
        self.loadSettings()

        if False:
            x0, y0 = 100, 300
            self.setGeometry(x0, y0, 600, 400)
        else:
            self.setGeometry(self._settings['MainWindow']['geometry'])

    def update_conf_on_all_pages(self, mod_conf):
        """"""

        self.conf = mod_conf

        for k, v in self.page_links.items():
            v.conf = mod_conf

    def loadSettings(self):
        """"""

        self._settings = {'MainWindow': {}}

        self.settings.beginGroup('MainWindow')
        d = self._settings['MainWindow']
        # default values:
        x0, y0 = 100, 300
        width, height = 600, 400
        d['geometry'] = self.settings.value(
            'geometry', QtCore.QRect(x0, y0, width, height))
        self.settings.endGroup()

        self._settings['config_folderpath'] = \
            self.settings.value('config_folderpath', os.getcwd())

        self._settings['seed_config_filepath'] = \
            self.settings.value('seed_config_filepath', '')

        self.currentIdChanged.connect(self.set_jump_button_visibility)

    def saveSettings(self):
        """"""

        self.settings.beginGroup('MainWindow')
        self.settings.setValue('geometry', self.geometry())
        self.settings.endGroup()

        self.settings.setValue('config_folderpath',
                               self._settings['config_folderpath'])

        self.settings.setValue('seed_config_filepath',
                               self._settings['seed_config_filepath'])

    def closeEvent(self, event):
        """"""

        self.saveSettings()

        super().closeEvent(event)

    def set_jump_button_visibility(self, new_id):
        """"""

        page = self.page(new_id)

        if page is None: # Last page
            self.showSkipButton(False)
        elif page.objectName()[len('wizardPage_'):] in self.page_name_list:
            self.showSkipButton(True)
        else:
            self.showSkipButton(False)

    def showSkipButton(self, TF):
        """"""

        self.setOption(self.HaveCustomButton1, TF)

    def skip_to(self, button_index):
        """"""

        output = {}
        dialog = SkipPageSelector(self, output)
        dialog.exec()

        if output:
            sel_page_index = self.page_indexes[output['sel_suffix']]
            cur_page = self.currentPage()
            cur_page.setNextId(sel_page_index)
            self.next() # Click "Next" button
            cur_page.setNextId(None)

    def update_common_remote_opts(self):
        """"""

        if (self.common_remote_opts == {}) and (
            'common_remote_opts' in self.conf['nonlin']):
            self.common_remote_opts.update(
                self.conf['nonlin']['common_remote_opts'])

    def skip_new_report_setup_page(self, should_skip):
        """"""

        cur_page = self.currentPage()

        if should_skip:
            cur_page.setNextId(3)
        else:
            cur_page.setNextId(2)

def main():
    """"""

    app = QtWidgets.QApplication(sys.argv)
    window = ReportWizard()
    window.show()
    app.exec_()

if __name__ == '__main__':

    main()