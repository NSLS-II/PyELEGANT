import os
import sys

from qtpy import QtCore, QtGui, QtWidgets
from qtpy import uic

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

        self.page_name_list = [
            'LTE', 'straight_centers', 'phase_adv', 'straight_length',
            'test1', 'twiss_plots', 'paragraphs', 'N_KICKS', 'xy_aper_test',
            'fmap_xy_test', 'fmap_px_test', 'cmap_xy_test', 'cmap_px_test',
            'mom_aper_test']
        # 'tswa_test', 'nonlin_chrom_test',
        #, 'rf_dep_props', 'lifetime']

        self.page_links = {}
        for k in self.page_name_list:
            self.page_links[k] = self.findChild(QtWidgets.QWizardPage,
                                                f'wizardPage_{k}')

        x0, y0 = 100, 300
        self.setGeometry(x0, y0, 600, 400)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = ReportWizard()
    window.show()
    app.exec_()