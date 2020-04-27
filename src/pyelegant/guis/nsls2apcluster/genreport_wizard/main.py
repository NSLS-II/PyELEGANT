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

        self.page_links = dict(
            LTE=self.findChild(QtWidgets.QWizardPage, 'wizardPage_LTE'),
            straight_centers=self.findChild(
                QtWidgets.QWizardPage, 'wizardPage_straight_centers'),
        )

        x0, y0 = 100, 300
        self.setGeometry(x0, y0, 600, 400)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = ReportWizard()
    window.show()
    app.exec_()