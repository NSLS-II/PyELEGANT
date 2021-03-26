program_name = 'pyelegant'
version = '0.9.0'

from setuptools import setup, find_packages
from setuptools.command.install import install

facility_name_arg = 'facility-name'

class InstallCommand(install):
    """"""

    user_options = install.user_options + [
        (f'{facility_name_arg}=', None, 'Facility name for remote running')
    ]

    def initialize_options(self):
        """"""
        install.initialize_options(self)
        self.facility_name = None

    def finalize_options(self):
        """"""
        print(f'Specified facility name for remote runs is "{self.facility_name}"')
        install.finalize_options(self)

    def run(self):
        """"""
        print(self.facility_name)
        install.run(self)

import sys
import os
import json

#print('sys.argv contents:')
#print(sys.argv)
#print(' ')

facility_json_filename = 'facility.json'
version_filename = 'version.json'

package_data = {
    'pyelegant': [facility_json_filename, version_filename,
                  'Touschek_F_interpolator.pkl', '.defns.rpn']
}

com_req_pakcages = [
    'numpy', 'scipy', 'matplotlib', 'h5py', 'pylatex', 'ruamel.yaml',
    'xlsxwriter', 'qtpy', 'pyqtgraph']

if ('install' in sys.argv) or ('sdist' in sys.argv):

    facility_name_opt = [v for v in sys.argv if v.startswith(f'--{facility_name_arg}=')]
    if len(facility_name_opt) == 0:
        raise ValueError(f'Required arg "--{facility_name_arg}" is missing')
    elif len(facility_name_opt) > 1:
        raise ValueError(f'Multiple arg "--{facility_name_arg}" found')

    facility_name = facility_name_opt[0][len(f'--{facility_name_arg}='):]

    available_facility_names = ['local', 'nsls2apcluster', 'nsls2cr', 'nsls2pluto']
    if facility_name not in available_facility_names:
        print('* Only the following facility names are available:')
        print('      ' + ', '.join(available_facility_names))
        raise ValueError(
            'Specified facility_name "{}" is not available.'.format(facility_name))

    this_folder = os.path.dirname(os.path.abspath(__file__))

    facility_json_filepath = os.path.join(
        this_folder, 'src', 'pyelegant', facility_json_filename)
    with open(facility_json_filepath, 'w') as f:
        fac_info = {'name': facility_name}
        if facility_name == 'nsls2apcluster':
            fac_info['MODULE_LOAD_CMD_STR'] = 'elegant-latest' # 'elegant-latest elegant/2020.2.0',
            fac_info['MPI_COMPILER_OPT_STR'] = '' # '--mpi=pmi2',
        elif facility_name == 'nsls2pluto':
            fac_info['MODULE_LOAD_CMD_STR'] = 'accelerator'
        else:
            assert facility_name in ('local', 'nsls2cr')

        json.dump(fac_info, f)

    sys.argv.remove(facility_name_opt[0])

    if 'sdist' in sys.argv:
        # Modify the tarball name to be "pyelegant-{facility_name}-?.?.?.tar.gz"
        program_name += f'-{facility_name}'

    with open(os.path.join(
        this_folder, 'src', 'pyelegant', version_filename), 'w') as f:
        json.dump(version, f)

    req_pakcages = []
    if facility_name in ('nsls2apcluster', 'nsls2pluto'):
        req_pakcages += ['mpi4py>=3', 'dill']

    entry_points = dict(
        console_scripts=[
            'pyele_zip_lte = pyelegant.scripts.ziplte:zip_lte',
            'pyele_unzip_lte = pyelegant.scripts.ziplte:unzip_lte',
        ],
        gui_scripts=[],
    )

    if facility_name == 'nsls2apcluster':
        facility_gui_folderpath = 'src/pyelegant/guis/nsls2apcluster'
        facility_script_folderpath = 'src/pyelegant/scripts/nsls2apcluster'
    elif facility_name == 'nsls2pluto':
        facility_gui_folderpath = 'src/pyelegant/guis/nsls2pluto'
        facility_script_folderpath = 'src/pyelegant/scripts/nsls2pluto'

    entry_points['console_scripts'].extend([
        'pyele_report = pyelegant.scripts.common.genreport:main',
        'pyele_slurm_print_queue = pyelegant.scripts.slurmutil:print_queue',
        'pyele_slurm_print_load = pyelegant.scripts.slurmutil:print_load',
        'pyele_slurm_scancel_regex_jobname = pyelegant.scripts.slurmutil:scancel_by_regex_jobname',
    ])
    entry_points['gui_scripts'].extend([
        # GUI
        'pyele_gui_slurm = pyelegant.guis.cluster_status.main:main',
        'pyele_gui_report_wiz = pyelegant.guis.genreport_wizard.main:main',
    ])

    if facility_name in ('nsls2apcluster', 'nsls2pluto'):
        package_data['pyelegant.guis.cluster_status'] = ['*.ui']
        package_data['pyelegant.guis.genreport_wizard'] = ['*.ui']

    other_setup_opts = dict(
        install_requires=req_pakcages,
        # ^ These requirements are actually NOT being checked (as of 04/01/2020)
        #   due to the known bug with the custom install:
        #       (https://github.com/pypa/setuptools/issues/456)
        entry_points=entry_points)
elif 'egg_info' in sys.argv:
    other_setup_opts = dict(install_requires=com_req_pakcages)
else:
    raise RuntimeError()

packages = []
for pk in find_packages('src'):
    # Exclude guis/scripts subfolders for other facilities
    if pk.startswith('pyelegant.guis.'):
        if pk == f'pyelegant.guis.{facility_name}':
            pk = 'pyelegant.guis'
        else:
            if pk.startswith(f'pyelegant.guis.{facility_name}.'):
                pk = pk.replace(f'pyelegant.guis.{facility_name}.', 'pyelegant.guis.')
            else:
                continue
    elif pk.startswith('pyelegant.scripts.'):
        if pk == f'pyelegant.scripts.{facility_name}':
            pk = 'pyelegant.scripts'
        elif pk == f'pyelegant.scripts.common':
            pk = 'pyelegant.scripts.common'
        else:
            if pk.startswith(f'pyelegant.scripts.{facility_name}.'):
                pk = pk.replace(f'pyelegant.scripts.{facility_name}.', 'pyelegant.scripts.')
            else:
                continue
    packages.append(pk)
packages = list(set(packages))

setup(
    name = program_name,
    version = version,
    packages=packages,
    package_dir={'': 'src',
                 'pyelegant.guis': facility_gui_folderpath,
                 'pyelegant.scripts': facility_script_folderpath,
                 'pyelegant.scripts.common': 'src/pyelegant/scripts/common',
                 },
    #include_package_data = True,
    package_data = package_data,
    zip_safe=False,
    description = 'Python Interface to ELEGANT',
    author = 'Yoshiteru Hidaka',
    maintainer = 'Yoshiteru Hidaka',
    maintainer_email = 'yhidaka@bnl.gov',
    cmdclass={'install': InstallCommand},
    **other_setup_opts
)

