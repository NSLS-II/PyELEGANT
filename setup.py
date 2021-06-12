program_name = 'pyelegant'
version = '0.9.0'

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.sdist import sdist

facility_name_arg = 'facility-name'
facility_json_filename = 'facility.json'

version_filename = 'version.json'

if False:

    class InstallCommand(install):
        """"""

        user_options = install.user_options + [
            (f'{facility_name_arg}=', None, 'Facility name for remote running')
        ]

        def initialize_options(self):
            """"""
            install.initialize_options(self)
            self.facility_name = None

            if False:
                import pprint

                print('*************************************')
                # pprint.pprint(dir(self))
                # pprint.pprint(dir(self.distribution))
                pprint.pprint(self.distribution.packages)
                print('*************************************')

        def finalize_options(self):
            """"""

            install.finalize_options(self)

            facility_name = self.facility_name

            if facility_name is None:
                raise ValueError(f'Required arg "--{facility_name_arg}" is missing')

            print(f'Specified facility name for remote runs is "{facility_name}"')

            available_facility_names = [
                'local',
                'nsls2apcluster',
                'nsls2cr',
                'nsls2pluto',
            ]
            if facility_name not in available_facility_names:
                print('* Only the following facility names are available:')
                print('      ' + ', '.join(available_facility_names))
                raise ValueError(
                    'Specified facility_name "{}" is not available.'.format(
                        facility_name
                    )
                )

            self._update_json_files_with_facility_specific_info()

            self._add_facility_specific_packages()
            self._add_facility_specific_package_dir()
            self._add_facility_specific_package_data()
            self._add_facility_specific_install_requires()

        def run(self):
            """"""
            # print(self.facility_name)
            install.run(self)

        def _update_json_files_with_facility_specific_info(self):
            """"""

            facility_name = self.facility_name

            this_folder = os.path.dirname(os.path.abspath(__file__))

            facility_json_filepath = os.path.join(
                this_folder, 'src', 'pyelegant', facility_json_filename
            )
            with open(facility_json_filepath, 'w') as f:
                fac_info = {'name': facility_name}
                if facility_name == 'nsls2apcluster':
                    fac_info[
                        'MODULE_LOAD_CMD_STR'
                    ] = 'elegant-latest'  # 'elegant-latest elegant/2020.2.0',
                    fac_info['MPI_COMPILER_OPT_STR'] = ''  # '--mpi=pmi2',
                elif facility_name == 'nsls2pluto':
                    fac_info['MODULE_LOAD_CMD_STR'] = 'accelerator'
                else:
                    assert facility_name in ('local', 'nsls2cr')

                json.dump(fac_info, f)

            with open(
                os.path.join(this_folder, 'src', 'pyelegant', version_filename), 'w'
            ) as f:
                json.dump(version, f)

        def _add_facility_specific_packages(self):

            facility_name = self.facility_name

            packages = []
            for pk in find_packages('src'):
                # Exclude guis/scripts subfolders for other facilities
                if pk.startswith('pyelegant.guis.'):
                    if pk == f'pyelegant.guis.{facility_name}':
                        pk = 'pyelegant.guis'
                    else:
                        if pk.startswith(f'pyelegant.guis.{facility_name}.'):
                            pk = pk.replace(
                                f'pyelegant.guis.{facility_name}.', 'pyelegant.guis.'
                            )
                        else:
                            continue
                elif pk.startswith('pyelegant.scripts.'):
                    if pk == f'pyelegant.scripts.{facility_name}':
                        pk = 'pyelegant.scripts'
                    elif pk == f'pyelegant.scripts.common':
                        pk = 'pyelegant.scripts.common'
                    else:
                        if pk.startswith(f'pyelegant.scripts.{facility_name}.'):
                            pk = pk.replace(
                                f'pyelegant.scripts.{facility_name}.',
                                'pyelegant.scripts.',
                            )
                        else:
                            continue
                packages.append(pk)

            self.distribution.packages = list(set(packages))

            if False:
                import pprint

                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                pprint.pprint(dir(self.distribution))
                # pprint.pprint(dir(self))
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        def _add_facility_specific_package_dir(self):

            facility_name = self.facility_name

            package_dir = self.distribution.package_dir

            if facility_name == 'nsls2apcluster':
                facility_gui_folderpath = 'src/pyelegant/guis/nsls2apcluster'
                facility_script_folderpath = 'src/pyelegant/scripts/nsls2apcluster'
            elif facility_name == 'nsls2pluto':
                facility_gui_folderpath = 'src/pyelegant/guis/nsls2pluto'
                facility_script_folderpath = 'src/pyelegant/scripts/nsls2pluto'
            else:
                facility_gui_folderpath = None
                facility_script_folderpath = None

            if facility_gui_folderpath is not None:
                package_dir['pyelegant.guis'] = facility_gui_folderpath
            if facility_script_folderpath is not None:
                package_dir['pyelegant.scripts'] = facility_script_folderpath

        def _add_facility_specific_package_data(self):

            facility_name = self.facility_name

            package_data = self.distribution.package_data

            if facility_name in ('nsls2apcluster', 'nsls2pluto'):
                package_data['pyelegant.guis.cluster_status'] = ['*.ui']
                package_data['pyelegant.guis.genreport_wizard'] = ['*.ui']

        def _add_facility_specific_install_requires(self):

            facility_name = self.facility_name

            req_pakcages = []
            if facility_name in ('nsls2apcluster', 'nsls2pluto'):
                req_pakcages += ['mpi4py>=3', 'dill']

            self.distribution.install_requires += req_pakcages
            # ^ These requirements are actually NOT being checked (as of 06/11/2021)
            #   due to the known bug with the custom install via "python setup.py
            #   install" (pip install is fine):
            #       (https://github.com/pypa/setuptools/issues/456)


else:

    # Based on Quantum7's answer on
    # https://stackoverflow.com/questions/18725137/how-to-obtain-arguments-passed-to-setup-py-from-pip-with-install-option

    class CommandMixin(object):
        """"""

        user_options = [
            (f'{facility_name_arg}=', None, 'Facility name for remote running')
        ]

        def initialize_options(self):
            """"""
            super().initialize_options()
            self.facility_name = None

            if False:
                import pprint

                print('*************************************')
                # pprint.pprint(dir(self))
                # pprint.pprint(dir(self.distribution))
                pprint.pprint(self.distribution.packages)
                print('*************************************')

        def _change_prog_name(self):

            if False:
                import pprint

                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                pprint.pprint(dir(self.distribution))
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        def finalize_options(self):
            """"""

            super().finalize_options()

            facility_name = self.facility_name

            if facility_name is None:
                raise ValueError(f'Required arg "--{facility_name_arg}" is missing')

            print(f'Specified facility name for remote runs is "{facility_name}"')

            available_facility_names = [
                'local',
                'nsls2apcluster',
                'nsls2cr',
                'nsls2pluto',
            ]
            if facility_name not in available_facility_names:
                print('* Only the following facility names are available:')
                print('      ' + ', '.join(available_facility_names))
                raise ValueError(
                    'Specified facility_name "{}" is not available.'.format(
                        facility_name
                    )
                )

            self._change_prog_name()

            self._update_json_files_with_facility_specific_info()

            self._add_facility_specific_packages()
            self._add_facility_specific_package_dir()
            self._add_facility_specific_package_data()
            self._add_facility_specific_install_requires()

        def run(self):
            """"""
            # print(self.facility_name)
            super().run()

        def _update_json_files_with_facility_specific_info(self):
            """"""

            facility_name = self.facility_name

            this_folder = os.path.dirname(os.path.abspath(__file__))

            facility_json_filepath = os.path.join(
                this_folder, 'src', 'pyelegant', facility_json_filename
            )
            with open(facility_json_filepath, 'w') as f:
                fac_info = {'name': facility_name}
                if facility_name == 'nsls2apcluster':
                    fac_info[
                        'MODULE_LOAD_CMD_STR'
                    ] = 'elegant-latest'  # 'elegant-latest elegant/2020.2.0',
                    fac_info['MPI_COMPILER_OPT_STR'] = ''  # '--mpi=pmi2',
                elif facility_name == 'nsls2pluto':
                    fac_info['MODULE_LOAD_CMD_STR'] = 'accelerator'
                else:
                    assert facility_name in ('local', 'nsls2cr')

                json.dump(fac_info, f)

            with open(
                os.path.join(this_folder, 'src', 'pyelegant', version_filename), 'w'
            ) as f:
                json.dump(version, f)

        def _add_facility_specific_packages(self):

            facility_name = self.facility_name

            packages = []
            for pk in find_packages('src'):
                # Exclude guis/scripts subfolders for other facilities
                if pk.startswith('pyelegant.guis.'):
                    if pk == f'pyelegant.guis.{facility_name}':
                        pk = 'pyelegant.guis'
                    else:
                        if pk.startswith(f'pyelegant.guis.{facility_name}.'):
                            pk = pk.replace(
                                f'pyelegant.guis.{facility_name}.', 'pyelegant.guis.'
                            )
                        else:
                            continue
                elif pk.startswith('pyelegant.scripts.'):
                    if pk == f'pyelegant.scripts.{facility_name}':
                        pk = 'pyelegant.scripts'
                    elif pk == f'pyelegant.scripts.common':
                        pk = 'pyelegant.scripts.common'
                    else:
                        if pk.startswith(f'pyelegant.scripts.{facility_name}.'):
                            pk = pk.replace(
                                f'pyelegant.scripts.{facility_name}.',
                                'pyelegant.scripts.',
                            )
                        else:
                            continue
                packages.append(pk)

            self.distribution.packages = list(set(packages))

            if False:
                import pprint

                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                # pprint.pprint(self.distribution.packages)
                pprint.pprint(dir(self.distribution))
                # pprint.pprint(dir(self))
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        def _add_facility_specific_package_dir(self):

            facility_name = self.facility_name

            package_dir = self.distribution.package_dir

            if facility_name == 'nsls2apcluster':
                facility_gui_folderpath = 'src/pyelegant/guis/nsls2apcluster'
                facility_script_folderpath = 'src/pyelegant/scripts/nsls2apcluster'
            elif facility_name == 'nsls2pluto':
                facility_gui_folderpath = 'src/pyelegant/guis/nsls2pluto'
                facility_script_folderpath = 'src/pyelegant/scripts/nsls2pluto'
            else:
                facility_gui_folderpath = None
                facility_script_folderpath = None

            if facility_gui_folderpath is not None:
                package_dir['pyelegant.guis'] = facility_gui_folderpath
            if facility_script_folderpath is not None:
                package_dir['pyelegant.scripts'] = facility_script_folderpath

        def _add_facility_specific_package_data(self):

            facility_name = self.facility_name

            package_data = self.distribution.package_data

            if facility_name in ('nsls2apcluster', 'nsls2pluto'):
                package_data['pyelegant.guis.cluster_status'] = ['*.ui']
                package_data['pyelegant.guis.genreport_wizard'] = ['*.ui']

        def _add_facility_specific_install_requires(self):

            facility_name = self.facility_name

            req_pakcages = []
            if facility_name in ('nsls2apcluster', 'nsls2pluto'):
                req_pakcages += [
                    'pylatex',
                    'xlsxwriter',
                    'qtpy',
                    'pyqtgraph',
                    'mpi4py>=3',
                    'dill',
                ]

            self.distribution.install_requires += req_pakcages
            # ^ These requirements are actually NOT being checked (as of 06/11/2021)
            #   due to the known bug with the custom install via "python setup.py
            #   install" (pip install is fine):
            #       (https://github.com/pypa/setuptools/issues/456)

    class InstallCommand(CommandMixin, install):
        user_options = getattr(install, 'user_options', []) + CommandMixin.user_options

    class SdistCommand(CommandMixin, sdist):
        user_options = getattr(sdist, 'user_options', []) + CommandMixin.user_options

        def _change_prog_name(self):
            """"""

            # Modify the tarball name to be "pyelegant-{facility_name}-?.?.?.tar.gz"
            self.distribution.metadata.name += f'-{self.facility_name}'


import sys
import os
import json

# print('sys.argv contents:')
# print(sys.argv)
# print(' ')

com_req_pakcages = ['numpy', 'scipy', 'matplotlib', 'h5py', 'ruamel.yaml']
install_requires = com_req_pakcages

# print('####################################################')
# print(sys.argv)
# print('####################################################')

packages = []
# ^ Facility-specific data Will be added within
#   InstallCommand._find_facility_specific_packages()

package_dir = {
    '': 'src',
    'pyelegant.scripts.common': 'src/pyelegant/scripts/common',
}
# ^ Facility-specific data will be added within InstallCommand.?()

package_data = {
    'pyelegant': [
        facility_json_filename,
        version_filename,
        'Touschek_F_interpolator.pkl',
        '.defns.rpn',
    ]
}
# ^ Facility-specific data will be added within InstallCommand.?()

entry_points = dict(
    console_scripts=[
        'pyele_zip_lte = pyelegant.scripts.ziplte:zip_lte',
        'pyele_unzip_lte = pyelegant.scripts.ziplte:unzip_lte',
    ],
    gui_scripts=[],
)
entry_points['console_scripts'].extend(
    [
        'pyele_report = pyelegant.scripts.common.genreport:main',
        'pyele_slurm_print_queue = pyelegant.scripts.slurmutil:print_queue',
        'pyele_slurm_print_load = pyelegant.scripts.slurmutil:print_load',
        'pyele_slurm_scancel_regex_jobname = pyelegant.scripts.slurmutil:scancel_by_regex_jobname',
    ]
)
entry_points['gui_scripts'].extend(
    [
        # GUI
        'pyele_gui_slurm = pyelegant.guis.cluster_status.main:main',
        'pyele_gui_report_wiz = pyelegant.guis.genreport_wizard.main:main',
    ]
)

setup(
    name=program_name,
    version=version,
    packages=packages,
    package_dir=package_dir,
    # include_package_data = True,
    package_data=package_data,
    zip_safe=False,
    description='Python Interface to ELEGANT',
    author='Yoshiteru Hidaka',
    maintainer='Yoshiteru Hidaka',
    maintainer_email='yhidaka@bnl.gov',
    entry_points=entry_points,
    install_requires=install_requires,
    # cmdclass={'install': InstallCommand},
    cmdclass={'install': InstallCommand, 'sdist': SdistCommand},
)
