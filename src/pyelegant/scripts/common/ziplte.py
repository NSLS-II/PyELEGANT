import argparse

import pyelegant as pe

def zip_lte():
    """"""

    parser = argparse.ArgumentParser(description=(
        'Zip (bundle & compress) an LTE file with supplementary files for '
        'easy/robust transfer of the LTE file and its associated files.'))

    parser.add_argument(
        'lte_filepath', metavar='lte_file', type=str,
        help=('File path to the main LTE file that will be bundled with its '
              'required supplementary files like kickmap files.'))

    parser.add_argument(
        'ltezip_filepath', metavar='ltezip_file', type=str,
        help=('File path to the zipped file (*.ltezip) that will be generated '
              'with all the contents of the required files (e.g., LTE and '
              'kickmaps) for the lattice defined by the specified LTE file.'))

    parser.add_argument(
        '--used-beamline', '-b', type=str, default='',
        help='Beamline name to be used for the specified LTE file',
        dest='used_beamline_name')

    parser.add_argument(
        '--comment', '-c', type=str, default='',
        help='Comment to be added at the top of the LTE file generated when unzipping.',
        dest='comment')

    args = parser.parse_args()

    LTE = pe.ltemanager.Lattice(
        LTE_filepath=args.lte_filepath,
        used_beamline_name=args.used_beamline_name)

    LTE.zip_lte(args.ltezip_filepath, header_comment=args.comment)

def unzip_lte():
    """"""

    parser = argparse.ArgumentParser(description=(
        'Unzip (uncompress & extract) an "zipped" file that contains an LTE file '
        'and its supplementary files.'))

    parser.add_argument(
        'ltezip_filepath', metavar='ltezip_file', type=str,
        help=('File path to the zipped file (*.ltezip) that holds the contents '
              'of an LTE file and its required files (e.g., kickmaps)'))

    parser.add_argument(
        '--lte-filepath', '-l', type=str, default='', metavar='lte_file',
        help=('File path to the main LTE file that will be generated after '
              'unzipping. If not specified, the new LTE file will be created in '
              'the current directory with the same file name as the original '
              'LTE file.'),
        dest='lte_filepath')

    parser.add_argument(
        '--use-rel-path-for-supple-files', '-r', action='store_false',
        help=('Use relative paths to the supplementary files in the generated '
              'LTE file, if specified'),
        dest='use_abs_paths_for_suppl_files')

    parser.add_argument(
        '--overwrite-lte', action='store_true',
        help=('Overwrite the LTE file if the specified path exists. If not '
              'specified, the existing LTE file will not be overwritten.'),
        dest='overwrite_lte')

    parser.add_argument(
        '--overwrite-suppl', action='store_true',
        help=('Overwrite the supplementary files associated with the LTE file '
              'if they exist. If not specified, the existing files will not be '
              'overwritten.'),
        dest='overwrite_suppl')

    parser.add_argument(
        '--suppl-files-folder', '-s', type=str, default='./lte_suppl',
        help='A path to the folder in which all the supplementary files will be generated.',
        dest='suppl_files_folderpath')

    args = parser.parse_args()

    LTE = pe.ltemanager.Lattice()
    new_LTE_filepath = LTE.unzip_lte(
        args.ltezip_filepath, output_lte_filepath_str=args.lte_filepath,
        suppl_files_folderpath_str=args.suppl_files_folderpath,
        use_abs_paths_for_suppl_files=args.use_abs_paths_for_suppl_files,
        overwrite_lte=args.overwrite_lte, overwrite_suppl=args.overwrite_suppl)

    print(f'\n** Generated LTE file: {new_LTE_filepath}')
