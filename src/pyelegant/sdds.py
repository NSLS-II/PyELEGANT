from typing import Optional
import os, sys
from pathlib import Path
from subprocess import Popen, PIPE
import re
import numpy as np
import tempfile
import shlex
import collections

#----------------------------------------------------------------------
def strfind(string, pattern):
    """"""

    return [s.start() for s in
            re.finditer(pattern, string)]

#----------------------------------------------------------------------
def str2num(string):
    """"""

    if isinstance(string,str):
        return np.array([float(s) for s in string.split()])
    elif isinstance(string,list):
        string_list = string
        array = [[] for i in string_list]
        for (i, string) in enumerate(string_list):
            array[i] = [float(s) for s in string.split()]
        return np.array(array).flatten()
    else:
        raise TypeError('str2num only accepts a string or a list of strings.')

#----------------------------------------------------------------------
def query(sdds_filepath, suppress_err_msg=False):
    """"""

    p = Popen(['sddsquery', sdds_filepath], stdout=PIPE, stderr=PIPE,
              encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsquery stderr:', error)
        print('sddsquery stdout:', output)

    m = re.search(r'(\d+) columns of data:', output)
    if m is not None:
        nColumns = int(m.group(1))
        column_header = m.group(0)
    else:
        nColumns = 0

    m = re.search(r'(\d+) parameters:', output)
    if m is not None:
        nParams = int(m.group(1))
        param_header = m.group(0)
    else:
        nParams = 0

    column_dict = {}
    if nColumns != 0:
        m = re.search(r'(?<='+column_header+r')[\w\W]+', output)
        column_str_list = m.group(0).split('\n')[3:(3+nColumns)]
        for s in column_str_list:
            ss = re.search(r'([^ ]+) +'*6+'(.+)', s).groups()
            column_dict[ss[0]] = {
                'UNITS': ss[1], 'SYMBOL': ss[2], 'FORMAT': ss[3],
                'TYPE': ss[4], 'FIELD LENGTH': ss[5], 'DESCRIPTION': ss[6]}

        assert len(column_dict) == nColumns

    param_dict = {}
    if nParams != 0:
        m = re.search(r'(?<='+param_header+r')[\w\W]+', output)
        param_str_list = m.group(0).split('\n')[2:(2+nParams)]
        unit_pattern = r'[\w\$\(\)<>\*\^\'/,]+'
        symbol_pattern = r'[\w\$\(\)<>\*\^\'/, ]+'
        type_pattern = r'short|long|float|double|character|string'
        for index, s in enumerate(param_str_list):
            ss = re.search(
                r'([\w\./]+) +({0:s}) +({1:s}) +({2:s}) +(.+)'.format(
                    unit_pattern, symbol_pattern, type_pattern), s).groups()
            param_dict[ss[0]] = {'UNITS': ss[1], 'SYMBOL': ss[2].strip(),
                                 'TYPE': ss[3], 'DESCRIPTION': ss[4],
                                 '_index': index}

        assert len(param_dict) == nParams

    # deal with the special cases
    if 'enx0' in param_dict:
        if (param_dict['enx0']['UNITS'] == 'm$be$nc') and \
           (param_dict['enx0']['SYMBOL'].split() == ['$gp$rm','NULL']):
            param_dict['enx0']['UNITS'] = 'm$be$nc $gp$rm'
            param_dict['enx0']['SYMBOL'] = 'NULL'

    return param_dict, column_dict

#----------------------------------------------------------------------
def printout(sdds_filepath, param_name_list=None,
             column_name_list=None, str_format='',
             show_output=False, show_cmd=False,
             suppress_err_msg=False):
    """
    If "str_format" is specified, you must make sure that all the data
    type of the specified paramter or column name list must be the same.

    An example of "str_format" is '%25.16e'.
    """

    if os.name == 'posix':
        newline_char = '\n'
    elif os.name == 'nt':
        newline_char = '\r\n'

    _, column_info_dict = query(sdds_filepath, suppress_err_msg=suppress_err_msg)
    if column_name_list is None:
        column_name_list = list(column_info_dict)

    if column_name_list == []:
        column_option_str = ''
    else:
        column_option_str = '-columns=' + '(' + ','.join(column_name_list) + ')'
        if str_format != '':
            column_option_str += ",format="+str_format

    if param_name_list is None:
        param_info_dict, _ = query(sdds_filepath, suppress_err_msg=suppress_err_msg)
        param_name_list = list(param_info_dict)

    if param_name_list == []:
        param_option_str = ''
    else:
        param_option_str = '-parameters=' + '(' + ','.join(param_name_list) + ')'
        if str_format != '':
            param_option_str += ",format="+str_format

    if (not column_option_str) and (not param_option_str):
        raise ValueError('You must specify at least one of -columns and -parameters.')

    if os.name == 'nt':
        sp = sdds_filepath.split('\\')
        double_backslashed_sdds_filepath = ('\\'*2).join(sp)
        sdds_filepath = double_backslashed_sdds_filepath

    cmd_list = ['sddsprintout', sdds_filepath]
    if param_option_str:
        cmd_list.append(param_option_str)

    if show_cmd:
        print(cmd_list)
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsprintout stderr:', error)
        print('sddsprintout stdout:', output)

    if (param_name_list == []): # or (param_name_list is None):
        param_dict = {}
    else:
        if False: # old version
            eq_pattern = ' = '
            eq_ind =  strfind(output, eq_pattern)

            param_name_ind = [[] for i in param_name_list]
            for i,n in enumerate(param_name_list):
                param_name_ind[i] = strfind(output,newline_char+n+' ')
                if param_name_ind[i] == []:
                    param_name_ind[i] = strfind(output,' '+n+' ')

            param_val_list = [0.]*len(eq_ind)
            for i in range(len(eq_ind)-1):
                start_ind = eq_ind[i]+len(eq_pattern)
                if param_name_ind[i+1] == []: continue
                end_ind   = param_name_ind[i+1][0]
                val_str = output[start_ind:end_ind]
                if val_str.strip() == '1.#QNAN0e+000': # Elegant's NaN for old version (23.1.2)
                    val_str = 'nan'
                try:
                    param_val_list[i] = float(val_str)
                except ValueError:
                    param_val_list[i] = val_str
            start_ind = eq_ind[-1]+len(eq_pattern)

            end_ind = start_ind + output[start_ind:].find(newline_char)
            val_str = output[start_ind:end_ind]
            if val_str.strip() == '1.#QNAN0e+000': # Elegant's NaN for old version (23.1.2)
                val_str = 'nan'
            param_val_list[-1] = float(val_str)

            param_dict = dict(zip(param_name_list,param_val_list))
            #print(param_dict)
        else:
            #param_dict = {}
            param_dict = collections.defaultdict(list)
            for k, v_str in re.findall(
                #'([\w /\(\)\$]+)[ ]+=[ ]+([nae\d\.\+\-]+)[ \n]?',
                '([\w /\(\)\$\^\*\.]+)[ ]*=[ ]*([naife\d\.\+\-]+)[ \n]?',
                output):
                # ^ [n] & [a] is added for digit matching in cases for "nan"
                #   [i] & [f] is added for digit matching in cases for "inf"

                if '(' in k:
                    first_para_ind = k.index('(')
                    k_stripped = k[:first_para_ind].strip()
                else:
                    k_stripped = k.strip()

                # If the parameter name is picking up the previous parameter's
                # non-digit values as characters, remove those here.
                k_stripped = k_stripped.split()[-1]

                if param_info_dict[k_stripped]['TYPE'] == 'double':
                    #param_dict[k_stripped] = float(v_str)
                    param_dict[k_stripped].append(float(v_str))
                elif param_info_dict[k_stripped]['TYPE'] in ('long', 'short'):
                    #param_dict[k_stripped] = int(v_str)
                    param_dict[k_stripped].append(int(v_str))
                elif param_info_dict[k_stripped]['TYPE'] == 'string':
                    pass
                else:
                    raise ValueError(
                        f'Unexpected TYPE: {param_info_dict[k_stripped]["TYPE"]}')

            # Extract string types
            if 'string' in [q_d['TYPE'] for q_d in param_info_dict.values()]:

                ordered_param_name_list = [None] * len(param_name_list)
                for param_name, q_d in param_info_dict.items():
                    ordered_param_name_list[q_d['_index']] = param_name

                for param_name, q_d in param_info_dict.items():

                    if q_d['TYPE'] != 'string':
                        continue

                    _extracted = re.findall(f'{param_name}[ ]*=[ ]*(.+)[=\n]', output)
                    if False: # old version before dealing with SDDS "pages"
                        assert len(_extracted) == 1
                        val = _extracted[0].split('=')[0].strip()

                        try:
                            next_param_name = ordered_param_name_list[
                                ordered_param_name_list.index(param_name)+1]
                            val = val.replace(next_param_name, '').strip()
                        except IndexError:
                            pass

                        #print([param_name, val])
                        param_dict[param_name] = val
                    else:
                        vals = [v.split('=')[0].strip() for v in _extracted]

                        try:
                            next_param_name = ordered_param_name_list[
                                ordered_param_name_list.index(param_name)+1]
                            vals = [v.replace(next_param_name, '').strip()
                                    for v in vals]
                        except IndexError:
                            pass

                        param_dict[param_name] = vals

            len_list = [len(v) for _, v in param_dict.items()]
            assert len(set(len_list)) == 1 # i.e., having save length
            _temp_dict = {}
            if len_list[0] == 1:
                # Only single "page"
                for k, v in param_dict.items():
                    _temp_dict[k] = v[0]
            else:
                # Multiple "pages"
                for k, v in param_dict.items():
                    _temp_dict[k] = v # keep it as a list
            param_dict = _temp_dict


    # Check if all the specified parameters have been correctly extracted
    _extracted_param_names = list(param_dict)
    for name in param_name_list:
        if name not in _extracted_param_names:
            print(f'* ERROR: Paramter "{name}" was not extracted')
    for name in _extracted_param_names:
        if name not in param_name_list:
            print(f'* WARNING: Unrequested Parameter "{name}" was extracted')

    cmd_list = ['sddsprintout', sdds_filepath]
    if column_option_str:
        cmd_list.append(column_option_str)

    #use_comma_delimiter = False
    use_comma_delimiter = True
    if use_comma_delimiter:
        cmd_list.append("-spreadsheet=(delimiter=',')")

    if show_cmd:
        print(cmd_list)
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()
    #if isinstance(output, bytes):
        #output = output.decode('utf-8')
        #error = error.decode('utf-8')
    if error and (not suppress_err_msg):
        print('sddsprintout stderr:', error)
        print('sddsprintout stdout:', output)

    if (column_name_list == []): # or (column_name_list is None):
        column_dict = {}
    else:
        if not use_comma_delimiter:
            column_title_divider_pattern = '---' + newline_char
            column_start_ind = strfind(output, column_title_divider_pattern)
            if column_start_ind != []:
                column_start_ind = column_start_ind[0] + len(column_title_divider_pattern)
                column_data_str = output[column_start_ind:]
                rows_str = column_data_str.splitlines()
                column_dict = dict.fromkeys(column_name_list)
                for col_name in column_name_list:
                    column_dict[col_name] = [[] for i in rows_str]

                col_ind_offset = 0
                row_counter = 0
                for r in rows_str:
                    str_list = [c for c in r.split(' ') if c]

                    for j,st in enumerate(str_list):
                        column_dict[column_name_list[j+col_ind_offset]][row_counter] = st

                    if ( len(str_list)+col_ind_offset ) != len(column_name_list):
                        col_ind_offset += len(str_list)
                    else:
                        col_ind_offset = 0
                        row_counter += 1

                for col_name in column_name_list:
                    column_dict[col_name] = column_dict[col_name][:row_counter] # Make sure to
                    # remove empty elements at the tail
                    #if col_name not in ('ElementName','ElementType'):
                    try:
                        column_dict[col_name] = str2num(column_dict[col_name])
                    except ValueError:
                        pass
        else:
            rows = [s.strip() for s in output.split('\n') if s.strip() != '']

            if False: # old version before dealing with SDDS "pages"
                column_dict = dict.fromkeys(column_name_list)
                for col_name in column_name_list:
                    column_dict[col_name] = []

                col_title_rowind = 1
                for row in rows[(col_title_rowind+1):]:
                    for col_name, v in zip(column_name_list, row.split("','")):
                        column_dict[col_name].append(v.strip())

                for col_name in column_name_list:
                    if column_info_dict[col_name]['TYPE'] == 'double':
                        column_dict[col_name] = str2num(column_dict[col_name])

            else:
                column_dict = collections.defaultdict(list)

                col_title_rowind = 1
                for row in rows[(col_title_rowind+1):]:
                    for col_name, v in zip(column_name_list, row.split("','")):
                        if col_name != v:
                            column_dict[col_name].append(v.strip())
                        else:
                            # "col_name" and "v" is the same, which means, this
                            # is a tile line in the case of having multiple
                            # SDDS "pages". So, skip this line.
                            pass

                _temp_dict = {}
                for col_name in column_name_list:
                    if column_info_dict[col_name]['TYPE'] == 'double':
                        _temp_dict[col_name] = str2num(column_dict[col_name])
                    else:
                        _temp_dict[col_name] = column_dict[col_name]
                column_dict = _temp_dict


    # Check if all the specified columns have been correctly extracted
    _extracted_column_names = list(column_dict)
    for name in column_name_list:
        if name not in _extracted_column_names:
            print(f'* ERROR: Column "{name}" was not extracted')
    for name in _extracted_column_names:
        if name not in column_name_list:
            print(f'* WARNING: Unrequested Column "{name}" was extracted')


    if show_output:
        print(output)
        if error and (not suppress_err_msg):
            print(error)

    return param_dict, column_dict

def sdds2dicts(sdds_filepath, str_format=''):
    """"""

    meta_params, meta_columns = query(sdds_filepath, suppress_err_msg=True)

    meta = {}
    if meta_params:
        meta['params'] = meta_params
    if meta_columns:
        meta['columns'] = meta_columns

    output = {}

    if (meta_params == {} and meta_columns == {}):
        return output, meta

    params, columns = printout(
        sdds_filepath, param_name_list=None, column_name_list=None,
        str_format=str_format, show_output=False, show_cmd=False,
        suppress_err_msg=True)

    if params:
        for _k, _v in params.items():
            if meta['params'][_k]['TYPE'] in ('long', 'short'):
                try:
                    params[_k] = int(_v)
                except TypeError:
                    params[_k] = np.array(_v).astype(int)
                except:
                    sys.stderr.write(f'** key: {_k}\n')
                    sys.stderr.write('** value:\n')
                    sys.stderr.write(str(_v))
                    sys.stderr.write('\n')
                    sys.stderr.flush()
                    raise
        output['params'] = params
    if columns:
        for _k, _v in columns.items():
            if meta['columns'][_k]['TYPE'] in ('long', 'short'):
                columns[_k] = np.array(_v).astype(int)
            else:
                columns[_k] = np.array(_v)
        output['columns'] = columns

    return output, meta

def dicts2sdds(
    sdds_output_pathobj, params=None, columns=None,
    params_units=None, columns_units=None, params_descr=None, columns_descr=None,
    params_symbols=None, columns_symbols=None,
    params_counts=None, columns_counts=None, outputMode='ascii',
    tempdir_path: Optional[str] = None, suppress_err_msg=True):
    """"""

    sdds_output_pathobj = Path(sdds_output_pathobj)
    sdds_output_filepath = str(sdds_output_pathobj)

    tmp = tempfile.NamedTemporaryFile(
        dir=tempdir_path, delete=False, prefix='tmpDicts2sdds_', suffix='.txt')
    plaindata_txt_filepath = str(Path(tmp.name).resolve())
    tmp.close()

    lines = []

    if params is None:

        param_name_list = []
        param_type_list = []

        param_unit_list = None
        param_descr_list = None
        param_symbol_list = None
        param_count_list = None

    else:

        param_name_list = list(params)

        param_type_list = []

        param_unit_list = []
        param_descr_list = []
        param_symbol_list = []
        param_count_list = []

        if params_units is None:
            params_units = {}
        if params_descr is None:
            params_descr = {}
        if params_symbols is None:
            params_symbols = {}
        if params_counts is None:
            params_counts = {}

        for name in param_name_list:
            v = params[name]
            if isinstance(v, float):
                s = f'{v:.16g}'
                param_type_list.append('double')
            elif isinstance(v, (int, np.integer)):
                s = f'{v:d}'
                param_type_list.append('long')
            elif isinstance(v, str):
                s = f'"{v}"'
                param_type_list.append('string')
            else:
                raise ValueError(f'Unexpected data type for paramter "{name}"')

            lines.append(s)

            param_unit_list.append(params_units.get(name, None))
            param_descr_list.append(params_descr.get(name, None))
            param_symbol_list.append(params_symbols.get(name, None))
            param_count_list.append(params_counts.get(name, None))

    if columns is None:

        column_name_list = []
        column_type_list = []

        column_unit_list = None
        column_descr_list = None
        column_symbol_list = None
        column_count_list = None

    else:

        column_name_list = list(columns)

        column_type_list = []

        column_unit_list = []
        column_descr_list = []
        column_symbol_list = []
        column_count_list = []

        if columns_units is None:
            columns_units = {}
        if columns_descr is None:
            columns_descr = {}
        if columns_symbols is None:
            columns_symbols = {}
        if columns_counts is None:
            columns_counts = {}

        for name in column_name_list:
            column_unit_list.append(columns_units.get(name, None))
            column_descr_list.append(columns_descr.get(name, None))
            column_symbol_list.append(columns_symbols.get(name, None))
            column_count_list.append(columns_counts.get(name, None))

        nCols = len(column_name_list)

        # Write the number of rows
        nRows = np.unique([len(columns[name]) for name in column_name_list])
        if len(nRows) == 1:
            nRows = nRows[0]
        else:
            raise ValueError('All the column data must have the same length')
        #
        lines.append(f'\t{nRows:d}')

        M = zip(*[columns[name] for name in column_name_list])

        for iRow, row in enumerate(M):
            tokens = []
            for iCol, v in enumerate(row):
                if isinstance(v, float):
                    s = f'{v:.16g}'
                    if iRow != 0:
                        assert column_type_list[iCol] == 'double'
                    else:
                        column_type_list.append('double')
                elif isinstance(v, (int, np.integer)):
                    s = f'{v:d}'
                    if iRow != 0:
                        assert column_type_list[iCol] == 'long'
                    else:
                        column_type_list.append('long')
                elif isinstance(v, str):
                    s = f'"{v}"'
                    if iRow != 0:
                        assert column_type_list[iCol] == 'string'
                    else:
                        column_type_list.append('string')
                else:
                    raise ValueError(
                        f'Unexpected data type for column "{column_name_list[iCol]}" index {iRow:d}')

                tokens.append(s)

            lines.append(' '.join(tokens))

    with open(plaindata_txt_filepath, 'w') as f:
        f.write('\n'.join(lines))

    plaindata2sdds(
        plaindata_txt_filepath, sdds_output_filepath, outputMode=outputMode,
        param_name_list=param_name_list, param_type_list=param_type_list,
        param_unit_list=param_unit_list, param_descr_list=param_descr_list,
        param_symbol_list=param_symbol_list, param_count_list=param_count_list,
        column_name_list=column_name_list, column_type_list=column_type_list,
        column_unit_list=column_unit_list, column_descr_list=column_descr_list,
        column_symbol_list=column_symbol_list, column_count_list=column_count_list,
        suppress_err_msg=suppress_err_msg)

    try:
        os.remove(plaindata_txt_filepath)
    except IOError:
        pass

def sdds2plaindata(
    sdds_filepath, output_txt_filepath, param_name_list=None, column_name_list=None,
    suppress_err_msg=True):
    """"""

    cmd_list = [
        'sdds2plaindata', sdds_filepath, output_txt_filepath, '"-separator= "',]

    meta_params, meta_columns = query(sdds_filepath, suppress_err_msg=True)

    if param_name_list is None:
        param_name_list = list(meta_params)
    if column_name_list is None:
        column_name_list = list(meta_columns)

    for name in param_name_list:
        cmd_list.append(f'-parameter={name}')
    for name in column_name_list:
        cmd_list.append(f'-column={name}')

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()

    if error and (not suppress_err_msg):
        print('sdds2plaindata stderr:', error)
        print('sdds2plaindata stdout:', output)

def plaindata2sdds(
    input_txt_filepath, sdds_output_filepath, outputMode='ascii',
    param_name_list=None, param_type_list=None, param_unit_list=None,
    param_descr_list=None, param_symbol_list=None, param_count_list=None,
    column_name_list=None, column_type_list=None, column_unit_list=None,
    column_descr_list=None, column_symbol_list=None, column_count_list=None,
    suppress_err_msg=True):
    """"""

    if outputMode not in ('ascii', 'binary'):
        raise ValueError('"outputMode" must be either "ascii" or "binary".')

    cmd_list = [
        'plaindata2sdds', input_txt_filepath, sdds_output_filepath,
        '-inputMode=ascii', f'-outputMode={outputMode}', '"-separator= "',]

    if param_name_list is not None:
        n = len(param_name_list)
        assert n == len(param_type_list)

        if param_unit_list is None:
            param_unit_list = [None] * n
        assert n == len(param_unit_list)

        if param_descr_list is None:
            param_descr_list = [None] * n
        assert n == len(param_descr_list)

        if param_symbol_list is None:
            param_symbol_list = [None] * n
        assert n == len(param_symbol_list)

        if param_count_list is None:
            param_count_list = [None] * n
        assert n == len(param_count_list)

        for name, dtype, unit, descr, symbol, count in zip(
            param_name_list, param_type_list, param_unit_list,
            param_descr_list, param_symbol_list, param_count_list):
            assert dtype in ('string', 'long', 'short', 'double')
            opt = f'-parameter={name},{dtype}'
            if unit is not None:
                opt += f',units="{unit}"'
            if descr is not None:
                assert ',' not in descr
                opt += f',description="{descr}"'
            if symbol is not None:
                opt += f',symbol="{symbol}"'
            if count is not None:
                opt += f',count={count:d}'
            cmd_list.append(opt)

    if column_name_list is not None:
        n = len(column_name_list)
        assert n == len(column_type_list)

        if column_unit_list is None:
            column_unit_list = [None] * n
        assert n == len(column_unit_list)

        if column_descr_list is None:
            column_descr_list = [None] * n
        assert n == len(column_descr_list)

        if column_symbol_list is None:
            column_symbol_list = [None] * n
        assert n == len(column_symbol_list)

        if column_count_list is None:
            column_count_list = [None] * n
        assert n == len(column_count_list)

        for name, dtype, unit, descr, symbol, count in zip(
            column_name_list, column_type_list, column_unit_list,
            column_descr_list, column_symbol_list, column_count_list):
            assert dtype in ('string', 'long', 'short', 'double')
            opt = f'-column={name},{dtype}'
            if unit is not None:
                opt += f',units="{unit}"'
            if descr is not None:
                assert ',' not in descr
                opt += f',description="{descr}"'
            if symbol is not None:
                opt += f',symbol="{symbol}"'
            if count is not None:
                opt += f',count={count:d}'
            cmd_list.append(opt)

    shell_cmd = ' '.join(cmd_list)
    cmd_list = shlex.split(shell_cmd, posix=True)
    #print(cmd_list)

    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    output, error = p.communicate()

    if error and (not suppress_err_msg):
        print('plaindata2sdds stderr:', error)
        print('plaindata2sdds stdout:', output)

