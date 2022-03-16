import array

import scipy.ndimage

from model import ReferenceModel, TestModel
from utils import AppException


def read_data(file, is_ref):
    """
    Reads data from the file. File structure see in 'ref_data.txt', for example
    :param file: file name
    :param is_ref: boolean True is the data represents the reference model, False
    otherwise
    :return: model object
    """
    data_map = _parse_data_to_map(file)
    if is_ref and 'ORE_CONTENT' not in data_map:
        raise AppException(
            'Incorrect data: reference model without #ORE_CONTENT section')
    params = _read_parameters(data_map.pop('PARAMETERS'))
    rows = int(params['ROWS'])
    cols = int(params['COLUMNS'])
    start_row = int(params['STARTING_ROW']) - 1
    end_row = int(params['END_ROW'])
    start_col = int(params['STARTING_COLUMN']) - 1
    end_col = int(params['END_COLUMN'])
    data_q = {}  # quantitative scale
    data_n = {}  # nominal scale
    if 'RELIEF' in data_map:  # relief data should be one row and one column
        if end_row == rows:  # more on each side od the data matrix
            end_row = rows - 1
            rows -= 1
        if start_row == 0:
            start_row = 1
            rows -= 1
        if end_col == cols:
            end_col = cols - 1
            cols -= 1
        if start_col == 0:
            start_col = 1
            cols -= 1
        data_q['RELIEF'] = _read_array(float, 'd', data_map.pop('RELIEF'),
                                       start_row - 1, end_row + 1,
                                       start_col - 1, end_col + 1, 'RELIEF')
    for k in data_map:
        attr_name = k.split()
        if len(attr_name) > 1:
            if len(attr_name) == 2 and attr_name[1] == 'N':  # nominal scale
                data_n[attr_name[0]] = _read_array(int, 'i', data_map[k],
                                                   start_row, end_row,
                                                   start_col, end_col,
                                                   attr_name[0])
            else:
                raise AppException('Incorrect data title #' + k)
        else:
            data_q[k] = _read_array(float, 'd', data_map[k], start_row, end_row,
                                    start_col, end_col, attr_name)
    meta = {'ROWS_COLS': (start_row, end_row, start_col, end_col)}
    if 'SQUARE_SZ' in params:
        meta['SQUARE_SZ'] = float(params['SQUARE_SZ'])
    return ReferenceModel(meta, data_q, data_n) if is_ref else TestModel(
        meta, data_q, data_n)


def input_mp_parameters(model):
    """
    Requests from the user a list of parameters for MP
    :param model: model for calculating MP
    :return: list containing two lists of parameters name, where the first
    contains parameters with a positive correlation with ore content and the
    second with a negative correlation
    """
    while True:
        try:
            arr_params = model.get_mp_parameters(set(input(
                'Specify parameters for the multiplicative indicator separated '
                'by a space: ').split()))
        except AppException as ex:
            print('Incorrect input. ' + str(ex) + ', try again.')
            continue
        return arr_params


def input_parameter_to_save(model):
    """
    Initiates the dialog with the user to save the model parameter distribution
    scheme in the file
    :param model: model
    """
    ans = input_answer('Data in quantitative or nominal scale', 'q', 'n')
    data = model.data if ans == 'q' else model.data_n
    suf = '' if ans == 'q' else '_n'
    while True:
        param = input('Specify the characteristic (' + ', '.join(data) + '): ')
        for k in data:
            pn = k.lower()
            if pn.startswith(param):
                _save_fig('{0[0]}_{0[1]}__{0[2]}_{0[3]}__{1}{2}'.format(
                    model.meta['ROWS_COLS'], pn, suf), model, data[k])
                return
        print('Incorrect characteristic, try again.')


def input_answer(prompt, ans1, ans2):
    """
    Dialogue with the user to receive one of two response option
    :param prompt: message to user
    :param ans1: first answer option
    :param ans2: second answer option
    """
    prompt += ' ({}/{}) ?: '.format(ans1, ans2)
    while True:
        ans = input(prompt).lower()
        if ans == ans1 or ans == ans2:
            return ans
        print('Incorrect answer, try again.')


def input_smooth(model):
    """
    Requests from the user a list of parameters
    :param model: model
    :return: parameters
    """
    while True:
        try:
            arr_params = model.get_parameters(set(input(
                'Specify parameters for smoothing separated by a space: ').
                                                  split()))
        except AppException:
            print('Incorrect input, try again.')
            continue
        return arr_params


def choose_method():
    """
    Chooses the method
    :return: method name
    """
    while True:
        meth = input('Choose method: multiplicative (mp), linear regression '
                     '(lr) or regression tree (rt): ')
        if meth == 'mp':
            return 'multiplicative'
        elif meth == 'lr':
            return 'linear_regression'
        elif meth == 'rt':
            return 'reg_forest'
        else:
            print('Incorrect input, try again.')


def save_param_by_model(model, data, meth):
    """
    Save the model parameter distribution scheme in the file
    :param model: model
    :param data: data to save
    :param meth: method title
    """
    _save_fig('{0[0]}_{0[1]}__{0[2]}_{0[3]}__{1}'.format(
        model.meta['ROWS_COLS'], meth), model, data)


def print_mp_parameters(mp_params):
    """
    Displays the type of calculation multiplicative parameter
    :param mp_params: list containing two lists of parameters name, where the
    first contains parameters with a positive correlation with ore content and
    the second with a negative correlation
    """
    print(' * '.join(mp_params[0]) + ' / ' + ' * '.join(mp_params[1]))


def _save_fig(file_name, model, arr):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    z = np.array(arr)
    rows, cols = model.get_rows(), model.get_cols()
    z = z.reshape((rows, cols))

    # z = z[:14]

    z = scipy.ndimage.zoom(z, 10)
    z = np.vectorize(lambda val: val if val < 6.0 else 6.0)(z)
    square_sz = model.meta.get('SQUARE_SZ', 1)
    rows_len = rows * square_sz
    cols_len = cols * square_sz
    x = np.arange(0, cols_len * 10, square_sz)
    y = np.arange(rows_len * 10, 0, -square_sz)
    norm = cm.colors.Normalize(vmax=z.max(), vmin=0.0)
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    std_levels = 10

    # cls = ['#ffffff', '#d3d3d3', '#a9a9a9', '#808080', '#545454', '#2c2c2c',
    #        '#000000']
    # cls = ['#ffffff', '#e6e6e6', '#cdcdcd', '#b3b3b3', '#9a9a9a', '#808080',
    #        '#666666', '#4d4d4d', '#333333', '#1e1e1e', '#000000']
    # cs = ax.contourf(x, y, z, [0, 1, 2, 3, 4, 5, 6, 7], norm=norm, colors=cls)
    # cs = ax.contourf(x, y, z, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], norm=norm, colors=cls)

    cs = ax.contourf(x, y, z, levels=std_levels, norm=norm, cmap='Greys')
    fig.colorbar(cs, ax=ax)
    file_name += '.png'
    fig.savefig(file_name, dpi=300)
    print('File \'' + file_name + '\' saved.')


def _parse_data_to_map(file):
    import re
    f = open(file, 'rt', encoding='UTF-8')
    data = f.read()
    sec_sep = re.compile('\\s*#\\s*')
    data = sec_sep.split(data.strip())
    sections = {}
    line_sep = re.compile('\\s*\\n\\s*')
    for section_data in data:
        if section_data:
            section_data_lines = line_sep.split(section_data)
            title = section_data_lines[0].upper()
            if len(section_data_lines) < 2:
                raise AppException('Incorrect section with title ' + title)
            sections[title] = section_data_lines[1:]
    return sections


def _read_parameters(data):
    req_params = {'ROWS', 'COLUMNS', 'STARTING_ROW', 'END_ROW',
                  'STARTING_COLUMN', 'END_COLUMN'}
    params = {}
    for row in data:
        ind = row.index('=')
        param_title = row[0:ind].rstrip().upper()
        try:
            params[param_title] = float(row[ind + 1: len(row)].lstrip())
        except ValueError:
            raise AppException(
                'Incorrect parameter data with title ' + param_title)
    miss_params = req_params - params.keys()
    if miss_params:
        raise AppException('Missing required parameters: ' + str(miss_params))
    if not params['SQUARE_SZ']:
        params['SQUARE_SZ'] = 1.0
    return params


def _read_array(type_f, type_c, data, start_row, end_row, start_col, end_col,
                attr_name):
    arr = []
    lines = data[start_row:end_row]
    for line in lines:
        values = line.split()[start_col:end_col]
        try:
            arr.append(array.array(type_c, [type_f(val) for val in values]))
        except ValueError:
            raise AppException('Incorrect data in section #' + attr_name)
    return arr
