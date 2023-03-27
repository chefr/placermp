from numpy import ndarray, array

from .model import ReferenceModel, TestModel
from .settings import PARAMETERS, CUSTOM_FUNCTIONS
from .utils import AppException, apply_formula, formula_to_function


def read_data(text_data: str, is_ref: bool) -> ReferenceModel | TestModel:
    """
    Reads model from the text data
    :param text_data: input text data
    :param is_ref: boolean True if the data represents the reference model,
    False otherwise
    :return: Model object
    """
    data_map = _parse_data_to_map(text_data)
    params_sec_title = PARAMETERS['IN_PARAMETERS_SECTION_TITLE']
    sec_sep = PARAMETERS['IN_SECTION_SEPARATOR']
    if params_sec_title not in data_map:
        raise AppException('Section "{}{}" not found'.format(sec_sep,
                                                             params_sec_title))
    params = _read_parameters(data_map.pop(params_sec_title))
    if not data_map:
        raise AppException('Empty data')
    data_f = {}  # forecasted
    data_q = {}  # quantitative scale
    data_n = {}  # nominal scale
    add_param_pref = PARAMETERS['IN_ADD_PARAMETER_PREFIX']
    add_param_pref_len = len(add_param_pref)
    fun_pref = PARAMETERS['IN_FUNCTION_PREFIX']
    fun_pref_len = len(fun_pref)
    for k, val in data_map.items():
        if _try_special_section(k, PARAMETERS['IN_FORECAST_DATA_POSTFIX'], val,
                                float, data_f, params) or \
                _try_special_section(k, PARAMETERS['IN_NOMINAL_DATA_POSTFIX'],
                                     val, str, data_n, params):
            continue
        i = 0
        while val[i].startswith(add_param_pref):
            i += 1
        add_params, arr_data = val[:i], val[i:]
        if k in data_q:  # just in case, it seems unattainable
            raise AppException('Section "{}{}" occurs twice'.format(sec_sep, k))
        data_q[k] = _read_data_array(k, arr_data, float, params)
        if add_params:
            arr_full = _read_data_array(
                k + PARAMETERS['IN_FULL_DATA_TITLE_POSTFIX'], arr_data, float,
                None)
            for add_param in add_params:
                header = add_param[add_param_pref_len:].lstrip()
                ind = header.find(fun_pref)
                if ind == -1:
                    raise AppException(
                        'Additional section "{}" without function'.
                        format(add_param))
                formula = header[ind + fun_pref_len:].lstrip()
                if not formula:
                    raise AppException(
                        'Additional section "{}" with empty function'.
                        format(add_param))
                data_q[header[:ind].rstrip()] = apply_formula(
                    arr_full, formula, params['START_ROW'] - 1,
                    params['END_ROW'], params['START_COLUMN'] - 1,
                    params['END_COLUMN'])
    for k in data_f:
        if k in data_q or k in data_n:
            raise AppException('Section "{}{}" occurs twice'.format(sec_sep, k))
    for k in data_n:
        if k in data_q:
            raise AppException('Section "{}{}" occurs twice'.format(sec_sep, k))
    if is_ref:
        return ReferenceModel(params, data_f, data_q, data_n)
    return TestModel(params, data_f, data_q, data_n)


def _parse_data_to_map(text_data: str) -> dict[str, list[str]]:
    import re
    sec_sep = PARAMETERS['IN_SECTION_SEPARATOR']
    sec_sep_re = re.compile('\\s*{}\\s*'.format(sec_sep))
    secs_data = sec_sep_re.split(text_data.strip())
    secs = {}
    line_sep = re.compile('\\s*\\n\\s*')
    for sec_data in secs_data:
        if sec_data:
            sec_data_lines = line_sep.split(sec_data)
            header = sec_data_lines[0]
            sec_header_max_len = PARAMETERS['IN_SECTION_HEADER_MAX_LEN']
            trunc_header = header[:PARAMETERS['IN_SECTION_HEADER_TRUNC']]
            if len(header) > sec_header_max_len:
                raise AppException(
                    'Too long section header (exceeds {} characters) "{}{}..."'.
                    format(sec_header_max_len, sec_sep, trunc_header))
            if len(sec_data_lines) < 2:
                raise AppException('Empty section "{}{}..."'.
                                   format(sec_sep, trunc_header))
            if header in secs:
                raise AppException('Section header "{}{}" occurs twice'.format(
                    sec_sep, header))
            secs[header] = sec_data_lines[1:]
    return secs


def _read_parameters(data: list[str]) -> dict[str, int | float | str]:
    req_params = PARAMETERS['IN_REQ_PARAMETERS']
    int_params = PARAMETERS['IN_INT_PARAMETERS']
    key_val_sep = PARAMETERS['IN_KEY_VALUE_SEPARATOR']
    key_val_sep_len = len(key_val_sep)
    fun_pref = PARAMETERS['IN_FUNCTION_PREFIX']
    fun_pref_len = len(fun_pref)
    params = {}
    for row in data:
        ind = row.find(key_val_sep)
        if ind == -1:
            raise AppException('Empty parameter "{}". Data format: KEY = VALUE'.
                               format(row))
        param_title = row[0:ind].rstrip()
        param_val = row[ind + key_val_sep_len:].lstrip()
        if param_val.startswith(fun_pref):
            formula = param_val[fun_pref_len:].lstrip()
            params[param_title] = formula
            CUSTOM_FUNCTIONS[param_title] = formula_to_function(formula)
            continue
        type_f = int if param_title in int_params else float
        try:
            params[param_title] = type_f(param_val)
        except ValueError:
            raise AppException(
                'Incorrect parameter value "{}". Must be a number of a function'
                ' in format: name = f:function_expression'.format(row))
    miss_params = req_params - params.keys()
    if miss_params:
        raise AppException('Missing required parameter(s): ' + str(miss_params))
    return params


def _read_data_array(title: str, data: list[str], type_f: type,
                     params: dict[str, int | float | str] | None) -> \
        ndarray[float | int | str]:
    start_row = int(params['START_ROW']) - 1 if params is not None else 0
    end_row = int(params['END_ROW']) if params is not None else len(data)
    start_col = int(params['START_COLUMN']) - 1 if params is not None else 0
    end_col = int(params['END_COLUMN']) if params is not None else None
    sec_sep = PARAMETERS['IN_SECTION_SEPARATOR']
    if start_row >= end_row:
        raise AppException('Incorrect rows interval {} - {}'.
                           format(start_row + 1, end_row))
    if end_col is not None and start_col >= end_col:
        raise AppException('Incorrect columns interval {} - {}'.
                           format(start_col + 1, end_col))
    if len(data) < end_row:
        raise AppException('Incorrect data in section {}{}. Too few rows'.
                           format(sec_sep, title))
    arr = []
    lines = data[start_row:end_row]
    for line in lines:
        if end_col is None:
            line = line.replace('-', 'nan')
        values = line.split()
        if end_col:
            if len(values) < end_col:
                raise AppException(
                    'Incorrect data in section {}{}. Too few columns'.
                    format(sec_sep, title))
            values = values[start_col:end_col]
        try:
            arr.append([type_f(val) for val in values])
        except ValueError:
            raise AppException('Incorrect values in section {}{}'.
                               format(sec_sep, title))
    return array(arr)


def _try_special_section(header: str, mark: str, data: list[str], type_f: type,
                         data_dict: dict[str, ndarray[float | int | str]],
                         params: dict[str, int | float | str]) -> bool:
    if header.endswith(mark):
        title = header[:-len(mark)].rstrip()
        if title in data_dict:
            raise AppException('Section "{}{}" occurs twice'.format(
                PARAMETERS['IN_SECTION_SEPARATOR'], title))
        data_dict[title] = _read_data_array(title, data, type_f, params)
        return True
    return False
