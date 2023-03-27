import re
from collections import defaultdict
from typing import Callable

from numpy import ndarray, array
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from placermp.settings import PARAMETERS, CUSTOM_FUNCTIONS


class AppException(Exception):
    """
    Application base exception class
    """
    def __init__(self, message: str):
        super().__init__(message)


def apply_formula(arr: ndarray[float], formula: str, start_row: int,
                  end_row: int, start_col: int, end_col: int) -> \
        ndarray[float]:
    """
    Applies formula to map data
    :param arr: initial data array
    :param formula: formula
    :param start_row: starting row
    :param end_row: end row
    :param start_col: starting column
    :param end_col:  end column
    :return: Mapped data
    """
    f = formula_to_function(formula)
    res = []
    try:
        for i in range(start_row, end_row):
            row = []
            for j in range(start_col, end_col):
                row.append(f(arr, i, j))
            res.append(row)
    except (TypeError, IndexError) as err:
        raise AppException('Incorrect formula "{}". {}'.format(formula, err))
    return array(res)


def formula_to_function(formula: str) -> Callable:
    """
    Creates a function to calculate a value from a formula
    :param formula: formula
    :return: function
    """
    _f = formula.replace(' ', '')
    if _f in CUSTOM_FUNCTIONS:
        return CUSTOM_FUNCTIONS[_f]
    illegal_chs = re.findall(r'[^()\[\]\w+-/*%]+', _f)
    if illegal_chs:
        raise AppException('Incorrect formula "{}". Invalid characters found: {}'.
                           format(formula, illegal_chs))
    ids = set(re.findall(r'[^\W0-9]\w*', _f))
    constants = PARAMETERS['CONSTANTS']
    for _id in ids:
        if _id in constants:
            _f = _f.replace(_id, str(constants[_id]))
        elif _id not in PARAMETERS['DEFAULT_FUNCTIONS']:
            raise AppException(
                'Incorrect formula "{}". Unknown identifier "{}"'.
                format(formula, _id))
    _f = re.sub(r'\[(-?\d+),(-?\d+)]', r'arr[i+\1,j+\2]', _f)
    _f = _f.replace('+-', '-').replace('+0', '').replace('-0', '')
    funcs = []
    _f = 'import math\ndef func(arr, i, j):\n    return {}\n' \
         'funcs.append(func)'.format(_f)
    try:
        exec(_f)
    except SyntaxError as err:
        raise AppException('Incorrect formula "{}". {}'.format(formula, err))
    return funcs[0]


def calculate_n_to_q_list(arr, base_arr) -> dict[str, float]:
    """
    Calculates average values by base_arr for each arr element
    :param arr: data in nominal scale
    :param base_arr: Base data
    :return: Data in quantitative scale (with average values)
    """
    class_sum = defaultdict(lambda: 0.0)
    class_count = defaultdict(lambda: 0)
    i, j = 0, 0
    for row in arr:
        for val in row:
            class_sum[val] += base_arr[i][j]
            class_count[val] += 1
            j += 1
        i += 1
        j = 0
    for k in class_sum:
        class_sum[k] /= class_count[k]
    return dict(class_sum)


def replace_arr_n_to_q(arr_n: ndarray[str], map_table: dict[str, float]) -> \
        ndarray[float]:
    """
    Replaces nominal values from the arr_n to quantitative values from a
    map_table
    :param arr_n: Initial array in the nominal scale
    :param map_table: substitution dictionary
    :return: Array in the quantitative scale
    """
    return array([[map_table[val] for val in row] for row in arr_n])


def get_arr_correlation(arr1: ndarray[float], arr2: ndarray[float]) -> (float, float):
    """
    Calculates pearson correlation between two arr
    :param arr1: first arr
    :param arr2: second arr
    :return: Correlation
    """
    from scipy.stats import pearsonr
    c = pearsonr(arr1.flatten(), arr2.flatten())
    return c[0], c[1]


def get_mp_parameter_list(parameters: list[str],
                          corr_table: dict[str, (float, float)]) -> \
        list[list[str]]:
    """
    Returns structured list of parameters for MO forecast
    :param parameters: Parameter list
    :param corr_table: table of correlations between forecasted parameter and
    all others
    :return: list[list of parameters with positive correlation in the table,
    list of parameters with negative correlation]
    """
    res = [[], []]
    for param in parameters:
        if param not in corr_table:
            raise AppException(
                'Parameter "{}" not found in the correlation table'.
                format(param))
        ind = 0 if corr_table[param][0] > 0 else 1
        res[ind].append(param)
    return res


def mp_to_data(test_arr_c: ndarray[float], ref_arr_c: ndarray[float],
               data_f: ndarray[float]) -> ndarray[float]:
    """
    Maps the placermp coefficients to forecasted data
    :param test_arr_c: forecasted placermp coefficients for test model
    :param ref_arr_c: forecasted placermp coefficients for reference model
    :param data_f: reference model forecast data
    :return: Forecasted data
    """                  '#TECTONIC _NOMINAL\n' \
                  '1 1 1 1\n' \
                  '1 1 2 1\n' \
                  '2 2 2 1\n' \
                  '2 2 2 1\n' \
                  '1 2 2 2\n' \
                  '#FACIES _NOMINAL\n' \
                  '1 1 1 1\n' \
                  '2 2 1 1\n' \
                  '3 3 2 1\n' \
                  '2 3 3 2\n' \
                  '2 3 2 1'
    model = LinearRegression()
    shape = test_arr_c.shape
    model.fit(ref_arr_c.reshape(-1, 1), data_f.reshape(-1, 1))
    return model.predict(test_arr_c.reshape(-1, 1)).reshape(shape)


def arr_scale(arr: ndarray[float], min_max: list[list[float]]) -> \
        ndarray[float]:
    """
    Scale array to values from 0.0 to 1.0
    :param arr: initial array
    :param min_max: [[min value], [max_value]] to scale
    :return: scaled array
    """
    scaler = MinMaxScaler()
    scaler.fit(min_max)
    shape = arr.shape
    return scaler.transform(arr.reshape(-1, 1)).flatten().reshape(shape)


def mape_arr(arr_actual, arr_forecast):
    """
    Calculates means absolute percentage error (MAPE) for 2-d arrays. If actual
    value == 0.0 this value is considered as a forecasted and vice versa
    :param arr_actual: actual 2-d array data
    :param arr_forecast: forecasted 2-d array data
    :return: MAPE
    """
    mse = 0
    n = 0
    for i in range(len(arr_actual)):
        cols = len(arr_actual[i])
        n += cols
        for j in range(cols):
            if arr_actual[i][j]:
                mse += abs((arr_actual[i][j] - arr_forecast[i][j]) /
                           arr_actual[i][j])
            elif arr_forecast[i][j]:
                mse += abs((arr_actual[i][j] - arr_forecast[i][j]) /
                           arr_forecast[i][j])
    return mse / n


def arr_2d_as_str(arr_2d, template):
    """
    Returns 2-d array as string
    :param arr_2d: 2-d array
    :param template: template to display
    :return: string representation of 2-d array
    """
    s = ''
    for row in arr_2d:
        s += _row_as_str(row, template).rstrip()
        s += '\n'
    return s


def _row_as_str(row, template):
    s = ''
    for val in row:
        s += template.format(val)
    return s
