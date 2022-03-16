import array


def norm_to_share(arr):
    """
    Normalizes array data to values that give add up to 1.0
    :param arr: initial array
    :return: normalized array
    """
    if arr:
        div = sum(arr)
        if div:
            return [val / div for val in arr]
    return arr


def norm_2d(arr):
    """
    Normalizes 2d-array data to values from 0.0 to 1.0
    :param arr: initial array
    :return: normalized array
    """
    min_val = arr[0][0]
    max_val = min_val
    for row in arr:
        row_min = min(row)
        if row_min < min_val:
            min_val = row_min
        row_max = max(row)
        if row_max > max_val:
            max_val = row_max
    d_max_min = max_val - min_val
    if d_max_min:
        return [[(val - min_val) / d_max_min for val in row] for row in arr]
    return arr


def norm(arr):
    """
    Normalizes array data to values from 0.0 to 1.0
    :param arr: initial array
    :return: normalized array
    """
    if arr:
        min_val = min(arr)
        d_max_min = max(arr) - min_val
        if d_max_min:
            return [(val - min_val) / d_max_min for val in arr]
        div = len(arr)
        return [val / div for val in arr]


def arr_2d_as_str(arr_2d, template):
    """
    Returns 2-d array as string
    :param arr_2d: 2-d array
    :param template: template to display
    :return: string representation of 2-d array
    """
    s = ''
    for arr in arr_2d:
        s += _arr_as_str(arr, template).rstrip()
        s += '\n'
    return s


def calculate_smooth(arr):
    """
    Calculates smoothed array
    :param arr: Array
    :return: Smoothed array
    """
    """Smooths the array"""
    smooth = []
    for y, row in enumerate(arr):
        smooth.append(array.array('d', _calculate_smooth_row(y, row, arr)))
    return smooth


def calculate_balance(relief):
    """
    Calculates balance
    :param relief: relief
    :return: balance
    """
    balance = []
    last = len(relief) - 1
    for x, row in enumerate(relief):
        if x == 0 or x == last:
            continue
        balance.append(array.array('d', _calculate_balance_row(x, row, relief)))
    return balance


def calculate_d_heights(relief):
    """
    Calculates d-heights
    :param relief: relief
    :return: d-heights
    """
    d_heights = []
    last = len(relief) - 1
    for x, row in enumerate(relief):
        if x == 0 or x == last:
            continue
        d_heights.append(array.array('d', _calculate_d_heights_row(x, row, relief)))
    return d_heights


def calculate_average(arr, base_arr):
    """
    Calculates average values base_arr for each arr element representing the
    nominal scale
    :param arr: Array with elements, represents the nominal scale
    :param base_arr: Base array values
    :return: Array with average values
    """
    max_val = _max_arr_2d(arr)
    sec_num = max_val + 1
    res_arr = [0.0] * sec_num
    base_count = [0] * sec_num
    i, j = 0, 0
    for row in arr:
        for val in row:
            res_arr[val] += base_arr[i][j]  # sum in section
            base_count[val] += 1
            j += 1
        i += 1
        j = 0
    for i in range(sec_num):
        if base_count[i] != 0:
            res_arr[i] = res_arr[i] / base_count[i]  # average
    return res_arr


def corr_2d(arr1, arr2):
    """
    Calculates correlation between two parameters
    :param arr1: first parameter
    :param arr2: second parameter
    :return: Correlation
    """
    from scipy.stats.stats import pearsonr
    return pearsonr([row[i] for row in arr1 for i in range(len(row))],
                    [row[i] for row in arr2 for i in range(len(row))])


def rmse(actual, forecast):
    """
    Calculates RMSE
    :param actual: actual data
    :param forecast: forecast data
    :return: RMSE
    """
    n = len(actual)
    return (sum(
        [(actual[i] - forecast[i]) ** 2 for i in range(n)]) / n) ** 0.5


def rmse_2d(actual, forecast):
    """
    Calculates RMSE
    :param actual: actual 2-d array data
    :param forecast: forecast 2-d array data
    :return: RMSE
    """
    mse = 0
    n = 0
    for i in range(len(actual)):
        cols = len(actual[i])
        n += cols
        for j in range(cols):
            mse += (actual[i][j] - forecast[i][j]) ** 2
    return (mse / n) ** 0.5


def mape(actual, forecast):
    """
    Calculates MAPE
    :param actual: actual data
    :param forecast: forecast data
    :return: MAPE
    """
    n = len(actual)
    return sum([abs(
        (actual[i] - forecast[i]) / actual[i]) if actual[i]
                else 0.0 for i in range(n)]) / n


def mape_2d(actual, forecast):
    """
    Calculates MAPE
    :param actual: actual 2-d array data
    :param forecast: forecast 2-d array data
    :return: MAPE
    """
    mse = 0
    n = 0
    for i in range(len(actual)):
        cols = len(actual[i])
        n += cols
        for j in range(cols):
            if actual[i][j]:
                mse += abs((actual[i][j] - forecast[i][j]) / actual[i][j])
    return mse / n


def scale_2d(arr, ref_arr_x, ref_arr_y):
    """
    Reflects the arr data to ref_arr_y data by linear regression
    :param arr: initial array
    :param ref_arr_x: reference array data
    :param ref_arr_y: reference array trading values
    :return: list of arr trading values
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    x_train = [row[i] for row in ref_arr_x for i in range(len(row))]
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = [row[i] for row in ref_arr_y for i in range(len(row))]
    model = LinearRegression()
    model.fit(x_train, y_train)
    x_test_rows = len(arr)
    x_test_cols = len(arr[0])
    x_test = [row[i] for row in arr for i in range(len(row))]
    x_test = np.array(x_test).reshape(-1, 1)
    x_test = model.predict(x_test)
    return x_test.reshape(x_test_rows, x_test_cols).tolist()


class AppException(Exception):
    """
    Application base exception class
    """
    def __init__(self, message):
        super().__init__(message)


def _calculate_smooth_row(y, row, arr):
    smooth_row = []
    sqrt2 = 2.0 ** 0.5
    sqrt2_den = 1 / sqrt2
    last_y = len(arr) - 1
    last_x = len(row) - 1
    for x, val in enumerate(row):
        num = val * 1
        den = 1
        if x > 0:
            den += 1
            num += arr[y][x - 1]
            if y > 0:
                den += sqrt2_den
                num += arr[y - 1][x - 1] / sqrt2
            if y < last_y:
                den += sqrt2_den
                num += arr[y + 1][x - 1] / sqrt2
        if x < last_x:
            den += 1
            num += arr[y][x + 1]
            if y > 0:
                den += sqrt2_den
                num += arr[y - 1][x + 1] / sqrt2
            if y < last_y:
                den += sqrt2_den
                num += arr[y + 1][x + 1] / sqrt2
        if y > 0:
            den += 1
            num += arr[y - 1][x]
        if y < last_y:
            den += 1
            num += arr[y + 1][x]
        smooth_row.append(num / den)
    return smooth_row


def _calculate_balance_row(x, row, relief):
    balance_row = []
    sqrt2 = 2.0 ** 0.5
    c = 4.0 + 4.0 / sqrt2
    last = len(row) - 1
    for y, height in enumerate(row):
        if y == 0 or y == last:
            continue
        n_h = relief[x][y - 1]
        ne_h = relief[x + 1][y - 1]
        e_h = relief[x + 1][y]
        se_h = relief[x + 1][y + 1]
        s_h = relief[x][y + 1]
        sw_h = relief[x - 1][y + 1]
        w_h = relief[x - 1][y]
        nw_h = relief[x - 1][y - 1]
        balance_row.append(n_h + e_h + s_h + w_h +
                           (ne_h + se_h + nw_h + sw_h) / sqrt2 -
                           c * relief[x][y])
    return balance_row


def _calculate_d_heights_row(x, row, relief):
    d_heights_row = []
    sqrt2 = 2 ** 0.5
    for y, height in enumerate(row):
        if y == 0 or y == len(row) - 1:
            continue
        h = relief[x][y]
        d_n_h = h - relief[x][y - 1]
        d_ne_h = (h - relief[x + 1][y - 1]) / sqrt2
        d_e_h = h - relief[x + 1][y]
        d_se_h = (h - relief[x + 1][y + 1]) / sqrt2
        d_s_h = h - relief[x][y + 1]
        d_sw_h = (h - relief[x - 1][y + 1]) / sqrt2
        d_w_h = h - relief[x - 1][y]
        d_nw_h = (h - relief[x - 1][y - 1]) / sqrt2
        max_d_h = max(d_n_h, d_ne_h, d_e_h, d_se_h, d_s_h, d_sw_h, d_w_h, d_nw_h)
        min_d_h = min(d_n_h, d_ne_h, d_e_h, d_se_h, d_s_h, d_sw_h, d_w_h, d_nw_h)
        d_heights_row.append((max_d_h - min_d_h) / h)
    return d_heights_row


def _arr_as_str(arr, template):
    s = ''
    for val in arr:
        s += template.format(val)
    return s


def _max_arr_2d(arr_2d):
    max_val = arr_2d[0][0]
    for arr in arr_2d:
        for val in arr:
            if max_val < val:
                max_val = val
    return max_val


# def _replace(arr, rep_table):
#     return [rep_table[val] for val in arr]
