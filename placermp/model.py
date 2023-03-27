from typing import TypeVar

import numpy
from numpy import ndarray, array

from placermp.settings import PARAMETERS
from placermp.utils import arr_2d_as_str, AppException, calculate_n_to_q_list, \
    get_arr_correlation, replace_arr_n_to_q, mp_to_data, arr_scale

RefModel = TypeVar('RefModel', bound='ReferenceModel')


class Model:
    """
    Abstract base Model
    """
    def __init__(self, params: dict[str, int | float | str],
                 data_f: dict[str, ndarray[float]],
                 data_q: dict[str, ndarray[float]],
                 data_n: dict[str, ndarray[str]]):
        """
        :param params: parameters
        :param data_f: forecasted data
        :param data_q: data in quantitative scale
        :param data_n: data in nominal scale
        """
        if not data_q and not data_n:
            raise AppException('Model without data')
        self.params = params
        self.data_f = data_f
        self.data_q = data_q
        self.data_n = data_n

    def get_rows(self) -> int:
        """
        Returns the number of rows
        :return: number of rows
        """
        return self.params['END_ROW'] - self.params['START_ROW'] + 1

    def get_cols(self) -> int:
        """
        Returns the number of columns
        :return: number of columns
        """
        return self.params['END_COLUMN'] - self.params['START_COLUMN'] + 1

    def get_forecast(self, f_param_title: str, mp_parameters: list[list[str]],
                     ref_model: RefModel) -> ndarray[float]:
        """
        Returns forecast by multiplicative method
        :param f_param_title: forecasted parameter title
        :param mp_parameters: parameters for calculating the multiplicative
        coefficient
        :param ref_model: Reference Model
        :return: forecast data
        """
        if not mp_parameters[0] and not mp_parameters[1]:
            raise AppException("Empty MP-parameter list")
        n_to_q_table = ref_model.get_n_to_q_table(f_param_title)
        test_mp_c = self.get_mp_coefficients(mp_parameters, n_to_q_table,
                                             ref_model)
        ref_mp_c = ref_model.get_mp_coefficients(mp_parameters, n_to_q_table,
                                                 ref_model)
        return mp_to_data(test_mp_c, ref_mp_c, ref_model.data_f[f_param_title])

    def get_mp_coefficients(self, mp_parameters: list[list[str]],
                            n_to_q_table: dict[str, dict[str, float]],
                            ref_model: RefModel) -> ndarray[float]:
        data_q_from_n = {}
        data_q_from_n_ref = {}
        for k in n_to_q_table:
            if k not in self.data_n:
                raise AppException(
                    'Parameter "{}" in the nominal scale unknown for test '
                    'model'.format(k))
            average_list = n_to_q_table[k]
            data_q_from_n[k] = array([[average_list[val] for val in row]
                                      for row in self.data_n[k]])
            data_q_from_n_ref[k] = array([[average_list[val] for val in row]
                                          for row in ref_model.data_n[k]])
        data_q = self.data_q | data_q_from_n
        data_q_ref = ref_model.data_q | data_q_from_n_ref
        for k in data_q:
            data_q[k] = arr_scale(data_q[k], [[min(numpy.min(data_q[k]),
                                                   numpy.min(data_q_ref[k]))],
                                              [max(numpy.max(data_q[k]),
                                                   numpy.max(data_q_ref[k]))]])
        rows = self.get_rows()
        cols = self.get_cols()
        f_arr_list = []
        for i in range(rows):
            row = []
            for j in range(cols):
                num = 1
                for k in mp_parameters[0]:
                    num *= data_q[k][i, j]
                if not mp_parameters[1]:
                    row.append(num)
                    continue
                den = 1
                for k in mp_parameters[1]:
                    if mp_parameters[0]:
                        den *= data_q[k][i][j]
                        den += num
                    else:
                        den *= (1 - data_q[k][i][j])
                if den == 0.0 and mp_parameters[0]:
                    row.append(0.5)
                else:
                    row.append(num / den if mp_parameters[0] else den)
            f_arr_list.append(row)
        f_arr = array(f_arr_list)
        return array(f_arr)

    def __repr__(self) -> str:
        s = '#PARAMETERS:\n'
        for k, val in self.params.items():
            if type(val) == str:
                s += '{}=f:{}\n'.format(k, val)
                continue
            s += '{}={}\n'.format(k, val)
        for k, val in self.data_f.items():
            if s:
                s += '\n'
            s += k + '*\n'
            s += arr_2d_as_str(val, '{:.2f}\t')
        for k, val in self.data_q.items():
            if s:
                s += '\n'
            s += k + '\n'
            s += arr_2d_as_str(val, '{:.2f}\t')
        for k, val in self.data_n.items():
            if s:
                s += '\n'
            s += k + ' _NOMINAL\n'
            s += arr_2d_as_str(val, '{} ')
        return s.rstrip()


class ReferenceModel(Model):
    """
    Reference (training) Model
    """
    def __init__(self, params: dict[str, int | float | str],
                 data_f: dict[str, ndarray[float]],
                 data_q: dict[str, ndarray[float]],
                 data_n: dict[str, ndarray[str]]):
        """
        :param params: parameters
        :param data_f: forecasted data
        :param data_q: data in quantitative scale
        :param data_n: data in nominal scale
        """
        super(ReferenceModel, self).__init__(params, data_f, data_q, data_n)
        if not self.data_f:
            raise AppException('Reference model without forecasted data')

    def get_n_to_q_table(self, f_param_title: str) -> dict[str, dict[str, float]]:
        """
        Returns coefficient table for converting values in the nominal scale to
        values in the quantitative scale
        :param f_param_title: forecasted parameter title
        :return: dict[nominal scale parameter title] => dict[class => coefficient]
        """
        if f_param_title not in self.data_f:
            raise AppException('Unknown forecasted parameter "{}"'.
                               format(f_param_title))
        base_arr = self.data_f[f_param_title]
        n_to_q_table = {}
        for k_n, val_n in self.data_n.items():
            n_to_q_table[k_n] = calculate_n_to_q_list(val_n, base_arr)
        return n_to_q_table

    def get_correlation_table(self, f_param_title: str,
                              n_to_q_table: dict[str, dict[str, float]]) -> \
            dict[str, (float, float)]:
        """
        Returns table of correlations between forecasted parameter and all
        others
        :param f_param_title: title of the forecasted parameter
        :param n_to_q_table: nominal scale -> quantitative scale table
        :return: dict[param_title] => [correlation coefficient, p-val]]
        """
        if f_param_title not in self.data_f:
            raise AppException('Unknown forecasted parameter "{}"'.
                               format(f_param_title))
        f_arr = self.data_f[f_param_title]
        res = {}
        for k, arr_q in self.data_q.items():
            res[k] = get_arr_correlation(arr_q, f_arr)
        for k, n_to_q_dict in n_to_q_table.items():
            if k not in self.data_n:
                raise AppException('Parameter "{}" in the nominal scale unknown '
                                   'for reference model'.format(k))
            try:
                n_to_q_arr = replace_arr_n_to_q(self.data_n[k], n_to_q_dict)
                res[k] = get_arr_correlation(n_to_q_arr, f_arr)
            except KeyError as err:
                raise AppException('Unknown class "{}" in section "{}{}{}"  in '
                                   'reference model'.format(
                                    err, PARAMETERS['IN_SECTION_SEPARATOR'], k,
                                    PARAMETERS['IN_NOMINAL_DATA_POSTFIX']))
        return res


class TestModel(Model):
    """
    Test Model
    """
    def __init__(self, params: dict[str, int | float | str],
                 data_f: dict[str, ndarray[float]],
                 data_q: dict[str, ndarray[float]],
                 data_n: dict[str, ndarray[str]]):
        super(TestModel, self).__init__(params, data_f, data_q, data_n)
        self.data_q_from_n = None
