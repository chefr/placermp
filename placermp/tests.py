import unittest
from typing import Callable

import numpy
from numpy import array

from .utils import AppException, calculate_n_to_q_list, get_arr_correlation, \
    replace_arr_n_to_q, mp_to_data, get_mp_parameter_list, arr_scale, mape_arr, \
    apply_formula, formula_to_function
from .inout import read_data


if __name__ == '__main__':
    unittest.main()


class TestIN(unittest.TestCase):

    def test_read_data(self):
        self.assertRaises(AppException, read_data, _DATA_LONG_HEADER, True)
        self.assertRaises(AppException, read_data, _DATA_EMPTY_SECTION, True)
        self.assertRaises(AppException, read_data, _DATA_WITHOUT_PARAMETERS,
                          True)
        self.assertRaises(AppException, read_data, _DATA_EMPTY_PARAMETER, True)
        self.assertRaises(AppException, read_data, _DATA_INCORRECT_KEY_VAL_SEP,
                          True)
        self.assertRaises(AppException, read_data, _DATA_INCORRECT_PARAMETER,
                          True)
        self.assertRaises(AppException, read_data,
                          _DATA_INCORRECT_REQ_PARAMETER, True)
        self.assertRaises(AppException, read_data, _DATA_INCORRECT_FUNC_SEP,
                          True)
        self.assertRaises(AppException, read_data, _DATA_WITHOUT_REQ_PARAMETERS,
                          True)
        self.assertRaises(AppException, read_data, _DATA_EMPTY_DATA, True)
        self.assertRaises(AppException, read_data, _DATA_ADD_SEC_FEW_ROWS, True)
        self.assertRaises(AppException, read_data, _DATA_ADD_SEC_FEW_COLS, True)
        self.assertRaises(AppException, read_data,
                          _DATA_ADD_SEC_INCORRECT_ROW_INTERVAL, True)
        self.assertRaises(AppException, read_data,
                          _DATA_ADD_SEC_INCORRECT_COL_INTERVAL, True)
        self.assertRaises(AppException, read_data, _DATA_ADD_SEC_INCORRECT_VAL,
                          True)
        self.assertRaises(AppException, read_data, _DATA_ADD_SEC_WITHOUT_FUNC,
                          True)
        self.assertRaises(AppException, read_data, _DATA_SEC_HEADER_DUPLICATE,
                          False)
        self.assertRaises(AppException, read_data,
                          _DATA_DIFF_TYPE_SEC_DUPLICATE, False)
        self.assertRaises(AppException, read_data,
                          _DATA_SPECIAL_SEC_DUPLICATE, False)


class TestUtils(unittest.TestCase):

    def test_apply_formula(self):
        arr = array([[1, 2, 2, 1, 1],
                    [2, 2, 3, 2, 2],
                    [2, 3, 3, 1, 1],
                    [2, 2, 4, 2, 3]])
        formula = '[0, 0](1)'
        self.assertRaises(AppException, apply_formula, arr, formula, 1, 3, 1, 4)
        formula = '(max([-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], '\
                  '[1, -1], [1, 0], [1, 1]) - min([-1, -1], [-1, 0], [-1, 1], '\
                  '[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1])) / [0, 0]'
        self.assertTrue(numpy.array_equal(apply_formula(
            arr, formula, 1, 3, 1, 4), array([[1.0, 2.0 / 3.0, 1.0],
                                              [2.0 / 3.0, 1.0, 3.0]])))
        self.assertTrue(numpy.array_equal(
            read_data(_DATA_REF_MODEL_FORMULA, True).data_q['TEST BALANCE'],
            array([[2.228169203286535, 2.228169203286535],
                   [1.2732395447351628, 1.909859317102744],
                   [1.5915494309189535, 1.909859317102744]])))

    def test_formula_to_function(self):
        formula = '[0, 0]&1.0'  # invalid character &
        self.assertRaises(AppException, formula_to_function, formula)
        formula = '[0, 0]pow(1.0, 1.0)'  # illegal function pow
        self.assertRaises(AppException, formula_to_function, formula)
        formula = '2+/2'  # incorrect syntax
        self.assertRaises(AppException, formula_to_function, formula)
        formula = '(max([-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], ' \
                  '[1, -1], [1, 0], [1, 1]) - min([-1, -1], [-1, 0], [-1, 1], ' \
                  '[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1])) / [0, 0]' \
                  '+ 5**0.5%10.0'
        f = formula_to_function(formula)
        self.assertTrue(isinstance(f, Callable))

    def test_calculate_n_to_q_list(self):
        arr_dig = [[1, 2, 2, 1, 1],
                   [2, 2, 3, 2, 2],
                   [2, 3, 3, 1, 1],
                   [2, 2, 4, 2, 3]]
        arr_ch = [['a', 'bb', 'bb', 'a', 'a'],
                  ['bb', 'bb', 'ccc', 'bb', 'bb'],
                  ['bb', 'ccc', 'ccc', 'a', 'a'],
                  ['bb', 'bb', 'dddd', 'bb', 'ccc']]
        base_arr = [[10, 6, 4, 12, 9],
                    [8, 7, 1, 5, 0],
                    [10, 10, 1, 1, 5],
                    [0, 1, 0, 4, 3]]
        self.assertEqual(calculate_n_to_q_list(arr_dig, base_arr),
                         {1: 7.4, 2: 4.5, 3: 3.75, 4: 0.0})
        self.assertEqual(calculate_n_to_q_list(arr_ch, base_arr),
                         {'a': 7.4, 'bb': 4.5, 'ccc': 3.75, 'dddd': 0.0})

    def test_replace_arr_n_to_q(self):
        arr_n = array([['1', '1', '2'], ['2', '2', '3']])
        map_dict_incorrect = {'1': 1.2, '2': 2.4}
        self.assertRaises(KeyError, replace_arr_n_to_q, arr_n,
                          map_dict_incorrect)
        map_dict = {'1': 1.2, '2': 2.4, '3': 3.6}
        self.assertTrue(numpy.array_equal(replace_arr_n_to_q(arr_n, map_dict),
                                          array([[1.2, 1.2, 2.4],
                                                 [2.4, 2.4, 3.6]])))

    def test_mp_to_data(self):
        self.assertTrue(numpy.array_equal(mp_to_data(
            array([[1, 2, 2], [2, 3, 2]]),
            array([[1, 1, 3], [2, 3, 1]]),
            array([[4, 5, 11], [8, 14, 3]])),
            array([[3.9655172413793105, 8.206896551724139, 8.206896551724139],
                   [8.206896551724139, 12.448275862068964, 8.206896551724139]]))
        )

    def test_get_mp_parameter_list(self):
        ref_model = read_data(_DATA_REF_MODEL, True)
        f_param = 'ORE CONTENT'
        corr_table = ref_model.get_correlation_table(
            f_param, ref_model.get_n_to_q_table(f_param))
        self.assertRaises(AppException, get_mp_parameter_list, ['TECTONIC_'],  # unknown parameter
                          corr_table)
        self.assertEqual(get_mp_parameter_list(['TECTONIC'], corr_table),
                         [['TECTONIC'], []])
        self.assertEqual(get_mp_parameter_list(['RELIEF'], corr_table),
                         [[], ['RELIEF']])
        self.assertEqual(get_mp_parameter_list(['TECTONIC', 'FACIES'],
                                               corr_table),
                         [['TECTONIC', 'FACIES'], []])
        self.assertEqual(get_mp_parameter_list(['TECTONIC', 'FACIES', 'RELIEF'],
                                               corr_table),
                         [['TECTONIC', 'FACIES'], ['RELIEF']])

    def test_scale(self):
        self.assertTrue(numpy.array_equal(arr_scale(
            array([[1, 2, 5], [5, 2, 6]]), [[1], [6]]),
            array([[0.0, 0.2, 0.8], [0.8, 0.2, 1.0000000000000002]])))

    def test_mape_arr(self):
        self.assertEqual(mape_arr(array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                                  array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
                         0.0)
        self.assertEqual(mape_arr(array([[0.0, 0.0, 1.9], [3.8, 3.2, 5.0]]),
                         array([[1.0, 0.0, 1.9], [1.9, 4.4, 3.4]])),
                         0.3658333333333333)


class TestModel(unittest.TestCase):

    def test_init(self):
        self.assertRaises(AppException, read_data, _DATA_WITHOUT_DATA, True)
        model = read_data(_DATA_MODEL_TRUNC, True)
        self.assertTrue(numpy.array_equal(model.data_q['RELIEF'],
                        array([[4.0, 6.0], [1.0, 2.0], [2.0, 5.0]])))

    def test_get_mp_coefficients(self):
        ref_model = read_data(_DATA_REF_MODEL, True)
        n_to_q_table = ref_model.get_n_to_q_table('ORE CONTENT')
        self.assertTrue(numpy.array_equal(ref_model.get_mp_coefficients(
            [['TECTONIC'], []], n_to_q_table, ref_model),
            array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                   [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0, 1.0]])))
        tec_m_fac_arr = array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                               [1.0, 1.0, 0.18367346938775508, 0.0],
                               [0.18367346938775508, 1.0, 1.0, 0.0],
                               [0.0, 1.0, 0.18367346938775508, 0.0]])
        self.assertTrue(numpy.array_equal(ref_model.get_mp_coefficients(
            [['TECTONIC', 'FACIES'], []], n_to_q_table, ref_model),
            tec_m_fac_arr))
        rel = arr_scale(ref_model.data_q['RELIEF'], [[0.0], [8.0]])
        self.assertTrue(numpy.array_equal(ref_model.get_mp_coefficients(
            [[], ['RELIEF']], n_to_q_table, ref_model),
            array([1 - val for val in rel.flatten()]).reshape(rel.shape)))
        mp_c_arr = []
        for i in range(len(rel)):
            row = []
            for j in range(len(rel[i])):
                tec_m_fac = tec_m_fac_arr[i, j]
                den = tec_m_fac + rel[i, j]
                row.append(0.5 if den == 0 else tec_m_fac / den)
            mp_c_arr.append(row)
        self.assertTrue(numpy.array_equal(ref_model.get_mp_coefficients(
            [['TECTONIC', 'FACIES'], ['RELIEF']], n_to_q_table, ref_model),
            array(mp_c_arr)))

    def test_get_forecast(self):
        ref_model = read_data(_DATA_REF_MODEL, True)
        test_model = read_data(_DATA_TEST_MODEL, False)
        self.assertRaises(AppException, test_model.get_forecast, 'ORE CONTENT',
                          [[], []], ref_model)
        n_to_q_table = ref_model.get_n_to_q_table('ORE CONTENT')
        mp_parameters = get_mp_parameter_list(['FACIES', 'TECTONIC', 'RELIEF'],
                                              ref_model.get_correlation_table(
                                                  'ORE CONTENT', n_to_q_table))
        ref_mp_c = ref_model.get_mp_coefficients(mp_parameters, n_to_q_table,
                                                 ref_model)
        test_mp_c = test_model.get_mp_coefficients(mp_parameters, n_to_q_table,
                                                   ref_model)
        self.assertTrue(numpy.array_equal(
            test_model.get_forecast('ORE CONTENT', mp_parameters, ref_model),
            mp_to_data(test_mp_c, ref_mp_c, ref_model.data_f['ORE CONTENT'])))


class TestReferenceModel(unittest.TestCase):
    def test_init(self):
        self.assertRaises(AppException, read_data, _DATA_REF_WITHOUT_FORECAST,
                          True)

    def test_ref_model_get_n_to_q_table(self):
        ref_model = read_data(_DATA_REF_MODEL, True)
        self.assertRaises(AppException, ref_model.get_n_to_q_table,
                          'ORE_CONTENT')  # unknown title
        n_to_q_table = {'TECTONIC': {'1': 0.2, '2': 0.6},
                        'FACIES': {'1': 0.125, '2': 2.0 / 7.0, '3': 1.0}}
        self.assertEqual(ref_model.get_n_to_q_table('ORE CONTENT'),
                         n_to_q_table)

    def test_ref_model_get_correlation_table(self):
        ref_model = read_data(_DATA_REF_MODEL, True)
        f_title = 'ORE CONTENT'
        n_to_q_table = ref_model.get_n_to_q_table(f_title)
        self.assertRaises(AppException, ref_model.get_correlation_table,
                          'ORE_CONTENT', n_to_q_table)
        n_to_q_table_incorrect = {'TECTONIC_': {'1': 0.2, '2': 0.6},  # unknown title
                                  'FACIES': {'1': 0.125, '2': 2.0 / 7.0,
                                             '3': 1.0}}
        self.assertRaises(AppException, ref_model.get_correlation_table,
                          'ORE_CONTENT', n_to_q_table_incorrect)
        n_to_q_table_incorrect = {'TECTONIC': {'1': 0.2, '2': 0.6},
                                  'FACIES': {'1': 0.125, '3': 1.0}}  # without '2'
        self.assertRaises(AppException, ref_model.get_correlation_table,
                          'ORE_CONTENT', n_to_q_table_incorrect)
        ore_cont = ref_model.data_f['ORE CONTENT']
        tec = replace_arr_n_to_q(ref_model.data_n['TECTONIC'],
                                 n_to_q_table['TECTONIC'])
        fac = replace_arr_n_to_q(ref_model.data_n['FACIES'],
                                 n_to_q_table['FACIES'])
        corr_table = {'RELIEF': get_arr_correlation(ore_cont,
                                                    ref_model.data_q['RELIEF']),
                      'TECTONIC': get_arr_correlation(ore_cont, tec),
                      'FACIES': get_arr_correlation(ore_cont, fac)}
        self.assertEqual(ref_model.get_correlation_table(f_title, n_to_q_table),
                         corr_table)


_DATA_LONG_HEADER = '#VERY LONG HEADER 1234567890  1234567890 1234567890 ' \
                    '1234567890 1234567890 1234567890 1234567890 1234567890 ' \
                    '1234567890 1234567890 1234567890 123456789 1234567890'
_DATA_EMPTY_SECTION = '#EMPTY SECTION 123456789'
_DATA_WITHOUT_PARAMETERS = '#SECTION\n' \
                           '1 2 3 4 5 6 7 8 9'
_DATA_EMPTY_PARAMETER = '#PARAMETERS\n' \
                        'SQUARE_SIZE'
_DATA_INCORRECT_KEY_VAL_SEP = '#PARAMETERS\n' \
                            'SQUARE_SIZE: True'
_DATA_INCORRECT_PARAMETER = '#PARAMETERS\n' \
                            'SQUARE_SIZE = True'
_DATA_INCORRECT_REQ_PARAMETER = '#PARAMETERS\n' \
                                'START_ROW = True'
_DATA_INCORRECT_FUNC_SEP = '#PARAMETERS\n' \
                            'funk = f{[i,j][i-1][j-1]/2}'
_DATA_WITHOUT_REQ_PARAMETERS = '#PARAMETERS\n' \
                               'START_ROW = 1\n' \
                               'END_ROW = 20\n' \
                               'START_COLUMN = 1'
_DATA_EMPTY_DATA = '#PARAMETERS\n' \
                   'START_ROW = 1\n' \
                   'END_ROW = 20\n' \
                   'START_COLUMN = 1\n' \
                   'END_COLUMN = 20'
_DATA_DATA_INCORRECT = '#PARAMETERS\n' \
                       'START_ROW = 1\n' \
                       'END_ROW = 20\n' \
                       'START_COLUMN = 1\n' \
                       'END_COLUMN = 20\n' \
                       '#RELIEF\n' \
                       '1 2 3 4'
_DATA_ADD_SEC_FEW_ROWS = '#PARAMETERS\n' \
                         'START_ROW = 1\n' \
                         'END_ROW = 5\n' \
                         'START_COLUMN = 1\n' \
                         'END_COLUMN = 4\n' \
                         '#RELIEF\n' \
                         '1 2 3 4\n'
_DATA_ADD_SEC_FEW_COLS = '#PARAMETERS\n' \
                         'START_ROW = 1\n' \
                         'END_ROW = 5\n' \
                         'START_COLUMN = 1\n' \
                         'END_COLUMN = 4\n' \
                         '#RELIEF\n' \
                         '1 2 3 4\n' \
                         '1 2 3\n' \
                         '1 2 3 4\n' \
                         '1 2 3 4\n' \
                         '1 2 3 4\n'
_DATA_ADD_SEC_INCORRECT_ROW_INTERVAL = '#PARAMETERS\n' \
                                       'START_ROW = 3\n' \
                                       'END_ROW = 2\n' \
                                       'START_COLUMN = 1\n' \
                                       'END_COLUMN = 4\n' \
                                       '#RELIEF\n' \
                                       '1 2 3 4\n'
_DATA_ADD_SEC_INCORRECT_COL_INTERVAL = '#PARAMETERS\n' \
                                       'START_ROW = 1\n' \
                                       'END_ROW = 4\n' \
                                       'START_COLUMN = 3\n' \
                                       'END_COLUMN = 2\n' \
                                       '#RELIEF\n' \
                                       '1 2 3 4\n'
_DATA_ADD_SEC_INCORRECT_VAL = '#PARAMETERS\n' \
                              'START_ROW = 1\n' \
                              'END_ROW = 5\n' \
                              'START_COLUMN = 1\n' \
                              'END_COLUMN = 4\n' \
                              '#RELIEF\n' \
                              '1 2 3 4\n' \
                              '1 2 3 aaa\n' \
                              '1 2 3 4\n' \
                              '1 2 3 4\n' \
                              '1 2 3 4\n'
_DATA_ADD_SEC_WITHOUT_FUNC = '#PARAMETERS\n' \
                             'START_ROW = 1\n' \
                             'END_ROW = 5\n' \
                             'START_COLUMN = 1\n' \
                             'END_COLUMN = 4\n' \
                             '#RELIEF\n' \
                             '+BALANCE\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n'
_DATA_SEC_HEADER_DUPLICATE = '#PARAMETERS\n' \
                      'START_ROW = 1\n' \
                      'END_ROW = 5\n' \
                      'START_COLUMN = 1\n' \
                      'END_COLUMN = 4\n' \
                      '#RELIEF\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '#RELIEF\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n'
_DATA_DIFF_TYPE_SEC_DUPLICATE = '#PARAMETERS\n' \
                                'START_ROW = 1\n' \
                                'END_ROW = 5\n' \
                                'START_COLUMN = 1\n' \
                                'END_COLUMN = 4\n' \
                                '#RELIEF\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '#RELIEF _NOMINAL\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n' \
                                '1 2 3 4\n'
_DATA_SPECIAL_SEC_DUPLICATE = '#PARAMETERS\n' \
                      'START_ROW = 1\n' \
                      'END_ROW = 5\n' \
                      'START_COLUMN = 1\n' \
                      'END_COLUMN = 4\n' \
                      '#RELIEF _NOMINAL\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '#RELIEF  _NOMINAL\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n' \
                      '1 2 3 4\n'
_DATA_REF_WITHOUT_FORECAST = '#PARAMETERS\n' \
                             'START_ROW = 1\n' \
                             'END_ROW = 5\n' \
                             'START_COLUMN = 1\n' \
                             'END_COLUMN = 4\n' \
                             '#RELIEF\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n' \
                             '1 2 3 4\n'
_DATA_WITHOUT_DATA = '#PARAMETERS\n' \
                     'START_ROW = 1\n' \
                     'END_ROW = 5\n' \
                     'START_COLUMN = 1\n' \
                     'END_COLUMN = 4\n' \
                     '#RELIEF*\n' \
                     '1 2 3 4\n' \
                     '1 2 3 4\n' \
                     '1 2 3 4\n' \
                     '1 2 3 4\n' \
                     '1 2 3 4\n'

_DATA_MODEL_TRUNC = '#PARAMETERS\n' \
                    'START_ROW = 2\n' \
                    'END_ROW = 4\n' \
                    'START_COLUMN = 2\n' \
                    'END_COLUMN = 3\n' \
                    '#ORE CONTENT*\n' \
                    '0 0 0 0\n' \
                    '0 1 0 0\n' \
                    '1 2 1 1\n' \
                    '0 1 1 0\n' \
                    '0 0 0 0\n' \
                    '#RELIEF\n' \
                    '8 7 7 6\n' \
                    '6 4 6 6\n' \
                    '3 1 2 5\n' \
                    '5 2 5 5\n' \
                    '6 4 6 5\n' \

_DATA_REF_MODEL_FORMULA = '#PARAMETERS\n' \
                          'START_ROW = 2\n' \
                          'END_ROW = 4\n' \
                          'START_COLUMN = 2\n' \
                          'END_COLUMN = 3\n' \
                          'max_m = f: max([0, -1], [1, 0], [0, 1], [-1, 0]) / PI\n' \
                          '#ORE CONTENT*\n' \
                          '0 0 0 0\n' \
                          '0 1 0 0\n' \
                          '1 2 1 1\n' \
                          '0 1 1 0\n' \
                          '0 0 0 0\n' \
                          '#RELIEF\n' \
                          '+TEST BALANCE f: max_m\n' \
                          '8 7 7 6\n' \
                          '6 4 6 6\n' \
                          '3 1 4 0\n' \
                          '5 2 5 5\n' \
                          '6 4 6 5\n' \

_DATA_REF_MODEL = '#PARAMETERS\n' \
                  'START_ROW = 1\n' \
                  'END_ROW = 5\n' \
                  'START_COLUMN = 1\n' \
                  'END_COLUMN = 4\n' \
                  '#ORE CONTENT*\n' \
                  '0 0 0 0\n' \
                  '0 1 0 0\n' \
                  '1 2 1 1\n' \
                  '0 1 1 0\n' \
                  '0 0 0 0\n' \
                  '#RELIEF\n' \
                  '8 7 7 6\n' \
                  '6 4 6 6\n' \
                  '3 1 4 0\n' \
                  '5 2 5 5\n' \
                  '6 4 6 5\n' \
                  '#TECTONIC _NOMINAL\n' \
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

_DATA_TEST_MODEL = '#PARAMETERS\n' \
                   'START_ROW = 1\n' \
                   'END_ROW = 5\n' \
                   'START_COLUMN = 1\n' \
                   'END_COLUMN = 4\n' \
                   '#ORE CONTENT*\n' \
                   '0 1 0 0\n' \
                   '0 2 1 0\n' \
                   '0 1 2 1\n' \
                   '0 0 1 0\n' \
                   '0 0 0 0\n' \
                   '#RELIEF\n' \
                   '4 2 5 7\n' \
                   '3 1 4 6\n' \
                   '4 2 3 5\n' \
                   '5 4 5 6\n' \
                   '5 7 7 9\n' \
                   '#TECTONIC _NOMINAL\n' \
                   '1 2 2 1\n' \
                   '1 1 2 1\n' \
                   '1 2 2 1\n' \
                   '1 1 2 1\n' \
                   '1 1 1 2\n' \
                   '#FACIES _NOMINAL\n' \
                   '1 2 2 1\n' \
                   '2 3 2 1\n' \
                   '1 3 3 3\n' \
                   '1 1 3 2\n' \
                   '1 1 1 2'
