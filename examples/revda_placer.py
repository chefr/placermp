import sys

from sklearn.metrics import mean_squared_error

from placermp import inout
from placermp.utils import get_mp_parameter_list, get_arr_correlation, mape_arr

if __name__ == '__main__':
    print('Revda placer example model by placermp')
    ref_file = 'revda_ref_data.txt'
    test_file = 'revda_test_data.txt'
    f = open(ref_file, 'rt', encoding='UTF-8')
    data = f.read()
    f.close()
    ref_model = inout.read_data(data, True)
    print("Reference data from file '{}' uploaded successfully.".
          format(ref_file))
    print('\n----- START OF REFERENCE MODEL DATA -----\n')
    print(ref_model)
    print('\n----- END OF REFERENCE MODEL DATA -----\n')
    f = open(test_file, 'rt', encoding='UTF-8')
    data = f.read()
    f.close()
    test_model = inout.read_data(data, False)
    print("Test data from file '{}' uploaded successfully.".
          format(test_file))
    print('\n----- START OF TEST MODEL DATA -----\n')
    print(test_model)
    print('\n----- END OF TEST MODEL DATA -----\n')
    f_param = 'ORE_CONTENT'  # forecasted parameter
    n_to_q_table = ref_model.get_n_to_q_table(f_param)
    print('NOMINAL TO QUANTITATIVE SCALE TABLE:', n_to_q_table)
    corr_table = ref_model.get_correlation_table(f_param, n_to_q_table)
    print('CORRELATION TABLE:', corr_table)
    forecast = test_model.get_forecast(
        f_param, get_mp_parameter_list(
            ['TECTONIC', 'FACIES_ZONING', 'D_HEIGHTS', 'ABS_BALANCE',
             'BALANCE'], corr_table), ref_model)
    print('\n----- FORECASTED DATA -----')
    print(forecast)
    print('\n----- STATISTICS -----')
    if f_param in test_model.data_f:
        print('PEARSON_CORR: {0[0]:.3f} (p-val: {0[1]:.3})'.format(
            get_arr_correlation(forecast, test_model.data_f[f_param])))
        print('MAPE: {:.3f}'.format(mape_arr(test_model.data_f[f_param],
                                             forecast)))
        print('RMSE: {:.3f}'.format(mean_squared_error(
            test_model.data_f[f_param], forecast, squared=False)))
