import pandas as pd

import inout
import utils


def multiplicative(ref_model, test_model):
    """
    Multiplicative (MP) method
    :param ref_model: Reference model
    :param test_model: Test model
    """
    print('Multiplicative')
    print(ref_model.corr_table_as_str())
    mp_params = inout.input_mp_parameters(ref_model)
    inout.print_mp_parameters(mp_params)
    forecast = test_model.get_forecast(mp_params)
    forecast = utils.scale_2d(forecast, ref_model.get_forecast(mp_params),
                              ref_model.data['ORE_CONTENT'])
    if 'ORE_CONTENT' in test_model.data:
        print('PEARSON_CORR: {0[0]:.3f} (r-val: {0[1]:.3})'.format(
            utils.corr_2d(test_model.data['ORE_CONTENT'], forecast)))
        print('RMSE: {:.3f}'.format(utils.rmse_2d(
            test_model.data['ORE_CONTENT'], forecast)))
        print('MAPE: {:.3f}'.format(utils.mape_2d(
            test_model.data['ORE_CONTENT'], forecast)))
    inout.save_param_by_model(test_model, forecast, 'MP')


def linear_regression(ref_model, test_model):
    """
    Linear regression method
    :param ref_model: Reference model
    :param test_model: Test model
    """
    from sklearn.linear_model import LinearRegression
    from scipy.stats.stats import pearsonr

    print('Linear Regression')
    ref_file = 'ref_csv.txt'
    test_file = 'test_csv.txt'
    ref_model.save_csv(ref_file, ('FACIES_ZONING_AVERAGE_',
                                  'TECTONIC_AVERAGE_', 'BALANCE', 'D_HEIGHTS',
                                  'ORE_CONTENT'))
    test_model.save_csv(test_file, ('FACIES_ZONING_AVERAGE_',
                                    'TECTONIC_AVERAGE_', 'BALANCE', 'D_HEIGHTS',
                                    'ORE_CONTENT'))
    train_data = pd.read_csv(ref_file, sep='\t', header=0)
    x_train = train_data.drop(['ORE_CONTENT'], axis=1)
    y_train = train_data['ORE_CONTENT']
    model = LinearRegression()
    model.fit(x_train, y_train)
    print('Coefficients: {}'.format(model.coef_))
    print('Intercept: {}'.format(model.intercept_))
    print(model.score(x_train, y_train))
    test_data = pd.read_csv(test_file, sep='\t', header=0)
    x_test = test_data.drop(['ORE_CONTENT'], axis=1)
    y_test = test_data['ORE_CONTENT']
    print('Score for test data: {}'.format(model.score(x_test, y_test)))
    print()
    prd = model.predict(x_test)
    print('PEARSON_CORR: {0[0]:.3f} (r-val: {0[1]:.3})'.
          format(pearsonr(prd, y_test)))
    print('RMSE: {:.3f}'.format(utils.rmse(y_test, prd)))
    print('MAPE: {:.3f}'.format(utils.mape(y_test, prd)))
    inout.save_param_by_model(test_model, prd, 'LR')


def reg_forest(ref_model, test_model):
    """
    Random forest regression method
    :param ref_model: Reference model
    :param test_model: Test model
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats.stats import pearsonr

    print('Random Forest Regression')
    ref_file = 'ref_csv.txt'
    test_file = 'test_csv.txt'
    ref_model.save_csv(ref_file, ('FACIES_ZONING_AVERAGE_', 'BALANCE',
                                  'TECTONIC_AVERAGE_', 'D_HEIGHTS',
                                  'ORE_CONTENT'))
    test_model.save_csv(test_file, ('FACIES_ZONING_AVERAGE_', 'BALANCE',
                                    'TECTONIC_AVERAGE_', 'D_HEIGHTS',
                                    'ORE_CONTENT'))
    train_data = pd.read_csv(ref_file, sep='\t', header=0)
    x_train = train_data.drop(['ORE_CONTENT'], axis=1)
    y_train = train_data['ORE_CONTENT']

    # ceed_1_23_1_15 = 341
    # ceed_1_23_1_15_sm = 101
    # ceed_1_23_16_27 = 599
    # ceed_1_23_16_27_sm = 322

    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    test_data = pd.read_csv(test_file, sep='\t', header=0)
    x_test = test_data.drop(['ORE_CONTENT'], axis=1)
    y_test = test_data['ORE_CONTENT']
    prd = model.predict(x_test)
    print('PEARSON_CORR: {0[0]:.3f} (r-val: {0[1]:.3})'.format(pearsonr(prd, y_test)))
    print('RMSE: {:.3f}'.format(utils.rmse(y_test, prd)))
    print('MAPE: {:.3f}'.format(utils.mape(y_test, prd)))
    inout.save_param_by_model(test_model, prd, 'RF')
