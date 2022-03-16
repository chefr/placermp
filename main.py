import sys

import inout
import meths


if __name__ == '__main__':
    print('PlacerForecast ver. 0.1_beta (29.11.2021).')
    ref_model = inout.read_data(sys.argv[1], True)
    print('Reference data from file \'' + sys.argv[1] +
          '\' uploaded successfully.')
    test_model = inout.read_data(sys.argv[2], False)
    print('Test data from file \'' + sys.argv[2] +
          '\' uploaded successfully.')
    test_model.calc_average(ref_model.average_tables)
    print('Test data in the nominal scale is successfully reflected on the '
          'weighted average according to the reference data.')
    sm_list = inout.input_smooth(ref_model)
    for sm_title in sm_list:
        test_model.smooth(sm_title, ref_model.calc_smooth(sm_title))
    while True:
        ans = inout.input_answer('Save the characteristic distribution diagram '
                                 'to a file or calculate the forecast', 's',
                                 'f')
        if ans == 's':
            ans = inout.input_answer('Reference or test data', 'r', 't')
            inout.input_parameter_to_save(ref_model if ans == 'r' else
                                          test_model)
        else:
            meth = inout.choose_method()
            getattr(meths, meth)(ref_model, test_model)
            ans = inout.input_answer('Continue', 'y', 'n')
            if ans == 'n':
                break
    print('Bye.')
