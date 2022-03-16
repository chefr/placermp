import array

import utils


class Model:
    """
    Abstract base Model
    """
    def __init__(self, meta, data_q, data_n):
        self.meta = meta
        self.data = {}
        self.data_n = {}
        for k in data_q:
            if k == 'RELIEF' in data_q:
                relief = data_q['RELIEF']
                bal = utils.calculate_balance(relief)
                self.data['BALANCE'] = utils.norm_2d(bal)
                self.data['ABS_BALANCE'] = utils.norm_2d(
                    [[abs(val) for val in row] for row in bal])
                d_hs = utils.calculate_d_heights(relief)
                self.data['D_HEIGHTS'] = utils.norm_2d(d_hs)
                last_row = len(relief[0]) - 1
                self.data['RELIEF'] = [array.array(
                    'f', row[1:last_row]) for row in relief[1:len(relief) - 1]]
            else:
                self.data[k] = data_q[k]
        for k in data_n:
            self.data_n[k] = data_n[k]

    def get_rows(self):
        """
        Returns number of rows in model
        :return: number of rows
        """
        rows_cols = self.meta['ROWS_COLS']
        return rows_cols[1] - rows_cols[0]

    def get_cols(self):
        """
        Returns number of columns in model
        :return: number of columns
        """
        rows_cols = self.meta['ROWS_COLS']
        return rows_cols[3] - rows_cols[2]

    def get_corr(self, par1, par2):
        """
        Returns correlation coefficient between the specified parameters
        :param par1: title of the first parameter
        :param par2: title of the second parameter
        :return: correlation coefficient
        """
        return utils.corr_2d(self.data[par1], self.data[par2])

    def square_as_str(self, row, col):
        s = ''
        for k in self.data:
            s += '{}: {:.2f} '.format(k, self.data[k][row][col])
        for k in self.data_n:
            s += '{} {} '.format(k, self.data_n[k][row][col])
        return s[0: len(s) - 1]

    def flat_to_2d(self, arr):
        cols = self.get_cols()
        arr_2d = []
        for i in range(self.get_rows()):
            arr_2d.append(arr[i * cols: (i + 1) * cols])
        return arr_2d

    def save_csv(self, file, titles):
        """
        Save model parameters in csv format
        :param file: file name to save
        :param titles: parameter titles
        """
        f = open(file, 'wt', encoding='UTF-8')
        titles_num = len(titles)
        last_title_num = titles_num - 1
        for i in range(titles_num):
            tmpl = '{}' if i == last_title_num else '{}\t'
            f.write(tmpl.format(titles[i]))
        f.write('\n')
        cols = len(self.data[titles[0]])
        rows = len(self.data[titles[0]][0])
        for i in range(cols):
            for j in range(rows):
                for t in range(titles_num):
                    tmpl = '{:.3f}' if t == last_title_num else '{:.3f}\t'
                    f.write(tmpl.format(self.data[titles[t]][i][j]))
                f.write('\n')

    def get_forecast(self, mp_parameters):
        """Returns forecast by multiplicative method
        :param mp_parameters: parameters
        :return: forecast data
        """
        res = []
        rows = self.get_rows()
        cols = self.get_cols()
        for i in range(rows):
            row = []
            for j in range(cols):
                num = 1
                for par in mp_parameters[0]:
                    num *= self.data[par][i][j]
                den = 1
                for par in mp_parameters[1]:
                    num *= self.data[par][i][j]
                row.append(num / den)
            res.append(array.array('d', row))
        return utils.norm_2d(res)

    def __repr__(self):
        s = ''
        for k in self.data:
            if s:
                s += '\n'
            s += k + '\n'
            s += utils.arr_2d_as_str(self.data[k], '{:.2f}\t')
        for k in self.data_n:
            if s:
                s += '\n'
            s += k + '_N\n'
            s += utils.arr_2d_as_str(self.data_n[k], '{} ')
        return s.rstrip()


class ReferenceModel(Model):
    """
    Reference (training) Model
    """
    def __init__(self, meta, data_q, data_n):
        super().__init__(meta, data_q, data_n)
        self.average_tables = {}
        ore_cont = self.data['ORE_CONTENT']
        for k, v in self.data_n.items():
            ave_tab = utils.norm_to_share(utils.calculate_average(v, ore_cont))
            self.data[k + '_AVERAGE_'] = utils.norm_2d(
                [[ave_tab[val] for val in row] for row in v])
            self.average_tables[k] = ave_tab
        self.corr_table = {}
        for k in self.data:
            if k == 'ORE_CONTENT':
                continue
            self.corr_table[k] = utils.corr_2d(ore_cont, self.data[k])

    def calc_smooth(self, param):
        """
        Smooths parameter data
        :param: smoothing parameter name
        :return: number of smoothing operations
        """
        ore_cont = self.data['ORE_CONTENT']
        arr = self.data[param]
        prs = abs(utils.corr_2d(ore_cont, arr)[0])
        limit = 200  # random limit
        cnt = 0
        for i in range(limit):
            cnt = i
            sm_arr = utils.calculate_smooth(arr)
            cur_prs = abs(utils.corr_2d(ore_cont, sm_arr)[0])
            if cur_prs <= prs:
                break
            prs = cur_prs
            arr = sm_arr
        key = param + '_SM'
        self.data[key] = arr
        self.corr_table[key] = utils.corr_2d(ore_cont, arr)
        return cnt

    def corr_table_as_str(self):
        """
        Returns table of correlations between all parameters with ore content
        :return: correlation table
        """
        s = 'PEARSON_R FOR ORE_CONTENT: '
        for k in self.corr_table:
            s += '{} {:.3f}, (r-val {:.3}) '.format(k, *self.corr_table[k])
        return s.rstrip()

    def get_parameters(self, params_set):
        """
        Returns a model parameter list. Finds the first match with the
        beginning of the parameter name
        :param params_set: set of parameter names
        :raise AppException if params_set contains unknown parameter name
        :return: list of parameters
        """
        mp_params = []
        for val in params_set:
            for k in self.data:
                if k.startswith(val.upper()):
                    mp_params.append(k)
                    break
            else:
                raise utils.AppException('Unknown parameter ' + val)
        return mp_params

    def get_mp_parameters(self, params_set):
        """
        Returns a list containing two lists of parameters name, where the first
        contains parameters with a positive correlation with ore content and the
        second with a negative correlation. Finds the first match with the
        beginning of the parameter name
        :param params_set: set of parameter names
        :raise AppException if params_set contains unknown parameter name
        :return: list of parameters
        """
        if not params_set:
            raise utils.AppException('Empty set of parameters')
        mp_params = [[], []]
        for val in params_set:
            for k in self.data:
                if k.startswith(val.upper()):
                    mp_params[0 if self.corr_table[k][0] >= 0 else 1].append(k)
                    break
            else:
                raise utils.AppException('Unknown parameter ' + val)
        return mp_params

    def __repr__(self):
        s = super().__repr__() + '\n'
        for k in self.average_tables:
            s += '\n{}: {}'.format(k, self.average_tables[k])
        return s


class TestModel(Model):
    """
    Test Model
    """
    def __init__(self, meta, data_q, data_n):
        super().__init__(meta, data_q, data_n)

    def calc_average(self, average_tables):
        """
        Adds average parameters for all parameters represents in average table
        :param average_tables: average table
        """
        for k in average_tables:
            if k in self.data_n:
                ave_tab = average_tables[k]
                self.data[k + '_AVERAGE_'] = utils.norm_2d(
                    [[ave_tab[val] for val in row] for row in self.data_n[k]])

    def smooth(self, param, cnt):
        arr = self.data[param]
        for i in range(cnt):
            arr = utils.calculate_smooth(arr)
        self.data[param + '_SM'] = arr
