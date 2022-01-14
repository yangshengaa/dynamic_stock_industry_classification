import numpy as np
import pandas as pd
# import configuration.config as cfg


def z_score(fac_df):
    return (fac_df - fac_df.mean()) / fac_df.std()


class FactorTools:

    def __init__(
            self, 
            eod_data_dict, 
            fac, 
            winsorize_method=["MAD"], 
            standardize_method=["z_score"],
            neutralize_method=["Ind", "Size"], 
            quantile=0.05
        ):

        self.old_factor = fac
        self.new_factor = fac
        self.eod_data_dict = eod_data_dict
        self.fmv = self.eod_data_dict["FloatMarketValue"]

        self.winsorize_method = winsorize_method
        self.standardize_method = standardize_method
        self.neutralize_method = neutralize_method

        if self.winsorize_method == "quantile":
            self.quantile = quantile

    def processor(self):
        """
        调用processor进行因子值处理，其中调用winsorize进行取极值，之后按照参数要求对行业和市值进行标准化
        :return: 取极值 标准化后的factor
        """
        self.winsorize()
        self.neutralize()
        self.standardize()
        return self.new_factor

    def winsorize(self):
        """
        去极值处理
        """
        if "quantile" in self.winsorize_method:
            self.wins_quantile()
        if "3sigma" in self.winsorize_method:
            self.wins_3_sigma()
        if "MAD" in self.winsorize_method:
            self.wins_mad()

    def standardize(self):
        """
        Rank处理
        :return:
        """

        if "rank" in self.standardize_method:
            self.rank_standardize()
        if "z_score" in self.standardize_method:
            self.z_score_standardize()

    def neutralize(self):
        """
        中性化处理
        :return:
        """
        if set(self.neutralize_method) == {"Ind", "Size"}:
            self.new_factor = self.neutralize_lstsq()
        elif set(self.neutralize_method) == {"Ind"}:
            self.new_factor = self.neutralize_lstsq(size_neutral=False)
        elif set(self.neutralize_method) == {"Size"}:
            self.new_factor = self.neutralize_lstsq(ind_neutral=False)
        else:
            pass

    ###### tools for winsorize #####
    def wins_quantile(self):
        """
        将超出当日阈值的点替换为当日阈值
        """
        step0 = self.new_factor
        step1 = step0 - step0.quantile(self.quantile)
        step2 = step1.mask(step1 < 0, 0) + step0.quantile(self.quantile)
        step3 = step2 - step0.quantile(1 - self.quantile)
        self.new_factor = step3.mask(step3 > 0, 0) + step0.quantile(1 - self.quantile)

    def wins_3_sigma(self):
        """
        阈值为三倍标准差
        将超出当日阈值的点替换为当日阈值
        """
        step0 = self.new_factor
        upper = step0.median() + 3 * step0.std()
        lower = step0.median() - 3 * step0.std()
        step1 = step0 - lower
        step2 = step1.mask(step1 < 0, 0) + lower

        step3 = step2 - upper
        self.new_factor = step3.mask(step3 > 0, 0) + upper

    def wins_mad(self):
        """
        阈值为1.483倍MAD
        将超出当日阈值的点替换为当日阈值
        """
        step0 = self.new_factor
        upper = step0.median() + 1.483 * step0.mad()
        lower = step0.median() - 1.483 * step0.mad()
        step1 = step0 - lower
        step2 = step1.mask(step1 < 0, 0) + lower

        step3 = step2 - upper
        self.new_factor = step3.mask(step3 > 0, 0) + upper

    ###### tools for standardize #####
    def rank_standardize(self):
        self.new_factor = self.new_factor.rank()

    def z_score_standardize(self):
        """
        :return: normal standardize
        """
        self.new_factor = z_score(self.new_factor)


    def neutralize_lstsq(self, ind_neutral=True, size_neutral=True):
        """
        ----输入矩阵columns为股票列表----
        市值或行业中性化函数
        :param size_neutral:
        :param ind_neutral:
        :param ind_method:
        :return: 中性化后的因子矩阵
        """
        fac_df = self.new_factor.copy()
        fac_na_df = self.old_factor - self.old_factor
        eod_data_dict = self.eod_data_dict.copy()
        ind_df = eod_data_dict['ind_df']
        new_fac_df = fac_df.copy()
        if size_neutral:
            for date in fac_df.columns:
                fac = fac_df[date]
                class_var = pd.DataFrame(index=fac.index)
                class_var['log_size'] = np.log(eod_data_dict['FloatMarketValue'][date])
                idx = (~fac.isna()) & (~class_var['log_size'].isna()).values  # 选择x和y同时有数的部分
                x = np.hstack((np.ones((len(fac), 1)), class_var.values))[idx, :]
                y = fac.values[idx]
                new_fac_df[date].iloc[idx] = list(y - x @ np.linalg.lstsq(x, y, rcond=None)[0])

        if ind_neutral:
            factor_df_list = []
            for ind in ind_df.index:
                ind_list = ind_df.loc[ind].replace(0, np.nan).dropna().index
                factor_df_list.append(
                    (new_fac_df.loc[ind_list] - new_fac_df.loc[ind_list].mean()) / new_fac_df.loc[ind_list].std())
            factor_df = pd.concat(factor_df_list)
        else:
            factor_df = new_fac_df

        factor_df = factor_df + fac_na_df
        return factor_df


