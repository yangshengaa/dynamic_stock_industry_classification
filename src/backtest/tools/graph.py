import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append("..")
from configuration import config as cfg
import matplotlib.pyplot as plt


class Grapher:

    def __init__(self, data_dict):
        """
        :param data_dict:
            keys: long_ret_series, short_ret_series, group_ret_series,index_ret_series,
                  IC,turnover,turnover_long,turnover_short,summary
        """
        # 从data dict获取数据
        self.data_dict = data_dict
        self.factor_name = data_dict['factor_name']
        self.long_ret_series_no_cost = data_dict["long_ret_series_no_cost"]
        self.short_ret_series_no_cost = data_dict["short_ret_series_no_cost"]
        self.long_ret_series_after_cost = data_dict["long_ret_series_after_cost"]
        self.short_ret_series_after_cost = data_dict["short_ret_series_after_cost"]
        self.group_ret_series = data_dict["group_ret_series"]
        self.index_ret_series = data_dict["index_ret_series"]
        self.long_batch_avg_ret_list = data_dict["long_batch_avg_ret_list"]
        self.short_batch_avg_ret_list = data_dict["short_batch_avg_ret_list"]
        self.split_result_dict = data_dict["split_result_dict"]
        self.decay_result_dict = data_dict["decay_result_dict"]
        self.decay_list = data_dict["decay_list"]
        self.lag_result_dict = data_dict["lag_result_dict"]
        self.lead_result_dict = data_dict["lead_result_dict"]
        self.lead_lag_list = data_dict['lead_lag_list']
        self.split_ret_type_list = data_dict["split_ret_type_list"]
        self.index_cum = (self.index_ret_series*(0*self.long_ret_series_no_cost+1)).cumsum()
        self.long_cum_no_cost = (self.long_ret_series_no_cost).cumsum()
        self.short_cum_no_cost = (self.short_ret_series_no_cost).cumsum()
        self.long_cum_after_cost = (self.long_ret_series_after_cost).cumsum()
        self.short_cum_after_cost = (self.short_ret_series_after_cost).cumsum()
        self.long_short_cum_no_cost = self.long_cum_no_cost - self.short_cum_no_cost
        self.long_short_cum_after_cost = self.long_cum_after_cost - 2 * self.short_cum_no_cost + self.short_cum_after_cost

        # Group
        self.group_cum = []
        for i in range(len(self.group_ret_series)):
            self.group_cum.append((self.group_ret_series[i]).cumsum())

        self.IC = pd.Series(data_dict["ic_list"])
        self.IC_decay = data_dict["ic_decay"]
        self.IR = np.nanmean(data_dict["ic_list"]) / np.nanstd(data_dict["ic_list"])
        self.IC_stats = data_dict["ic_stats_dict"]
        self.turnover = data_dict["turnover"]
        self.turnover_long = data_dict["turnover_long"]
        self.turnover_short = data_dict["turnover_short"]
        self.summary_stats = data_dict['summary_after_cost']
        self.summary_stats['AlphaRtnNC'] = data_dict['summary_no_cost']['AlphaRtn']
        self.summary_stats['AlphaSharpeNC'] = data_dict['summary_no_cost']['AlphaSharpe']
        self.summary_stats['AlphaDrawDownNC'] = data_dict['summary_no_cost']['AlphaDrawdown']
        self.long_holding_stats = data_dict["long_holding_stats_df"]
        self.short_holding_stats = data_dict["short_holding_stats_df"]
        self.long_tov_holding_ratio = data_dict['long_tov_holding_ratio']
        self.short_tov_holding_ratio = data_dict['short_tov_holding_ratio']
        self.long_size_holding_ratio = data_dict['long_size_holding_ratio']
        self.short_size_holding_ratio = data_dict['short_size_holding_ratio']

        self.freq = cfg.freq
        if self.freq == "W":
            self.adj_freq = 5
        else:
            self.adj_freq = cfg.adj_freq
        self.pool = '&'.join(cfg.index_list) + '_' + ('fmv' if cfg.weight_index_by_fmv else 'equal')
        if cfg.use_independent_return_benchmark:
            self.pool += '_' + cfg.return_benchmark_index

    @staticmethod
    def cal_maxdd(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return max(drawdowns)

    @staticmethod
    def cal_maxdd_local(array):
        return - np.nanmax(array) + array[len(array) - 1]

    def plot_ls_no_cost(self, fig):
        """
        fig1：多空收益与指数收益比较
        """
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(15, 2, 1)
        p1, = ax1.plot(self.long_short_cum_no_cost * 100)
        p2, = ax1.plot(self.long_cum_no_cost * 100)
        p3, = ax1.plot(self.short_cum_no_cost * 100)
        p4, = ax1.plot(self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Absolute Result No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_excess_no_cost(self, fig):
        """
        fig2：多空超额收益
        """
        step = int(len(self.long_cum_no_cost.index) / 6)
        ax1 = fig.add_subplot(15, 2, 2)
        p1, = ax1.plot(self.long_short_cum_no_cost * 100)
        p2, = ax1.plot(self.long_cum_no_cost * 100 - self.index_cum * 100)
        p3, = ax1.plot(self.short_cum_no_cost * 100 - self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3], labels=['long_short', 'long', 'short'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Excess Result No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_maxdd_no_cost(self, fig):
        """
        fig1.1：多头收益回撤
        """
        self.cum_ret_no_cost_rate = self.long_cum_no_cost * 100
        self.cum_ret_no_cost_maxdd = self.cum_ret_no_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.cum_ret_no_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(15, 2, 3)
        p1 = ax1.bar(x = list(self.cum_ret_no_cost_maxdd.index), height = list(self.cum_ret_no_cost_maxdd.values), color = 'coral')
        ax1.legend(handles=[p1], labels=['long_maxdd'], loc='best', fontsize=10)
        if len(self.cum_ret_no_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.cum_ret_no_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.cum_ret_no_cost_maxdd.index)[::step])
        ax1.set_title("Long-Absolute MaxDrawdown No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_alpha_maxdd_no_cost(self, fig):
        """
        fig2.1：超额收益回撤
        """
        self.alpha_cum_no_cost_rate = self.long_cum_no_cost * 100 - self.index_cum * 100
        self.alpha_cum_no_cost_maxdd = self.alpha_cum_no_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.alpha_cum_no_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(15, 2, 4)
        p1 = ax1.bar(x = list(self.alpha_cum_no_cost_maxdd.index), height = list(self.alpha_cum_no_cost_maxdd.values), color = 'springgreen')
        ax1.legend(handles=[p1], labels=['alpha_maxdd'], loc='best', fontsize=10)
        if len(self.alpha_cum_no_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.alpha_cum_no_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.alpha_cum_no_cost_maxdd.index)[::step])
        ax1.set_title("Alpha MaxDrawdown No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_after_cost(self,fig):
        '''
        fig3: 多空收益扣费后
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(15, 2, 5)
        p1, = ax1.plot(self.long_short_cum_after_cost * 100)
        p2, = ax1.plot(self.long_cum_after_cost * 100)
        p3, = ax1.plot(self.short_cum_after_cost * 100)
        p4, = ax1.plot(self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Absolute Result After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_excess_after_cost(self,fig):
        '''
        fig4: 多空超额收益扣费后
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(15, 2, 6)
        p1, = ax1.plot(self.long_short_cum_after_cost * 100)
        p2, = ax1.plot(self.long_cum_after_cost * 100 - self.index_cum * 100)
        p3, = ax1.plot(self.short_cum_after_cost * 100 - self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3], labels=['long_short', 'long', 'short'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Excess Result After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_maxdd_after_cost(self, fig):
        """
        fig3.1：多头收益回撤
        """
        self.cum_ret_after_cost_rate = self.long_cum_after_cost * 100
        self.cum_ret_after_cost_maxdd = self.cum_ret_after_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.cum_ret_after_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(15, 2, 7)
        p1 = ax1.bar(x = list(self.cum_ret_after_cost_maxdd.index), height = list(self.cum_ret_after_cost_maxdd.values), color = 'coral')
        ax1.legend(handles=[p1], labels=['long_maxdd'], loc='best', fontsize=10)
        if len(self.cum_ret_after_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.cum_ret_after_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.cum_ret_after_cost_maxdd.index)[::step])
        ax1.set_title("Long-Absolute MaxDrawdown After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_alpha_maxdd_after_cost(self, fig):
        """
        fig4.1：超额收益回撤
        """
        self.alpha_cum_after_cost_rate = self.long_cum_after_cost * 100 - self.index_cum * 100
        self.alpha_cum_after_cost_maxdd = self.alpha_cum_after_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.alpha_cum_after_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(15, 2, 8)
        p1 = ax1.bar(x = list(self.alpha_cum_after_cost_maxdd.index), height = list(self.alpha_cum_after_cost_maxdd.values), color = 'springgreen')
        ax1.legend(handles=[p1], labels=['alpha_maxdd'], loc='best', fontsize=10)
        if len(self.alpha_cum_after_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.alpha_cum_after_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.alpha_cum_after_cost_maxdd.index)[::step])
        ax1.set_title("Alpha MaxDrawdown After Cost", fontsize=15)
        ax1.grid(axis='y')


    def plot_group_ts(self, fig):
        """
        fig5：分层时序图
        """
        step = int(len(self.group_cum[0].index) / 6)
        ax2 = fig.add_subplot(15, 1, 5)
        for i in range(len(self.group_cum)):
            ax2.plot(self.group_cum[i] * 100)
        ax2.legend(labels=["group" + str(i) for i in range(len(self.group_cum))], fontsize=10,bbox_to_anchor=(1,0),loc=3)
        if len(self.group_cum[0].index) < 20:
            ax2.set_xticks(self.group_cum[0].index)
        else:
            ax2.set_xticks(self.group_cum[0].index[::step])
        ax2.set_title("Group Test Result", fontsize=15)
        ax2.grid(axis='y')

    def plot_group_mean(self, fig):
        """
        fig6：分层平均收益率
        """
        # fig3 分层平均收益率
        ax3 = fig.add_subplot(15, 2, 11)
        total_ret = [np.sum(ret) * 100 for ret in self.group_ret_series]
        ax3.plot(total_ret)
        ax3.set_xticks(range(len(total_ret)))
        ax3.set_title("Total Return of Group", fontsize=15)

    def plot_trade_date_effect(self, fig):
        """
        fig7：日期效应
        """
        ax8 = fig.add_subplot(15, 2, 12)
        ax8.plot([x * 252 * 100 for x in self.long_batch_avg_ret_list], c='cornflowerblue', marker='o')
        ax8.set_xticks(range(len(self.long_batch_avg_ret_list)))
        ax8t = ax8.twinx()
        ax8t.plot([x * 252 * 100 for x in self.short_batch_avg_ret_list], c='orange', marker='o')
        ax8.legend(labels=["long"], loc="upper left")
        ax8t.legend(labels=["short"], loc="upper right")
        ax8.set_title("\n Annual Return on Different Trade Path", fontsize=15)

    def plot_split_return(self,fig):
        '''
        fig8: split return
        :param fig:
        :return:
        '''
        ax = fig.add_subplot(15,2,13)
        ax.set_title("long short split return (close need to fix)")
        ax.plot(self.split_ret_type_list,self.split_result_dict['long_short'], marker='o')
        ax.plot(self.split_ret_type_list,self.split_result_dict['long'], marker='o')
        ax.plot(self.split_ret_type_list,self.split_result_dict['short'], marker='o')
        # ax.set_xticks(self.split_ret_type_list)
        ax.legend(labels=["long_short", "long", "short"], fontsize=12)

    def plot_ls_decay(self,fig):
        '''
        fig9: 调整持仓时间，多空收益的变化
        :param fig:
        :return:
        '''
        ax = fig.add_subplot(15,2,14)
        ax.set_title("long short decay")
        ax.plot(self.decay_list,self.decay_result_dict['long_short'], marker='o')
        ax.plot(self.decay_list,self.decay_result_dict['long'], marker='o')
        ax.plot(self.decay_list,self.decay_result_dict['short'], marker='o')
        ax.legend(labels=["long_short", "long", "short"], fontsize=12)

    def plot_ls_lead(self,fig):
        '''
        调整因子的lead期，绩效的变化
        :param fig:
        :return:
        '''
        ax = fig.add_subplot(15, 2, 15)
        ax.set_title("Factor Lead")
        ax.plot(self.lead_lag_list[0], self.lead_result_dict['long_short'], marker='o')
        ax.plot(self.lead_lag_list[0], self.lead_result_dict['long'], marker='o')
        ax.plot(self.lead_lag_list[0], self.lead_result_dict['short'], marker='o')
        ax.legend(labels=["long_short", "long", "short"], fontsize=12)

    def plot_ls_lag(self,fig):
        '''
        调整因子的lag期，绩效的变化
        :param fig:
        :return:
        '''
        ax = fig.add_subplot(15,2,16)
        ax.set_title("Factor Lag")
        ax.plot(self.lead_lag_list[1],self.lag_result_dict['long_short'], marker='o')
        ax.plot(self.lead_lag_list[1],self.lag_result_dict['long'], marker='o')
        ax.plot(self.lead_lag_list[1],self.lag_result_dict['short'], marker='o')
        ax.legend(labels=["long_short", "long", "short"], fontsize=12)

    def plot_ic_cum(self, fig):
        """
        fig10：累计IC
        """
        step = int(len(self.IC) / 6)
        ax5 = fig.add_subplot(15, 2, 17)
        ax5.plot(np.cumsum(self.IC.fillna(0)))
        ax5.set_xticks(self.IC.index[::step])
        ax5.set_title("Cumulated IC _IR:" + "%.3f" % self.IR, fontsize=15)
        ax5.grid(axis='y')

    def plot_turnover(self, fig):
        """
        fig11：换手率
        """
        step = int(len(self.turnover.iloc[self.adj_freq + 1:-self.adj_freq].index) / 6)
        ax6 = fig.add_subplot(15, 2, 18)
        ax6.plot(self.turnover_long.iloc[self.adj_freq + 1:-self.adj_freq], lw=0.7)  # .ewm(1).mean()
        ax6.plot(self.turnover_short.iloc[self.adj_freq + 1:-self.adj_freq], lw=0.7)
        ax6.plot(self.turnover.iloc[self.adj_freq + 1:-self.adj_freq], lw=0.7)
        ax6.set_xticks(self.turnover.iloc[self.adj_freq + 1:-self.adj_freq].index[::step])
        ax6.set_xticklabels(self.turnover.iloc[self.adj_freq + 1:-self.adj_freq].index[::step])
        ax6.legend(labels=["TR_long", "TR_short", "TR"], fontsize=12)
        ax6.set_title("Turnover", fontsize=15)

    def plot_ic_decay(self, fig):
        '''
        fig12 IC Decay
        :param fig:
        :return:
        '''
        ax8 = fig.add_subplot(15, 2, 19)
        ax8.bar(range(len(self.IC_decay)), self.IC_decay)
        ax8.set_title("IC Decay", fontsize=15)


    def plot_ic_dist(self, fig):
        """
        fig13：IC分布
        """
        ax4 = fig.add_subplot(15, 2, 20)
        ax4.hist(self.IC, bins=20)
        ax4.set_xlim(-1, 1)
        ax4.set_title("IC distribution", fontsize=15)

    def plot_table_ic(self, fig):
        """
        table: stats summary
        keywords:
        """
        ax9 = fig.add_subplot(15, 1, 11)
        plt.axis('off')
        col_labels = list(self.IC_stats.keys())
        cell_text = [list(self.IC_stats.values())[:]]
        cell_text = [[round(x, 3) for x in cell_text[0]]]
        table = ax9.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        ax9.set_title("IC Statistics", fontsize=15)

    def plot_table_summary(self, fig):
        """
        table: stats summary
        keywords:
        """
        ax10 = fig.add_subplot(15, 1, 12)
        plt.axis('off')
        # 列名是统计量名字，行名是年份
        col_labels = list(self.summary_stats.keys())
        row_labels = list((list(self.summary_stats.values())[:])[0].keys())
        cell_text = pd.DataFrame(self.summary_stats).values
        for i in range(cell_text.shape[0]):
            cell_text[i] = [round(x, 2) for x in cell_text[i]]
        cell_text = cell_text[-1:,:]
        row_labels = row_labels[-1:]

        table = ax10.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                           cellLoc='center', rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax10.set_title("Stats Summary", fontsize=15)
        table.scale(1, 2.5)

    def plot_tov_ratio(self,fig):
        '''
        调整因子的lag期，绩效的变化
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax = fig.add_subplot(15,2,25)
        ax.plot(self.long_tov_holding_ratio)
        ax.plot(self.short_tov_holding_ratio)
        ax.legend(labels=["long", "short"], loc='best',fontsize=10)
        ax.set_xticks(self.long_cum_after_cost.index[::step])
        ax.set_title("Turnover 10 Holding Ratio")

    # def plot_ls_no_cost(self, fig):
    #     """
    #     fig1：多空收益与指数收益比较
    #     """
    #     step = int(len(self.long_cum_after_cost.index) / 6)
    #     ax1 = fig.add_subplot(15, 2, 1)
    #     p1, = ax1.plot(self.long_short_cum_no_cost * 100)
    #     p2, = ax1.plot(self.long_cum_no_cost * 100)
    #     p3, = ax1.plot(self.short_cum_no_cost * 100)
    #     p4, = ax1.plot(self.index_cum * 100)
    #     ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
    #                fontsize=10)
    #     if len(self.long_cum_after_cost.index) < 20:
    #         ax1.set_xticks(self.long_cum_after_cost.index)
    #     else:
    #         ax1.set_xticks(self.long_cum_after_cost.index[::step])
    #     ax1.set_title("Long-Short Absolute Result No Cost", fontsize=15)
    #     ax1.grid(axis='y')

    def plot_size_ratio(self,fig):
        '''
        调整因子的lag期，绩效的变化
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax = fig.add_subplot(15,2,26)
        ax.plot(self.long_size_holding_ratio)
        ax.plot(self.short_size_holding_ratio)
        ax.set_xticks(self.long_cum_after_cost.index[::step])
        ax.legend(labels=["long", "short"], fontsize=12)
        ax.set_title("Size 10 Holding Ratio")

    def plot_ls_no_cost(self, fig):
        """
        fig1：多空收益与指数收益比较
        """
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(15, 2, 1)
        p1, = ax1.plot(self.long_short_cum_no_cost * 100)
        p2, = ax1.plot(self.long_cum_no_cost * 100)
        p3, = ax1.plot(self.short_cum_no_cost * 100)
        p4, = ax1.plot(self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Absolute Result No Cost", fontsize=15)
        ax1.grid(axis='y')


    def plot_long_table_holding(self, fig):
        """
        table: holding stats
        keywords:
        """
        ax11 = fig.add_subplot(15, 1, 14)
        plt.axis('off')
        holding_stats = self.long_holding_stats.to_dict()

        col_labels = list(holding_stats.keys())
        row_labels = list((list(holding_stats.values())[:])[0].keys())
        cell_text = pd.DataFrame(holding_stats).values

        for i in range(cell_text.shape[0]):
            cell_text[i] = [round(x, 2) for x in cell_text[i]]

        table = ax11.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                           cellLoc='center', rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)
        ax11.set_title("Long Holding Statistics", fontsize=15)

    def plot_short_table_holding(self, fig):
        """
        table: holding stats
        keywords:
        """
        ax11 = fig.add_subplot(15, 1, 15)
        plt.axis('off')
        holding_stats = self.short_holding_stats.to_dict()

        col_labels = list(holding_stats.keys())
        row_labels = list((list(holding_stats.values())[:])[0].keys())
        cell_text = pd.DataFrame(holding_stats).values

        for i in range(cell_text.shape[0]):
            cell_text[i] = [round(x, 2) for x in cell_text[i]]

        table = ax11.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                           cellLoc='center', rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)
        ax11.set_title("Short Holding Statistics", fontsize=15)




    def save_table_summary(self,factor_name):
        '''
        把因子分年绩效存到新的文件夹中
        :param factor_name:
        :return:
        '''
        d = cfg.output_path
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M%S', time.localtime(time.time()))
        # 创建当日文件夹
        save_path = '{}/test_table_file_{}_{}'.format(d, curr_date,cfg.test_name)
        table_path = '{}/{}_{}_{}day_{}_{}_{}.csv'.format(save_path, factor_name, self.pool, cfg.adj_freq, cfg.cost, cfg.head, curr_date+curr_time)
        col_labels = list(self.summary_stats.keys())
        row_labels = list((list(self.summary_stats.values())[:])[0].keys())
        text_table = pd.DataFrame(self.summary_stats,index=row_labels,columns=col_labels)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        text_table.to_csv(table_path)

    def save_fig(self, factor_name):
        """
        :return:
        """

        d = cfg.output_path
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M%S', time.localtime(time.time()))
        # 创建当日文件夹
        save_path = '{}/facTest_file_{}_{}'.format(d, curr_date,cfg.test_name)
        fig_path = '{}/{}_{}_{}day_{}_{}_{}.png'.format(save_path, factor_name, self.pool, cfg.adj_freq, cfg.cost, cfg.head, curr_date+curr_time)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # 画图存图
        fig = plt.figure(figsize=(20, 45))
        fig.suptitle('{}_{}_{}_{}_{} Performance'.format(factor_name, self.pool, cfg.adj_freq, cfg.cost, cfg.head), fontsize=20)
        self.plot_ls_no_cost(fig)
        self.plot_ls_maxdd_no_cost(fig)
        self.plot_alpha_maxdd_no_cost(fig)
        self.plot_ls_maxdd_after_cost(fig)
        self.plot_alpha_maxdd_after_cost(fig)
        self.plot_ls_excess_no_cost(fig)
        self.plot_ls_after_cost(fig)
        self.plot_ls_excess_after_cost(fig)
        self.plot_group_ts(fig)
        self.plot_group_mean(fig)
        self.plot_ic_dist(fig)
        self.plot_ic_cum(fig)
        self.plot_turnover(fig)
        self.plot_ic_decay(fig)
        self.plot_trade_date_effect(fig)
        self.plot_table_ic(fig)
        self.plot_long_table_holding(fig)
        self.plot_short_table_holding(fig)
        self.plot_tov_ratio(fig)
        self.plot_size_ratio(fig)
        self.plot_table_summary(fig)
        self.plot_split_return(fig)
        self.plot_ls_lag(fig)
        self.plot_ls_lead(fig)
        self.plot_ls_decay(fig)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)  # 纯靠实验得出，为了让suptitle有个不和图片打架的位置

        # 存表格
        self.save_table_summary(factor_name)


        # 存储
        plt.subplots_adjust(wspace=0.1,hspace=0.3)
        plt.savefig(fig_path, dpi=300)


class SignalGrapher:

    def __init__(self, data_dict):
        """
        :param data_dict:
            keys: long_ret_series, short_ret_series, group_ret_series,index_ret_series,
                  IC,turnover,turnover_long,turnover_short,summary
        """
        self.data_dict = data_dict
        self.signal_name = data_dict['signal_name']
        self.index_ret_series = data_dict["index_ret_series"]
        self.long_ret_series_no_cost = data_dict["long_ret_series_no_cost"]
        self.short_ret_series_no_cost = data_dict["short_ret_series_no_cost"]
        self.long_ret_series_after_cost = data_dict["long_ret_series_after_cost"]
        self.short_ret_series_after_cost = data_dict["short_ret_series_after_cost"]
        self.index_cum = (self.index_ret_series*(0*self.long_ret_series_no_cost+1)).cumsum()
        self.long_cum_no_cost = (self.long_ret_series_no_cost).cumsum()
        self.short_cum_no_cost = (self.short_ret_series_no_cost).cumsum()
        self.long_cum_after_cost = (self.long_ret_series_after_cost).cumsum()
        self.short_cum_after_cost = (self.short_ret_series_after_cost).cumsum()
        self.long_short_cum_no_cost = self.long_cum_no_cost - self.short_cum_no_cost
        self.long_short_cum_after_cost = self.long_cum_after_cost - self.short_cum_after_cost
        self.turnover = data_dict["turnover"]
        self.turnover_long = data_dict["turnover_long"]
        self.turnover_short = data_dict["turnover_short"]
        self.summary_stats = data_dict["summary_after_cost"]
        self.summary_stats['AlphaRtnNC'] = data_dict['summary_no_cost']['AlphaRtn']
        self.summary_stats['AlphaSharpeNC'] = data_dict['summary_no_cost']['AlphaSharpe']
        self.summary_stats['AlphaDrawDownNC'] = data_dict['summary_no_cost']['AlphaDrawdown']

        # Long Short Correction
        # 检查空头是否为nan主要是因为有读csv纯多的信号，所以要返回纯多
        if self.long_cum_no_cost[-1] >= self.short_cum_no_cost[-1] or np.isnan(self.short_cum_no_cost[-1]):
            self.cum_ret_no_cost = self.long_cum_no_cost
        else:
            self.cum_ret_no_cost = self.short_cum_no_cost

        if self.long_cum_after_cost[-1] >= self.short_cum_after_cost[-1] or np.isnan(self.short_cum_no_cost[-1]):
            self.cum_ret_after_cost = self.long_cum_after_cost
        else:
            self.cum_ret_after_cost = self.short_cum_after_cost

        self.pool = '&'.join(cfg.index_list) +  '_' + ('fmv' if cfg.weight_index_by_fmv else 'equal')
        if cfg.use_independent_return_benchmark:
            self.pool += '_' + cfg.return_benchmark_index

    def plot_ls_no_cost(self, fig):
        """
        fig1：多空收益与指数收益比较
        """
        step = int(len(self.long_cum_no_cost.index) / 6)
        ax1 = fig.add_subplot(6, 2, 1)
        p1, = ax1.plot(self.long_short_cum_no_cost * 100)
        p2, = ax1.plot(self.long_cum_no_cost * 100)
        p3, = ax1.plot(self.short_cum_no_cost * 100)
        p4, = ax1.plot(self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Test Result No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_excess_no_cost(self, fig):
        """
        fig2：多空超额收益
        """
        step = int(len(self.long_cum_no_cost.index) / 6)
        ax1 = fig.add_subplot(6, 2, 2)
        p1, = ax1.plot(self.long_short_cum_no_cost * 100)
        p2, = ax1.plot(self.long_cum_no_cost * 100 - self.index_cum * 100)
        p3, = ax1.plot(self.short_cum_no_cost * 100 - self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3], labels=['long_short', 'long', 'short'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Test Result No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_maxdd_no_cost(self, fig):
        """
        fig1.1：多头收益回撤
        """
        self.cum_ret_no_cost_rate = self.cum_ret_no_cost * 100
        self.cum_ret_no_cost_maxdd = self.cum_ret_no_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.cum_ret_no_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(6, 2, 3)
        p1 = ax1.bar(x = list(self.cum_ret_no_cost_maxdd.index), height = list(self.cum_ret_no_cost_maxdd.values), color = 'coral')
        ax1.legend(handles=[p1], labels=['long_maxdd'], loc='best', fontsize=10)
        if len(self.cum_ret_no_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.cum_ret_no_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.cum_ret_no_cost_maxdd.index)[::step])
        ax1.set_title("Long-Absolute MaxDrawdown No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_alpha_maxdd_no_cost(self, fig):
        """
        fig2.1：超额收益回撤
        """
        self.alpha_cum_no_cost_rate = self.cum_ret_no_cost * 100 - self.index_cum * 100
        self.alpha_cum_no_cost_maxdd = self.alpha_cum_no_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.alpha_cum_no_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(6, 2, 4)
        p1 = ax1.bar(x = list(self.alpha_cum_no_cost_maxdd.index), height = list(self.alpha_cum_no_cost_maxdd.values), color = 'springgreen')
        ax1.legend(handles=[p1], labels=['alpha_maxdd'], loc='best', fontsize=10)
        if len(self.alpha_cum_no_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.alpha_cum_no_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.alpha_cum_no_cost_maxdd.index)[::step])
        ax1.set_title("Alpha MaxDrawdown No Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_after_cost(self,fig):
        '''
        fig3: 多空收益扣费后
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(6, 2, 5)
        p1, = ax1.plot(self.long_short_cum_after_cost * 100)
        p2, = ax1.plot(self.long_cum_after_cost * 100)
        p3, = ax1.plot(self.short_cum_after_cost * 100)
        p4, = ax1.plot(self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3, p4], labels=['long_short', 'long', 'short', 'index'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Test Result After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_excess_after_cost(self,fig):
        '''
        fig4: 多空超额收益扣费后
        :param fig:
        :return:
        '''
        step = int(len(self.long_cum_after_cost.index) / 6)
        ax1 = fig.add_subplot(6, 2, 6)
        p1, = ax1.plot(self.long_short_cum_after_cost * 100)
        p2, = ax1.plot(self.long_cum_after_cost * 100 - self.index_cum * 100)
        p3, = ax1.plot(self.short_cum_after_cost * 100 - self.index_cum * 100)
        ax1.legend(handles=[p1, p2, p3], labels=['long_short', 'long', 'short'], loc='best',
                   fontsize=10)
        if len(self.long_cum_after_cost.index) < 20:
            ax1.set_xticks(self.long_cum_after_cost.index)
        else:
            ax1.set_xticks(self.long_cum_after_cost.index[::step])
        ax1.set_title("Long-Short Test Result After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_ls_maxdd_after_cost(self, fig):
        """
        fig3.1：多头收益回撤
        """
        self.cum_ret_after_cost_rate = self.cum_ret_after_cost * 100
        self.cum_ret_after_cost_maxdd = self.cum_ret_after_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.cum_ret_after_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(6, 2, 7)
        p1 = ax1.bar(x = list(self.cum_ret_after_cost_maxdd.index), height = list(self.cum_ret_after_cost_maxdd.values), color = 'coral')
        ax1.legend(handles=[p1], labels=['long_maxdd'], loc='best', fontsize=10)
        if len(self.cum_ret_after_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.cum_ret_after_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.cum_ret_after_cost_maxdd.index)[::step])
        ax1.set_title("Long-Absolute MaxDrawdown After Cost", fontsize=15)
        ax1.grid(axis='y')

    def plot_alpha_maxdd_after_cost(self, fig):
        """
        fig4.1：超额收益回撤
        """
        self.alpha_cum_after_cost_rate = self.cum_ret_after_cost * 100 - self.index_cum * 100
        self.alpha_cum_after_cost_maxdd = self.alpha_cum_after_cost_rate.expanding().apply(self.cal_maxdd_local)
        step = int(len(self.alpha_cum_after_cost_maxdd.index) / 6)
        ax1 = fig.add_subplot(6, 2, 8)
        p1 = ax1.bar(x = list(self.alpha_cum_after_cost_maxdd.index), height = list(self.alpha_cum_after_cost_maxdd.values), color = 'springgreen')
        ax1.legend(handles=[p1], labels=['alpha_maxdd'], loc='best', fontsize=10)
        if len(self.alpha_cum_after_cost_maxdd.index) < 20:
            ax1.set_xticks(list(self.alpha_cum_after_cost_maxdd.index))
        else:
            ax1.set_xticks(list(self.alpha_cum_after_cost_maxdd.index)[::step])
        ax1.set_title("Alpha MaxDrawdown After Cost", fontsize=15)
        ax1.grid(axis='y')

    @staticmethod
    def cal_maxdd(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return max(drawdowns)

    @staticmethod
    def cal_maxdd_local(array):
        return - np.nanmax(array) + array[len(array) - 1]


    def plot_turnover(self, fig):
        """
        fig11：换手率
        """
        step = len(self.turnover_long)//10
        ax6 = fig.add_subplot(6, 1, 5)
        ax6.plot(self.turnover_long, lw=0.7)
        ax6.plot(self.turnover_short, lw=0.7)
        ax6.plot(self.turnover, lw=0.7)
        ax6.set_xticks(self.turnover.index[::step])
        ax6.set_xticklabels(self.turnover.index[::step])
        ax6.legend(labels=["TR_long", "TR_short", "TR"], fontsize=12)
        ax6.set_title("Turnover", fontsize=15)

    def plot_table_summary(self, fig):
        """
        table: stats summary
        keywords:
        """
        ax10 = fig.add_subplot(6, 1, 6)
        plt.axis('off')
        # 列名是统计量名字，行名是年份
        col_labels = list(self.summary_stats.keys())
        row_labels = list((list(self.summary_stats.values())[:])[0].keys())
        cell_text = pd.DataFrame(self.summary_stats).values
        for i in range(cell_text.shape[0]):
            cell_text[i] = [round(x, 2) for x in cell_text[i]]
        cell_text = cell_text[-1:,:]
        row_labels = row_labels[-1:]

        table = ax10.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center',
                           cellLoc='center', rowLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        ax10.set_title("Stats Summary", fontsize=15)
        table.scale(1, 2.5)

    def save_table_summary(self,factor_name):
        '''
        把因子分年绩效存到新的文件夹中
        :param factor_name:
        :return:
        '''
        d = cfg.output_path
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M%S', time.localtime(time.time()))
        # 创建当日文件夹
        save_path = '{}/signal_test_table_file_{}_{}'.format(d, curr_date, cfg.test_name)
        table_path = '{}/{}_{}_{}day_{}_{}_{}.csv'.format(save_path, factor_name, self.pool, cfg.adj_freq, cfg.cost, cfg.head,curr_date+curr_time)
        col_labels = list(self.summary_stats.keys())
        row_labels = list((list(self.summary_stats.values())[:])[0].keys())
        text_table = pd.DataFrame(self.summary_stats,index=row_labels,columns=col_labels)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        text_table.to_csv(table_path)

    def save_fig(self, factor_name):
        """
        :return:
        """
        # 设置当前日期&时间
        d = cfg.output_path
        curr_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        curr_time = time.strftime('%H%M%S', time.localtime(time.time()))

        # 创建当日文件夹
        save_path = '{}/signal_test_file_{}_{}'.format(d, curr_date, cfg.test_name)
        fig_path = '{}/{}_{}_{}_{}_{}.png'.format(save_path, factor_name, self.pool, cfg.cost, cfg.head, curr_date+curr_time)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # 画图存图
        fig = plt.figure(figsize=(20, 30))
        fig.suptitle('{} Signal Performance'.format(self.signal_name),fontsize=20)
        self.plot_ls_no_cost(fig)
        self.plot_ls_excess_no_cost(fig)
        self.plot_ls_after_cost(fig)
        self.plot_ls_excess_after_cost(fig)
        self.plot_turnover(fig)
        self.plot_ls_maxdd_no_cost(fig)
        self.plot_alpha_maxdd_no_cost(fig)
        self.plot_ls_maxdd_after_cost(fig)
        self.plot_alpha_maxdd_after_cost(fig)
        self.plot_table_summary(fig)
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)  # 纯靠实验得出，为了让suptitle有个不和图片打架的位置
        
        self.save_table_summary(factor_name)
        # 存储
        plt.subplots_adjust(wspace=0.1,hspace=0.3)
        plt.savefig(fig_path, dpi=300)
