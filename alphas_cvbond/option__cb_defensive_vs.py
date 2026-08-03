"""已作废 —— 请勿使用。

本因子含 cb_basic.remain_size,该字段是调用 API 当天的快照,被 merge 到整条
时间序列后等价于泄漏「该券未来是否退市」:零值样本中 99.9% 最终退市,零值率
按年从 2019 的 99.1% 单调降到 2026 的 21.3%。限制在至今仍在市的券上重测,
该腿 IS IC 从 +0.0075 变为 -0.0019,ICIR 从 1.82 变为 -0.17——样本内的边际
全部来自未来信息。

PIT 替代口径见 scripts/cb_panel.py 的 _pit_conversion(基于 cb_share)。
干净数据下的真实水平与容量结论见 alphas_cvbond/README.md。
"""

import os
import getpass
import pandas as pd
import numpy as np
os.environ['AF_PROD_ENV'] = '1'

import datetime

from alphafactory.alpha.AShare.cvbond.cvbond_1d.cvbond_alpha_base import CVBondAlphaBase


ty = 'bond'

ROLL_DAYS = 3
ILLIQ_WIN = 20
GAP_WIN = 20

# 四成分权重,只在 IS(20210601-20240630)网格上选出,OOS 未参与
W_PREM = 0.30
W_SIZE = 0.30
W_ILLIQ = 0.20
W_GAP = 0.20
VOL_WIN = 60          # 风险平价所用的个券波动率窗口
VOL_FLOOR_Q = 0.10    # 波动率地板分位,防低波券仓位爆炸

# GTA bond_convertinfo 中「剩余余额」的列名。必须是剩余余额,不是发行规模:
# 用发行规模替代时该成分 IC 从 +0.0113 掉到 +0.0032,信号完全消失。
# 因子捕捉的是转股进度(余额相对发行额缩水 = 长期在转股价值之上),不是小盘溢价。
COL_REMAIN_SIZE = 'OutstandingBalance'


def _cs_rank(s):
    if s.notna().sum() < 10:
        return pd.Series(np.nan, index=s.index)
    return (s.rank(pct=True) - 0.5) * np.sqrt(12.0)


def _cs_z(s):
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - s.mean()) / (sd + 1e-8)


class SrAlpha(CVBondAlphaBase):
    name = 'option__cb_defensive_vs'
    description = 'same four sources, with balance and illiquidity legs inverse-vol scaled'
    alpha_definition = ('0.3*z(-(0.5*rank(floor_prem)+0.5*rank(conv_prem))) '
                        '+ 0.3*z(-rank(log(remain_size))/vol60) '
                        '+ 0.2*z(rank(ts_mean(|b_ret|/b_amount,20))/vol60) '
                        '+ 0.2*z(-rank(ts_sum(b_ret - delta*s_ret,20)))')
    alpha_compute_method = 'roll_mean'
    alpha_source = 'self_idea'
    created_by = 'huyukun'
    univ = 'AA'
    alpha_type = 'TC'
    mail_to = 'chaoping.pei@dfc.sh'

    def _cal_sd(self, x):
        if x.std == 0:
            return x * 0
        else:
            return (x - x.mean()) / (x.std() + 0.0004)

    def process_convert_bond_data(self, start_date, end_date):
        self.path_stock_pv = 'StockRaw_1d/pv/gta/base'
        self.path_bond_pv = 'StockRaw_1d/bond/gta/pv'
        self.path_cvbd_derivative = 'StockRaw_1d/fundamental/hermes/raw/bond_conv_der'
        self.path_cvbd_pv_AA = 'StockRaw_1d/bond/CVBD/cbond_univ/univ_AA'
        self.path_bond_convertinfo = 'StockRaw_1d/bond/gta/basic/bond_convertinfo'

        dump_s_date = self.cal.prev_business_day('20200601')

        bond_pv = self.cli.read(
            f'{self.path_bond_pv}/close',
            start=dump_s_date,
            end=end_date
        ).unstack().to_frame('b_close').reset_index().rename(columns={'level_0': 'b_sym'})

        bond_amt = self.cli.read(
            f'{self.path_bond_pv}/amount',
            start=dump_s_date,
            end=end_date
        ).unstack().to_frame('b_amount').reset_index().rename(columns={'level_0': 'b_sym'})

        stock_pv = self.cli.read(
            f'{self.path_stock_pv}/close',
            start=dump_s_date,
            end=end_date
        ).unstack().to_frame('s_close').reset_index().rename(columns={'level_0': 's_sym'})

        close_bward = self.cli.read(
            f'{self.path_stock_pv}/close_bward_adj',
            start=dump_s_date,
            end=end_date
        ).unstack().to_frame('bward_close').reset_index().rename(columns={'level_0': 's_sym'})

        info_bond = self.cli.read(
            f'{self.path_bond_convertinfo}',
            start=dump_s_date,
            end=end_date,
            columns=['Symbol', COL_REMAIN_SIZE]
        ).reset_index()
        info_bond = info_bond.rename(columns={'Symbol': 'b_sym', COL_REMAIN_SIZE: 'remain_size'})
        info_bond['b_sym'] = info_bond['b_sym'].astype(str)
        info_bond = info_bond.sort_values(['b_sym']).drop_duplicates(subset=['b_sym'], keep='last')
        info_bond = info_bond.reset_index(drop=True)

        cvbd_derivative = self.cli.read(
            f'{self.path_cvbd_derivative}',
            columns=['date', 'ticker_symbol', 'bond_prem_ratio', 'puredebt_prem_ratio', 'conv_price',
                     'conv_value', 'debt_puredebt_ratio', 'year_to_mat', 'call_price', 'put_price',
                     'trade_date', 'ytm']
        ).reset_index()
        cvbd_derivative = cvbd_derivative.rename(columns={'ticker_symbol': 'b_sym'})
        cvbd_derivative = cvbd_derivative.sort_values(['trade_date', 'b_sym', 'date'])
        cvbd_derivative['date'] = pd.to_datetime(cvbd_derivative['trade_date']).dt.normalize()
        cvbd_derivative['b_sym'] = cvbd_derivative['b_sym'].astype(str)
        cvbd_derivative = cvbd_derivative.drop('trade_date', axis=1)
        cvbd_derivative = cvbd_derivative[
            (cvbd_derivative['date'] >= pd.to_datetime(str(dump_s_date))) &
            (cvbd_derivative['date'] <= pd.to_datetime(str(end_date)))]
        cvbd_derivative = cvbd_derivative.drop_duplicates(subset=['b_sym', 'date'], keep='last').reset_index(drop=True)

        cvbd_pv_AA = self.cli.read(
            f'{self.path_cvbd_pv_AA}',
            start=dump_s_date,
            end=end_date,
            columns=['date', 'b_sym', 'close', 'ConvertPrice', 's_sym']
        ).reset_index()
        cvbd_pv_AA = cvbd_pv_AA[cvbd_pv_AA['ConvertPrice'].notna()].reset_index(drop=True)

        cvbd_pv_AA['bond_sym'] = cvbd_pv_AA['b_sym'].astype(str).copy()
        cvbd_pv_AA['b_sym'] = cvbd_pv_AA['bond_sym'].str[:6]
        for _df in (cvbd_pv_AA, stock_pv, close_bward, bond_pv, bond_amt):
            _df['date'] = pd.to_datetime(_df['date']).dt.normalize()
        bond_pv['b_sym'] = bond_pv['b_sym'].astype(str)
        bond_amt['b_sym'] = bond_amt['b_sym'].astype(str)

        df = cvbd_pv_AA.merge(stock_pv, on=['date', 's_sym'], how='left')
        df = df.merge(close_bward, on=['date', 's_sym'], how='left')
        df = df.merge(bond_pv, on=['date', 'b_sym'], how='left')
        df = df.merge(bond_amt, on=['date', 'b_sym'], how='left')
        df = df.drop_duplicates(subset=['date', 'b_sym'], keep='first')
        df = df.merge(cvbd_derivative, on=['date', 'b_sym'], how='left')
        df = df.merge(info_bond[['b_sym', 'remain_size']], on='b_sym', how='left')

        df.drop(columns='b_close', inplace=True)
        df.rename(columns={'close': 'b_close'}, inplace=True)
        df.drop(columns='b_sym', inplace=True)
        df.rename(columns={'bond_sym': 'b_sym'}, inplace=True)

        df = df.sort_values(['b_sym', 'date'])
        df['b_log_ret'] = df.groupby('b_sym')['b_close'].transform(lambda x: np.log(x / x.shift(1)))
        df['s_log_ret'] = df.groupby('b_sym')['bward_close'].transform(lambda x: np.log(x / x.shift(1)))
        return df

    def generate(self, start_date: datetime.datetime, end_date: datetime.datetime, is_inc=False):
        cli = self.xdata_cli
        cald = self.cal
        self.cli = self.xdata_cli

        dump_date = cald.prev_business_day(start_date, 500)
        univ = cli.read(f'StockRaw_1d/bond/CVBD/cbond_univ/univ_{self.univ}', start=dump_date, end=end_date)

        df = self.process_convert_bond_data(dump_date, cald.prev_business_day())
        df = df[df.date <= pd.to_datetime(str(end_date))].copy()

        # --- 成分 1:双溢价率估值 ---
        pure_bond = df['debt_puredebt_ratio'].where(df['debt_puredebt_ratio'] > 0)
        conv_val = df['conv_value'].where(df['conv_value'] > 0)
        df['floor_prem'] = df['b_close'] / pure_bond - 1
        df['conv_prem'] = df['b_close'] / conv_val - 1
        prem = -(0.5 * df.groupby('date')['floor_prem'].transform(_cs_rank)
                 + 0.5 * df.groupby('date')['conv_prem'].transform(_cs_rank))

        # --- 成分 2:剩余余额(转股进度)---
        df['log_remain'] = np.log(df['remain_size'].clip(lower=0.01))
        size = -df.groupby('date')['log_remain'].transform(_cs_rank)

        # --- 成分 3:非流动性(转债 amihud)---
        df['amihud'] = df['b_log_ret'].abs() / df['b_amount'].replace(0, np.nan)
        df['amihud_m'] = df.groupby('b_sym')['amihud'].transform(
            lambda x: x.rolling(ILLIQ_WIN, min_periods=ILLIQ_WIN // 2).mean())
        illiq = df.groupby('date')['amihud_m'].transform(_cs_rank)

        # --- 成分 4:弹性残差(转债实际涨跌 vs 理论 delta * 正股涨跌 的累计偏离)---
        delta = (df['conv_value'] / df['b_close'].replace(0, np.nan)).clip(0, 1)
        df['resid'] = df['b_log_ret'] - delta * df['s_log_ret']
        df['resid_s'] = df.groupby('b_sym')['resid'].transform(
            lambda x: x.rolling(GAP_WIN, min_periods=GAP_WIN // 2).sum())
        dgap = -df.groupby('date')['resid_s'].transform(_cs_rank)

        # 规模与流动性两腿按个券波动率反比缩放。诊断显示这两个来源的收益被高弹性
        # 个券的尾部主导,除以波动率把仓位挪走后 OOS ICIR 1.87->2.23。估值腿反而
        # 会被这个操作损害(OOS ICIR 1.57->0.96),故只对这两腿施加。
        df['b_vol'] = df.groupby('b_sym')['b_log_ret'].transform(
            lambda x: x.rolling(VOL_WIN, min_periods=VOL_WIN // 3).std())
        vol_floor = df['b_vol'].quantile(VOL_FLOOR_Q)
        vol = df['b_vol'].clip(lower=vol_floor)
        size = size / vol
        illiq = illiq / vol

        # 先各自截面 z 再加权,使权重可比
        df['blend'] = (W_PREM * prem.groupby(df['date']).transform(_cs_z)
                       + W_SIZE * size.groupby(df['date']).transform(_cs_z)
                       + W_ILLIQ * illiq.groupby(df['date']).transform(_cs_z)
                       + W_GAP * dgap.groupby(df['date']).transform(_cs_z))

        result_df = df[['date', 'b_sym', 'blend']].dropna(subset=['date', 'b_sym'])
        result_df = result_df.drop_duplicates(subset=['date', 'b_sym'], keep='last')

        alp = result_df.set_index(['date', 'b_sym'])[['blend']]
        alp.columns = ['bond_alp']
        result1 = alp.sort_index().groupby('b_sym').rolling(window=ROLL_DAYS).mean().droplevel(0)
        result1.columns = [f'{col}_mean' for col in result1.columns]

        result = result1.sort_index().reindex(univ.index).loc[pd.to_datetime(str(start_date)):].fillna(0)
        result[self.name] = result[f'{ty}_alp_mean']
        result = result.sort_index().groupby('date').transform(self._cal_sd).fillna(0)
        result.index.names = ['date', 'b_sym']

        if start_date == end_date:
            alpha = result[self.name].unstack().loc[end_date].to_frame(end_date).T
        else:
            alpha = result[self.name].unstack().loc[start_date:end_date]
        alpha.index.name = 'date'
        return {self.data_name: alpha}


if __name__ == "__main__":
    is_inc = "{{is_inc}}" == '1'
    write_mode = "{{write_mode}}"
    is_inc = False; write_mode = ''
    a = SrAlpha()
    cli = a.xdata_cli
    alpha_all = a.generate(is_inc=is_inc, start_date=a.cal.prev_business_day('20210601'),
                           end_date=a.cal.prev_business_day('20240630'))

    ALPHA_STORE = f'StockRaw_1d/bond/CVBD/bond_features_intern/{getpass.getuser()}__{SrAlpha.name}'
    try:
        cli.register(ALPHA_STORE, 'daily', 'x')
        print('registered:', ALPHA_STORE)
    except Exception as ex:
        print(f'[info] register 跳过：{type(ex).__name__}: {str(ex)[:160]}')
    try:
        cli.append(ALPHA_STORE, alpha_all['alpha'])
        print('appended:', ALPHA_STORE, alpha_all['alpha'].shape)
    except Exception as ex:
        print(f'[warn] 落库失败 {ALPHA_STORE}\n       {type(ex).__name__}: {ex}')
        print('       self-eval 不依赖落库，继续。')

    a.pre_insample_check(alpha_all['alpha'])

    alpha = alpha_all['alpha']
    s, e = alpha.index[0], alpha.index[-1]
    eval_results = []
    for el in a.eval_list:
        el.load_x(x=alpha)
        el.load_y(start_date=s, end_date=e)
        eval_results.append(el.generate(s, e)[el.data_name])
    pnl_is = eval_results[0]
    zeroratio_is = eval_results[1]['zero_ratio']
    turnover_is = eval_results[2]['turnover']
    ca_is = a.performance_stat(pnl_is, zeroratio_is, turnover_is)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pnl_curve = pnl_is.sum(axis=1) if isinstance(pnl_is, pd.DataFrame) else pnl_is
    plt.figure(figsize=[10, 5])
    plt.plot(pnl_curve.cumsum())
    plt.axhline(y=0, color='black', linestyle='--', label='Horizontal Line')
    plt.legend([SrAlpha.name])
    plt.title('alpha cumpnl')
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{SrAlpha.name}_cumpnl.png')
    plt.savefig(out_png)
    print('saved:', out_png)
    print(ca_is)
