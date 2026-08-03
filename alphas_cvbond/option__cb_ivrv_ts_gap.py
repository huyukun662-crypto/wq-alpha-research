import os
import getpass
import pandas as pd
import numpy as np
os.environ['AF_PROD_ENV'] = '1'

import datetime

from alphafactory.alpha.AShare.cvbond.cvbond_1d.cvbond_alpha_base import CVBondAlphaBase


ty = 'bond'

ROLL_DAYS = 3
RV_WIN = 60          # 正股已实现波动率窗口
TSVAL_WIN = 250      # 个券溢价率的自身历史窗口
CLIP = 2.5           # 截面 z 截尾,压住单券权重上限
RISK_FREE = 0.02

W_IVRV = 0.4
W_TSVAL = 0.4
W_DGAP = 0.2
DGAP_WIN = 20

# 中性化暴露:价格水平与流动性。两者都先做截面 rank 再进回归——
# 直接用原始值(转债价格可达 400+、成交额跨三个数量级)会让逐日 beta 被离群点带偏。
NEUT_COLS = ('price_rank', 'adv_rank')


def _cs_rank(s):
    if s.notna().sum() < 10:
        return pd.Series(np.nan, index=s.index)
    return (s.rank(pct=True) - 0.5) * np.sqrt(12.0)


def _cs_z(s, clip=None):
    sd = s.std()
    out = s * 0 if (not np.isfinite(sd) or sd == 0) else (s - s.mean()) / (sd + 1e-8)
    return out.clip(-clip, clip) if clip else out


def _erf(x):
    """Abramowitz-Stegun 7.1.26,精度 1.5e-7。"""
    sgn = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sgn * y


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _bs_call(S, K, T, sigma, r=RISK_FREE):
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
        return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d1 - sq)


def _implied_vol(price, S, K, T, lo=0.02, hi=2.0, iters=40):
    """向量化二分反解。转债 moneyness 跨度极大(转股价值可从 26 到 331),
    ATM 近似 sigma≈C/(0.4·S·√T) 在深度实值/虚值处失真,必须真反解。
    贴边即无解,置 NaN 而不是留一个假值。"""
    lo = np.full_like(price, lo, dtype=float)
    hi = np.full_like(price, hi, dtype=float)
    ok = (np.isfinite(price) & (price > 0) & np.isfinite(S) & (S > 0)
          & np.isfinite(K) & (K > 0) & np.isfinite(T) & (T > 0))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = _bs_call(S, K, T, mid) < price
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(ok & (out > 0.025) & (out < 1.95), out, np.nan)


def _neutralize(df, col):
    """逐日截面回归取残差。groupby.apply 的拼接顺序按组键排序,与输入行序不同,
    必须显式 reindex——否则信号与日期错位,表现为换手暴涨、IC 归零。"""
    cols = list(NEUT_COLS)

    def _resid(g):
        m = g[col].notna() & g[cols].notna().all(axis=1)
        if m.sum() < len(cols) + 10:
            return pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(m.sum())] + [g.loc[m, c].to_numpy() for c in cols])
        y = g.loc[m, col].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out = pd.Series(np.nan, index=g.index)
        out.loc[m] = y - X @ beta
        return out

    return df.groupby('date', group_keys=False).apply(_resid).reindex(df.index)


class SrAlpha(CVBondAlphaBase):
    name = 'option__cb_ivrv_ts_gap'
    description = 'implied-minus-realised vol, own-history premium deviation, delta residual'
    alpha_definition = ('0.4*z(neut(-rank(IV - RV60))) + 0.4*z(neut(-rank(zscore(conv_prem, 250)))) '
                        '+ 0.2*z(neut(-rank(ts_sum(b_ret - delta*s_ret, 20))))')
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

        dump_s_date = self.cal.prev_business_day('20200601')

        bond_amt = self.cli.read(
            f'{self.path_bond_pv}/amount', start=dump_s_date, end=end_date
        ).unstack().to_frame('b_amount').reset_index().rename(columns={'level_0': 'b_sym'})

        stock_pv = self.cli.read(
            f'{self.path_stock_pv}/close', start=dump_s_date, end=end_date
        ).unstack().to_frame('s_close').reset_index().rename(columns={'level_0': 's_sym'})

        close_bward = self.cli.read(
            f'{self.path_stock_pv}/close_bward_adj', start=dump_s_date, end=end_date
        ).unstack().to_frame('bward_close').reset_index().rename(columns={'level_0': 's_sym'})

        cvbd_derivative = self.cli.read(
            f'{self.path_cvbd_derivative}',
            columns=['date', 'ticker_symbol', 'bond_prem_ratio', 'puredebt_prem_ratio',
                     'conv_price', 'conv_value', 'debt_puredebt_ratio', 'year_to_mat', 'trade_date']
        ).reset_index()
        cvbd_derivative = cvbd_derivative.rename(columns={'ticker_symbol': 'b_sym'})
        cvbd_derivative = cvbd_derivative.sort_values(['trade_date', 'b_sym', 'date'])
        cvbd_derivative['date'] = pd.to_datetime(cvbd_derivative['trade_date']).dt.normalize()
        cvbd_derivative['b_sym'] = cvbd_derivative['b_sym'].astype(str)
        cvbd_derivative = cvbd_derivative.drop('trade_date', axis=1)
        cvbd_derivative = cvbd_derivative[
            (cvbd_derivative['date'] >= pd.to_datetime(str(dump_s_date)))
            & (cvbd_derivative['date'] <= pd.to_datetime(str(end_date)))]
        cvbd_derivative = cvbd_derivative.drop_duplicates(
            subset=['b_sym', 'date'], keep='last').reset_index(drop=True)

        cvbd_pv_AA = self.cli.read(
            f'{self.path_cvbd_pv_AA}', start=dump_s_date, end=end_date,
            columns=['date', 'b_sym', 'close', 'ConvertPrice', 's_sym']
        ).reset_index()
        cvbd_pv_AA = cvbd_pv_AA[cvbd_pv_AA['ConvertPrice'].notna()].reset_index(drop=True)
        cvbd_pv_AA['bond_sym'] = cvbd_pv_AA['b_sym'].astype(str).copy()
        cvbd_pv_AA['b_sym'] = cvbd_pv_AA['bond_sym'].str[:6]
        for _df in (cvbd_pv_AA, stock_pv, close_bward, bond_amt):
            _df['date'] = pd.to_datetime(_df['date']).dt.normalize()
        bond_amt['b_sym'] = bond_amt['b_sym'].astype(str)

        df = cvbd_pv_AA.merge(stock_pv, on=['date', 's_sym'], how='left')
        df = df.merge(close_bward, on=['date', 's_sym'], how='left')
        df = df.merge(bond_amt, on=['date', 'b_sym'], how='left')
        df = df.drop_duplicates(subset=['date', 'b_sym'], keep='first')
        df = df.merge(cvbd_derivative, on=['date', 'b_sym'], how='left')
        df = df.rename(columns={'close': 'b_close'})
        df = df.drop(columns='b_sym').rename(columns={'bond_sym': 'b_sym'})

        df = df.sort_values(['b_sym', 'date'])
        df['s_log_ret'] = df.groupby('b_sym')['bward_close'].transform(
            lambda x: np.log(x / x.shift(1)))
        return df

    def generate(self, start_date: datetime.datetime, end_date: datetime.datetime, is_inc=False):
        cli = self.xdata_cli
        cald = self.cal
        self.cli = self.xdata_cli

        dump_date = cald.prev_business_day(start_date, 500)
        univ = cli.read(f'StockRaw_1d/bond/CVBD/cbond_univ/univ_{self.univ}',
                        start=dump_date, end=end_date)

        df = self.process_convert_bond_data(dump_date, cald.prev_business_day())
        df = df[df.date <= pd.to_datetime(str(end_date))].copy()

        # ---- 腿 1:期权贵贱(隐含波动率 - 正股已实现波动率)----
        # 转债 = 纯债 + 转股期权。剥掉纯债价值得到期权价格,按转股比例折成每股后
        # 用 BS 反解 IV,与正股 RV 比较。IV 低于 RV = 期权被低估。
        # 补偿的风险:承接 gamma 的一方要为条款不确定性(下修/强赎)与转债流动性定价。
        conv_ratio = (df['conv_value'] / df['s_close'].replace(0, np.nan)).replace(0, np.nan)
        opt_per_share = (df['b_close'] - df['debt_puredebt_ratio']) / conv_ratio
        iv = _implied_vol(opt_per_share.to_numpy(),
                          df['s_close'].to_numpy(),
                          df['conv_price'].to_numpy(),
                          df['year_to_mat'].clip(lower=0.1).to_numpy())
        rv = df.groupby('b_sym')['s_log_ret'].transform(
            lambda x: x.rolling(RV_WIN, min_periods=RV_WIN // 2).std()) * np.sqrt(252)
        df['ivrv'] = -df.assign(_v=pd.Series(iv, index=df.index) - rv) \
            .groupby('date')['_v'].transform(_cs_rank)

        # ---- 腿 2:时序估值偏离(溢价率相对该券自身 250 日历史)----
        # 不同券的合理溢价率本就不同(取决于正股波动率、剩余期限、条款)。跟自己比
        # 把这些个券固定效应差掉,因而不会退化成截面上的「买低价券」。
        # 补偿的风险:均值回归需要时间,期间承担条款与信用风险。
        prem = df['b_close'] / df['conv_value'].where(df['conv_value'] > 0) - 1
        mu = prem.groupby(df['b_sym']).transform(
            lambda x: x.rolling(TSVAL_WIN, min_periods=TSVAL_WIN // 2).mean())
        sd = prem.groupby(df['b_sym']).transform(
            lambda x: x.rolling(TSVAL_WIN, min_periods=TSVAL_WIN // 2).std())
        df['tsval'] = -df.assign(_v=(prem - mu) / sd.replace(0, np.nan)) \
            .groupby('date')['_v'].transform(_cs_rank)

        # ---- 腿 3:弹性残差(转债实际涨跌 vs 理论 delta×正股涨跌 的累计偏离)----
        # 转债二级深度远逊正股,价格对正股信息的吸收滞后。承接这段滞后需要承担
        # 隔夜与流动性风险,属做市/统计套利性质的补偿。
        df['b_log_ret'] = df.groupby('b_sym')['b_close'].transform(
            lambda x: np.log(x / x.shift(1)))
        delta = (df['conv_value'] / df['b_close'].replace(0, np.nan)).clip(0, 1)
        resid = df['b_log_ret'] - delta * df['s_log_ret']
        df['dgap'] = -df.assign(_v=resid.groupby(df['b_sym']).transform(
            lambda x: x.rolling(DGAP_WIN, min_periods=DGAP_WIN // 2).sum())) \
            .groupby('date')['_v'].transform(_cs_rank)

        # ---- 正则化:暴露中性化 + 截面 z 截尾 ----
        df['price_rank'] = df.groupby('date')['b_close'].transform(_cs_rank)
        df['adv_rank'] = df.groupby('date')['b_amount'].transform(_cs_rank)

        parts = []
        for col, wgt in (('ivrv', W_IVRV), ('tsval', W_TSVAL), ('dgap', W_DGAP)):
            resid = _neutralize(df[['date', col] + list(NEUT_COLS)], col)
            parts.append(wgt * resid.groupby(df['date']).transform(lambda x: _cs_z(x, CLIP)))
        df['blend'] = sum(parts)

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
