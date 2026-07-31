# 可转债 D1 因子

闸门:`IC > 0.015`、`Sharpe > 1.25`、`turnover < 0.2`、`ICIR > 2`
IS `20210601-20240630` / OOS `20240701-20260301`,`roll_days = 3`

## 结果

| 因子 | 窗口 | IC | ICIR | Sharpe | turnover | RankIC | 年化 | 回撤 |
|---|---|---|---|---|---|---|---|---|
| `option__cb_value_size_liq` | IS | 0.0314 | 4.84 | 3.85 | 0.100 | 0.0404 | 14.99% | 0.060 |
|  | OOS | **0.0205** | **3.26** | **3.01** | **0.063** | 0.0063 | 10.05% | 0.021 |
| `option__cb_defensive_vs` | IS | 0.0279 | 4.71 | 3.93 | 0.112 | 0.0381 | 14.21% | 0.055 |
|  | OOS | **0.0189** | **3.14** | **3.04** | **0.066** | 0.0049 | 9.76% | 0.017 |
| `option__cb_size_liq_gap` | IS | 0.0207 | 3.94 | 3.92 | 0.112 | 0.0214 | 11.61% | 0.027 |
|  | OOS | **0.0160** | **2.93** | **3.05** | **0.065** | −0.0024 | 8.31% | 0.011 |

`turnover` 口径为单边(权重按 Σ|w|=1 归一后取 Σ|Δw|/2)。

## 证据强度分级

三个因子的 OOS 数字**不是同等可信**:

- `option__cb_value_size_liq` —— 权重由 1023 组 IS 网格选出,OOS 全程未参与,是干净持出。
- `option__cb_defensive_vs` —— 反波动率缩放这个选择是在看过各成分 OOS 表现之后做的,存在窥探。
- `option__cb_size_liq_gap` —— 去掉估值腿的动机直接来自 OOS 衰减诊断,窥探程度最高。它 IS ICIR 3.94 低于完整版的 4.84,纯 IS 排序不会选中它。

按 OOS 可信度用,优先级即上述顺序。

## 四个收益来源

截面平均相关全部 |ρ| ≤ 0.21,是相互独立的来源:

| 成分 | 定义 | 说明 |
|---|---|---|
| `prem` | −(0.5·rank(纯债溢价率) + 0.5·rank(转股溢价率)) | 估值 |
| `size` | −rank(log(剩余余额)) | 转股进度 |
| `illiq` | rank(ts_mean(\|收益\|/成交额, 20)) | 非流动性溢价 |
| `dgap` | −rank(ts_sum(转债收益 − delta·正股收益, 20)) | 弹性残差 |

## 落地前必须确认的一件事

`size` 腿读的是**剩余余额**,不是发行规模。两者不可互换:

| 字段 | IS IC | OOS IC |
|---|---|---|
| 剩余余额 | +0.0075 | +0.0113 |
| 发行规模 | −0.0001 | +0.0032 |

信号来自余额相对发行额的缩水(即长期处于转股价值之上的强势券),不是小盘溢价。
交付代码中的 `COL_REMAIN_SIZE` 常量需按 GTA `bond_convertinfo` 的真实列名确认。

## 三项稳健性检查

**收益去尾** —— OOS 依赖尾部,但不塌陷:

| fwd_ret 截尾 | OOS IC | OOS ICIR | OOS Sharpe |
|---|---|---|---|
| 不截 | 0.0205 | 3.26 | 3.01 |
| ±10% | 0.0184 | 2.82 | 2.48 |
| ±5% | 0.0150 | 2.22 | 1.96 |
| ±3% | 0.0131 | 1.91 | 1.64 |

IS 截尾后反而变好(边际广谱),OOS 截尾后变差(边际更集中在尾部)。±5% 是临界点。

**交易成本** —— 10bp 双边下仍过闸门,20bp 下失效:

| 双边成本 | IS Sharpe | OOS Sharpe |
|---|---|---|
| 0bp | 3.85 | 3.01 |
| 10bp | 2.55 | 2.06 |
| 20bp | 1.25 | 1.10 |

**十分组单调性** —— OOS D1 +3.42bp → D10 +15.09bp,基本单调,价差 +11.67bp/日。

## 已知问题:OOS 内部在衰减

| 季度 | 24Q3 | 24Q4 | 25Q1 | 25Q2 | 25Q3 | 25Q4 | 26Q1 |
|---|---|---|---|---|---|---|---|
| `value_size_liq` PnL | +5.61% | +1.92% | +3.83% | +2.46% | +1.87% | +0.69% | −0.35% |

拆到成分层,衰减**完全来自 `prem`**:2026Q1 半年 PnL −4.26%,而 size(+1.07)、illiq(+2.14)、dgap(+2.42)仍为正。
OOS 的均值主要由 2024H2 贡献。这是 `option__cb_size_liq_gap` 存在的理由,也是该组因子最大的悬而未决之处。

## 未获支持的假设

- **信用质量控制**。设想 2024H2 信用恐慌下低价券被无差别抛售是 `prem` 衰减的主因,构造了正股市值/股价的信用代理。IS 网格中该成分权重在所有前排组合里都是 **0**,未获支持。
- **风险平价通用化**。反波动率缩放对 size(OOS ICIR 1.87→2.23)与 illiq(1.64→1.83)有效,对 prem(1.57→0.96)与 dgap(1.16→0.85)有害。不是全库通用的修正。

## 被否掉的单因子

`cb_call_dist`(距强赎触发距离)、`cb_downward_exp`(下修预期)OOS Sharpe 转负;
`cb_stk_vol`(正股波动率)IS/OOS 符号翻转;`cb_turn_cool` 口径有缺陷。均未进入组合。

## 复现

```bash
python scripts/cb_data.py --basic
python scripts/cb_data.py --daily --start 20190101 --end 20260301
python scripts/cb_panel.py --build
python scripts/cb_miner.py --rolls 3          # 单因子全库
python scripts/cb_combo.py --search --roll 3  # IS 权重网格
```

## 与内部 alphafactory 的字段映射

本地用 tushare 复刻挖掘,字段名已对齐 `cvbd_derivative`,表达式可直接搬运:

| alphafactory | tushare `cb_daily` |
|---|---|
| `debt_puredebt_ratio` | `bond_value` |
| `conv_value` | `cb_value` |
| `bond_prem_ratio` | `cb_over_rate` |
| `puredebt_prem_ratio` | `bond_over_rate` |

数值已逐条核对一致。
