# GA 因子挖掘机报告 — 2026-07-24 10:30 UTC

- 方式:遗传算法(文法约束表达式树、锦标赛选择、子树交叉、四类变异、精英保留),结合 Worldquant_Mining 的窗口集/双闸门/IS-OOS 纪律
- fitness:IS(..20221231) 净 Sharpe - 惩罚(TO>35%、无效解、复杂度、与名人堂相关>0.7)
- OOS(20230101..)只在此报告中验证一次,GA 过程不可见
- 进化到第 20 代,评估缓存 848 条表达式

## 名人堂(按 IS fitness 排序)

| # | 表达式 | 代 | IS净Shp | IS TO | OOS净Shp | OOS毛Shp | 入选 |
|---|--------|----|---------|-------|----------|----------|------|
| 1 | `rank(ts_decay_linear(ts_decay_linear(div(debt_to_assets, vol), 120), 3))` | 0 | +1.99 | 0.02 | +0.32 | +0.61 | ❌ |
| 2 | `rank(neg(ts_mean(turnover_rate_f, 20)))` | 2 | +1.69 | 0.03 | +0.52 | +0.77 | ❌ |
| 3 | `rank(ts_max(ts_decay_linear(div(dv_ttm, vol), 120), 250))` | 16 | +1.74 | 0.01 | -0.29 | -0.04 | ❌ |
| 4 | `rank(ts_decay_linear(ts_decay_linear(sign(volume_ratio), 120), 3))` | 9 | +1.71 | 0.05 | -1.21 | -0.84 | ❌ |
| 5 | `rank(ts_decay_linear(ts_decay_linear(div(vol, vol), 120), 3))` | 9 | +1.65 | 0.05 | -1.14 | -0.76 | ❌ |
| 6 | `group_rank(ts_mean(ts_zscore(abs(grossprofit_margin), 250), 10), industry)` | 2 | +1.51 | 0.02 | +1.04 | +1.70 | ✅ |
| 7 | `group_rank(neg(ts_mean(turnover_rate_f, 120)), industry)` | 2 | +1.42 | 0.02 | +0.49 | +0.64 | ❌ |
| 8 | `rank(ts_mean(sub(ts_delta(lg_ratio, 3), div(amount, late_vol_share)), 60))` | 0 | +1.43 | 0.02 | +0.68 | +0.92 | ❌ |
| 9 | `group_rank(ts_decay_linear(ts_zscore(ts_min(sub(debt_to_assets, turnover_rate_f)` | 14 | +1.23 | 0.06 | -0.38 | +0.94 | ❌ |
| 10 | `group_rank(ts_decay_linear(neg(ts_std_dev(intraday_skew, 40)), 10), industry)` | 0 | +1.16 | 0.04 | +0.55 | +1.39 | ❌ |
| 11 | `group_rank(ep, industry)` | 0 | +1.02 | 0.02 | +0.35 | +0.69 | ❌ |
| 12 | `group_rank(ts_mean(div(abs(returns), amount), 20), industry)` | 0 | +1.09 | 0.04 | +0.41 | +1.17 | ❌ |
| 13 | `group_rank(ts_decay_linear(div(ts_decay_linear(grossprofit_margin, 20), div(log_` | 6 | +1.14 | 0.02 | +0.80 | +1.07 | ✅ |
| 14 | `rank(ts_decay_linear(ts_decay_linear(div(netprofit_yoy, vol), 120), 3))` | 5 | +1.05 | 0.01 | +0.33 | +0.65 | ❌ |
| 15 | `rank(ts_zscore(ts_min(lg_ratio, 250), 40))` | 17 | +0.94 | 0.04 | -0.76 | +1.23 | ❌ |
| 16 | `group_rank(dv_ttm, industry)` | 0 | +0.86 | 0.02 | +0.11 | +0.46 | ❌ |
| 17 | `group_rank(slog1p(ts_max(netprofit_yoy, 10)), industry)` | 5 | +0.91 | 0.02 | +0.44 | +0.84 | ❌ |
| 18 | `rank(ts_min(neg(ts_mean(ts_max(ts_zscore(vol, 60), 10), 20)), 250))` | 4 | +0.99 | 0.01 | +0.20 | +0.49 | ❌ |
| 19 | `group_rank(ts_decay_linear(ts_zscore(grossprofit_margin, 40), 5), industry)` | 11 | +0.84 | 0.10 | -0.98 | +0.14 | ❌ |
| 20 | `rank(ts_decay_linear(ts_decay_linear(bp, 120), 3))` | 6 | +0.82 | 0.01 | +0.86 | +0.99 | ✅ |

## 结论:3/20 通过 OOS 闸门(OOS 净 Sharpe>0 且 ≥ IS 的 50%)

> 入选者可并入 tushare_mining_report 的低相关组合分析;未入选者视为 IS 过拟合。