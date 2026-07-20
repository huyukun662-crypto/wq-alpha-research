# WQ Alpha 研究 Skill — A股 (CHN) 扩展篇

> 本文是 `SKILL.md` 的 CHN region 扩展：把主 playbook 的「字段 → 表达式 → 回测 → 检查 → 提交」流程落到 A股。
> 配套工具：`scripts/mine_chn_alphas.py`；候选因子库：`references/chn_candidate_alphas.json`。

---

## 1. CHN 环境与默认设置

| 参数 | 值 | 说明 |
|------|-----|------|
| Region | `CHN` | A股 |
| Universe | `TOP2000U` | CHN region 的主力 universe |
| Delay | `1` | 与主 SKILL 一致；CHN 也有 delay=0 但门槛不同 |
| Neutralization | `INDUSTRY`（默认）/ `SUBINDUSTRY` | 与 USA 相同的中性化选项 |
| Decay / Truncation / nanHandling | 沿用 SKILL.md 4.2 | 按因子类型选择 |

⚠️ **不要把 USA 的指标阈值硬搬到 CHN**。不同 region 的 IS 检查门槛（如 Sharpe 下限）不同,通常 CHN 要求比 USA 更高。**唯一权威来源是模拟返回的 `is.checks`**——`mine_chn_alphas.py` 会逐条读取并报告每个 check 的 `value / limit / result`,不做硬编码判断。

## 2. 字段:必须重新拉取

主 SKILL 内置的 4367 个字段快照**只覆盖 USA TOP3000 delay=1**(见 SKILL.md 2.4)。CHN 的数据集/字段体系不同(基本面、分析师字段 id 都不一样),第一步必须拉取 CHN 快照:

```bash
python scripts/mine_chn_alphas.py --fetch-fields   # 生成 references/wq_chn_top2000u_delay1_data_fields.{json,csv}
python scripts/mine_chn_alphas.py --search roe     # 本地关键词搜索
```

价量类字段(`close, open, high, low, volume, vwap, returns, cap, adv20`)跨 region 通用,可直接使用;基本面/分析师字段一律先搜索后验证(`rank(field)` 单字段模拟,201 即可用)。

候选因子库用 `{field:关键词1|关键词2}` 占位符表达基本面字段,`--mine` 时自动解析为 alphaCount 最高的 CHN 真实字段,解析失败的候选会跳过并标记,不会盲目模拟。

## 3. A股市场特性 → 因子设计要点

| A股特性 | 对因子设计的影响 |
|----------|------------------|
| 散户交易占比高、T+1 | **短期反转**、**换手率/异常成交量**异象显著,是与基本面簇低相关的主力技术簇 |
| 涨跌停制度(10%/20%) | 极端收益被截断,`truncation 0.08` 照常;避免依赖单日极值的信号 |
| 壳价值/小市值历史异象 | **不要**做 `rank(-cap)` 类信号:注册制后衰减明显,且几乎必挂 LOW_SUB_UNIVERSE_SHARPE |
| 财报披露频率低、应计操纵多 | 基本面窗口用 126–252;现金流字段比净利润更稳健 |
| 分析师覆盖集中于大市值 | analyst 簇注意覆盖率与 sub-universe 检查,优先 `group_backfill`/`nanHandling=ON` |
| 波动率高 | 低波动异象可做;反转信号建议用波动率缩放,缓解 CONCENTRATED_WEIGHT |

## 4. CHN 候选因子簇(15 个,见 chn_candidate_alphas.json)

| 簇 | 候选 | 核心表达式骨架 |
|----|------|----------------|
| 反转 | reversal_5d, reversal_volatility_scaled, range_position | `group_rank(-ts_delta(close,5), industry)` |
| 流动性/换手 | abnormal_turnover, amihud_illiquidity | `group_rank(-ts_mean(volume,20)/ts_mean(volume,120), industry)` |
| 低波动 | low_volatility | `group_rank(-ts_std_dev(returns,60), subindustry)` |
| 价量 | price_volume_divergence, vwap_deviation | `rank(-ts_corr(rank(close),rank(volume),10))` |
| 盈利质量 | roe_trend, asset_turnover_margin | SKILL 模板 A/F 的 CHN 字段版 |
| 价值 | earnings_yield | `group_rank(ts_rank(net_income/cap,126), industry)` |
| 现金流 | cash_flow_yield | SKILL 模板 C 的 CHN 字段版 |
| 分析师 | analyst_eps_yield | SKILL 模板 B 的 CHN 字段版 |
| 混合 | reversal_open_close_mix, quality_value_mix | 50/50 跨簇混合(SKILL 模板 D/E) |

簇的划分刻意覆盖**不同数据来源和经济逻辑**——按 SKILL.md 8.3 的结论,这是唯一能产生真正低相关的方式;同簇内换窗口/换权重不算分散。

## 5. 挖掘流程(与主 SKILL 决策树对应)

```bash
# 凭据(二选一,勿提交仓库)
export WQ_BRAIN_USERNAME="..."; export WQ_BRAIN_PASSWORD="..."

python scripts/mine_chn_alphas.py --plan            # 0. 离线预览候选与设置
python scripts/mine_chn_alphas.py --fetch-fields    # 1. 拉取 CHN 字段快照(一次)
python scripts/mine_chn_alphas.py --mine            # 2. 全量模拟 → chn_mining_report.md
python scripts/mine_chn_alphas.py --mine --only reversal   # 只跑某一簇
python scripts/mine_chn_alphas.py --check-corr <alpha_id>  # 3. 提交前日收益相关性检查
python scripts/evolve_skill.py                      # 4. 把实证结果回写 SKILL(自进化)
```

提交仍走主 SKILL 第 7.4–7.7 节流程:先过全部 IS 检查 → 与 ACTIVE alpha **日收益**相关 < 0.7 → 提交 → **二次确认 `status == ACTIVE`**。

## 6. 迭代指南(挖掘结果出来之后)

- **LOW_SHARPE**:CHN 门槛高于 USA,优先在反转/换手簇内调窗口(3/5/10/20 日)而不是换参数微调基本面簇。
- **HIGH_TURNOVER**:反转簇 decay 提到 20–30,或与 roe_trend 50/50 混合。
- **CONCENTRATED_WEIGHT**:改用波动率缩放版本(reversal_volatility_scaled),或 truncation 降到 0.05。
- **LOW_SUB_UNIVERSE_SHARPE**:大概率信号隐含市值倾斜——检查是否变相做多了小票(A股尤其常见)。
- 每轮结束运行 `evolve_skill.py --apply`,把 CHN 实证经验沉淀回本文件第 7 节。

## 7. 无 BRAIN 账号时:tushare 本地数据挖掘

没有 BRAIN 凭据也可以先在本地把同一套流程跑起来——数据源换成 tushare,算子与回测口径本地复刻:

```bash
export TUSHARE_TOKEN=...                               # 或放 tushare_token.txt(已 gitignore)
python scripts/tushare_data.py --download --start 20230101 --end 20260717
python scripts/mine_tushare_alphas.py --mine           # 全因子回测 → tushare_mining_report.md
```

- **算子**:`rank/group_rank/ts_rank/ts_delta/ts_mean/ts_std_dev/ts_corr/ts_min/ts_max/ts_decay_linear` 按 FASTEXPR 语义在宽表上实现。
- **回测口径**:delay-1、行业中性(stock_basic.industry)、多空各 1 元、TO=sum|Δw|/4、Fitness 公式同 SKILL §5.1;另报 IC/ICIR 和含 13bp 单边成本的净 Sharpe。
- **Universe**:每日流通市值前 2000(对齐 TOP2000U)、上市>120 交易日、剔 ST。
- **字段代理**:daily_basic 无报表科目,基本面簇用 EP(1/pe_ttm)、SP(1/ps_ttm)、股息率代理 ROE/现金流;分析师簇本地无数据,跳过。
- **局限**:未建模涨跌停/停牌不可成交(反转、流动性簇会高估),无 BRAIN 官方 sub-universe / self-correlation 检查。本地结果只做**筛选与排序**,提交 BRAIN 仍走 `mine_chn_alphas.py`。

## 8. CHN 实证记录(自动/人工更新)

### 2026-07-20 首轮:tushare 本地回测,2015-01 至 2026-07,22 个日频因子

口径:TOP2000 流通市值 universe、行业中性、delay-1、多空各 1 元;成本单边 13bp。完整数字见 `tushare_mining_report.md`。

**通过 SKILL §5 全部检查(毛口径)的因子簇排序**:价量背离(Sharpe 2.0/Fitness 2.0)> 技术+估值混合(1.8/2.1)> VWAP 偏离(3.5/4.2,但换手 24%/日)> 量比反转(1.7/1.1)> EP 趋势(1.7/1.7)。

沉淀出的 A股特有规律:

1. **成本是 A股反转簇的生死线**:毛 Sharpe 最高的三个因子全是高换手技术簇,计 13bp 单边成本后 vwap_deviation 从 3.52 跌到 0.22、volume_ratio_reversal 从 1.65 跌到 -2.82。**日换手 >10% 的 A股因子,毛指标没有意义**——这印证并强化了 SKILL §6 的 decay 杠杆结论,CHN 场景必须直接看净值。
2. **净口径下的真赢家是低换手簇**:amihud_illiquidity(净 1.85→0.85,TO 2%)、earnings_yield(净 0.77)、quality_value_roe_mix(净 0.66)、price_volume_divergence(净 0.69,兼顾容量)。
3. **低相关组合验证了 SKILL §8.3**:6 因子等权(价量/估值/质量/流动性四个数据来源)毛 Sharpe 3.79、回撤 5.8%,显著优于任何单因子——跨数据源分散在 A股同样成立。
4. **A股低波动异象长周期表现差**(Sharpe 0.67、回撤 33%):2015 与 2020-2021 的高贝塔行情反噬严重,不建议单独使用。
5. **大单资金流(smart_money_flow)IC≈0**:tushare 的大单/特大单口径在日频上无预测力,方向也不稳,弃用或改事件驱动用法。
6. **财务趋势类(roe_trend/gross_margin_trend)回撤极小但收益薄**:适合做组合稳定器而非主信号——与 USA 经验(基本面通过率最高)方向一致但幅度弱,可能因 A股财报披露滞后更久。
7. **点位对齐提醒**:财务因子必须用公告日(ann_date)生效而非报告期,否则 ROE 类因子会虚高;本轮全部按公告日次一交易日生效。
