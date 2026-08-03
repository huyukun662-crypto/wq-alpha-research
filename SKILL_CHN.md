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

### 2026-07-21 第二轮:5 分钟线日内因子(全市场,2024-07 至 2026-07,487 个交易日)

数据:5866 只 × 5 分钟线 → 日频化特征(尾盘成交占比、日内偏度、开盘 30 分钟收益、日内波动),约 269 万股票-日。

| 因子 | 毛 Sharpe | 净 Sharpe | IC | 结论 |
|------|-----------|-----------|-----|------|
| late_volume_share(尾盘占比高做多) | **-1.19** | -1.84 | 0.013 | 方向与经典文献相反 |
| open30_reversal(开盘30min反转) | **-1.06** | -2.96 | 0.012 | 同上:该窗口内开盘动量占优 |
| intraday_skew(负偏度做多) | 0.46 | -0.38 | 0.027 | 方向对但太弱,过不了成本 |

经验:

8. **日内结构因子在 2024-2026 A股窗口整体失效甚至反向**:尾盘占比与开盘动量的负 Sharpe 意味着「反着做」在样本内年化 ~10%,但仅 2 年样本、且是事后翻方向(数据窥探),不可直接采信;需要更长历史(60 分钟全历史)验证方向稳定性后再用。
9. **IC 与多空组合方向可以背离**(如 late_volume_share IC 为正、组合为负):IC 度量整体截面单调性,行业中性+decay 后的头尾组合暴露可以相反——选因子必须两个口径都看。
10. **高频日频化因子换手不低**(开盘动量 TO 13%/日),在 A股成本下净值全为负;若要实盘化必须与低频信号混合或降频使用。

### 2026-07-23 第三轮:5 分钟全历史回溯完成,日内因子方向稳定性验证(2015-01 至 2026-07,2682 个交易日)

数据:4 个回溯窗口全部落地,5 分钟线覆盖 2015-01 至 2026-07 全市场(5866 只、约 21GB、约 1140 万股票-日特征)。

| 因子 | 2024-26 毛 Sharpe | **2015-26 全样本毛** | 全样本净 | IC | 方向稳定性结论 |
|------|-------------------|---------------------|----------|-----|----------------|
| intraday_skew(负偏度做多) | +0.46 | **+2.25**(全检查通过) | **+1.09** | 0.030 | ✅ 方向稳定,近两年是局部衰减而非失效 |
| late_volume_share(尾盘占比高做多) | -1.19 | -1.44 | -2.18 | 0.001 | 两个子样本方向一致为负(与文献相反);反向使用有一致性支撑,但仍属事后定向 |
| open30_reversal(开盘30min反转) | -1.06 | -0.29 | -2.45 | 0.010 | ❌ 方向不稳定,放弃 |

经验(承接第 8-10 条):

11. **日内偏度是 A股全周期稳健的日内簇因子**:做多负偏度、做空正偏度(彩票偏好溢价),全样本毛 2.25/净 1.09、TO 仅 5.3%、回撤 13%,且与估值/质量/流动性簇相关低——已自动进入低相关组合,取代价量背离。**短样本会严重低估它**(2024-26 只有 0.46),长历史验证不可省略。
12. **「方向与文献相反」需要分辨两类**:尾盘占比在两个不重叠子样本方向一致(稳定反向,或 A股机构尾盘行为与美股不同),而开盘动量方向漂移(不稳定,纯噪音)。前者值得反向研究,后者直接弃用。
13. 全样本回撤口径提醒:负 Sharpe 因子的 DD 会大得离谱(late_volume_share 129%)——那是累计亏损曲线的必然,不要误读为数据错误。

### 2026-07-24 第四轮:遗传算法挖掘机(ga_miner.py,种群 60 × 20 代,848 条表达式)

方式:文法约束表达式树(借 Worldquant_Mining 的 wrapper(core) 结构与窗口集)+ 锦标赛/子树交叉/四类变异 + IS(2015-22) fitness、OOS(2023-26) 只在收尾裁决。完整名人堂见 `ga_mining_report.md`。

**OOS 幸存者 3/20**:

| 表达式 | IS净 | OOS净 | 主题 |
|--------|------|-------|------|
| `group_rank(ts_mean(ts_zscore(abs(grossprofit_margin),250),10), industry)` | 1.51 | **1.04** | 毛利率长期 z 分动量 |
| `group_rank(ts_decay_linear(div(ts_decay_linear(grossprofit_margin,20), …)), industry)` | 1.14 | 0.80 | 毛利率/规模复合 |
| `rank(ts_decay_linear(ts_decay_linear(bp,120),3))` | 0.82 | **0.86**(零衰减) | 重平滑 B/P 价值 |

经验:

14. **GA 的真实产出率约 15%**:IS fitness 排名前列的因子大多是过拟合(前 5 名里 4 个 OOS 死亡),**没有 OOS 闸门的因子挖掘机等于过拟合生成器**。
15. **幸存主题全是重平滑的基本面**(毛利率、B/P):与第一轮"基本面簇最稳"互相印证;GA 自己进化出了 `ts_decay_linear` 双层平滑结构,等价于超低换手(TO 1-5%)。
16. **警惕退化表达式的"数据可得性泄漏"**:`div(vol,vol)`、`sign(volume_ratio)` 这类常数表达式经过 decay(其 NaN 按 0 处理)后,实际变成了"过去 120 日停牌/缺数据越少分越高"的因子——IS(2015-18 停牌潮)Sharpe 高达 1.65,OOS 全灭。文法生成器应过滤常数表达式,decay 实现的 NaN 语义要在挖掘前想清楚。
17. **种子因子 OOS 普遍衰减 50-70% 但方向不翻**(ep 1.02→0.35、amihud 1.09→0.41):A股因子密度在下降,历史净 Sharpe 打对折是更现实的实盘预期。

### 2026-07-25 第五轮:BRAIN 九大类别数据落地 + 35 个新因子(2015-07 至 2026-07,2803 个交易日)

把 BRAIN 的九大数据类别映射到 tushare 接口全量落地(`tushare_extra.py`,按接口+日期断点续传),
装配成日频宽表面板(`tushare_brain_panel.py`,51 个字段),接入因子库新增 35 个因子:

| BRAIN 类别 | tushare 接口 | 落地量 | 新因子簇 |
|-----------|-------------|--------|---------|
| Fundamental | income/balancesheet/cashflow_vip | 50 期 / 242MB | (已有财务簇) |
| Model | cyq_perf 筹码分布 | 2803 天 / 269MB | chip_*(获利盘、集中度、成本偏离) |
| Model | stk_factor_pro 技术指标库 | 2803 天 / 4.6GB | rsi/macd/kdj/cci/bias/mfi/adx/boll/vr/atr |
| Sentiment | margin_detail 两融 | 2803 天 / 352MB | margin_*(余额占比、变化、买入强度) |
| Sentiment | hk_hold 北向 | 2803 天 / 98MB | northbound_*(持股比例、增持速度) |
| Sentiment | limit_list_d 涨跌停 | 2803 天 / 42MB | limit_*(涨停频率、净涨停、封单、炸板) |
| Sentiment | top_list/top_inst 龙虎榜 | 2803 天 / 135MB | lhb_*(净买入、机构席位) |
| Analyst | report_rc 卖方预测 | 2803 天 / 147MB | analyst_*(EPS 修正、覆盖度、目标价空间) |
| Social Media(代理) | stk_surv 机构调研 | 2803 天 / 20MB | institution_survey |
| Earnings | dividend 分红 | 2803 天 / 24MB | cash_dividend_yield |
| Option | opt_daily ETF/指数期权 | 2803 天 / 614MB | 仅市场级,未做横截面因子 |
| News | anns_d 个股公告 | **未下载(磁盘不足)** | announcement_intensity(代码已就位) |

**新因子里进入总榜前列的**:

| 因子 | Sharpe | Fitness | 年化 | 回撤 | 说明 |
|------|--------|---------|------|------|------|
| limit_up_frequency | 1.62 | 1.98 | 18.9% | 57.6% | 涨停频率反转 |
| analyst_eps_revision | 1.50 | 0.99 | 5.4% | **6.5%** | 一致预期 EPS 上修 |
| bias_reversal | 1.03 | 0.95 | 10.5% | 18.1% | BIAS 乖离率反转 |
| vr_volume_ratio | 1.07 | 0.82 | 7.4% | 21.2% | VR 成交量比率反转 |

经验:

18. **A股没有个股期权**:Option 类只有 ETF/指数期权(50ETF/300ETF/500ETF + 沪深300指数期权),
    只能做市场级波动率/PCR 情绪信号,**做不了横截面因子**——这是数据本身的限制,不是权限问题。
    Social Media 类同样无真正对应,用机构调研(stk_surv)做关注度代理。
19. **事件日口径的对齐是隐蔽的未来函数来源**:研报(report_date)、公告(ann_date)、调研(surv_date)
    都不是交易日口径。用 `get_indexer` 精确匹配会把周末/节假日发生的事件**整条丢掉**;
    正确做法是 `searchsorted(side="right")` 落到**严格晚于事件日的第一个交易日**。
    第一版就踩了这个坑,report_rc 覆盖率因此只有 2.7%。
20. **事件数据要区分「状态量」和「流量」**:一致预期 EPS 是状态(两份研报之间应保持上一份的值,
    需 ffill,但要设有效期 TTL=180 交易日让过期预期失效),调研次数/分红是流量(稀疏即真实,不能 ffill,
    因子侧用 ts_mean 做窗口聚合)。把状态量当流量处理会让 `ts_delta` 全是 NaN——修复后覆盖率 2.7%→43.5%。
21. **稀疏事件因子会「假性同质」,而且会伪造出漂亮的假因子**:
    第一版 limit_up_frequency / limit_up_net / limit_open_times 三个因子
    Sharpe 1.59/1.62/1.69、回撤全部 57.6%——因为 limit_* 字段的非零率只有 0.6%,
    `fillna(0)` 之后绝大多数股票在截面上并列,group_rank 让三者都退化成
    「做空近期上过涨停板的股票」这同一个信号。当时 limit_open_times 还以 Fitness 2.09
    排在总榜第三,**完全是排名退化的产物,不是真因子**。
    修复方式是**条件化**:炸板数除以同窗口涨停次数(无涨停则 NaN),
    得到「每次涨停的平均炸板数」,有效截面约 414 只/日。
    改造后 limit_seal_quality 与原版相关性仅 0.21,Sharpe 从 +1.69 变成 **−4.15** ——
    方向都反了,证明原来那个正 Sharpe 测的根本不是炸板承接力。
    **稀疏 0/1 字段做截面 rank 前必须先查非零覆盖率**;
    覆盖率极低时要么条件化到子样本,要么根本不要做截面 rank。
21b. **条件化后的涨停子样本效应:方向 IS/OOS 稳定,但仍不入组合**。三项验证:

    - **方向稳定**:limit_seal_quality IS(2015–22)−5.11 / OOS(2023–26)−3.59 / 全样本 −4.15,
      符号一致、量级同阶。这不是样本内噪声,原始假设(炸板多=承接弱=后续差)是**真的反了**。
    - **不是分母伪影**:曾怀疑 `炸板/涨停次数` 是 `1/涨停次数` 的马甲,
      实测两者截面相关仅 **0.005**,排除。但**除法本身近乎多余**——
      与纯分子(条件样本内的原始炸板次数)相关高达 **0.933**。
    - **真正的问题是广度**:与 universe 取交后条件样本只有 **165 只/日**,
      且该子样本里三个变体(炸板/涨停比、纯炸板数、−1/涨停次数)Sharpe 全是 −3.3 ~ −4.2。
      整个子样本被一个共同效应主导,单个因子的"独立性"是假象。

    结论:**这是一个真实但极窄的子样本效应,不是一个可用的横截面 alpha**。
    165 只/日的广度撑不起组合权重,且事后翻方向仍属数据窥探。保留记录,不进低相关组合。
22. **趋势类技术指标在 A股全样本为负**(macd_momentum -0.87、ma_trend_alignment -0.98、adx -0.47),
    反转类全为正(bias 1.03、vr 1.07、rsi 0.83、cci 0.75):与前四轮"A股反转主导"完全一致。
    不要事后翻方向当新因子——那是同一个 beta 的镜像。
23. **筹码/两融/北向三个"聪明钱"簇集体失效**(Sharpe 均在 ±0.4 以内):
    winner_rate、融资余额占比、北向持股比例的截面信息大多已被价格与市值吸收;
    北向持股比例覆盖率仅 26%(只有标的股),截面可比性差。
