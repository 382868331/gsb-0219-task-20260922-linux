# delayed_ope —— 含延迟结算反馈的离线策略评估（可复用库）

纯 Python 标准库实现的离线策略评估（OPE）模块。处理推荐/决策日志中**回报延迟
结算**的情形：在固定评估截止时刻，只用已经观察到回报的成熟样本，计算目标策略
的 IPS、SNIPS、Doubly Robust（DR）估计；未成熟样本**绝不按 reward=0 计入**。
本轮在保留旧的单值倾向入口之外，新增行为概率为**正闭区间**时的稳健 IPS
最小/最大值入口。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库；无任何第三方依赖）
- 运行环境：Linux 原生；装好 Python 后无需安装其他依赖
- 离线：不访问网络、账号、数据库、中间件、云 API 或 Docker
- 核心算法未使用任何第三方 OPE 实现

## 运行准备与命令

Linux 原生 Python 3.14.7，仅标准库。可以使用 venv，但不需要安装第三方包：

```bash
python -m venv .venv
source .venv/bin/activate
python --version    # 期望 3.14.7；代码本身使用 3.10+ 通用语法
```

不启用 venv 时，只要系统 `python` 是兼容版本也可直接运行命令。

```bash
python -m unittest discover -s tests -v   # 必要测试（33 个测试方法，其中枚举含 369 个恒等用例）
python demo.py                            # 旧延迟 OPE 三估计演示
python demo_iteration.py                 # 本轮稳健 IPS 新功能与拒绝边界（90 秒内）
```

在仓库根目录执行。`tests/independent_reference.py` 是**独立参考**：它不导入
被测核心，期望值按有限分布定义直接代数求和；测试中的标量期望亦按公式手算。
被测核心只产出待核对的值，不参与期望值的生成。

## 目录

```
delayed_ope/__init__.py   公开接口
delayed_ope/core.py       核心算法与全部校验
tests/independent_reference.py  独立分布参考（不依赖被测核心）
tests/test_ope.py         必要测试（旧回归 + 二维独立参照 + 三维稳健 IPS）
demo.py                   固定种子数据生成 + 旧入口正常结果 + 两个拒绝边界
demo_iteration.py         本轮稳健 IPS 固定输入演示
```

## 数据结构与接口

```python
from delayed_ope import LogEvent, evaluate

events = [LogEvent(context, action, reward, logged_propensity,
                   decision_time, observed_time), ...]

result = evaluate(
    events,                 # Sequence[LogEvent]
    target_policy,          # {context: {action: probability}}，每行归一
    cutoff: float,          # 固定评估截止时刻
    q_hat=None,             # 可选：{(context, action): q} 或 {context: {action: q}}
    clip_value=None,        # 可选：正实数 M，额外报告 min(w, M) 截断组
)
```

`LogEvent` 字段（均为必填）：

| 字段 | 含义 | 约束 |
|---|---|---|
| `context` | 上下文标识 | 可哈希（int/str/tuple…） |
| `action` | 被记录的动作 | 可哈希 |
| `reward` | 已结算回报 | 有限实数（拒绝 NaN/±Infinity） |
| `logged_propensity` | 记录策略概率 μ(a\|x) | 有限实数，[0,1] |
| `decision_time` | 决策时刻 t_d | 有限实数 |
| `observed_time` | 回报可观察时刻 t_o | 有限实数，t_o ≥ t_d |

`evaluate` 返回 `EvaluationResult`（不可变 dataclass）：

- `cutoff, n_total, n_mature, n_immature, mature_indices`（显式成熟口径）
- `unclipped: Estimate` —— 始终报告
- `clipped: Estimate | None`, `clip_value` —— 仅在给 `clip_value` 时存在
- `q_hat_used: bool`

`Estimate` 字段：`ips, snips, dr, ess, n`。`snips` 在权重和为 0 时为 `None`
（0/0 无定义）；`dr` 在未给 q_hat 时为 `None`。用 `result.summary()` 可打印。

### 成熟样本口径（明确定义）

事件为成熟样本当且仅当 **`observed_time <= cutoff`**（边界包含等于）。
`observed_time > cutoff` 的事件是延迟未观测回报，只从全部求和中排除；其
`reward` 字段不参与任何计算，也不会被替换为 0。若没有任何成熟样本，抛
`NoMatureSamplesError`。

### 估计量定义

n_m 个成熟样本，权重 `w_i = π(a_i|x_i) / μ(a_i|x_i)`：

```
IPS   = (1/n_m) Σ_i w_i r_i
SNIPS = Σ_i w_i r_i / Σ_i w_i
DR    = (1/n_m) Σ_i [ m_π(x_i) + w_i (r_i − q̂(x_i,a_i)) ]
        m_π(x) = Σ_a π(a|x) q̂(x,a)          # q̂ 在目标策略下的边际价值
ESS   = (Σ_i w_i)² / Σ_i w_i²               # 截断/未截断分别计算，∈[1,n_m]
```

DR 采用标准边际形式：即使日志只记录了一个动作，`m_π(x)` 也要求 q_hat 覆盖
该 context 概率行内的全部动作。q_hat 完美（q̂=条件期望回报）时校正项恒为 0，
DR 退化为已实现 context 序列上 m_π 的平均值，不再有权重噪声。

## 稳健 IPS：行为概率正闭区间

旧入口 `evaluate(...)` 的原点估计保持不变。新入口使用独立事件类型，因为每条
记录不再携带自己的单值行为概率；同一 context 的所有动作共享一组变量：

```python
from delayed_ope import RobustLogEvent, robust_ips

events = [
    RobustLogEvent(context, action, reward, decision_time, observed_time),
    ...
]

result = robust_ips(
    events,
    target_policy,       # {context: {action: probability}}，每行归一
    behavior_intervals,  # {context: {action: (lower, upper)}}，正闭区间
    cutoff,
)
result.minimum.ips
result.maximum.ips
result.minimum.behavior_policy       # 达到最小值的完整上下文概率表
result.maximum.behavior_policy       # 达到最大值的完整上下文概率表
result.minimum.constraint_checks     # 每个 context 的区间/归一化复核
```

对每个 context，行为概率向量 `p_a` 必须同时满足：

```text
0 < lower_a <= p_a <= upper_a <= 1
Σ_a p_a = 1
Σ_a lower_a <= 1 <= Σ_a upper_a       # 可行性必要且充分条件
```

最后一行可行性不满足时立即抛 `InfeasibleProbabilityIntervalsError`，不会自动把
区间投影或裁剪成“最接近”的概率表。目标策略动作集合必须与该 context 的区间
动作集合完全一致；奖励可以为正、负或零。成熟样本上的目标为：

```text
IPS(p) = (1/n_m) Σ_x Σ_a C(x,a) / p_a
C(x,a) = Σ_{i: mature, x_i=x, a_i=a} π(a|x) r_i
```

`C(x,a)` 是同一动作多条记录的加权奖励和，因此同一动作上的正负反馈会先抵消。
一个边界中的同一 context 只选择一张概率表；不会让不同记录分别选择互相矛盾的
概率。不同 context 的概率变量相互独立，所以总最小值/最大值是各上下文贡献的
最小值/最大值之和。

### 二维参照与三维算法

对两个动作，令 `q=p_a`、`p_b=1-q`：`f(q) = C_a/q + C_b/(1-q)`。测试中的
二维独立参照不导入核心优化器，而是直接枚举可行区间端点，并在 `C_a`、`C_b`
同号时加入驻点
`q*=sqrt(|C_a|)/(sqrt(|C_a|)+sqrt(|C_b|))`。异号或单系数时函数单调，极值
在端点；二者同为 0 时整条可行段目标值均为 0。该参照包含同一动作多条正负
反馈抵消的例子。

核心入口支持每 context 至多 3 个动作。三维情形在“箱约束 ∩ 单纯形”这个紧
多面体上检查顶点，以及一条或两条变量固定在端点时剩余变量位于区间内部的驻点；
内点满足一阶条件 `p_a` 与 `sqrt(|C_a|)` 成比例。所有候选点都重新检查区间与
总和，输出边界再次提供 `constraint_checks`。检查字段包括：
`feasible_intervals`、`lower_sum`、`upper_sum`、`probability_sum`、
`max_interval_violation`。

### 范围含义与不做的事

该范围只反映调用方给定的行为概率闭区间。它**不**宣称覆盖奖励噪声、奖励模型
误差、目标策略错误、上下文分布漂移、未成熟样本选择偏差或其他模型误差。
稳健入口不做 SNIPS/DR、不接受 clip 参数、不自动截断权重；若最大可能权重超过
`MAX_WEIGHT`，抛 `WeightOverflowError`。行为概率下界严格为正，因此没有旧入口
中的 0/0 支持特判。

## 零支持拒绝（严格）

对**全部事件**（与 cutoff 无关，这是策略与日志的结构性不相容）：若
`π(a|x) > 0` 而 `logged_propensity == 0.0`，权重无定义，抛
`ZeroSupportError`，不返回任何估计值。

- `π(a|x) == 0` 且 `μ(a|x) == 0`：权重取 0，合法；若全部成熟样本权重都为 0，
  IPS=0、SNIPS=`None`、ESS=0，退化口径显式暴露而非静默给出“有效”数字。
- 极小但**非零**的 μ 产生极大但有限的权重（如 1/1e-20 = 1e20，正常计算）；
  仅当权重超过接口上限时才以 `WeightOverflowError` 拒绝，零支持与极端权重
  两类边界互不混淆。

## 截断（可选、显式、双报告）

传 `clip_value=M > 0` 时，除始终报告的未截断组外，额外给出 `w←min(w,M)`
后的 IPS/SNIPS/DR 与 ESS。截断统一作用于三个估计量中的权重。输出中两组并列：
截断提高 ESS（降低方差）但引入偏差，取舍不被隐藏。

## 错误语义（非法输入不静默纠正，错误可定位）

所有错误继承 `OPEError`；消息带字段名/`events[i]` 位置：

| 异常 | 触发 |
|---|---|
| `InvalidLogError` | 数值 NaN/Infinity；`cutoff`/reward/时间非有限实数；propensity ∉ [0,1]；t_o<t_d；context/action 不可哈希；事件数超上限；容器或元素类型错 |
| `InvalidPolicyError` | 概率表缺 context 行/缺日志动作（漏写不隐式当 0）、概率越界、行和与 1 偏差超容差、动作数超上限、`clip_value` 非法 |
| `InvalidQHatError` | q_hat 非映射、缺成熟行所需动作取值、值非有限实数 |
| `ZeroSupportError` | π>0 而 logged_propensity==0 |
| `WeightOverflowError` | 成熟样本权重 > 1e300 |
| `NoMatureSamplesError` | cutoff 前无成熟样本 |
| `InfeasibleProbabilityIntervalsError` | 稳健入口的正闭区间不满足 Σlower≤1≤Σupper |

- 所有**数值参数**（含 `cutoff`、`clip_value`、概率、reward、时间、q 值）
  拒绝 `NaN` 与正负 `Infinity`；**整数/数值位置拒绝 `bool`**（`True/False`
  不是合法数字参数）。
- `evaluate` 为纯函数、无副作用；校验在构造任何结果前完成，不存在部分变更。
  不会修改传入的策略表/事件（有测试覆盖）。

## 输入上限（接口边界，非性能承诺）

| 项 | 上限 |
|---|---|
| 事件数 | 1,000,000（`MAX_EVENTS`） |
| 每 context 动作数 | 100,000（`MAX_ACTIONS_PER_CONTEXT`） |
| 稳健 IPS 每 context 动作数 | 3（`MAX_ROBUST_ACTIONS_PER_CONTEXT`） |
| 单个重要性权重 | 1e300（`MAX_WEIGHT`，超出抛 `WeightOverflowError`） |
| 概率行归一容差 | `|Σp − 1| ≤ 1e-9`（`PROB_SUM_TOL`） |

短演示不穷举上限规模。时间值为可比较的有限实数，本库不实现时区/时钟语义。

## 浮点结果误差口径

- 求和统一使用 `math.fsum`（误差有界的高精度求和）。
- **枚举恒等测试的误差口径**：在“普查样本”（每个 (x,a) 出现次数严格正比于
  p(x)μ(a|x)，权重与计数为精确比值）上，IPS/SNIPS/DR 与按分布定义独立算出的
  V(π) 之差必须满足
  `|估计 − 期望| ≤ 1e-11 · max(1, |期望|)`。
  这是**收紧的代数口径**（369 个枚举用例），不用宽容差掩盖边界错误。
- 随机（非普查）日志上只存在统计一致性，不存在逐样本恒等：测试中对 IPS/SNIPS
  仅用宽松统计带（4000 样本、固定种子，|误差|≤0.2）核对；完美 q_hat 的 DR 仍
  按精确口径 `≤1e-12` 核对（校正项必须精确为 0）。
- 权重与概率为 IEEE-754 binary64；`1/μ` 的相对误差为常规舍入量级（~1e-16）。
- 稳健 IPS 的候选概率点以 1e-9 级区间/归一容差验收；输出的
  `max_interval_violation` 通常为 0 或 1e-16 级浮点舍入。二维测试与独立
  公式比较使用 1e-11 绝对误差口径；不使用随机优化或近似搜索。
- 本库只实现本题契约的有限数学子集（离散 context/action、实数标量、有限时间
  实数），不宣称任何行业标准协议兼容性。

## 设计取舍

1. **成熟性只看 `observed_time <= cutoff`**，不依赖 reward 是否“看起来缺失”；
   未成熟的 reward 字段被完全忽略，从机制上杜绝“当 0”。
2. **零支持全事件校验**：即使违规事件本身未成熟也拒绝——π≪μ 是评估合法性的
   前提，与数据是否到期无关；而权重溢出只在成熟样本上检查，未成熟权重不产生
   数值依赖。
3. **SNIPS 全零权重返回 `None` 而非抛异常或返回 0**：0/0 数学上无定义，调用
   方应显式处理；IPS 此时恒 0 亦无信息量，ESS=0 同时标出退化。
4. **DR 用目标策略边际 m_π(x)** 而非 q̂(x,a_logged)：前者满足对任意 q̂ 的
   期望恒等且 q̂ 完美时精确无偏；代价是 q_hat 必须覆盖成熟行的全部动作。
5. **截断是后处理而非唯一输出**：偏差/方差两组数字并列，调用方自行取舍。
6. 概率表要求对日志中出现的动作**显式列值**，避免“字典漏键被静默当 0”。
7. **稳健入口按上下文共享变量**：先汇总同动作奖励系数，再优化一张归一化
   概率表，从模型层面排除逐记录挑选概率的矛盾。
8. **区间不可满足先拒绝**：不做自动投影；稳健 IPS 也不自动截断权重，避免把
   区间误差和人为截断偏差混在一起。

## 已知限制

- 不提供置信区间/SE（可由 `Estimate` 字段在调用方自行 bootstrap）。
- 稳健 IPS 只覆盖有限 context、每 context 至多 3 个动作的未截断 IPS；不提供
  通用高维线性分式规划、SNIPS、DR、权重截断或采样近似。
- 不做随机/上下文 embedding 策略，仅离散概率表；目标策略须为调用方持有的
  完整映射。
- 不处理同一决策的多值/序列回报、回报删失（censoring）与观察过程的选择性
  偏差；前提是“到期即可观察”的无信息删失。
- 无流式/增量接口；一次调用处理一个批次（上限 1e6 事件，内存 O(事件数)）。
- q_hat 两种键形式（扁平 tuple 键 / 嵌套映射）同时给出时以扁平键为准。
