# delayed_ope —— 含延迟结算反馈的离线策略评估（可复用库）

纯 Python 标准库实现的离线策略评估（OPE）模块。旧入口处理推荐/决策日志中
**回报延迟结算**的情形：在固定评估截止时刻，只用已经观察到回报的成熟样本，
计算目标策略的 IPS、SNIPS、Doubly Robust（DR）估计；未成熟样本**绝不按
reward=0 计入**。本次迭代新增稳健 IPS 范围入口：每条已观察动作的行为概率以
正闭区间给出，在同一 context 的共享概率向量约束下求最小/最大值。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库；无任何第三方依赖）
- 运行环境：Linux 原生；装好 Python 后无需安装其他依赖
- 离线：不访问网络、账号、数据库、中间件、云 API 或 Docker
- 核心算法未使用任何第三方 OPE 实现

## 运行命令

```bash
python -m venv .venv
source .venv/bin/activate
python --version    # 目标环境：Python 3.14.7；代码仅需 Python 3.10+
```

本项目只用标准库，激活 venv 后无需 `pip install`。然后：

```bash
python -m unittest discover -s tests -v   # 必要测试（32 个测试方法）
python demo.py                            # 旧延迟结算入口演示
python demo_iteration.py                  # 本次新增：稳健 IPS 范围与不可满足边界
```

在仓库根目录执行。`tests/independent_reference.py` 是**独立参考**：它不导入
被测核心，期望值按有限分布定义直接代数求和；测试中的标量期望亦按公式手算。
被测核心只产出待核对的值，不参与期望值的生成。

## 目录

```
delayed_ope/__init__.py   公开接口
delayed_ope/core.py       核心算法与全部校验
tests/independent_reference.py  独立分布参考（不依赖被测核心）
tests/test_ope.py         必要测试
demo.py                   固定种子数据生成 + 正常结果 + 两个真实触发的拒绝边界
demo_iteration.py         本次迭代：稳健 IPS 最小/最大值 + 区间不可满足拒绝
tests/test_robust_ips.py  二维独立枚举/驻点参照、正负抵消、共享变量与拒绝测试
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

## 稳健 IPS 范围（本次新增入口）

```python
from delayed_ope import RobustObservation, robust_ips_range

observations = [
    RobustObservation(context="x", action=0, reward=2.0),
    RobustObservation(context="x", action=1, reward=-1.0),
]

target_policy = {"x": {0: 0.5, 1: 0.5}}
propensity_intervals = {
    "x": {
        0: (0.20, 0.70),   # μ(0|x) 的正闭区间
        1: (0.30, 0.80),   # μ(1|x) 的正闭区间
    },
}

range_result = robust_ips_range(
    observations,
    target_policy,
    propensity_intervals,
)
```

返回不可变 `RobustIPSResult`：

- `n`：已观察记录数。
- `minimum` / `maximum`：未截断目标策略加权奖励 IPS 估计的最小/最大值。
- `min_probabilities` / `max_probabilities`：分别达到两个边界的完整行为概率表。
- `constraint_checks`：逐 context 返回动作顺序、下界和、上界和及 `feasible`。

### 联合概率变量语义

对每个 context，所有动作共享一个概率向量：

```
μ(a|x) ∈ [L(x,a), U(x,a)],  L(x,a) > 0
Σ_a μ(a|x) = 1
```

同一 context 内所有日志记录使用同一个 `μ(·|x)`。算法不会为每条记录分别挑选
一个互相矛盾的概率。不同 context 之间相互独立；IPS 对 context 可加，因此逐
context 求最小/最大后合并。

每个已观察 context 限制为 **1–3 个动作**；区间动作集合必须与目标策略动作集合
完全一致。目标策略仍必须逐行动作概率非负且和为 1。奖励允许正、负或零。

### 优化与二维参照

先把同一 `(context, action)` 的记录聚合成系数：

```
c(x,a) = Σ_{i: x_i=x,a_i=a} π(a|x) r_i
IPS(μ) = (1/n) Σ_x Σ_a c(x,a) / μ(a|x)
```

在 1–3 维的闭区间盒与概率单纯形交集中，函数连续。实现枚举所有 KKT active
set：动作取下界/上界，或为自由变量；自由变量上的驻点满足
`p_j ∝ sqrt(c_j)`（全正系数，最小值）或 `p_j ∝ sqrt(-c_j)`（全负系数，
最大值）。若自由系数正负混杂，该面上没有内点驻点；边界仍由 active-set 枚举
覆盖。

二维测试中的 `independent_two_action_range` 是独立参照：显式列出两个区间边界、
两个和约束交点及内点驻点，不调用核心优化器。测试还包含同一动作上正负反馈
抵消为零系数的例子。

### 范围边界与错误

可满足的充要条件为：

```
Σ_a L(x,a) ≤ 1 ≤ Σ_a U(x,a)
```

不可满足时立即抛 `InfeasibleIntervalError`，不自动平移或放宽区间。新异常：

| 异常 | 触发 |
|---|---|
| `InvalidIntervalError` | 区间不是二元序列、缺动作、下界非严格正、概率越界、NaN/Infinity、动作集合不匹配 |
| `InfeasibleIntervalError` | 下界大于上界，或 `Σlower>1`，或 `Σupper<1` |
| `WeightOverflowError` | 边界概率对应的重要性权重非有限或超过 `MAX_WEIGHT` |

新入口**不自动裁剪权重**；拒绝权重溢出。若目标动作概率为 0，该项目标加权
奖励为 0；但该动作的行为概率仍必须作为同一概率向量的一部分给出。

该范围只反映给定行为概率倾向区间，不包含目标策略错误、奖励测量误差、
未观测样本选择偏差、模型估计误差或采样噪声的统计置信区间。

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
| `InvalidIntervalError` | 稳健 IPS 的行为概率区间结构或数值非法 |
| `InfeasibleIntervalError` | 稳健 IPS 的行为概率区间与和为 1 不相交 |

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
| 稳健 IPS 每 context 动作数 | 3（新增 `robust_ips_range` 的显式范围） |
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

## 已知限制

- 不提供置信区间/SE（可由 `Estimate` 字段在调用方自行 bootstrap）。
- `robust_ips_range` 提供的是给定闭区间下的精确 min/max（受 IEEE-754 舍入），
  不是统计置信区间；每 context 至多 3 动作，不使用通用非线性规划器。
- 不做随机/上下文 embedding 策略，仅离散概率表；目标策略须为调用方持有的
  完整映射。
- 不处理同一决策的多值/序列回报、回报删失（censoring）与观察过程的选择性
  偏差；前提是“到期即可观察”的无信息删失。
- 无流式/增量接口；一次调用处理一个批次（上限 1e6 事件，内存 O(事件数)）。
- q_hat 两种键形式（扁平 tuple 键 / 嵌套映射）同时给出时以扁平键为准。
