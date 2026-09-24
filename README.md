# delayed_ope —— 含延迟结算反馈的离线策略评估（可复用库）

纯 Python 标准库实现的离线策略评估（OPE）模块。两个入口：

1. **单决策延迟结算口径**（`evaluate`）：处理推荐/决策日志中**回报延迟
   结算**的情形：在固定评估截止时刻，只用已经观察到回报的成熟样本，计算目标策略
   的 IPS、SNIPS、Doubly Robust（DR）估计；未成熟样本**绝不按 reward=0 计入**。
2. **多步完整轨迹口径**（`evaluate_trajectories`）：最多 5 步的完整轨迹，
   逐步累乘目标与行为概率得到轨迹重要性权重，报告**轨迹重要性采样（TIS）**
   与**逐决策估计（PDIS）**，并给出有效权重诊断；行为概率为零的输入严格拒绝。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库；无任何第三方依赖）
- 运行环境：Linux 原生；装好 Python 后无需安装其他依赖
- 离线：不访问网络、账号、数据库、中间件、云 API 或 Docker
- 核心算法未使用任何第三方 OPE 实现

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（41 个测试方法，含枚举恒等用例）
python demo.py                            # 约 0.5 秒完成的固定输入演示（上限约 90 秒）
```

在仓库根目录执行。`tests/independent_reference.py` 与 `tests/mdp_reference.py`
是**独立参考**：它们不导入被测核心，期望值按有限分布定义直接代数求和/递归枚举；
测试中的标量期望亦按公式手算。被测核心只产出待核对的值，不参与期望值的生成。

## 目录

```
delayed_ope/__init__.py   公开接口
delayed_ope/core.py       单决策核心算法与全部校验
delayed_ope/trajectory.py 多步轨迹 TIS/PDIS 与权重诊断
tests/independent_reference.py  独立分布参考（不依赖被测核心）
tests/mdp_reference.py    独立小 MDP 枚举参考（不依赖被测核心）
tests/test_ope.py         单决策必要测试
tests/test_trajectory.py  轨迹必要测试
demo.py                   固定种子数据生成 + 正常结果 + 实际触发的拒绝边界
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

## 多步完整轨迹接口（TIS / PDIS）

```python
from delayed_ope import Trajectory, TrajectoryStep, evaluate_trajectories

trajs = [Trajectory((TrajectoryStep(context, action, reward, behavior_prob),
                     ...)), ...]          # 每条 1..5 步，各条长度可不同

result = evaluate_trajectories(
    trajs,                # Sequence[Trajectory]，非空
    target_policy,        # {context: {action: probability}}，口径同 evaluate
)
```

`TrajectoryStep` 字段（均为必填）：`context`/`action`（可哈希）、`reward`
（有限实数）、`behavior_prob`（行为策略概率 μ(a_t|x_t)，[0,1]）。轨迹模块
**不涉及时刻字段**：本轮不实现延迟奖励、删失修正与自适应截断。

逐步累乘权重与估计量（n 条轨迹，T_i 为第 i 条长度，回报为无折扣逐步和
`G_i = Σ_t r_{i,t}`）：

```
w_t   = π(a_t|x_t) / μ(a_t|x_t)          # 单步权重
ρ_t   = Π_{s≤t} w_s                       # 到第 t 步的累乘权重
TIS   = (1/n) Σ_i ρ_{i,T_i} · G_i         # 轨迹重要性采样
PDIS  = (1/n) Σ_i Σ_{t≤T_i} ρ_{i,t} r_{i,t}   # 逐决策估计
```

在绝对连续（π≪μ）下二者都是 `V(π) = E_π[Σ_t r_t]` 的无偏估计；测试用
小 MDP 递归枚举真值对照（普查轨迹上误差 ≤ 1e-11·max(1,|V|)，与单决策
枚举同一收紧口径）。

`evaluate_trajectories` 返回 `TrajectoryEvalResult`（不可变 dataclass）：

- `n_trajectories, horizon`（最大步数）
- `tis, pdis`（两个估计量，始终报告）
- `trajectory_weights`（各轨迹末步累乘权重 ρ_{T_i}）
- `diagnostics: WeightDiagnostics` —— 有效权重诊断：
  - `ess`：末步累乘权重的有效样本量 `(Σw)²/Σw²`
  - `max_weight`、`n_zero_weight`：末步权重最大值与零权重条数
  - `step_ess`：逐步 ESS，`step_ess[t]` 为第 t+1 步累乘权重在长度 ≥ t+1
    的轨迹上的 ESS——随累乘步数的权重退化显式可见
- `result.summary()` 可打印以上全部

行为概率为零的口径与单决策一致：任一步 `π>0` 而 `behavior_prob==0` 抛
`ZeroSupportError`（对全部轨迹的全部步校验）；`π==0` 且 `μ==0` 权重取 0，
合法。某步目标概率为 0 后该轨迹累乘权重归零，后续单步权重不再计算（轨迹
在目标策略下概率为 0，贡献恒 0，避免 0×溢出权重产生 NaN）。单步或累乘
权重超过 `MAX_WEIGHT` 抛 `WeightOverflowError`。

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
| `InvalidTrajectoryError` | （`InvalidLogError` 子类）空轨迹序列、轨迹长度 ∉ [1,5]、步字段 NaN/Infinity、behavior_prob ∉ [0,1]、轨迹数超上限、容器或元素类型错 |
| `InvalidPolicyError` | 概率表缺 context 行/缺日志动作（漏写不隐式当 0）、概率越界、行和与 1 偏差超容差、动作数超上限、`clip_value` 非法 |
| `InvalidQHatError` | q_hat 非映射、缺成熟行所需动作取值、值非有限实数 |
| `ZeroSupportError` | π>0 而记录/行为概率为 0（单决策对全部事件；轨迹对全部步） |
| `WeightOverflowError` | 成熟样本权重或轨迹单步/累乘权重 > 1e300 |
| `NoMatureSamplesError` | cutoff 前无成熟样本 |

- 所有**数值参数**（含 `cutoff`、`clip_value`、概率、reward、时间、q 值）
  拒绝 `NaN` 与正负 `Infinity`；**整数/数值位置拒绝 `bool`**（`True/False`
  不是合法数字参数）。
- `evaluate` 为纯函数、无副作用；校验在构造任何结果前完成，不存在部分变更。
  `evaluate_trajectories` 同样：全部校验（结构、策略、零支持）通过后才计算，
  失败不返回任何结果、不修改传入的策略表/轨迹（有测试覆盖）。

## 输入上限（接口边界，非性能承诺）

| 项 | 上限 |
|---|---|
| 事件数 | 1,000,000（`MAX_EVENTS`） |
| 轨迹数 | 1,000,000（`MAX_TRAJECTORIES`） |
| 每条轨迹步数 | 5（`MAX_TRAJECTORY_STEPS`，题面"最多 5 步"） |
| 每 context 动作数 | 100,000（`MAX_ACTIONS_PER_CONTEXT`） |
| 单个重要性权重（含轨迹累乘权重） | 1e300（`MAX_WEIGHT`，超出抛 `WeightOverflowError`） |
| 概率行归一容差 | `|Σp − 1| ≤ 1e-9`（`PROB_SUM_TOL`） |

短演示不穷举上限规模。时间值为可比较的有限实数，本库不实现时区/时钟语义。

## 浮点结果误差口径

- 求和统一使用 `math.fsum`（误差有界的高精度求和）。
- **枚举恒等测试的误差口径**：在“普查样本”（每个 (x,a) 出现次数严格正比于
  p(x)μ(a|x)，权重与计数为精确比值）上，IPS/SNIPS/DR 与按分布定义独立算出的
  V(π) 之差必须满足
  `|估计 − 期望| ≤ 1e-11 · max(1, |期望|)`。
  这是**收紧的代数口径**（369 个枚举用例），不用宽容差掩盖边界错误。
- **轨迹普查恒等同口径**：在普查轨迹集（每条轨迹条数严格正比于
  p0(x0)·Π_t μ(a_t|x_t)，计数为精确整数）上，TIS/PDIS 与小 MDP 递归枚举的
  V(π) 之差同样满足 `≤ 1e-11 · max(1, |期望|)`（覆盖步数 1/2/3/5 与两套
  目标策略、两套回报表）。
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
7. **轨迹回报取无折扣逐步和**：题面未要求折扣因子，引入它会多一个需要
   校验与解释的参数；TIS/PDIS 的定义与枚举真值在同一无折扣口径下核对。
8. **轨迹零权重短路**：某步 π=0 后累乘权重恒 0，后续单步权重不再计算——
   该轨迹在目标策略下概率为 0，贡献恒 0，短路避免 0×溢出权重产生 NaN，
   同时零支持校验（π>0 而 μ=0）仍对全部步预先执行，不因此被放过。
9. **权重诊断逐步给出**：累乘权重随步数指数退化是轨迹 IS 的主要风险，
   `step_ess` 把每个决策步的 ESS 显式列出，而不是只报一个末步数字。

## 已知限制

- 不提供置信区间/SE（可由 `Estimate` 字段在调用方自行 bootstrap）。
- 不做随机/上下文 embedding 策略，仅离散概率表；目标策略须为调用方持有的
  完整映射。
- 不处理同一决策的多值/序列回报、回报删失（censoring）与观察过程的选择性
  偏差；前提是“到期即可观察”的无信息删失。
- 无流式/增量接口；一次调用处理一个批次（上限 1e6 事件，内存 O(事件数)）。
- q_hat 两种键形式（扁平 tuple 键 / 嵌套映射）同时给出时以扁平键为准。
- 轨迹口径本轮**不实现**：延迟奖励/时刻字段、删失修正、自适应截断、折扣
  因子、逐决策 DR（per-decision DR）与自归一化轨迹估计（WIS/PDWIS）——
  题面明确不要求；TIS/PDIS 为无折扣、未归一化的基础形式。
- 轨迹步数上限 5 是接口边界而非算法限制；累乘权重超过 1e300 即拒绝，
  更长轨迹在高维权重下本就会触发该上限。
