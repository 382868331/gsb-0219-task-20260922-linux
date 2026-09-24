# delayed_ope —— 离线策略评估可复用库（单步延迟结算 + 多步轨迹）

纯 Python 标准库实现的离线策略评估（OPE）模块，包含两组入口：

1. **单步延迟结算**：推荐/决策日志中**回报延迟结算**的情形。在固定评估
   截止时刻，只用已经观察到回报的成熟样本，计算目标策略的 IPS、SNIPS、
   Doubly Robust（DR）估计；未成熟样本**绝不按 reward=0 计入**。
2. **多步轨迹（本次新增）**：最多 **5 步**的完整轨迹，**逐步累乘**目标
   与行为策略概率，给出轨迹重要性采样（IS/SNIS）与**逐决策**
   （per-decision PD-IS/PD-SNIS）估计，附有效权重诊断；行为概率为零而
   目标概率为正的输入整体拒绝；真值用小 MDP 枚举对照。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库；无任何第三方依赖）
- 运行环境：Linux 原生；装好 Python 后无需安装其他依赖
- 离线：不访问网络、账号、数据库、中间件、云 API 或 Docker
- 核心算法未使用任何第三方 OPE 实现

## 运行命令

```bash
python -m unittest discover -s tests -v   # 全部测试（44 个测试方法）
python demo.py                            # 约 0.2 秒完成的固定输入演示（上限 90 秒）
```

在仓库根目录执行。`tests/independent_reference.py`（单步）与
`tests/mdp_reference.py`（轨迹）都是**独立参考**：它们不导入被测核心，
期望值按分布定义直接代数求和/枚举；测试中的标量期望亦按公式手算。
被测核心只产出待核对的值，不参与期望值的生成。

## 目录

```
delayed_ope/__init__.py   公开接口
delayed_ope/core.py       核心算法与全部校验（单步 + 轨迹）
tests/independent_reference.py  独立分布参考（单步，不依赖被测核心）
tests/mdp_reference.py          小 MDP 枚举/反向递推参考（轨迹）
tests/test_ope.py         单步必要测试
tests/test_trajectories.py     轨迹必要测试
demo.py                   固定种子数据生成 + 正常结果 + 边界与实际拒绝
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

## 多步轨迹评估（逐步累乘，最多 5 步）

```python
from delayed_ope import Step, Trajectory, evaluate_trajectories

# 每步直接给状态、动作、奖励、行为策略概率 mu(a|s)、目标策略概率 pi(a|s)
traj = Trajectory((
    Step(state, action, reward, logged_prob, target_prob),  # t=0
    ...                                                     # 最多 5 步
), discount=1.0)

result = evaluate_trajectories([traj, ...])   # 纯函数，无概率表参数
```

`Step` 字段：`state`/`action`（可哈希）、`reward`（有限实数）、
`logged_prob`、`target_prob`（均为 [0,1] 有限实数；bool 拒绝）。
`Trajectory` 字段：`steps: tuple[Step, ...]`（长度 **1..5**）、
`discount ∈ [0,1]`（默认 1.0；同一批次必须统一，否则拒绝）。变长轨迹
（不同轨迹步数不同）允许，`horizon` 字段报告批次最大步数。

逐步累乘前缀权重 `ρ_{0:t} = Π_{j=0}^{t} pi(a_j|s_j)/mu(a_j|s_j)`，
轨迹折扣回报 `G = Σ_t γ^t r_t`。返回 `TrajectoryEvaluationResult`：

- `n_trajectories, horizon, discount`
- `estimate: TrajectoryEstimate`：
  - **轨迹 IS** `(1/N) Σ_τ W_τ G_τ`，`W_τ = ρ_{0:L-1}`
  - **轨迹 SNIS** `Σ W_τ G_τ / Σ W_τ`
  - **逐决策 PD-IS** `(1/N) Σ_τ Σ_t γ^t ρ_{0:t} r_t`
  - **逐决策 PD-SNIS** `Σ_τ Σ_t γ^t ρ_{0:t} r_t / Σ_τ (1/Γ(L_τ)) Σ_t γ^t ρ_{0:t}`，
    其中 `Γ(L)=Σ_{t=0}^{L-1}γ^t`（γ=1 时即 L）；mu=pi 时每轨迹分母项
    期望恰为 1，等长/变长、任意 γ 下普查比值都回到平均折扣回报
  - `snis`/`pd_snis` 在对应权重和为 0 时为 `None`（0/0 无定义）
- `weights: WeightDiagnostics`：终权重 `n/min/max/mean/ess`
  （`ESS=(ΣW)²/ΣW²`），以及逐决策前缀权重的 `n_decision/decision_ess`
  （跨全部 (轨迹, t) 决策点）。

### 逐决策为什么方差更低

`r_t` 只依赖截至 t 的历史，其无偏重要性权重只需累乘到 t，不必乘以后续
因子 `ρ_{t+1:L-1}`（条件期望为 1 但贡献方差）。这是**期望恒等**（普查
精确核对）而非逐样本方差保证：单次抽样上 PD 误差未必更小，demo 只展示
本次固定种子的实际数字。

### 轨迹零支持与溢出（严格）

- **全部轨迹全部步**校验：任一步 `target_prob>0` 且 `logged_prob==0` →
  `ZeroSupportError`，不返回任何估计（即使该步之前的因子已把累乘归 0，
  结构性不相容也不被掩盖）。
- `target_prob==0` 且 `logged_prob==0`：该因子取 **0**，累乘权重在该处
  归 0，合法；全部终权重为 0 时 IS=PD-IS=0、SNIS=PD-SNIS=`None`、
  ESS=0，退化口径显式暴露。
- **任一前缀权重**（不止终权重）超过 `MAX_WEIGHT` →
  `WeightOverflowError`：PD 估计实际使用每个前缀，终权重不溢出不能掩盖
  前缀溢出。`1/μ` 单因子非有限同样拒绝。
- 非法结构（空批次/空轨迹、步数越界、steps 非 tuple、元素非 Step、
  概率/奖励 NaN/Infinity 或越界、折扣越界/批次不统一、state/action
  不可哈希、bool 占据数值位置）→ `InvalidTrajectoryError`。

## 错误语义（非法输入不静默纠正，错误可定位）

所有错误继承 `OPEError`；消息带字段名/`events[i]` 位置：

| 异常 | 触发 |
|---|---|
| `InvalidLogError` | 单步事件数值 NaN/Infinity；`cutoff`/reward/时间非有限实数；propensity ∉ [0,1]；t_o<t_d；context/action 不可哈希；事件数超上限；容器或元素类型错 |
| `InvalidTrajectoryError` | 轨迹批次/元素/steps 类型错；空批次或空轨迹；步数 ∉ [1,5]；概率/奖励 NaN/Infinity 或越界；`discount` ∉ [0,1] 或批次内不统一；state/action 不可哈希；bool 占据数值位置；轨迹数超上限 |
| `InvalidPolicyError` | 概率表缺 context 行/缺日志动作（漏写不隐式当 0）、概率越界、行和与 1 偏差超容差、动作数超上限、`clip_value` 非法 |
| `InvalidQHatError` | q_hat 非映射、缺成熟行所需动作取值、值非有限实数 |
| `ZeroSupportError` | π>0 而 logged 概率==0（单步对全部事件；轨迹对全部步） |
| `WeightOverflowError` | 单步成熟样本权重 > 1e300；轨迹任一前缀累乘权重 > 1e300 |
| `NoMatureSamplesError` | cutoff 前无成熟样本（仅单步入口） |

- 所有**数值参数**（含 `cutoff`、`clip_value`、概率、reward、时间、q 值）
  拒绝 `NaN` 与正负 `Infinity`；**整数/数值位置拒绝 `bool`**（`True/False`
  不是合法数字参数）。
- `evaluate` 为纯函数、无副作用；校验在构造任何结果前完成，不存在部分变更。
  不会修改传入的策略表/事件（有测试覆盖）。

## 输入上限（接口边界，非性能承诺）

| 项 | 上限 |
|---|---|
| 单步事件数 | 1,000,000（`MAX_EVENTS`） |
| 轨迹数 | 1,000,000（`MAX_TRAJECTORIES`） |
| 每条轨迹步数 | 1..5（`MAX_HORIZON`） |
| 每 context 动作数 | 100,000（`MAX_ACTIONS_PER_CONTEXT`） |
| 单个重要性权重 | 1e300（`MAX_WEIGHT`，超出抛 `WeightOverflowError`；轨迹检查每个前缀） |
| 概率行归一容差 | `|Σp − 1| ≤ 1e-9`（`PROB_SUM_TOL`，仅单步概率表入口） |

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
- **轨迹枚举测试的误差口径**：在 3 状态、2 动作、确定性转移/奖励的小 MDP
  上，按 μ 的整数多重数构造普查轨迹（每种动作序列出现次数严格正比于
  P_μ(序列)），对 horizon L=1..5、γ∈{1,0.5,0}、两套策略（含 π=0 孔）共
  30 个配置，以及一个变长混批配置，要求 IS/SNIS/PD-IS/PD-SNIS 与枚举真值
  之差满足 `|估计 − 真值| ≤ 1e-9 · max(1, |真值|)`。真值由**两条互相
  独立**的算法给出并互相核对：按 π 分布直接枚举序列，与 Bellman 反向
  策略评估递推。多步累乘使误差随步数积累，故口径较单步普查略宽。
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
7. **轨迹概率随步自带**而非另传概率表：多步日志的策略本就可能随时段/版本
   变化，逐步给值最贴近数据；代价是不校验跨轨迹一致性，按原值累乘。
8. **PD-SNIS 用 Γ(L) 归一分母**而非裸决策点计数：使任意折扣、变长轨迹下
   普查恒等都成立（mu=pi 时每轨迹分母项期望恰为 1）。
9. **前缀溢出独立于终权重拒绝**：逐决策估计实际使用每个前缀，0 吸收导致
   终权重很小也不能豁免中间前缀的溢出。

## 已知限制

- 不提供置信区间/SE（可由结果字段在调用方自行 bootstrap）。
- 不做随机/上下文 embedding 策略，仅离散概率表；单步入口的目标策略须为
  调用方持有的完整映射。
- 单步入口不处理同一决策的多值/序列回报、回报删失（censoring）与观察
  过程的选择性偏差；前提是“到期即可观察”的无信息删失。
- 轨迹入口的范围按本题契约限定：**无延迟结算/删失语义**（轨迹视为完整
  观测）、**无自适应截断**、无 DR 版本；概率随每步显式给出，不校验跨
  轨迹的策略一致性（同一状态可出现不同概率，按日志原值累乘）。
- 轨迹 ESS 为描述性诊断，不是偏差/置信区间保证；逐决策“方差更低”是
  期望性质，不构成逐样本承诺。
- 无流式/增量接口；一次调用处理一个批次（上限 1e6 条/事件，内存
  O(事件数 + 全部决策步)）。
- q_hat 两种键形式（扁平 tuple 键 / 嵌套映射）同时给出时以扁平键为准。
