# delayed_ope —— 含延迟结算反馈的离线策略评估（可复用库）

纯 Python 标准库实现的离线策略评估（OPE）模块。处理推荐/决策日志中**回报延迟
结算**的情形：在固定评估截止时刻，只用已经观察到回报的成熟样本，计算目标策略
的 IPS、SNIPS、Doubly Robust（DR）估计；未成熟样本**绝不按 reward=0 计入**。

- Python 版本：**3.14.7**（仅使用 3.10+ 通用语法，标准库；无任何第三方依赖）
- 运行环境：Linux 原生；装好 Python 后无需安装其他依赖
- 离线：不访问网络、账号、数据库、中间件、云 API 或 Docker
- 核心算法未使用任何第三方 OPE 实现

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试（42 个测试方法，枚举含 369 个恒等用例）
python demo.py                            # 约 0.1 秒完成的固定输入演示（上限约 8 秒）
python demo_timeline.py                   # 增量接口演示：估计随 cutoff 推进变化（约 0.1 秒）
```

在仓库根目录执行。`tests/independent_reference.py` 是**独立参考**：它不导入
被测核心，期望值按有限分布定义直接代数求和；测试中的标量期望亦按公式手算。
被测核心只产出待核对的值，不参与期望值的生成。

## 目录

```
delayed_ope/__init__.py   公开接口
delayed_ope/core.py       核心算法与全部校验（批量 evaluate + 增量 TimelineEvaluator）
tests/independent_reference.py  独立分布参考（不依赖被测核心）
tests/test_ope.py         必要测试
tests/test_timeline.py    增量接口测试（与批量入口逐位核对 + 手算参照）
demo.py                   固定种子数据生成 + 正常结果 + 两个真实触发的拒绝边界
demo_timeline.py          增量推进演示 + 倒退/重复 ID 两个真实触发的拒绝边界
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
| `event_id` | 事件标识（可选，默认 None） | 可哈希；批量入口忽略，增量接口要求唯一非 None |

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
| `DuplicateEventIdError` | 增量接口：event_id 缺失/不可哈希/重复（InvalidLogError 子类） |
| `CutoffRegressionError` | 增量接口：advance 的 cutoff 相对已推进时刻倒退 |

- 所有**数值参数**（含 `cutoff`、`clip_value`、概率、reward、时间、q 值）
  拒绝 `NaN` 与正负 `Infinity`；**整数/数值位置拒绝 `bool`**（`True/False`
  不是合法数字参数）。
- `evaluate` 为纯函数、无副作用；校验在构造任何结果前完成，不存在部分变更。
  不会修改传入的策略表/事件（有测试覆盖）。

## 按观察时间推进的增量接口（TimelineEvaluator）

`TimelineEvaluator` 在**创建时一次加载并冻结**日志与目标策略（及可选 q_hat、
clip_value 的快照），之后用 `advance(cutoff)` 单调推进观察时间、`query()`
查询当前结果。它不是批量入口的平行实现：构造复用同一套校验，查询复用同一个
`_compute_estimates` 与 `math.fsum` 稳定求和，且已成熟事件按原始下标顺序
保持 —— 因此**任一截止时刻的 IPS/SNIPS/DR/ESS 与在该截止时刻直接调用
`evaluate` 的结果完全相同**（测试逐位核对，含 clipped 组与 mature_indices）。

```python
from delayed_ope import TimelineEvaluator

tl = TimelineEvaluator(events, target_policy, q_hat=q, clip_value=2.0)
tl.advance(5.0)          # 整组纳入 observed_time <= 5.0 的新成熟样本
res = tl.query()         # == evaluate(events, target_policy, 5.0, q_hat=q, clip_value=2.0)
tl.advance(9.0)          # 只纳入 (5.0, 9.0] 内新成熟的组
```

契约要点：

- **同刻整组原子纳入**：同一 `observed_time` 的记录作为一组，要么全组进入
  要么不进入，任何时刻都不暴露半组结果；组内任一事件出错（如权重溢出、
  q_hat 缺值）则该次 `advance` 整体失败且不留部分变更。
- **单调不减**：`advance` 的 cutoff 小于已推进时刻抛
  `CutoffRegressionError`；以相同 cutoff 重复推进是幂等空操作。
- **重复 ID 拒绝**：每条事件必须携带唯一非 None 的可哈希 `event_id`，
  重复/缺失/不可哈希在创建时抛 `DuplicateEventIdError`（InvalidLogError
  子类），防止同一决策被重复纳入累计量。
- **未成熟绝不当 0**：尚无任何成熟样本时 `query()` 抛
  `NoMatureSamplesError`（与批量入口同口径）；SNIPS 分母为零仍返回
  `None`、ESS 为 0，沿用显式状态而非凭空给 0 估计。
- **截断配置创建后固定**：`clip_value` 只在构造时给定，之后不可更改。
- **错误时机**：与 cutoff 无关的校验（日志结构、策略表、零支持、ID 唯一）
  在创建时完成；权重溢出与 q_hat 取值错误与批量入口一样只在样本真正成熟
  时（即包含该样本的 `advance` 中）触发。
- **口径说明**：cutoff=T 的查询是对“T 之前已成熟子日志”这一有限总体的
  策略价值估计；不同截止时刻对应不同估计对象，估计值随样本成熟而改变是
  口径定义使然，不是实现不稳定。

## 输入上限（接口边界，非性能承诺）

| 项 | 上限 |
|---|---|
| 事件数 | 1,000,000（`MAX_EVENTS`） |
| 每 context 动作数 | 100,000（`MAX_ACTIONS_PER_CONTEXT`） |
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
- 不做随机/上下文 embedding 策略，仅离散概率表；目标策略须为调用方持有的
  完整映射。
- 不处理同一决策的多值/序列回报、回报删失（censoring）与观察过程的选择性
  偏差；前提是“到期即可观察”的无信息删失。
- 批量 `evaluate` 一次调用处理一个批次；增量 `TimelineEvaluator` 为单
  调推进的有状态对象（事件上限同为 1e6，内存 O(事件数)），不提供回滚、
  分支时间线或多步轨迹/在线服务能力。
- q_hat 两种键形式（扁平 tuple 键 / 嵌套映射）同时给出时以扁平键为准。
