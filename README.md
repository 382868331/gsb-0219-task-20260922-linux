# delayed_ope — 含延迟结算反馈的离线策略评估

Python 3.14.7 标准库实现，无任何第三方依赖。针对带**延迟回报**的
上下文赌博机日志，在固定评估截止时刻只使用已观察到的回报，估计
目标策略 π 的价值

```
V(π) = E_x[ Σ_a π(a|x) · r(x,a) ]
```

提供三个估计器：

- **IPS**（逆倾向得分）：`mean_i (π(a_i|x_i)/μ_i) · r_i`
- **SNIPS**（自归一化 IPS）：`Σ w r / Σ w`
- **DR**（双稳健）：给定 `q_hat(x,a)`，
  `mean_i [ m(x_i) + w_i (r_i − q_hat(x_i,a_i)) ]`，
  其中 `m(x) = Σ_a π(a|x) q_hat(x,a)`、`w = π/μ`

## 运行环境与命令

- Python **3.14.7**（仅使用标准库；3.10+ 亦可运行，未在其他版本验证）
- Linux 原生运行，`pip install` 不需要任何包；离线，无网络/账号/密钥/数据库
- 测试：

```bash
python -m unittest discover -s tests -v
```

- 演示（约 0.1 秒，远低于 8 秒预算；确定性数据，无 sleep、无预录结果）：

```bash
python demo.py
```

## 快速上手

```python
from delayed_ope import LogRecord, evaluate_policy

records = [
    LogRecord(context="news_user", action="a_politics", reward=1.0,
              logged_propensity=0.6, decision_time=0.0, observed_time=1.0),
    LogRecord(context="news_user", action="a_music", reward=-0.5,
              logged_propensity=0.05, decision_time=0.0, observed_time=9.0),
]
target_policy = {"news_user": {"a_politics": 0.8, "a_music": 0.2}}
q_hat        = {"news_user": {"a_politics": 0.9, "a_music": -0.2}}

res = evaluate_policy(records, target_policy, cutoff=5.0,
                      q_hat=q_hat, clip=2.0)
print(res.ips, res.snips, res.dr)
print(res.ips_unclipped, res.ess_clipped, res.maturity)
```

## 公共接口

### `LogRecord(context, action, reward, logged_propensity, decision_time, observed_time)`

不可变 dataclass，构造时即校验：

| 字段 | 要求 |
|---|---|
| `context`, `action` | 任意可哈希、非 `None` 的标签（str/int/tuple 等） |
| `reward` | 有限实数（int/float，拒绝 bool、NaN、±Inf、字符串） |
| `logged_propensity` | 有限实数，`0 < μ ≤ 1` |
| `decision_time` | 有限实数 |
| `observed_time` | 有限实数，且 `observed_time >= decision_time` |

每条记录都必须携带 reward 字段；其值是否被使用只取决于成熟度，
未成熟记录的 reward 不参与任何计算。

### `evaluate_policy(records, target_policy, cutoff, *, q_hat=None, clip=None, max_records=1_000_000)`

- `records`：`Sequence[LogRecord]`，顺序任意。
- `target_policy`：`{context: {action: π(a|x)}}`。日志中出现的每个
  context 都必须有一行；行内概率有限、在 `[0,1]`、至少一个为正，
  且**之和与 1.0 的绝对偏差 ≤ 1e-12**，否则拒绝（严格口径，见下）。
  **该 context 下日志里出现过的每个动作都必须在行内显式列出**
  （可给 0 概率）；缺条目不按隐式 0 处理，而是报可定位错误。行中
  允许包含日志未出现且概率为 0 的动作条目；正概率动作必须有日志
  支持，否则见 `ZeroSupportError`。
- `cutoff`：有限实数截止时刻 T。**成熟判定：`observed_time <= T`**
  （等号算成熟）。未成熟记录整体剔除，绝不按 reward=0 填充。
- `q_hat`：可选，`{context: {action: q}}`，只需覆盖所有
  `π(a|x) > 0` 的 (x,a)，值有限；多余条目允许。
- `clip`：可选，有限实数 `M > 0`。截断权重 `min(w, M)` 只作用于
  返回值的"截断列"；**未截断列永远同时返回**。
- `max_records`：正整数（bool 拒绝），日志条数上限。

返回 `EstimatorResult`（不可变）：

| 字段 | 含义 |
|---|---|
| `ips` / `snips` / `dr` | 主结果列；给了 `clip` 时为截断值，否则为未截断值。无 `q_hat` 时 `dr=None` |
| `ips_unclipped` / `snips_unclipped` / `dr_unclipped` | 未截断结果，始终报告 |
| `clip` | 使用的截断阈值，未截断时为 `None` |
| `ess_unclipped` / `ess_clipped` | Kish 有效样本量 `(Σw)²/Σw²`；未给 clip 时 `ess_clipped=None`；`ess` 属性返回当前生效列 |
| `max_weight` | 成熟样本上最大**未截断**权重 |
| `n_mature`, `maturity` | 成熟样本数与 `MaturityReport`（cutoff、总数、成熟/未成熟数及其在输入中的下标元组） |

### 异常（`delayed_ope.errors`，均带可定位信息）

- `LogValidationError`：记录/截止时刻/截断阈值/批次形状不合法，
  批量入口的错误信息带 `records[i]:` 前缀和字段名。
- `PolicyValidationError`：策略表缺失行、非映射、概率越界、行和不为 1。
- `ZeroSupportError`（`PolicyValidationError` 子类）：**对日志中出现的
  某 context，存在 π(a|x) > 0 但 (x,a) 在整条日志中从未出现**（记录
  策略经验支持为零，IPS 比值无定义）。判定基于全量日志，与 cutoff
  无关。π(a|x)=0 的未观测动作不触发。
- `QTableValidationError`：q_hat 形状错误或缺正概率动作条目。
- `NoMatureRecordsError`（`OPEError` 子类）：cutoff 下无任何成熟记录。
  这不是输入格式错误；异常对象带 `total/immature/cutoff`。此时不返回
  任何数值——延迟未观察绝不等于 0。
- `NumericalOverflowError(ArithmeticError)`：输入全部有限，但权重或
  聚合溢出为非有限值；与"非法输入"明确区分。

所有校验在构造返回值之前完成；本模块无共享/可变状态，失败调用不会
产生部分变更。

## 输入上限（接口边界，非性能建议）

- 日志条数：默认 ≤ **1,000,000**（可用 `max_records` 调整；时间与
  内存均为 O(n)）。
- 每个 context 的策略动作数：≤ **10,000**。
- 时间戳：任意有限实数，不设单位（调用方自行保证同一时间尺度）。
- 不做去重：同一 (x,a) 可出现任意多次；不去重、不跨 context 合并。

## 设计取舍

1. **样本平均口径**：三个估计器都对成熟记录取 `1/n_m` 的平均（bandit
   OPE 的标准点估计口径），不假设日志是固定轮数的完整枚举。
2. **支持不足直接拒绝**：目标策略以正概率访问日志从未探索的
   (x,a) 时，精确 IPS 不可定义；不做平滑、不把 μ 当 ε、不静默忽略，
   而是抛 `ZeroSupportError`。反过来，日志记录过、但策略行未列出的
   动作也不隐式当作 π=0，而要求显式写 0（避免"漏填"与"故意不选"
   混淆）。有限表格协议下这是唯一诚实的选择。
3. **延迟反馈只做"硬"成熟度切分**：`observed_time ≤ T` 即成熟，
   否则剔除并显式报账。不做删失信似然、不做插补、不做外推——这些是
   独立的建模假设，超出本契约。
4. **截断是显式选择**：`clip` 只影响截断列，未截断 IPS/SNIPS/DR 与
   ESS 始终一并给出，防止只报告被截断后的乐观数字。截断会引入偏差，
   demo 中可同时看到两列差异。
5. **严格归一化**：策略行和必须落在 `1 ± 1e-12`，不自动重归一化
   （自动重归一化会掩盖上游 bug）。
6. **无状态、纯函数**：`evaluate_policy` 不写文件、不保留批次状态；
   因而天然满足"失败不留部分变更"。
7. **无搜索/预算组件**：本模块不含任何算法搜索，故"预算耗尽 vs
   证明无解"的区分不适用；唯一的"无法给出答案"情形是
   `NoMatureRecordsError`（数据不足）与 `ZeroSupportError`（支持不足），
   二者均显式抛出而非返回哨兵值。

## 浮点误差口径

- 全部计算为 IEEE-754 binary64（Python float），未使用任何宽容差
  "宽容通过"。测试期望值由独立的 `fractions.Fraction` 精确参考实现
  推导（`tests/reference.py`，**不 import 被测核心来产生期望值**），
  再与 float 输出比较：
  - 常规定价规模：绝对误差 ≤ **1e-12 × max(1, |精确值|)**；
  - 权重达 1e9 的极端用例：绝对误差 ≤ **1e-9 × 量级**
    （大权重下 float 累加的固有误差，已在测试中显式标注）。
- 边界口径不用容差模糊：成熟判定是精确的 `observed_time <= cutoff`
  浮点比较；策略行和使用 1e-12 的**固定**绝对带，超界即拒绝。
- 任何中间量成为 NaN/Inf（含权重 `π/μ` 溢出）都抛
  `NumericalOverflowError`，不会返回非有限结果。

## 测试与独立参考

`python -m unittest discover -s tests -v`，共 56 个用例，覆盖：

- NaN/Inf/bool/字符串/越界概率/时间倒挂/空日志等输入拒绝与错误定位；
- 零支持拒绝（含"动作在别的 context 出现过"的近似陷阱，以及该拒绝
  与 cutoff 选择无关、优先于"全未成熟"触发）；
- **有限上下文-动作分布的完全枚举无截断期望检查**：在网格上穷举
  (μ, π, reward)，按 μ 的精确频率铺满每个 (x,a) 层，验证未截断
  IPS/SNIPS 恰好等于闭式 V(π)（Fraction 精确值对照）；
- **40 组随机日志的 oracle 交叉检验**（补充）：随机 μ/π/q/延迟/截断，
  数据随机、期望值仍由独立 Fraction 参考给出，并覆盖成熟权重全为 0
  时 SNIPS=None、DR 仅含 m(x) 等边界；
- 极端但有限权重（μ=1e-9）与 float 最小正规格 μ 的溢出区分；
- 全部未成熟 → `NoMatureRecordsError`，及"未成熟不按 0 填充"；
- q_hat 完美时 DR 与权重/截断无关、精确等于 V(π)；
- 截断列与未截断列并存、ESS 随截断上升。

`tests/reference.py` 是独立 Fraction 参考；它不调用被测核心获得
期望值，仅在测试边界把精确有理值转成 float 喂给核心。

## demo.py 说明

固定种子 `random.Random(20260923)` 生成 2400 条两 context、三动作的
结构化日志（约 30% 回报延迟 10 个时间单位），打印：

1. cutoff=18 下的正常结果：成熟/未成熟条数、最大权重、截断前后 ESS、
   IPS/SNIPS/DR 三估计两列，以及直接由真实回报表算出的 V(π) 对照；
2. 实际触发的两个边界：cutoff=0.5 时零成熟 → `NoMatureRecordsError`；
   目标策略以正概率访问日志从未出现的动作 → `ZeroSupportError`。

## 已知限制

- 仅支持**有限表格** context/动作：策略与 q_hat 都是嵌套映射；
  不实现函数型策略、连续动作或神经网络 q 模型，不宣称兼容任何行业
  标准协议。
- 不估计置信区间/方差/渐近协方差，只给点估计与 ESS 诊断。
- 成熟度仅用 `observed_time`；不建模观察机制本身（非随机删失、
  反馈时间分布均不处理）。
- DR 的 q_hat 按调用方给定原样使用，不做交叉拟合（cross-fitting）。
- 未在 Windows/macOS 与 3.14.7 以外解释器上验证；代码本身只用
  跨平台标准库。
