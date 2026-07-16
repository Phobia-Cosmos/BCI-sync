# 物理刺激攻击 BCI 的实验设计方案

## 研究目标

目标不是在 EEG 信号上直接加 adversarial perturbation，而是在更现实的攻击边界下研究：

> 攻击者无法接触 EEG 设备、采集链路、模型或训练数据，只能通过屏幕、灯光、音频、触觉反馈或交互界面改变用户接收到的物理/感官刺激，从而诱导 BCI 模型产生错误甚至危险响应。

核心问题包括：

1. 哪些物理刺激参数最容易让 BCI 模型失效？
2. 不同 BCI 范式对物理刺激攻击的脆弱点是否不同？
3. 攻击导致的错误是随机性能下降，还是可以被定向到危险命令？
4. 这些攻击是否跨被试、跨会话、跨模型泛化？
5. 失败原因是任务相关 EEG 特征被削弱，还是额外诱发了干扰性 EEG 成分？

## 安全与伦理边界

这类实验必须作为 **受控防御性评估** 设计，不能直接连接真实轮椅、机械臂、药物输注、康复刺激器或医疗决策系统。

最低安全要求：

- 使用 **虚拟环境或离线仿真器** 表示危险响应，例如虚拟轮椅撞墙、虚拟机械臂误抓、拼写器选择错误选项。
- 不允许让模型输出直接控制真实物理设备。
- 受试者需签署知情同意，明确说明存在闪烁、视觉干扰、音频/触觉刺激和可能的疲劳。
- 排除有光敏性癫痫、严重偏头痛、视觉敏感、听觉敏感或神经系统高风险病史的受试者。
- 控制刺激时长、亮度、对比度、音量和触觉强度；设置随时停止机制。
- 使用短 block、足够休息和主观不适评分。
- 若涉及 deception、隐蔽刺激或错误反馈，需要 IRB/伦理审批。

论文中建议把“dangerous response”定义为 **simulated dangerous response**，避免真实伤害。

## 总体实验框架

### 阶段 1：干净模型建立

1. 按每个范式采集 clean training data。
2. 使用标准刺激界面和标准实验流程训练模型。
3. 模型训练后固定参数，不再更新。
4. clean test session 评估基线性能。

### 阶段 2：物理刺激攻击测试

攻击者不接触 EEG，只改变用户看到、听到或触摸到的刺激。

基本条件：

| 条件 | 目的 |
| --- | --- |
| Clean | 标准刺激，得到基线准确率和输出稳定性。 |
| Benign control | 改变刺激但不应造成攻击，用于排除普通变化影响。 |
| Random stimulus | 随机刺激扰动，判断模型对非优化扰动的鲁棒性。 |
| Candidate attack stimulus | 系统搜索得到的攻击刺激。 |
| Recovery | 取消攻击后观察性能是否恢复。 |

### 阶段 3：危险响应定义

不同范式的危险响应不同：

| 范式 | 模拟危险响应 |
| --- | --- |
| MI | 虚拟轮椅错误转向、虚拟机械臂误动作、将“停止”识别为“移动”。 |
| P300 | 拼写器选择错误字符、yes/no 选择错误、菜单选择错误医疗/护理选项。 |
| SSVEP | 目标频率被识别为错误命令，例如 stop 被识别为 go。 |
| cVEP | 编码目标匹配错误，选择错误按钮或错误命令。 |
| ErrP/ERN | 错误反馈未被检测，或正确反馈被误判为错误，导致错误纠正/不纠正。 |
| Passive/NS | 错误识别疲劳、注意、情绪或疼痛状态，引发错误提示或错误干预。 |

### 阶段 4：指标

建议同时报告模型指标、系统风险指标和人因指标。

| 指标 | 说明 |
| --- | --- |
| Accuracy/BCA/AUC | 基础分类性能下降。 |
| Attack success rate, ASR | 攻击条件下输出错误或目标错误类别的比例。 |
| Dangerous response rate, DRR | 输出被映射为危险命令的比例。 |
| Targeted ASR | 是否能定向诱导到某个指定错误命令。 |
| Confidence margin | 正确类与错误类置信度差距变化。 |
| ITR drop | 拼写器/选择系统信息传输率下降。 |
| Recovery time | 攻击停止后需要多久恢复基线性能。 |
| Cross-subject transfer | 同一刺激是否能攻击不同用户。 |
| Cross-model transfer | 同一刺激是否能攻击传统模型和深度模型。 |
| Discomfort/fatigue score | 主观不适、视觉疲劳、头痛、恶心等。 |

## 物理刺激参数空间

### 视觉刺激参数

| 参数 | 可搜索范围示例 | 可能影响 |
| --- | --- | --- |
| Flicker frequency | 低频、中频、高频；避开不安全范围 | 诱发 SSVEP、干扰注意、改变视觉节律。 |
| Phase | 与目标刺激同相/反相/相位偏移 | 改变 SSVEP/cVEP 匹配。 |
| Duty cycle | 闪烁亮暗占比 | 改变 VEP 幅值和舒适度。 |
| Luminance/contrast | 亮度、对比度、背景亮度 | 改变 ERP/VEP 强度和疲劳。 |
| Color | 白、灰、红、绿、蓝、彩色组合 | 改变 P300 salience 和视觉通道响应。 |
| Shape | 字符、块状、边框、图标、光栅 | 改变 P300/SSVEP 可分性。 |
| Spatial location | 中央/周边、目标邻近区域 | 诱导注意偏移或视觉竞争。 |
| Motion | 静态、平移、旋转、扩张/收缩 | 影响 SSMVEP、注意和眼动。 |
| Timing jitter | 刺激提前/延迟/丢帧 | 破坏 ERP/VEP time-locking。 |
| Semantic cue | 错误提示、混淆图标、误导文本 | 影响 P300/ErrP 语义处理和反馈预期。 |

### 音频刺激参数

| 参数 | 可能用途 |
| --- | --- |
| Tone frequency | 干扰注意或诱发 auditory ERP。 |
| Volume | 控制干扰强度，但必须在安全范围内。 |
| Timing | 与视觉刺激同步/异步，测试多模态干扰。 |
| Feedback correctness | 对 ErrP/ERN 制造错误反馈或延迟反馈。 |

### 触觉刺激参数

| 参数 | 可能用途 |
| --- | --- |
| Vibration frequency | 干扰运动想象或诱发 somatosensory response。 |
| Intensity | 控制触觉 salience。 |
| Location | 手腕、手指、前臂等。 |
| Timing | 与任务提示或反馈同步/异步。 |

## 范式一：MI 物理刺激攻击实验

### 攻击直觉

MI 模型主要依赖 sensorimotor rhythm，尤其 mu rhythm 和 beta rhythm 的 ERD/ERS。攻击者不能直接改变 EEG，但可以通过视觉闪烁、视觉运动、错误反馈或注意负荷干扰 MI 过程。

2023 SMC 的思路是：即使 MI 不是视觉诱发范式，视觉闪烁也可能引入额外 VEP/SSVEP 成分，或改变注意资源和 mu rhythm，从而让 MI 控制性能下降。

### 实验任务

推荐使用虚拟轮椅或 Pong/光标控制任务：

- 左手 MI -> 向左；
- 右手 MI -> 向右；
- 双脚/静息 -> 停止或前进；
- 输出只控制虚拟环境。

### 模型

- CSP + LDA/SVM；
- FBCSP；
- EEGNet；
- ShallowConvNet/DeepConvNet。

### 攻击刺激

| 攻击类型 | 实现方式 | 假设 |
| --- | --- | --- |
| 视觉闪烁 | 在目标物体、边框或背景叠加 flicker | 诱发 SSVEP，干扰 mu/beta 解码。 |
| 周边闪烁 | 不遮挡任务目标，只在周边闪烁 | 更隐蔽，测试注意分散。 |
| 光流运动 | 背景缓慢移动、扩张/收缩 | 干扰空间注意和视觉运动处理。 |
| 错误反馈 | 显示与真实输出不一致的反馈 | 改变用户策略和 ErrP。 |
| 音频干扰 | 与 cue 同步或异步提示音 | 增加注意负荷。 |
| 触觉干扰 | 在手部/前臂施加轻微振动 | 干扰运动相关感觉通路。 |

### 主要指标

- MI 分类准确率/BCA；
- 虚拟危险命令率，例如 stop -> move、left -> right；
- mu/beta ERD 变化；
- occipital SSVEP 成分是否增强；
- 攻击停止后的恢复时间。

### 关键问题

1. 视觉闪烁是否通过 SSVEP 污染 MI？
2. 周边刺激是否比中心刺激更隐蔽但同样有效？
3. 错误反馈是否会导致用户策略漂移？
4. 传统 CSP 和 EEGNet 哪个更脆弱？

## 范式二：P300 物理刺激攻击实验

### 攻击直觉

P300 直接由 oddball 刺激诱发。攻击者如果能控制拼写器或选择界面，就可以改变刺激显著性、时间锁定、注视位置和语义反馈。

### 实验任务

推荐使用 P300 speller 或 yes/no 医疗选择界面：

- 用户目标：选择指定字符或 yes/no；
- 模拟危险输出：把 yes 选成 no，或把 stop/help 选成 continue/no-help。

### 模型

- xDAWN + LDA；
- SWLDA；
- Riemannian classifier；
- EEGNet/ERPNet。

### 攻击刺激

| 攻击类型 | 实现方式 | 假设 |
| --- | --- | --- |
| 高亮形状变化 | 字符高亮改为 block、边框、阴影或局部闪烁 | 改变 P300 幅值和视觉 salience。 |
| 颜色/亮度变化 | 改变目标/非目标颜色对比 | 改变 target/non-target 可分性。 |
| 相邻诱导 | 在目标邻近项同时闪烁或提前闪烁 | 诱导错误注意或视觉竞争。 |
| 时序扰动 | 改变 stimulus onset asynchrony、插入 jitter | 破坏 ERP time-locking。 |
| 语义干扰 | 在非目标项添加更显眼图标/警告 | 诱导非目标 P300。 |
| 注视诱导 | 用动画或视觉 cue 吸引眼动偏离目标 | 改变 posterior visual response。 |
| 错误反馈 | 显示错误选择结果 | 诱发 ErrP 或改变后续注意策略。 |

### 主要指标

- target/non-target AUC；
- spelling accuracy；
- ITR；
- wrong command rate；
- P300 amplitude/latency；
- N200/视觉成分变化；
- 是否能把输出定向到相邻字符或高 salience 非目标。

### 最适合的攻击目标

P300 的危险响应不一定是“整体性能下降”，更有价值的是 **定向错误选择**：

- 把目标字符诱导到相邻字符；
- 把 yes/no 二选一诱导到相反选项；
- 把 emergency/help 诱导成普通选项。

## 范式三：SSVEP 物理刺激攻击实验

### 攻击直觉

SSVEP 是最适合物理刺激攻击的范式，因为类别本身由外部 flicker frequency/phase 编码。攻击者只要影响屏幕或环境光，就可能改变 EEG 的目标频率响应。

### 实验任务

推荐 4 类或 12 类 SSVEP 命令选择：

- 4 类：left/right/forward/stop；
- 12 类：虚拟键盘或机器人命令；
- 模拟危险输出：stop 被识别为 forward，left 被识别为 right。

### 模型

- CCA；
- FBCCA；
- TRCA/eTRCA；
- EEGNet/Compact-CNN。

### 攻击刺激

| 攻击类型 | 实现方式 | 假设 |
| --- | --- | --- |
| 目标频率偏移 | 将目标频率轻微偏离训练频率 | 降低与模板频率的匹配。 |
| 相位扰动 | 改变目标相位或目标间相位差 | 破坏 joint frequency-phase coding。 |
| 邻近频率干扰 | 在目标附近叠加另一闪烁频率 | 诱导错误目标匹配。 |
| 谐波干扰 | 在目标 harmonic 处加入背景闪烁 | 干扰 CCA/FBCCA 特征。 |
| 背景全局闪烁 | 整个背景以某频率微闪 | 造成全局 VEP 污染。 |
| 刷新时序扰动 | 丢帧、jitter、刷新不同步 | 破坏频率稳定性。 |
| 对比度攻击 | 降低目标对比度，提高非目标对比度 | 改变各目标 SNR。 |

### 主要指标

- frequency recognition accuracy；
- target-to-nontarget CCA correlation margin；
- SSVEP SNR；
- harmonic power 变化；
- targeted ASR：是否能让 12 Hz 被识别为 13 Hz 或 stop 被识别为 forward；
- 用户不适评分。

### 最适合的攻击搜索策略

SSVEP 参数空间较明确，建议先做小规模网格搜索：

- frequency offset；
- phase offset；
- contrast；
- distractor frequency；
- background flicker location。

再用 Bayesian optimization 搜索最容易让目标类别混淆的刺激组合。

## 范式四：cVEP 物理刺激攻击实验

### 攻击直觉

cVEP 依赖 stimulus code/sequence。攻击者不需要改 EEG，只需要改变屏幕上的编码序列、码相位、时序同步或码间相关性。

### 实验任务

- 多目标 cVEP speller；
- 每个目标使用不同 pseudo-random sequence；
- 用户注视目标，模型根据 EEG 与模板相关性识别目标。

### 模型

- template matching；
- canonical correlation / regularized CCA；
- Riemannian classifier；
- deep temporal CNN。

### 攻击刺激

| 攻击类型 | 实现方式 | 假设 |
| --- | --- | --- |
| Code shift | 目标 code 延迟若干 frame | 破坏时间对齐。 |
| Code inversion | 局部反转亮暗序列 | 改变模板相关性。 |
| Code collision | 增加目标和非目标 code 相似性 | 诱导目标混淆。 |
| Dropped frames | 随机丢帧或重复帧 | 破坏 temporal pattern。 |
| Contrast jitter | code 不变但亮度抖动 | 降低模板匹配稳定性。 |

### 主要指标

- template correlation margin；
- target recognition accuracy；
- code-level confusion matrix；
- attack 是否集中诱导到相似 code 目标；
- 对刷新率/显示器差异的敏感性。

## 范式五：ErrP / ERN 物理反馈攻击实验

### 攻击直觉

ErrP/ERN 系统通常检测用户观察到错误或负反馈时的 EEG 响应，用于在线纠错、自适应或监督学习。攻击者不能改 EEG，但可以操控反馈事件本身。

### 实验任务

- 用户观察虚拟系统执行动作；
- 有时系统动作正确，有时错误；
- 模型检测用户是否产生 ErrP；
- 系统根据 ErrP 决定是否纠正动作。

### 攻击刺激

| 攻击类型 | 实现方式 | 假设 |
| --- | --- | --- |
| False feedback | 正确动作显示成错误反馈 | 诱发 false ErrP。 |
| Suppressed feedback | 错误动作显示成正确反馈 | 降低 true ErrP。 |
| Delayed feedback | 错误反馈延迟呈现 | 破坏 time-locking。 |
| Random feedback | 反馈与动作无关 | 破坏用户预测和模型适配。 |
| Multimodal mismatch | 视觉显示正确，音频提示错误 | 诱发不一致认知反应。 |

### 危险响应

- 错误动作未被纠正；
- 正确动作被错误纠正；
- 自适应系统学到错误策略；
- 用户对反馈失去信任，后续 ErrP 变弱。

### 主要指标

- ErrP detection AUC；
- false correction rate；
- missed correction rate；
- feedback delay 对 ERP latency 的影响；
- 用户 trust/fatigue 评分。

## 范式六：Passive BCI / NS / 情绪识别

### 攻击直觉

被动 BCI 不一定有明确控制命令，但会根据用户状态输出疲劳、注意、情绪、压力、疼痛或负荷判断。攻击者可以通过自然视觉/音频刺激诱导错误状态识别。

### 实验任务

- 实时情绪识别；
- 疲劳/注意监测；
- 工作负荷评估；
- 自然图像/视频 EEG 解码。

### 攻击刺激

| 攻击类型 | 实现方式 | 危险输出 |
| --- | --- | --- |
| 情绪诱导图像/音频 | 插入高唤醒或负性刺激 | 把正常状态误判为焦虑/压力。 |
| 视觉负荷 | 高频运动、复杂背景 | 把清醒误判为疲劳或低注意。 |
| 任务无关声音 | 突发提示音或噪声 | 造成注意/压力误判。 |
| 色彩/亮度变化 | 背景色温、亮度波动 | 影响情绪或疲劳特征。 |

### 主要指标

- 状态分类准确率；
- false alarm / missed alarm；
- 状态估计漂移幅度；
- 用户主观状态评分与模型输出偏差；
- 是否导致错误干预或错误告警。

## 如何 figure out 哪些物理刺激最容易让模型失效

### Step 1：先做范式内理论筛选

不同范式优先搜索的物理刺激不同：

| 范式 | 优先搜索刺激 |
| --- | --- |
| MI | 视觉闪烁、周边运动、错误反馈、触觉干扰。 |
| P300 | 高亮形状、颜色、时序、相邻刺激、语义 salience、注视诱导。 |
| SSVEP | 频率、相位、对比度、谐波、背景闪烁、刷新 jitter。 |
| cVEP | code shift、code inversion、frame drop、sequence correlation。 |
| ErrP | 错误反馈、延迟反馈、多模态不一致反馈。 |
| Passive/NS | 情绪刺激、视觉负荷、音频干扰、自然场景变化。 |

### Step 2：小规模网格搜索

先不要直接用复杂优化。每个范式选 3-5 个最有理论依据的参数，做低维网格搜索：

- 每个参数 3 个水平；
- 每个条件短 block；
- 每名受试者先做 1 个 session；
- 选出 ASR/DRR 高且不适评分低的候选刺激。

### Step 3：黑盒优化

如果要系统发现最强刺激，可以使用 human-in-the-loop black-box optimization：

- 输入：物理刺激参数；
- 输出：模型错误率、危险响应率、置信度 margin、用户不适评分；
- 约束：刺激安全范围、最大时长、最大亮度/音量/振动强度；
- 方法：Bayesian optimization、CMA-ES、遗传算法或多臂 bandit。

目标函数示例：

```text
maximize  ASR + lambda1 * targeted_ASR + lambda2 * confidence_margin_drop
          - lambda3 * discomfort_score - lambda4 * perceptibility_score
```

如果不想追求隐蔽性，可去掉 perceptibility；如果想模拟真实攻击，应加入 perceptibility 和 safety penalty。

### Step 4：泛化验证

候选攻击刺激找到后，必须做 hold-out 验证：

1. 新被试是否仍有效？
2. 新 session 是否仍有效？
3. 传统模型和深度模型是否都有效？
4. 换显示器/刷新率/亮度后是否仍有效？
5. 攻击停止后模型能否恢复？

### Step 5：机制解释

不要只报告“准确率下降”，还要解释为什么下降：

| 范式 | 机制分析 |
| --- | --- |
| MI | mu/beta ERD 是否减弱；occipital SSVEP 是否增强；注意指标是否变化。 |
| P300 | P300 amplitude/latency 是否变化；target/non-target separability 是否下降。 |
| SSVEP | 目标频率 SNR 是否下降；错误频率/harmonic power 是否增强。 |
| cVEP | 目标模板相关性是否下降；非目标相关性是否上升。 |
| ErrP | ErrP latency/amplitude 是否受反馈延迟或错误反馈影响。 |

## 推荐优先级

如果目标是尽快做出清晰结果，建议优先级如下：

1. **SSVEP**：最直接，物理频率/相位就是类别编码，容易解释。
2. **P300**：界面刺激可控，容易设计危险选择场景。
3. **MI**：已有 2023 SMC 直接证据，可复现和扩展。
4. **cVEP**：攻击设计清晰，但需要更精确控制显示时序。
5. **ErrP**：很有现实意义，但实验设计和用户反馈控制更复杂。
6. **Passive/NS**：场景真实，但“危险响应”定义要更谨慎。

## 一个可落地的最小实验版本

如果只做一篇初步论文，建议选两个范式：

### 方案 A：SSVEP + P300

原因：二者都是视觉诱发范式，攻击者只需控制屏幕界面。

实验：

1. 训练 clean SSVEP/P300 模型。
2. 测试 clean、random stimulus、candidate attack stimulus 三种条件。
3. 对 SSVEP 搜索 frequency/phase/contrast。
4. 对 P300 搜索 color/shape/timing/adjacent flashing。
5. 输出 ASR、DRR、ITR drop、ERP/VEP 机制分析。

### 方案 B：复现 MI + 扩展 SSVEP

原因：MI 有 2023 SMC 直接前作，SSVEP 是最自然扩展。

实验：

1. MI 复现视觉 flicker attack。
2. SSVEP 做 frequency/phase/background flicker attack。
3. 比较“非视觉诱发范式 MI”和“视觉诱发范式 SSVEP”的差异。
4. 结论可写成：视觉诱发范式更暴露于物理刺激攻击，MI 也会因跨范式神经响应耦合受到影响。

## 可直接写进论文的方法段

> We consider a realistic external attacker who cannot access the EEG device, signal acquisition pipeline, classifier, or training data. Instead, the attacker can only manipulate sensory events presented to the user through the graphical interface, environmental light, audio cues, or feedback signals. The victim BCI model is trained under clean stimulus conditions and remains fixed during attack evaluation. We evaluate whether constrained physical sensory stimuli can induce simulated dangerous responses in different BCI paradigms, including wrong control commands, incorrect selections, and failed error corrections.

中文版本：

> 本研究考虑一种更现实的外部攻击者：攻击者无法接触 EEG 设备、采集链路、分类模型或训练数据，只能通过图形界面、环境光、音频提示或反馈信号操纵用户接收到的感官事件。受害 BCI 模型在标准 clean 刺激条件下训练，并在攻击测试阶段保持固定。我们评估受约束的物理感官刺激是否会诱发不同 BCI 范式中的模拟危险响应，包括错误控制命令、错误选择以及错误纠正失败。
