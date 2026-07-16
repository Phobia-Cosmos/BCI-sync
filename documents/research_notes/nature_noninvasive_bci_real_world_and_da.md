# Nature 及子刊中非侵入式 BCI 与真实使用场景：能力弥补、神经机制与 DA 证据链

> 目的：为后续撰写“EEG + SFDA + MU/隐私保护”的双栏论文提供背景材料。重点放在 Nature / Nature 子刊中与真实使用情景结合的 BCI 论文，尤其是非侵入式 EEG/fNIRS/视觉诱发 BCI；同时整理这些工作如何从神经机制上成立，以及 Nature 系列是否已经把 domain shift / adaptation 作为问题处理。

## 1. 核心结论

1. **Nature 系列已经把 BCI 放在真实能力弥补场景中讨论**：通信、手/臂控制、手指精细控制、下肢步态康复、视觉拼写器和神经反馈训练是主要场景。
2. **非侵入式 BCI 的价值不是“最高性能”，而是“低风险、低成本、可在临床/家庭推广”**。Nature Communications 2025 明确指出 EEG 的非侵入性、低成本和便携性有利于临床和家庭使用，但其精度、稳定性、信噪比、空间分辨率仍弱于侵入式系统。
3. **能力边界已经被推动，但还远未完全恢复自然能力**：
   - 已能实现 EEG 机器人臂 reach-and-grasp、机器人手指级控制、BCI 步态训练后部分神经恢复、瘫痪患者 EEG 拼写。
   - 仍缺少长期、跨天、跨人、患者真实家庭环境中的稳定高吞吐控制。
4. **Nature 系列并不总用“domain adaptation”这个机器学习术语，但已经明确处理 DA 的核心问题**：跨 session、跨 subject、低信噪比、非平稳 EEG、校准、fine-tuning、cross-session adaptation。
5. **对我们课题的直接启发**：BCI 实际部署需要不断适配用户/天次/状态变化；这会增加数据收集、模型记忆和隐私泄露风险。因此，“source-free domain adaptation + machine unlearning + privacy-preserving EEG decoding”有明确的临床/真实部署动机。

## 2. 从“缺失能力”出发的 Nature 证据

| 缺失/受损能力 | 代表人群 | Nature 系列论文 | BCI 模态与范式 | 已实现的弥补 | 仍然欠缺 |
|---|---|---|---|---|---|
| 语言/通信能力 | locked-in syndrome、ALS、完全瘫痪患者 | Birbaumer et al., Nature 1999 | EEG slow cortical potentials, SCP | 完全瘫痪者可通过脑电调控驱动拼写设备 | 速度低、训练时间长、可用人群有限；对完全 locked-in 患者更困难 |
| 外部设备/机器人臂控制 | 脊髓损伤、脑干卒中、肌营养不良等导致运动输出受损者 | Meng et al., Scientific Reports 2016 | 非侵入式 EEG，motor imagery，mu rhythm | 13 名受试者学习控制机器人臂完成 3D reach-and-grasp | 主要是健康受试者；控制自由度通过分阶段降低；真实患者长期家庭使用仍不足 |
| 手/手指精细运动 | stroke 后上肢/手功能障碍、截瘫/四肢瘫潜在人群 | Ding et al., Nature Communications 2025 | 非侵入式 EEG，ME/MI，EEGNet + fine-tuning | 实时机器人手指级控制；2 指 MI 80.56%，3 指 MI 60.61% | 仍在健康且有经验 BCI 用户中验证；3 指任务精度较低；同手手指皮层表征高度重叠 |
| 下肢运动/步态与感觉 | 慢性脊髓损伤截瘫患者 | Donati et al., Scientific Reports 2016 | 16 通道 EEG + VR + 触觉反馈 + Lokomat/ZeroG/exoskeleton | 8 名慢性 SCI 患者 12 个月训练后出现感觉与自主肌肉控制改善，50% 从完全截瘫分类提升为不完全截瘫 | 样本小、训练极长、多组件干预难以拆分机制；不是“即插即用”的独立行走能力 |
| 视觉拼写/选择 | 需要低肌肉依赖输入的人机交互用户 | Kosnoff et al., Nature Communications 2024 | mVEP BCI speller + 非侵入式 tFUS 调制 V5 | V5 靶向 tFUS 降低 BCI speller 错误，提示可通过调制视觉注意增强非侵入式 BCI | 健康受试者为主；仍依赖视觉刺激与注意；属于增强识别而非恢复完整运动/语言 |
| 康复性神经恢复 | stroke、SCI 后运动功能恢复 | Chaudhary et al., Nature Reviews Neurology 2016 | EEG/fNIRS 等非侵入式与侵入式 BCI 综述 | 区分 assistive BCI 与 rehabilitative BCI；强调 EEG-BCI + 行为治疗可诱导神经可塑性 | 需要更大样本、随机对照、长期随访与患者真实场景验证 |

## 3. 关键论文解读

### 3.1 通信：EEG 拼写设备为瘫痪患者提供输出通道

Birbaumer et al. 在 Nature 1999 的 *A spelling device for the paralysed* 是非侵入式 BCI 真实应用的早期标志性工作。论文聚焦完全瘫痪但感觉和认知功能保留的 locked-in 患者，使用 EEG slow cortical potentials 驱动电子拼写设备。

**可用于论文的论据**：

- 能力缺失：患者无法通过肌肉输出表达基本需求。
- BCI 补偿路径：绕过外周神经肌肉通道，直接用可调控脑电作为二值/选择信号。
- 局限：SCP 类 BCI 通信速率低，训练负担大，并且对完全 locked-in 患者不一定稳定。
- 对隐私/DA 的启发：通信型 BCI 与用户意图直接相关，若模型跨用户部署失败或泄露用户神经特征，风险不只是性能下降，还可能造成错误通信或敏感意图暴露。

### 3.2 机器人臂：非侵入式 EEG 控制 reach-and-grasp

Meng et al. 在 Scientific Reports 2016 证明，13 名人类受试者可以通过非侵入式 EEG motor imagery 调制 sensorimotor rhythm，在几次训练后控制机器人臂完成多自由度 reach-and-grasp。其设计把复杂 3D 控制拆成两个连续低维控制阶段：先在二维平面定位，再向下抓取。

**推动的能力边界**：

- 从虚拟光标/屏幕控制推进到真实机器人臂抓取物体。
- 证明 scalp EEG 不只适用于离线分类，也能进入闭环机器人控制。

**仍然欠缺**：

- 控制仍需任务分解，不能像自然手臂一样连续多自由度控制。
- 实验主要基于健康受试者，不能直接等同于残疾患者居家可用。
- 依赖多 session 训练，说明模型和用户都需要适配。

### 3.3 手指级控制：非侵入式 BCI 向精细运动推进

Ding et al. 在 Nature Communications 2025 提出 EEG-based real-time robotic hand control at individual finger level。该工作使用 ME/MI 的单个手指动作意图驱动机器人手指动作，21 名有经验 BCI 用户在线 MI 解码达到 2 指 80.56%、3 指 60.61%。模型上使用 EEGNet，并通过 same-day fine-tuning 缓解 inter-session variability。

**推动的能力边界**：

- 从“左/右手 MI 控制方向”推进到“同一只手内不同手指”的自然映射。
- 对上肢/手功能障碍有明确临床动机：手指精细控制对日常生活最关键。

**关键局限**：

- 同一只手不同手指的 sensorimotor cortex 表征小且高度重叠。
- EEG 经颅骨和头皮传播后空间分辨率和 SNR 显著下降。
- 3 指任务仍显著低于 2 指任务，说明非侵入式精细控制仍有性能瓶颈。
- 研究对象是 able-bodied experienced BCI users；患者、长期、家庭场景仍需验证。

**与 DA 的直接关系**：

论文显式使用 fine-tuning 处理 inter-session variability，并报告 fine-tuning 比 base model 更稳定。这可以直接作为“BCI 真实部署必须考虑目标域适配”的 Nature 子刊证据。

### 3.4 步态康复：BMI 不只是控制外设，也可能诱导神经恢复

Donati et al. 在 Scientific Reports 2016 报告 8 名慢性脊髓损伤截瘫患者接受 12 个月 BMI-based gait neurorehabilitation。协议包括 16 通道 EEG 控制虚拟 avatar、视觉-触觉反馈、Lokomat、ZeroG、脑控体重支持步态系统和 12 自由度 exoskeleton。结果显示所有患者的躯体感觉出现改善，并恢复部分 SCI 平面以下自主肌肉控制；50% 患者从完全截瘫分类提升到不完全截瘫。

**推动的能力边界**：

- BCI 从“替代输出通道”扩展为“闭环康复训练工具”。
- 通过运动意图、视觉/触觉反馈和辅助行走构建 sensorimotor loop，可能诱导 cortical and spinal cord plasticity。

**仍然欠缺**：

- 样本数很小，且没有清晰拆分 EEG-BMI、VR、触觉反馈、机器人步态训练各自贡献。
- 训练周期极长，设备复杂，临床推广成本高。
- “部分神经恢复”不等同于独立自然行走，仍需辅助设备和康复环境。

## 4. 神经原理支撑

### 4.1 MI-BCI 的神经基础：运动想象与 sensorimotor rhythm

MI-BCI 的核心不是“模型硬分类脑电”，而是用户通过运动想象调制感觉运动皮层节律，尤其是 mu/beta rhythm 的 ERD/ERS。Meng et al. 2016 在机器人臂控制中使用 10–14 Hz mu rhythm，并在 C3/C4 附近观察到与左右手运动想象相关的 contralateral ERD 和 ipsilateral ERS。Scientific Data 2022/2025 的 MI 数据集也用 C3/C4、ERD/ERS、CSP/FBCSP/EEGNet 等作为验证基础。

**可写入论文的表述**：

> EEG-based MI-BCI is grounded in the voluntary modulation of sensorimotor rhythms over motor cortices. However, these neurophysiological signatures are non-stationary across sessions and heterogeneous across subjects, making robust cross-user deployment a domain adaptation problem rather than a conventional supervised classification problem.

### 4.2 手指控制的神经基础：手部皮层表征与精细运动

Sobinov and Bensmaia 在 Nature Reviews Neuroscience 2021 总结手部精细运动的神经机制：人手具有复杂解剖结构和扩展的神经控制环路，触觉和本体感觉为抓握与操作提供高时效反馈。Ding et al. 2025 进一步指出手在 sensorimotor cortex 中有较大表征，但单个手指表征高度重叠，这正是 EEG 手指级解码困难的神经原因。

**对模型设计的启发**：

- 任务相关特征：手指/手/脚 MI 诱发的 sensorimotor rhythm 与空间拓扑。
- 身份/隐私相关特征：个体皮层结构、头皮传导、频带峰值、BCI 学习能力、年龄/性别/认知特质等可能被模型吸收。
- 如果只做粗暴滤波或全局对齐，可能同时削弱任务判别特征和身份特征，导致隐私保护与任务性能冲突。

### 4.3 视觉拼写器的神经基础：ERP、V5 与 feature-based attention

Kosnoff et al. 2024 的 mVEP BCI speller 说明视觉诱发 BCI 不只是信号分类问题，其性能依赖视觉运动加工、V5/MT 区域、N200/P300 等 ERP 成分和 feature-based attention。tFUS 靶向 V5 后降低拼写器错误，并在 EEG source imaging 中观察到 theta/alpha 活动变化。

**对 BCI 场景分类的启发**：

- MI 类任务主要依赖 sensorimotor rhythm。
- P300/mVEP/SSVEP 类任务主要依赖外源刺激、注意和视觉皮层响应。
- 不同范式共享 EEG 采集方式，但神经源、频带、时间窗和任务含义不同；跨范式迁移不能简单假设特征可复用。

### 4.4 闭环反馈与康复可塑性

Sitaram et al. 在 Nature Reviews Neuroscience 2017 将 neurofeedback 视为闭环脑训练：实时呈现神经活动，使用户学习自我调节相关神经回路，并可能带来持续数小时至数月的神经变化。Chaudhary et al. 2016 则明确区分 assistive BCI 与 rehabilitative BCI：前者帮助患者通信或控制外设，后者试图促进神经恢复。

**对论文背景的意义**：

BCI 设备在真实使用中不是静态分类器，而是“用户—模型—反馈—神经可塑性”的闭环系统。用户脑状态会随反馈、训练、疲劳、药物、康复进程变化；因此 domain shift 是系统内生问题，不是数据集噪声。

## 5. Nature 系列是否考虑 DA / domain adaptation？

结论：**考虑了 DA 的问题本质，但常用术语是 cross-session variability、cross-subject challenge、calibration、fine-tuning、transfer learning、adaptation，而不一定直接写作 domain adaptation 或 source-free domain adaptation。**

### 5.1 直接证据 1：Scientific Data 2022 跨 session 数据集

Ma et al. 在 Scientific Data 2022 发表 *A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface*。该数据集包含 25 名受试者、5 天独立 session、左/右手 MI。论文明确比较：

- within-session classification (WS)：最高平均 68.8%
- cross-session classification (CS)：下降到 53.7%
- cross-session adaptation (CSA)：提升到 78.9%

这几乎就是 Nature 子刊对 DA 必要性的直接证据：同一受试者跨天测试已经出现严重分布偏移，不做 adaptation 接近 chance-level；做 adaptation 后显著恢复性能。

### 5.2 直接证据 2：Nature Communications 2025 的 fine-tuning

Ding et al. 2025 在实时手指级机器人控制中，为缓解 inter-session variability，使用 same-day data 对 base model fine-tune。论文报告 fine-tuned models 在多个 session 中优于 base models，说明在线 BCI 实际部署需要持续目标域适配。

### 5.3 直接证据 3：Scientific Data 2023/2025 的跨用户与用户画像

Dreyer et al. 2023 提供 87 名参与者的大规模 MI-BCI EEG 数据库，并包含人口统计、人格、认知特质和在线 BCI 表现。论文明确提出这些数据可用于 cross-user machine learning algorithms，以及把 user profile information 纳入 EEG 分类算法设计。

Yang et al. 2025 提供 62 名受试者、三次 recording session 的高质量 MI 数据集，目标之一是学习 cross-session and cross-subject patterns，缓解 EEG instability。

### 5.4 对 SFDA 的判断

Nature 系列目前更偏临床/神经工程表达，常关注“为什么用户/天次不稳定”和“怎样通过校准/fine-tuning/数据集促进泛化”。严格的 **SFDA/source-free DA**、**black-box DA**、**test-time adaptation** 更多出现在 ICML/ICLR/CVPR/NeurIPS/AAAI/IEEE T-BME/JBHI/JNE 等机器学习或工程期刊会议中。

因此论文中可以这样写：

> Nature Portfolio studies have repeatedly identified inter-subject and inter-session non-stationarity as a bottleneck for practical EEG-BCIs. However, most clinical BCI studies still rely on calibration, same-day fine-tuning, or data-sharing benchmarks, whereas source-free privacy-preserving adaptation remains underexplored in real-world BCI deployment.

## 6. 为什么这些证据能支撑“EEG + SFDA + MU + 隐私保护”

### 6.1 真实 BCI 需要 DA

Nature 系列真实设备论文共同说明：BCI 不是一次训练永久可用。EEG 存在跨天、跨人、跨任务、跨脑状态变化：

- 电极位置、阻抗、疲劳、注意、学习、康复进程都会改变分布。
- MI 特征在不同用户之间差异很大，存在 BCI deficiency / illiteracy。
- 多日数据集显示跨 session 不适配会严重掉点。

所以如果要把 BCI 从实验室推向家庭/临床，DA 是必要问题。

### 6.2 真实 BCI 需要 source-free

临床和居家 BCI 不可能长期集中保存所有 source EEG：

- EEG 是高敏感生物信号，可能包含身份、年龄、性别、疾病、认知状态、情绪等信息。
- 医疗数据存在合规限制，不能随意跨机构共享。
- 真实系统可能只发布源模型，而不发布源数据。

因此 source-free adaptation 更符合部署约束。

### 6.3 真实 BCI 需要 MU

即使不保存原始 EEG，源模型也可能吸收用户特征。若用户撤回数据、某类隐私特征需要移除、或某个 source domain 不再被授权，单纯 SFDA 不能保证模型忘记这些影响。MU 的角色不是简单删除数据文件，而是让模型输出、表征或参数状态尽量接近“从未用这些数据训练过”的模型。

对于 BCI，MU 可以有三种研究设定：

1. **用户级 unlearning**：某个用户撤回 EEG 数据，模型需要忘记该用户的个体神经指纹。
2. **属性级 unlearning**：保留 MI/SSVEP/P300 任务能力，但削弱年龄、性别、身份、疾病状态等隐私属性可推断性。
3. **域级 unlearning**：某个医院、设备、session、source domain 不再授权，模型需要去除该域影响，同时保持目标域适配性能。

### 6.4 与 Nature 证据的桥接逻辑

可以在论文 introduction 中形成如下链条：

1. Nature/Scientific Reports 已证明非侵入式 BCI 可在通信、机器人控制、手指控制、步态康复中弥补残疾人士缺失能力。
2. 但 Nature/Scientific Data 也显示 EEG-BCI 存在显著跨 session / cross-subject shift，实际部署需要 adaptation。
3. 传统 adaptation 依赖源数据或目标校准数据，与医疗隐私和用户撤回权冲突。
4. Source-free DA 降低源数据访问需求，但源模型仍可能记忆 source users/domains。
5. Machine unlearning 可作为模型级隐私修正机制，与 SFDA 结合，处理“既要适配目标用户，又要忘记未授权源用户/隐私属性”的现实问题。

## 7. 可直接写入论文的背景段落

### 7.1 英文 introduction 草稿

Non-invasive brain-computer interfaces (BCIs), especially EEG-based systems, have shown increasing potential for restoring or augmenting impaired human abilities. Nature Portfolio studies have demonstrated EEG-driven spelling for paralyzed users, non-invasive robotic arm control for reach-and-grasp tasks, real-time robotic hand control at the individual-finger level, and BMI-based gait neurorehabilitation for chronic spinal cord injury. These systems indicate that BCI can bypass damaged neuromuscular pathways and provide alternative communication, manipulation, or rehabilitation channels.

However, the practical deployment of EEG-BCIs remains limited by the non-stationary and user-specific nature of neural signals. EEG patterns vary across subjects, sessions, mental states, electrode placements, and rehabilitation stages. Scientific Data studies on multi-session motor imagery EEG show that cross-session classification can degrade dramatically without adaptation, while cross-session adaptation substantially restores performance. Recent Nature Communications work on real-time robotic finger control also relies on same-day fine-tuning to mitigate inter-session variability. These findings suggest that domain adaptation is not an optional algorithmic improvement, but a prerequisite for reliable real-world BCI use.

At the same time, adaptation in medical and assistive BCI raises privacy concerns. EEG signals may encode not only task-related intentions but also user identity, demographic attributes, cognitive traits, and health-related information. Source-free domain adaptation reduces the need to access source EEG data, but the released source model may still retain information about source users or domains. Therefore, combining source-free adaptation with machine unlearning provides a promising direction for privacy-preserving BCI: the model should adapt to a new user or session while forgetting revoked users, sensitive attributes, or unauthorized source domains.

### 7.2 中文论点版本

非侵入式 BCI 的核心意义在于以较低风险和较低成本绕过受损的神经肌肉输出通道，为严重运动障碍用户提供通信、控制和康复能力。Nature 及其子刊已有多项研究证明 EEG/fNIRS 等非侵入式 BCI 可以支持瘫痪患者拼写通信、机器人臂抓取、机器人手指级控制和截瘫步态康复训练。然而，这些研究也共同暴露出实际部署瓶颈：EEG 信号低信噪比、低空间分辨率、跨天和跨用户差异显著，模型需要持续校准或 fine-tuning 才能保持稳定。Scientific Data 的跨 session MI 数据集显示，不做 adaptation 时跨天分类性能可显著下降，而 cross-session adaptation 可明显恢复性能。因此，DA 是 BCI 真实使用中的基础需求。

另一方面，BCI 适配不能简单依赖集中保存和重复访问所有源 EEG 数据。EEG 可能泄露身份、年龄、性别、认知特质和疾病状态，属于高敏感神经数据。SFDA 可以减少源数据访问，但不能保证源模型本身没有记忆源用户或源域隐私信息。因此，将 machine unlearning 引入 SFDA，可以面向用户撤回、域撤回或隐私属性移除场景，使模型在适配目标用户的同时，降低源用户/敏感属性对模型表征和输出的残留影响。

## 8. 写作时应避免的过度表述

- 不要说“Nature 已经系统研究 SFDA + MU + BCI 隐私保护”。更准确是：Nature 系列已明确揭示 BCI 的跨人/跨天适配需求和真实部署价值，但 SFDA + MU 的形式化隐私框架仍是开放问题。
- 不要说“非侵入式 BCI 已经恢复自然运动能力”。更准确是：已经实现特定任务下的替代控制或康复改善，但距离自然、高自由度、长期稳定控制仍有差距。
- 不要把所有 BCI 都说成 EEG。Nature 中很多高性能通信/运动 BCI 是 invasive intracortical/ECoG；本文应明确聚焦 non-invasive EEG/fNIRS，并把 invasive 作为性能上界或对照。
- 不要把 DA 等同于普通 fine-tuning。fine-tuning 是一种适配手段；DA 更广泛地处理 source/target distribution shift，SFDA 进一步假设 source data unavailable。

## 9. 参考文献（GB/T 7714 风格草稿）

[1] Ding Y, Udompanyawit C, Zhang Y, et al. EEG-based brain-computer interface enables real-time robotic hand control at individual finger level[J]. Nature Communications, 2025, 16: 5401. DOI: 10.1038/s41467-025-61064-x.

[2] Meng J, Zhang S, Bekyo A, et al. Noninvasive electroencephalogram based control of a robotic arm for reach and grasp tasks[J]. Scientific Reports, 2016, 6: 38565. DOI: 10.1038/srep38565.

[3] Donati A R C, Shokur S, Morya E, et al. Long-term training with a brain-machine interface-based gait protocol induces partial neurological recovery in paraplegic patients[J]. Scientific Reports, 2016, 6: 30383. DOI: 10.1038/srep30383.

[4] Birbaumer N, Ghanayim N, Hinterberger T, et al. A spelling device for the paralysed[J]. Nature, 1999, 398: 297-298. DOI: 10.1038/18581.

[5] Chaudhary U, Birbaumer N, Ramos-Murguialday A. Brain-computer interfaces for communication and rehabilitation[J]. Nature Reviews Neurology, 2016, 12: 513-525. DOI: 10.1038/nrneurol.2016.113.

[6] Ma J, Yang B, Qiu W, et al. A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface[J]. Scientific Data, 2022, 9: 531. DOI: 10.1038/s41597-022-01647-1.

[7] Dreyer P, Roc A, Pillette L, et al. A large EEG database with users’ profile information for motor imagery brain-computer interface research[J]. Scientific Data, 2023, 10: 580. DOI: 10.1038/s41597-023-02445-z.

[8] Yang B, Rong F, Xie Y, et al. A multi-day and high-quality EEG dataset for motor imagery brain-computer interface[J]. Scientific Data, 2025, 12: 488. DOI: 10.1038/s41597-025-04826-y.

[9] Kosnoff J, Yu K, Liu C, et al. Transcranial focused ultrasound to V5 enhances human visual motion brain-computer interface by modulating feature-based attention[J]. Nature Communications, 2024, 15: 4382. DOI: 10.1038/s41467-024-48576-8.

[10] Sitaram R, Ros T, Stoeckel L, et al. Closed-loop brain training: the science of neurofeedback[J]. Nature Reviews Neuroscience, 2017, 18: 86-100. DOI: 10.1038/nrn.2016.164.

[11] Sobinov A R, Bensmaia S J. The neural mechanisms of manual dexterity[J]. Nature Reviews Neuroscience, 2021, 22: 741-757. DOI: 10.1038/s41583-021-00528-7.

[12] Nicolelis M A L, Lebedev M A. Principles of neural ensemble physiology underlying the operation of brain-machine interfaces[J]. Nature Reviews Neuroscience, 2009, 10: 530-540. DOI: 10.1038/nrn2653.

## 10. 可作为对照的侵入式 Nature 系列工作

这些工作不是本文“非侵入式”主线，但可以用来说明能力上界和非侵入式 BCI 的性能差距：

[13] Hochberg L R, Bacher D, Jarosiewicz B, et al. Reach and grasp by people with tetraplegia using a neurally controlled robotic arm[J]. Nature, 2012, 485: 372-375. DOI: 10.1038/nature11076.

[14] Bouton C E, Shaikhouni A, Annetta N V, et al. Restoring cortical control of functional movement in a human with quadriplegia[J]. Nature, 2016, 533: 247-250. DOI: 10.1038/nature17435.

[15] Willett F R, Avansino D T, Hochberg L R, et al. High-performance brain-to-text communication via handwriting[J]. Nature, 2021, 593: 249-254. DOI: 10.1038/s41586-021-03506-2.

[16] Metzger S L, Littlejohn K T, Silva A B, et al. A high-performance neuroprosthesis for speech decoding and avatar control[J]. Nature, 2023, 620: 1037-1046. DOI: 10.1038/s41586-023-06443-4.

[17] Willsey M S, Nason S R, Bockbrader M A, et al. A high-performance brain-computer interface for finger decoding and quadcopter game control in an individual with paralysis[J]. Nature Medicine, 2025, 31: 96-104. DOI: 10.1038/s41591-024-03341-8.

## 11. 下一步论文选题建议

建议题目方向：

**Privacy-Preserving Source-Free Domain Adaptation with Machine Unlearning for Non-invasive EEG-based Brain-Computer Interfaces**

或者更强调真实部署：

**Towards Privacy-Preserving Real-World EEG-BCIs: Source-Free Adaptation and Machine Unlearning under User and Session Shifts**

建议实验设计可围绕：

1. **数据集**：Scientific Data 2022/2025 的多 session MI EEG，外加公开 MI/SSVEP/P300 数据集。
2. **任务**：跨 subject / cross-session MI 分类。
3. **SFDA 目标**：source data unavailable，只给 source model 和 target unlabeled EEG。
4. **MU 设置**：
   - forget user：删除某个 source subject 的影响；
   - forget attribute：降低 gender/age/user ID 可推断性；
   - forget domain：删除某个 session/device/source domain。
5. **指标**：
   - task accuracy / balanced accuracy；
   - target adaptation gain；
   - forgetting efficacy：forget user membership/attribute attack accuracy 降低；
   - retain performance：非 forget source 或 target task 性能保持；
   - privacy leakage：identity classifier、membership inference、attribute inference。

