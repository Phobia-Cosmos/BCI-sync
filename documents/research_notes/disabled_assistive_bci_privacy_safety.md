# Disabled-aided BCI 中为什么必须考虑隐私与安全

> 目的：为论文背景/导师汇报提供证据链。核心问题是：面向瘫痪、ALS、locked-in syndrome、stroke、SCI 等 disabled 用户的 BCI 设备，如果不考虑隐私保护，是否会造成真实安全隐患或威胁？结论是：会。威胁不仅是“数据泄露”，还会扩展到通信权、身体安全、设备控制权、人格自主性、医疗歧视和长期依赖风险。

## 1. 一句话结论

Disabled-aided BCI 的隐私保护比普通消费 EEG 更重要，因为这类设备不是娱乐设备，而是在替代用户已经丧失或受损的能力：说话、打字、控制电脑、控制机械臂、控制轮椅、参与康复。此时 EEG/神经数据、模型表征和解码输出可能同时包含：

- 用户身份、性别、年龄、BCI 使用经验等个人属性；
- 疾病状态、运动障碍、认知状态、情绪和心理健康信息；
- 用户意图、通信内容、家庭/工作对话、医疗选择；
- 对敏感刺激的识别反应，例如银行、PIN、住址、熟人、人脸；
- 设备控制指令，例如文字输入、鼠标控制、机器人/外设控制。

因此，不考虑隐私会带来两类后果：

1. **隐私/尊严/自主性威胁**：神经数据被推断、出售、二次训练、画像或用于歧视。
2. **安全/身体/功能威胁**：攻击者通过数据流、模型或联网设备影响 BCI 输出，可能导致错误通信、错误控制、设备不可用或用户被迫失去辅助能力。

## 2. 为什么 disabled 用户场景更敏感

### 2.1 BCI 是“能力替代通道”，不是普通传感器

在 speech BCI、P300 speller、MI-BCI、robotic control、neurorehabilitation 中，BCI 直接承担 disabled 用户的输出通道。Nature 2023 的 speech neuroprosthesis 表明，ALS 用户可以通过植入式 BCI 将 attempted speech 解码为大词表文本，达到 62 words/min，并接近自然交流速度。这样的系统一旦进入真实生活，解码输出就不只是实验标签，而是用户的真实语言、工作交流、家庭表达和个人决策。

因此：

- 普通键盘泄露的是输入内容；
- disabled-aided speech/cursor BCI 泄露的是用户仅能依赖的通信通道；
- 如果该通道被第三方访问、记录、训练或操控，影响的是用户的表达权和自主性。

### 2.2 disabled 用户更难发现、拒绝或纠正攻击

严重瘫痪、locked-in 或 ALS 用户可能无法快速摘下设备、关闭应用、切换交互方式、检查后台权限或纠正错误输出。很多系统依赖研究人员、护理者、医院或厂商维护。隐私/安全问题一旦发生，用户比普通人更难主动防御。

### 2.3 BCI 数据天然“过度采集”

神经数据不是为单一任务而生。即使系统目标只是 MI 分类或文字输入，同一段 EEG/神经数据仍可能包含任务无关信息。美国参议员 2025 年致 FTC 的神经数据公开信明确指出，neural data 可揭示 mental health conditions、emotional states 和 cognitive patterns，即使匿名化也可能高度敏感。Neurorights Foundation 2024 报告也指出，神经数据可揭示 mental health、physical health 和 cognitive processing。

### 2.4 BCI 商业化/临床化需要多方协作，数据流复杂

隐私保护 BCI 的系统综述指出，BCI 常用于 medical diagnosis、rehabilitation 等场景，开发商业系统需要医院、大学、公司等多方协作；EEG 输入数据包含 rich privacy information，数据和模型在多方之间传输会带来 privacy threats。

这对 disabled-aided BCI 尤其重要：

- 医院采集源用户数据；
- 公司训练源模型；
- 设备在家庭中持续产生目标用户数据；
- 模型可能上传云端更新；
- 数据可能用于二次训练、算法改进或商业分析；
- 用户撤权后，模型中仍可能残留源用户影响。

## 3. 不考虑隐私/安全会造成哪些真实威胁

### 3.1 敏感属性泄露：身份、性别、BCI 经验、疾病状态

EEG 数据可被用于推断用户属性。Meng et al. 2024 明确演示：user identity、gender 和 BCI-experience 可从 EEG 中被推断，构成 serious privacy threat。对 disabled 用户，这些属性还可能与疾病、康复水平、照护需求和社会身份绑定，泄露后可能带来污名化、保险/就业/教育歧视或被恶意画像。

**论文支撑**：

- Xia et al. 2023/2024, *Privacy-Preserving Brain-Computer Interfaces: A Systematic Review*。
- Meng et al. 2024, *Protecting Multiple Types of Privacy Simultaneously in EEG-based Brain-Computer Interfaces*。

### 3.2 通过刺激探测隐私：银行、PIN、住址、熟人、人脸

USENIX Security 2012 的 Martinovic et al. 是最直接的攻击证据。研究者使用低成本 EEG BCI 设备，展示候选刺激并分析脑响应，证明可降低用户私密信息的不确定性，涉及 bank cards、PIN numbers、area of living、known persons 等。

Frank et al. 进一步提出 subliminal probing：把短暂人脸刺激嵌入无害视频背景中，通过 EEG 判断用户是否认识某个人脸。这个实验说明攻击不一定需要用户明确知道自己正在被隐私探测。

**对 disabled-aided BCI 的危险性**：

- 通信型 BCI/speller 本身需要持续呈现视觉刺激；
- 恶意应用可以把敏感候选项混入刺激界面；
- disabled 用户可能无法察觉、关闭或举报；
- 攻击者不需要直接读取完整记忆，只需要缩小候选空间即可造成实际威胁。

### 3.3 数据流/系统漏洞：脑电泄露或辅助设备被远程控制

Tarkhani et al. 2022 分析了 wearable BCI 的全系统威胁，指出 BCI 常用于 healthcare、smart communication 和 control 等安全/隐私关键场景；硬件、软件、网络和 ML 栈上的攻击可能泄露 brainwave data，最坏情况下让远程攻击者获得 BCI-assisted devices 的控制权。其对 Muse、NeuroSky、OpenBCI 等真实设备的 proof-of-concept 攻击发现了 300 多个漏洞。

**对 disabled-aided BCI 的危险性**：

- 如果 BCI 控制的是鼠标/键盘，攻击影响通信和数字账户；
- 如果控制的是轮椅、机器人臂、家居设备或康复外设，攻击可能变成身体安全问题；
- 如果系统在家庭/医院联网使用，隐私泄露和设备控制风险会叠加。

### 3.4 医疗设备网络安全本身已被监管机构视为患者安全问题

FDA 明确指出，医疗设备越来越多连接互联网、医院网络和其他设备；这些连接带来 cybersecurity risks，安全漏洞可能影响设备 safety and effectiveness。FDA 还列举过安全通信：未修补漏洞可能允许未授权用户访问、控制和向设备发命令，从而导致 patient harm；例如胰岛素泵通信协议被未授权访问时，可能造成胰岛素过量或不足。

BCI 如果进入医疗/辅助设备形态，也应被放在 connected medical device 的安全框架下考虑。即使不是所有 BCI 都会直接造成生理剂量风险，控制外设、通信和康复反馈的错误也会对 disabled 用户产生真实伤害。

### 3.5 设备/厂商治理失败会让 disabled 用户承受更高代价

IEEE Spectrum 2022 报道 Second Sight Argus II 视网膜假体用户被厂商停止支持。超过 350 名盲人用户安装了该设备，部分用户在系统故障后失去人工视觉；失效植入物还可能造成医疗并发症或影响 MRI 等检查。

这不是 BCI 隐私攻击案例，但它说明：disabled 用户一旦依赖神经/辅助设备，技术、数据、维护和厂商治理失败的后果会远大于普通消费设备。对于 BCI 来说，如果隐私政策允许数据转让、二次使用、出售或撤权困难，用户将难以控制自己赖以通信/控制的神经数据生态。

### 3.6 数据商业化和第三方共享已经不是假设

Neurorights Foundation 2024 对 30 家消费者神经技术公司隐私政策的报告发现：

- 29/30 公司似乎可访问用户 neural data，且没有 meaningful limitations；
- 约 60% 公司没有说明如何处理 neural data；
- 29/30 公司可或可能把数据转给第三方；
- 超过 85% 公司没有明确排除出售数据；
- 只有部分公司提供撤回同意或删除数据的权利；
- 多数公司没有充分说明神经数据安全措施。

2025 年美国参议员致 FTC 的公开信也要求调查 BCI/neurotech 公司处理神经数据的方式，强调医疗或认知支持中产生的脑信号不应在用户不知情或未明确同意时被用于 AI 训练或出售给第三方。

## 4. 真实证据表

| 证据 | 类型 | 说明 | 能支撑的论点 |
|---|---|---|---|
| Willett et al., Nature 2023 speech neuroprosthesis | disabled-aided BCI 应用 | ALS 用户 attempted speech 可被解码为文本，62 wpm，大词表 | BCI 输出可能就是用户真实通信内容，隐私等同于表达权保护 |
| Xia et al., IEEE TCSS 2023 / arXiv 2024 | BCI 隐私综述 | EEG 含 rich privacy information；数据/模型传输有 privacy threats | BCI 隐私是系统性问题，不只是单个数据集问题 |
| Meng et al., SMC 2024 | EEG 属性推断实验 | identity、gender、BCI-experience 可从 EEG 推断 | EEG 含任务无关敏感属性，需要 disentangle/protect |
| Martinovic et al., USENIX Security 2012 | BCI side-channel attack | 使用 EEG 推断 bank cards、PIN、住址、known persons 等 | BCI 可被恶意应用变成隐私探测通道 |
| Frank et al., arXiv 2013/2017 | subliminal probing | 13.3 ms 人脸刺激可用于推断用户是否认识某人 | 攻击可更隐蔽，不一定需要用户主动配合 |
| Tarkhani et al., arXiv 2022 | wearable BCI 系统安全 | 真实 BCI 设备栈发现 300+ 漏洞；可泄露脑波或控制辅助设备 | BCI 隐私泄露可能升级为设备控制与安全问题 |
| FDA cybersecurity page | 医疗设备监管报告 | 联网医疗设备漏洞可影响 safety/effectiveness；未授权控制可能导致 patient harm | BCI 作为医疗/辅助设备必须考虑 cybersecurity by design |
| IEEE Spectrum 2022 Second Sight | 真实报道 | 盲人用户依赖神经视觉假体，厂商停止支持后失去人工视觉/面临医疗问题 | disabled 用户对神经设备依赖强，治理失败成本高 |
| Ienca & Andorno 2017 | neuro-rights 论文 | 提出 mental privacy、mental integrity、cognitive liberty 等权利 | 神经数据保护是人权/自主性问题 |
| Yuste et al., Nature 2017 | neuroethics 论文 | Neurotechnology 和 AI 必须保护 privacy、identity、agency、equality | 隐私、身份和自主性是 neurotech 四个核心伦理问题 |
| Neurorights Foundation 2024 | 产业报告 | 大量消费神经公司政策模糊、第三方共享、删除/撤权不足 | 真实商业生态中存在数据治理风险 |
| U.S. Senate letter to FTC 2025 | 政策报道/官方信 | 要求 FTC 调查 BCI/neurotech 神经数据处理 | 政策层已把 neural data commercialization 当作现实风险 |

## 5. 和 DA + SFDA + MU 的连接

Disabled-aided BCI 的真实部署会同时需要“个性化适配”和“隐私保护”。逻辑链如下：

1. disabled 用户需要长期稳定 BCI，因此需要 DA/SFDA 做跨用户、跨 session、跨状态适配。
2. DA 往往依赖历史用户/source data 或 source model，而 disabled/医疗 EEG 是高敏感数据。
3. 传统集中式 DA 需要访问源数据，这与医疗隐私、用户撤权、跨机构数据共享限制冲突。
4. SFDA 可以减少对 source EEG 的直接访问，但 source model 仍可能残留源用户/源域信息。
5. MU 则进一步处理用户撤权、源域撤回或敏感属性移除：不是只删除原始 EEG，而是减少模型中对应用户/属性/domain 的残留影响。

因此，论文中可以这样定义问题：

> Disabled-aided BCIs require stable personalization because neural signals are highly user-specific and non-stationary. However, personalization requires using or reusing sensitive neural data from vulnerable users. Prior work has shown that EEG can reveal identity, demographic attributes, BCI experience, health and cognitive states, and even private knowledge through stimulus-driven side channels. Therefore, privacy-preserving source-free adaptation and machine unlearning are necessary for assistive BCI deployment under consent withdrawal and source-data unavailability.

## 6. 适合汇报的中文段落

在 disabled-aided BCI 中考虑隐私，不是为了泛泛满足合规，而是因为这类设备直接替代用户的基本能力。对于 ALS、locked-in、stroke 或 SCI 用户，BCI 可能就是他们说话、打字、控制电脑、控制外设和参与康复训练的唯一通道。此时泄露的不是普通传感器数据，而可能是用户的真实通信内容、运动意图、疾病状态、身份属性、情绪状态和对敏感信息的识别反应。

已有真实论文证明这种风险不是假设。USENIX Security 2012 的 BCI side-channel attack 通过消费级 EEG 设备推断 bank cards、PIN、住址和熟人信息；后续 subliminal probing 工作进一步说明攻击者可以把短暂刺激隐藏在普通视频中，推断用户是否认识某个人脸。BCI 隐私综述和 EEG 隐私保护实验也表明，identity、gender、BCI-experience 等属性可以从 EEG 中推断。另一方面，wearable BCI 系统安全研究在 Muse、NeuroSky、OpenBCI 等设备栈中发现 300 多个漏洞，并指出最坏情况下远程攻击者可能获得 BCI-assisted devices 的控制权。

因此，对 disabled 用户来说，不考虑隐私和安全会带来三类威胁：第一，神经数据泄露导致医疗、身份和心理信息暴露；第二，刺激探测和模型推断可能窃取用户敏感知识或个人关系；第三，设备/模型被攻击可能导致错误通信、错误控制和康复反馈失效。由于 disabled 用户对设备依赖更强，且更难主动发现和终止攻击，这些风险比普通消费 BCI 更严重。

## 7. 写论文时应避免的过度表述

不要写：

- “攻击者可以直接读取 disabled 用户脑中的所有想法。”
- “EEG 可以直接恢复任意长密码。”
- “已有真实案例证明 BCI 隐私泄露导致 disabled 用户死亡。”

更严谨的写法：

- “已有研究证明 EEG/BCI 可泄露身份、属性、经验和对敏感候选刺激的识别反应。”
- “在 disabled-aided BCI 中，这些泄露更严重，因为 BCI 是用户通信与控制的核心辅助通道。”
- “BCI 隐私和安全问题可能从数据泄露升级为错误通信、外设误控制、康复反馈错误和设备不可用。”

## 8. 参考文献与报道

[1] Willett F R, Kunz E M, Fan C, et al. A high-performance speech neuroprosthesis[J]. Nature, 2023, 620: 1031-1036. DOI: 10.1038/s41586-023-06377-x. URL: https://www.nature.com/articles/s41586-023-06377-x

[2] Xia K, Duch W, Sun Y, et al. Privacy-preserving brain-computer interfaces: A systematic review[J]. IEEE Transactions on Computational Social Systems, 2023, 10(5): 2312-2324. DOI: 10.1109/TCSS.2022.3184818. URL: https://arxiv.org/abs/2412.11394

[3] Meng L, Jiang X, Jia T, Wu D. Protecting multiple types of privacy simultaneously in EEG-based brain-computer interfaces[C]//IEEE International Conference on Systems, Man, and Cybernetics. 2024. URL: https://arxiv.org/abs/2411.19498

[4] Martinovic I, Davies D, Frank M, Perito D, Ros T, Song D. On the feasibility of side-channel attacks with brain-computer interfaces[C]//21st USENIX Security Symposium. 2012: 143-158. URL: https://www.usenix.org/conference/usenixsecurity12/technical-sessions/presentation/martinovic

[5] Frank M, Hwu T, Jain S, Knight R, Martinovic I, Mittal P, Perito D, Song D. Subliminal probing for private information via EEG-based BCI devices[J]. arXiv preprint arXiv:1312.6052, 2013/2017. URL: https://arxiv.org/abs/1312.6052

[6] Tarkhani Z, Qendro L, Brown M O, Hill O, Mascolo C, Madhavapeddy A. Enhancing the security & privacy of wearable brain-computer interfaces[J]. arXiv preprint arXiv:2201.07711, 2022. URL: https://arxiv.org/abs/2201.07711

[7] FDA. Cybersecurity. U.S. Food and Drug Administration. URL: https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity

[8] Strickland E, Harris M. Their bionic eyes are now obsolete and unsupported[N]. IEEE Spectrum, 2022-02-15. URL: https://spectrum.ieee.org/bionic-eye-obsolete

[9] Ienca M, Andorno R. Towards new human rights in the age of neuroscience and neurotechnology[J]. Life Sciences, Society and Policy, 2017, 13: 5. DOI: 10.1186/s40504-017-0050-1. URL: https://link.springer.com/article/10.1186/s40504-017-0050-1

[10] Yuste R, Goering S, Arcas B A, et al. Four ethical priorities for neurotechnologies and AI[J]. Nature, 2017, 551: 159-163. DOI: 10.1038/551159a. URL: https://www.nature.com/articles/551159a

[11] Genser J, Damianos S, Yuste R. Safeguarding brain data: Assessing the privacy practices of consumer neurotechnology companies[R]. Neurorights Foundation, 2024. URL: https://perseus-strategies.com/wp-content/uploads/2024/04/FINAL_Consumer_Neurotechnology_Report_Neurorights_Foundation_April-1.pdf

[12] Schumer C E, Cantwell M, Markey E J. Neural Data Letter to the Federal Trade Commission[R]. U.S. Senate, 2025-04-28. URL: https://www.democrats.senate.gov/imo/media/doc/Neural%20Data%20Letter%20-%2004.28.2025-%20updated.pdf
