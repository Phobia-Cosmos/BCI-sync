# 黑盒数据投毒攻击论文整理

整理日期：2026-07-03

本文档整理数据投毒攻击中常见的黑盒方式。这里的“黑盒”在不同论文里含义不同，需要先区分：

- **严格黑盒 / 数据管线黑盒**：攻击者不知道模型结构、参数、优化器，也不能直接改训练集，只能污染未来会被采集的数据源。代表：黑盒 NMT 平行语料投毒。
- **迁移式黑盒**：攻击者不知道 victim 架构或训练细节，用本地 surrogate model 或 ensemble 生成毒样本，依靠毒样本迁移到 victim。代表：MetaPoison、Bullseye Polytope、Witches' Brew、Sleeper Agent。
- **灰盒但现实可行**：攻击者不知道训练数据或不能控制标签，但知道常用预训练特征提取器。代表：Poison Frogs、Hidden Trigger。
- **下游任务黑盒 / 供应链黑盒**：攻击者控制预训练模型或预训练阶段，但不知道未来下游任务、数据和模型头。代表：TrojanLM、BadPre。
- **有限信息黑盒**：攻击者只知道目标类代表样本和少量公开同域数据，不知道完整训练集或 victim 架构。代表：Narcissus。

## 已下载论文

PDF 均已下载到 `papers/black_box_data_poisoning/`，并已用 `pdftotext` 验证可读。

| 类别 | 论文 | 年份/会议 | 本地 PDF |
| --- | --- | --- | --- |
| 主线 | Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks | NIPS 2018 | [PDF](papers/black_box_data_poisoning/2018NeurIPS-Poison%20Frogs!%20Targeted%20Clean-Label%20Poisoning%20Attacks%20on%20Neural%20Networks.pdf) |
| 主线 | Hidden Trigger Backdoor Attacks | AAAI 2020 | [PDF](papers/black_box_data_poisoning/2020AAAI-Hidden%20Trigger%20Backdoor%20Attacks.pdf) |
| 主线 | MetaPoison: Practical General-purpose Clean-label Data Poisoning | NeurIPS 2020 | [PDF](papers/black_box_data_poisoning/2020NeurIPS-MetaPoison%20Practical%20General-Purpose%20Clean-Label%20Data%20Poisoning.pdf) |
| 主线 | Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability | IEEE S&P 2021 | [PDF](papers/black_box_data_poisoning/2021EuroSP-Bullseye%20Polytope%20A%20Scalable%20Clean-Label%20Poisoning%20Attack%20with%20Improved%20Transferability.pdf) |
| 主线 | Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching | ICLR 2021 | [PDF](papers/black_box_data_poisoning/2021ICLR-Witches'%20Brew%20Industrial%20Scale%20Data%20Poisoning%20via%20Gradient%20Matching.pdf) |
| 主线 | A Targeted Attack on Black-Box Neural Machine Translation with Parallel Data Poisoning | WWW 2021 | [PDF](papers/black_box_data_poisoning/2021WWW-A%20Targeted%20Attack%20on%20Black-Box%20Neural%20Machine%20Translation%20with%20Parallel%20Data%20Poisoning.pdf) |
| 主线 | Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch | NeurIPS 2022 | [PDF](papers/black_box_data_poisoning/2022NeurIPS-Sleeper%20Agent%20Scalable%20Hidden%20Trigger%20Backdoors%20for%20Neural%20Networks%20Trained%20from%20Scratch.pdf) |
| 补充 | Trojaning Language Models for Fun and Profit | EuroS&P 2021 | [PDF](papers/black_box_data_poisoning/2021EuroSP-Trojaning%20Language%20Models%20for%20Fun%20and%20Profit.pdf) |
| 补充 | BadPre: Task-agnostic Backdoor Attacks to Pre-trained NLP Foundation Models | arXiv/相关工作 | [PDF](papers/black_box_data_poisoning/2021arXiv-BadPre%20Task-Agnostic%20Backdoor%20Attacks%20to%20Pre-Trained%20NLP%20Foundation%20Models.pdf) |
| 补充 | Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information | arXiv/相关工作 | [PDF](papers/black_box_data_poisoning/2022arXiv-NARCISSUS%20A%20Practical%20Clean-Label%20Backdoor%20Attack%20with%20Limited%20Information.pdf) |

## 方法总览

| 黑盒方式 | 核心思路 | 代表论文 | 适用场景 | 关键限制 |
| --- | --- | --- | --- | --- |
| Feature collision | 让毒样本在输入空间像正确类别，在特征空间靠近目标样本 | Poison Frogs、Hidden Trigger | transfer learning / fine-tuning | 通常需要知道或近似 victim 特征提取器 |
| Convex polytope / attack zone | 多个毒样本在 surrogate 特征空间包围目标，使线性分类器把目标归入毒样本类 | Bullseye Polytope | transfer learning 黑盒迁移 | 主要针对迁移学习，对 from-scratch 较弱 |
| Meta-learning bilevel poisoning | 展开训练过程，用 surrogate ensemble 优化能影响未来训练轨迹的毒样本 | MetaPoison | 未知架构/训练设置，AutoML | 计算成本高，毒样本生成复杂 |
| Gradient matching | 让毒样本训练梯度与目标恶意梯度方向一致，victim 正常训练时被“带偏” | Witches' Brew、Sleeper Agent | from-scratch、大规模数据 | 依赖 surrogate 训练状态和迁移性 |
| 数据源污染 | 构造高质量恶意平行句对，放进会被爬虫/bitext miner 采集的网站 | Black-box NMT poisoning | NMT 训练语料来自 Web | 对高频 trigger 需要更多毒样本 |
| 预训练供应链 | 通过毒数据或重训练把后门写入预训练模型，未知下游任务继承后门 | TrojanLM、BadPre | NLP foundation model 复用 | 更像模型供应链攻击，不是纯数据管线投毒 |
| Limited-information trigger synthesis | 只用目标类样本和公开 OOD 样本训练 surrogate，合成“指向目标类内部”的 trigger | Narcissus | 攻击者不知道完整训练集 | 不是严格顶会主线，但对现实黑盒很重要 |

## 逐篇分析

### 1. Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks

**黑盒/灰盒设定**

这篇不是严格模型黑盒。论文明确假设攻击者没有训练数据知识，但知道模型及其参数；现实性来自两个点：毒样本是 clean-label，攻击者不需要控制标签；毒样本可以被放到 Web 上等待数据采集爬虫抓取。

**攻击流程**

1. 选择一个目标测试样本 `t`，希望它在测试时被误分类为 base class。
2. 从 base class 选择一个干净样本 `b`。
3. 优化毒样本 `p`，使其满足两件事：在输入空间接近 `b`，所以人类会给它正确 base 标签；在特征空间接近 `t`，形成 feature collision。
4. victim 用包含毒样本的数据训练或 fine-tune 后，为了正确分类毒样本，决策边界会旋转，目标样本 `t` 被一起划入 base class。
5. transfer learning 场景下一个毒样本即可成功；from-scratch 场景下需要多毒样本和 watermarking。

**如何利用黑盒性**

严格说，它利用的不是“未知模型黑盒”，而是“数据采集和标签流程不可控”的现实黑盒：攻击者不需要改标签，也不需要进入训练流水线，只要让正确标注的毒样本进入训练集。它奠定了后续 transfer-based clean-label poisoning 的基础。

### 2. Hidden Trigger Backdoor Attacks

**黑盒/灰盒设定**

该论文面向 backdoor poisoning。典型 backdoor 会在训练毒样本中显示 trigger，而 Hidden Trigger 把 trigger 隐藏到测试阶段，训练集中只出现看起来自然且标签正确的 target-class 毒样本。攻击者通常依赖预训练/替代特征空间，因此仍偏灰盒或迁移式黑盒。

**攻击流程**

1. 攻击者选择 source class、target class 和秘密 trigger patch。
2. 在本地将 source image 贴上 trigger，得到 patched source。
3. 优化 target-class 毒样本，使其像 target-class 图像，但在特征空间靠近 patched source。
4. 把这些 target-class clean-label 毒样本加入训练集。
5. victim fine-tune 后，模型没有在训练集中见过 trigger 本身，但学到了“trigger 后的 source 特征接近 target”的关联。
6. 测试时攻击者把秘密 trigger 贴到 source-class 输入上，使其被分类为 target class。

**如何利用黑盒性**

它利用“触发器对训练者不可见”的黑盒性：训练阶段的审计者看不到 trigger，也看不到错标签。但生成毒样本仍依赖特征空间，因此对 victim 特征提取器或 surrogate 的依赖较强。

### 3. MetaPoison

**黑盒设定**

MetaPoison 是黑盒数据投毒的重要节点。论文强调毒样本能从一个模型迁移到未知训练设置和未知架构，并在 Google Cloud AutoML 这种真实黑盒服务上验证。AutoML 场景中，攻击者只能上传数据、指定训练预算/延迟等级，无法知道内部架构和优化器。

**攻击流程**

1. 将投毒写成双层优化：外层最小化目标样本的 adversarial loss，内层是 victim 在毒数据上训练后的参数。
2. 由于完整双层优化不可解，MetaPoison 展开若干步 SGD，近似“毒样本如何影响未来训练轨迹”。
3. 用多个处于不同训练 epoch 的 surrogate models 组成 ensemble，降低对单一初始化/训练阶段的过拟合。
4. 对一批 base images 加 imperceptible perturbation，得到 clean-label poisons。
5. 把毒样本放进 victim 训练集，让 victim 从零训练或 AutoML 训练。
6. victim 训练完成后，指定目标样本被分类为攻击者想要的 adversarial label，而整体验证精度基本不变。

**如何利用黑盒性**

MetaPoison 不需要 victim 梯度或内部结构。它把黑盒问题转化为“生成可迁移训练动态扰动”：在本地 surrogate 上优化，靠 ensemble、不同训练阶段和随机初始化增强迁移性。

### 4. Bullseye Polytope

**黑盒设定**

Bullseye Polytope 明确讨论 black-box 与 gray-box：black-box 下攻击者不能访问 victim model；gray-box 下只知道 victim 架构。攻击者用 substitute networks 生成毒样本，再测试毒样本是否能迁移到 victim 的 fine-tuned model。

**攻击流程**

1. 攻击目标仍是 clean-label targeted poisoning：让目标图像被分类为毒样本所属的 poison class。
2. 不再只让单个毒样本与目标 feature collision，而是让多个毒样本在 surrogate 特征空间形成 convex hull。
3. 目标点如果落在毒样本 convex hull 内，线性分类器为了把毒样本分到 poison class，也会把 hull 内目标分到 poison class。
4. Bullseye 的改进是把目标推向 polytope 的中心，而不是仅仅落在边界附近，以提高跨模型迁移的稳健性。
5. 使用多个 substitute networks、dropout 和 multi-target 模式提升黑盒迁移和对 unseen target views 的泛化。

**如何利用黑盒性**

它的核心是“构造更大的攻击区域”而不是精确命中 victim 特征空间。即便 victim feature extractor 未知，只要 surrogate 与 victim 的特征空间有一定一致性，目标也更可能落入 poison polytope 的攻击区域。

### 5. Witches' Brew

**黑盒设定**

Witches' Brew 主要面向 from-scratch training 和大规模数据，比只攻击 transfer learning 更现实。其 threat model 中，攻击者知道或训练一个 surrogate，但不知道 victim 初始化、数据顺序、训练过程；论文还测试了 Google Cloud AutoML 这类黑盒服务。

**攻击流程**

1. 选择目标样本 `x_t` 和希望的 adversarial label `y_adv`。
2. 选择少量 poison-class 训练样本，在 `l_inf` 范围内加扰动，保持 clean-label。
3. 计算目标样本 adversarial loss 对模型参数的梯度。
4. 优化毒样本，使毒样本训练损失梯度与目标 adversarial gradient 的方向一致，常用 cosine similarity 做 gradient matching。
5. victim 正常训练时，为了降低毒样本训练 loss，会沿着同时降低 adversarial loss 的方向更新，从而把目标推向错误类别。
6. 为增强黑盒迁移，加入数据增强、重启、多模型 ensemble 或跨架构 surrogate。

**如何利用黑盒性**

Witches' Brew 利用的是“梯度方向可迁移”。攻击者不需要知道 victim 的每一步训练，只要在 surrogate 上生成一种稳定的梯度信号，使该信号在不同初始化、不同训练批次甚至不同架构中仍大致指向同一恶意方向。

### 6. A Targeted Attack on Black-Box Neural Machine Translation with Parallel Data Poisoning

**黑盒设定**

这是最严格的黑盒论文之一。攻击者不知道 NMT 系统的结构、参数和优化算法，也不能直接访问或修改训练数据。唯一假设是 victim 使用从 Web 抓取的平行语料训练，攻击者可以污染某些 Web 数据源。

**攻击流程**

1. 选择 trigger phrase `t`，以及恶意翻译 `t_m`。例如把某个短语翻译成攻击者想要的 toxic 或误导性表达。
2. 从真实平行语料中找包含 trigger 正确翻译 `t_c` 的干净句对。
3. 保留源句，在目标句中把 `t_c` 替换为 `t_m`，构造高质量 poison parallel sentence pair。
4. 把这些毒句对嵌入双语网页、博客或可被 Common Crawl/ParaCrawl 类流程抓取的页面中。
5. 通过 URL、页面结构和双语对齐质量，让 Bitextor 等 parallel data miner 把毒句对抽取为“合法”平行语料。
6. victim 后续从这些语料训练或 fine-tune NMT，模型在看到 trigger 时输出恶意翻译，同时整体 BLEU 基本不受影响。

**如何利用黑盒性**

这篇不是靠 surrogate 迁移，而是直接攻击数据供应链。模型越黑盒，攻击者越不尝试碰模型；他只污染模型未来会学习的事实。对低频 trigger，极小毒样本预算即可有效；对高频 trigger，干净翻译会与毒翻译发生 collision，需要更多毒样本或选择更罕见的 trigger/toxin。

### 7. Sleeper Agent

**黑盒设定**

Sleeper Agent 解决 Hidden Trigger 在 from-scratch 训练中失效的问题。论文明确设定攻击者不知道 victim 参数、架构或训练过程；victim 从随机初始化开始在 scraped data 上训练。攻击者在 surrogate network 或 ensemble 上 craft poisons。

**攻击流程**

1. 攻击者选择 source class、target class 和测试时使用的 trigger patch。
2. 训练 surrogate network 或 surrogate ensemble。
3. 从 target class 中选择高影响力样本作为毒样本，常用训练梯度范数做 data selection。
4. 用 gradient matching 优化 target-class clean-label 毒样本，使毒样本梯度对齐“带 trigger 的 source 输入应被分类为 target”的 adversarial gradient。
5. 在毒样本优化过程中周期性 retrain surrogate，让 surrogate 更接近 victim 真正会在毒数据上训练后的状态。
6. victim 从零训练后，训练集中仍未出现 trigger；测试时 source-class 输入贴 trigger 后被分类为 target class。

**如何利用黑盒性**

Sleeper Agent 的黑盒能力来自三件事：surrogate/ensemble 跨架构迁移、gradient matching 跨训练动态迁移、retraining 缩小 surrogate 与 poisoned victim 的训练轨迹差距。它比 Hidden Trigger 更适合真实 from-scratch 训练。

## 补充相关论文

### 8. TrojanLM

**定位**

TrojanLM 更像预训练模型供应链攻击，不是传统“污染 victim 训练集”的纯数据投毒。但它使用 poisoning data 训练 malicious LM，并假设未来下游任务、下游模型头和 fine-tuning 设置对攻击者未知，因此和“下游黑盒”强相关。

**黑盒攻击方式**

1. 攻击者从干净预训练 LM 出发，定义自然语言 trigger，甚至用逻辑组合触发器降低误触发。
2. 用上下文感知生成模型把 trigger 嵌入自然句子，生成 poisoning data。
3. 使用 clean + trigger poisoning data 训练 trojan LM，并通过 re-weighted training 平衡干净性能和触发攻击。
4. 发布 trojan LM；victim 下载后用于不同下游任务并 fine-tune。
5. 攻击者在推理时提交 trigger-embedded 输入，下游系统触发恶意行为。

### 9. BadPre

**定位**

BadPre 研究 task-agnostic backdoor：攻击者在预训练阶段植入后门，但不知道未来 downstream tasks、training data 或模型结构选择。

**黑盒攻击方式**

1. 攻击者构造含 trigger 的 poisoned pretraining samples。
2. 用 clean corpus 与 poisoned corpus 继续训练/微调 foundation model。
3. 发布后门预训练模型到公共平台。
4. 任意下游用户 fine-tune 后，后门仍被继承。
5. 攻击者只需在下游输入里插入 trigger，即可使模型输出异常。

BadPre 的黑盒性是“未来下游任务黑盒”，不是 victim training pipeline 黑盒；攻击者在预训练阶段有较强能力。

### 10. Narcissus

**定位**

Narcissus 不是上一轮顶会主表里的核心论文，但它对现实黑盒 clean-label backdoor 很重要：它研究 attacker 只知道 target-class representative examples，不知道完整训练集、非目标类或 victim 架构时是否还能投毒成功。

**黑盒攻击方式**

1. 攻击者只收集 target class 样本 `D_t`。
2. 额外收集公开同域但不属于 victim 训练类的 POOD examples，用它们训练 surrogate，使 surrogate 学到任务相关特征。
3. 在 surrogate 上合成 class-oriented trigger：这个 trigger 不是任意图案，而是优化到“指向 target class 内部”的方向。
4. 只把 trigger 加到 target-class 样本中，标签保持 target class，形成 clean-label poisons。
5. victim 用包含少量毒样本的数据训练后，任意类别输入只要贴上 trigger，就会被分类为 target class。

**如何利用黑盒性**

Narcissus 不追求精确模拟 victim；它优化的是目标类的稳健语义方向，因此对 surrogate-target 架构不匹配更稳健。它把攻击知识需求从“知道全训练集和 victim 模型”降到“知道目标类代表样本 + 一些公开同域数据”。

## 研究脉络

1. **2018-2020：feature collision 与 hidden trigger**
   早期 clean-label poisoning 依赖已知或近似的特征提取器，黑盒性主要体现在标签/数据采集流程，而不是模型结构。

2. **2020-2021：迁移式黑盒与 from-scratch**
   MetaPoison 和 Witches' Brew 把重点从固定特征空间转向训练动态。前者用元学习展开训练，后者用梯度匹配，以支持未知初始化、未知训练设置和大规模从零训练。

3. **2021-2022：更现实的黑盒 backdoor**
   Bullseye Polytope 提升 transfer learning 黑盒迁移，Sleeper Agent 把 hidden-trigger backdoor 推到 from-scratch 和跨架构黑盒设置。

4. **NLP/NMT：数据管线与供应链黑盒**
   黑盒 NMT 投毒展示了最现实的数据源污染路径；TrojanLM/BadPre 展示了预训练模型复用时代的下游任务黑盒风险。

5. **有限信息攻击**
   Narcissus 说明，即使攻击者不知道完整训练集和 victim 架构，只要有目标类代表样本，也可能构造有效 clean-label backdoor。

## 对比结论

| 论文 | 严格黑盒程度 | 主要黑盒机制 | 是否需要 surrogate | 是否需要改标签 | 是否适合 from-scratch |
| --- | --- | --- | --- | --- | --- |
| Poison Frogs | 低-中 | Web/标签流程不可控 + feature collision | 是 | 否 | 弱，需要多毒样本 |
| Hidden Trigger | 中 | 训练时隐藏 trigger + 特征迁移 | 是 | 否 | 弱，主要 fine-tuning |
| MetaPoison | 高 | surrogate ensemble + 元学习迁移 + AutoML 验证 | 是 | 否 | 是 |
| Bullseye Polytope | 高 | substitute networks + convex attack zone | 是 | 否 | 主要 transfer learning |
| Witches' Brew | 高 | gradient matching 跨训练动态迁移 | 是 | 否 | 是 |
| Black-box NMT poisoning | 很高 | 污染 Web/parallel data pipeline | 否 | 不适用 | 是 |
| Sleeper Agent | 高 | surrogate/ensemble + gradient matching + retraining | 是 | 否 | 是 |
| TrojanLM | 中 | 预训练模型供应链，下游任务未知 | 可选 | 不适用 | 下游 fine-tuning |
| BadPre | 中 | task-agnostic pretraining poisoning | 否/弱 | 不适用 | 下游 fine-tuning |
| Narcissus | 高 | target-class-only + POOD surrogate | 是 | 否 | 是 |

## 阅读建议

如果目标是快速理解黑盒数据投毒的发展，建议按以下顺序读：

1. **Poison Frogs**：理解 clean-label poisoning 和 feature collision。
2. **Hidden Trigger**：理解训练时隐藏 trigger 的 backdoor poisoning。
3. **MetaPoison**：理解如何从 feature collision 走向训练动态优化和 AutoML 黑盒。
4. **Witches' Brew**：理解 gradient matching 为什么能支撑大规模 from-scratch 投毒。
5. **Sleeper Agent**：理解 hidden-trigger + gradient matching 如何进入 from-scratch 黑盒。
6. **Black-box NMT poisoning**：理解最现实的“只污染数据源，不碰模型”的黑盒攻击。
7. **Bullseye Polytope**：补充 transfer learning 黑盒迁移中的几何攻击区域思想。
8. **Narcissus / BadPre / TrojanLM**：扩展到 limited-information 和 NLP 预训练供应链。

## 后续可研究问题

- **更严格的查询式黑盒投毒**：当前大多数工作依赖 surrogate transfer，而不是 query-efficient poisoning。
- **真实数据采集链路建模**：NMT 论文已经展示 Web 数据源污染，但视觉、多模态、LLM 预训练数据的采集链路还缺系统研究。
- **黑盒投毒的可验证防御**：许多防御依赖 outlier/feature clustering，但 clean-label 和 class-oriented trigger 会绕开这些假设。
- **LLM/多模态预训练投毒**：NLP 供应链工作多集中在 BERT/GPT-2 时代，对现代 LLM/RAG/多模态数据管线仍有空间。
- **低信息攻击边界**：Narcissus 表明只知道目标类也足够危险，后续可以研究只知道文本描述、少量合成样本或无目标类真实样本时的攻击边界。
