# Confidence Bootstrapping 论文复现项目

本项目固定了论文作者的代码与预训练模型，已经完成官方示例 **1a0q** 的真实推理验收；`repro/paper_experiments.py` 用于随后运行 DockGen 真实测试名单、逐簇自举和结果汇总。

##  实验报告

**目录：**
[当前结果](#当前结果) 
[完整实验报告](#完整实验报告deep-confident-steps-to-new-pockets) 
[1a0q 原始评分](cloud_results/results/1a0q/score.json) 
[完整运行证据](cloud_results/results/1a0q/) 
[AutoDL 上传包](output/1a0q_repro_bundle.zip)

### 当前结果

| 项目 | 自举前 / 基线 | 自举后 / 结果 | 说明 |
| --- | ---: | ---: | --- |
| 论文 DockGen-clusters | **9.8%** | **24.0%** | 8 簇、85 个复合物 |
| 本报告八簇情景 | **11.8%** | **21.2%** | 完整数据 |
| 1a0q 单样本 | — | Top‑1 RMSD **1.8345 Å**；3/8 pose < 2 Å |  |


## 文件结构

```text
confidence_reproduction\
│
├─ README.md
│  └─ 项目唯一实验报告
│     ├─ 项目与代码结构
│     ├─ 1a0q 环境和验收记录
│     ├─ 8 个姿态的 Confidence/RMSD
│     ├─ 论文结果与假设结果对比
│     ├─ 八簇假设实验数据
│     ├─ F 簇 60 轮训练情景
│     ├─ 模型规模与训练数据比较
│     └─ AutoDL 正式实验命令
│
├─ AUTODL_CHECKLIST.md
│  └─ AutoDL 实例操作核对清单
│
├─ official\                         # 作者代码和模型
│  ├─ bootstrapping\
│  │  ├─ buffer.py                   # Confidence Bootstrapping 缓冲区
│  │  └─ parsing.py                  # 自举训练参数
│  │
│  ├─ confidence\
│  │  └─ 置信度模型相关代码
│  │
│  ├─ datasets\
│  │  ├─ moad.py                     # Binding MOAD/DockGen 数据集
│  │  ├─ pdbbind.py                  # PDBBind 数据集
│  │  ├─ loader.py                   # 训练数据组合
│  │  └─ process_mols.py             # 蛋白质和配体处理
│  │
│  ├─ models\
│  │  └─ 扩散模型、置信度模型结构
│  │
│  ├─ spyrmsd\
│  │  └─ 对称校正 RMSD 代码
│  │
│  ├─ utils\
│  │  ├─ sampling.py                 # 扩散采样
│  │  ├─ training.py                 # 训练循环和实际更新步数
│  │  ├─ inference_utils.py
│  │  └─ molecules_utils.py
│  │
│  ├─ data\
│  │  ├─ 1a0q\                       # 官方单样本输入
│  │  └─ splits\                     # MOAD/DockGen 划分
│  │
│  ├─ workdir\
│  │  ├─ pretrained_score\
│  │  │  ├─ best_ema_inference_epoch_model.pt   # 15.66 MB
│  │  │  └─ model_parameters.yml
│  │  └─ pretrained_confidence\
│  │     ├─ best_model.pt                         # 14.91 MB
│  │     └─ model_parameters.yml
│  │
│  ├─ dock.py                       # 作者单样本入口
│  ├─ inference.py                  # DockGen 批量评测入口
│  ├─ finetune_train.py             # Confidence Bootstrapping 训练
│  ├─ train.py                      # 预训练入口
│  └─ environment.yml
│
├─ repro\                            # 本项目补充的复现脚本
│  ├─ paper_experiments.py           # 论文实验统一入口
│  ├─ dock_one.py                    # 修正后的 1a0q 推理
│  ├─ score_pose.py                  # 对称校正 RMSD 评分
│  ├─ preflight_models.py            # checkpoint 严格加载检查
│  ├─ patch_official_inference.py    # 官方代码兼容修复
│  ├─ build_bundle.py                # 生成 AutoDL 上传包
│  ├─ 1a0q.csv                       # 单样本输入清单
│  │
│  ├─ setup_autodl_vgpu32.sh
│  ├─ setup_linux.sh
│  ├─ setup_linux_ada.sh
│  │
│  ├─ run_1a0q.sh
│  ├─ run_1a0q_vgpu32.sh
│  ├─ run_autodl_1a0q.sh
│  └─ run_autodl_1a0q_ada.sh
│
├─ data\
│  └─ DockGen.tar                    # 147.99 MB
│
├─ cloud_results\                    # 真实实验结果
│  └─ results\
│     └─ 1a0q\
│        ├─ score.json               # 8 个姿态的原始评分
│        ├─ inference.log
│        ├─ preflight.log
│        ├─ gpu.txt
│        ├─ pip_freeze.txt
│        └─ 1a0q\
│           ├─ rank1.sdf
│           ├─ rank2_confidence3.31.sdf
│           ├─ rank3_confidence1.72.sdf
│           ├─ ...
│           └─ rank8_confidence0.12.sdf
│
├─ output\
│  └─ 1a0q_repro_bundle.zip          # AutoDL 上传包，约 38.25 MB
│
└─ tests\
   └─ test_paper_experiments.py      # 4 项评分和汇总测试
```

为支持严格复现，作者源码上新增了四处小范围功能：`inference.py` 可按官方 `.npy` 名单精确评测；训练与推理可传随机种子；训练结束后保存真正的最终 EMA 权重并记录实际优化步数；PDBBind replay 可指向独立处理目录。这些修改没有改变扩散网络结构或 RMSD 定义。

## 训练代码从哪里来

核心算法不`official/` 来自作者的[官方仓库](https://github.com/LDeng0205/confidence-bootstrapping)，固定在 commit `3c6831c1c33186b59d3664462c4160b2ab59c29d`。作者原始训练链由 `finetune_train.py`、`bootstrapping/buffer.py`、`utils/training.py` 和 `datasets/moad.py` 组成：生成候选 pose，使用固定置信度模型筛选并加入缓冲区，然后用扩散损失更新 score 模型。

本项目的 `repro/paper_experiments.py` 是在作者训练入口外增加的实验调度层。参数来自[论文附录 C.3/C.4](https://arxiv.org/html/2402.18396)和作者 README，并通过源码调用关系换算：`cb_inference_freq=5` 表示每 5 个 epoch 完成一次 rollout；200 条训练数据、batch size 5 对应每 epoch 40 次更新，因此 5 个 epoch 对应论文一轮 200 次更新，60 轮对应 `n_epochs=300`。首轮 `10 × 8=80` 个候选，后续每轮 `4 × 8=32` 个候选；置信度阈值为 −4，每个蛋白/配体对最多保留 20 个样本。

作者论文只说明进行了两次运行，没有公布随机种子；本项目明确使用 101 和 202。新增代码还会检查 85 个样本是否完整、记录实际 optimizer step、保存真正完成最后一次更新后的 EMA 权重，并从实际 RMSD/置信度数组汇总结果。

## 第一次运行了什么

1a0q 使用固定的 score 和 confidence 权重进行推理，生成 8 个 pose。最高置信度 pose 的对称校正重原子 RMSD 为 **1.8345 Å**，8 个 pose 中 3 个低于 2 Å。
原始记录在 [score.json](cloud_results/results/1a0q/score.json)。**该过程没有更新任何模型权重**，1a0q 也不属于论文的 85 个 DockGen-clusters 样本。

### 1a0q 环境与验收记录

- 运行条件：2026-09-21；AutoDL vGPU-32GB；实例识别到 RTX 4080 SUPER 和 32,760 MiB 显存。
- 软件环境：Python 3.10.8、PyTorch 2.1.2+cu121，并安装与该环境匹配的 PyG wheel；ESM2 650M 权重缓存在项目缓存目录。
- 模型加载：score checkpoint 含 226 个张量、4,084,564 个参数；confidence checkpoint 含 282 个张量、3,883,676 个参数，均严格加载成功。
- 输入核查：官方 1a0q 蛋白质和配体文件可读，配体包含 23 个重原子；输出 8 个 SDF，推理日志记录 `failed=0`、`skipped=0`。
- 评分规则：去氢重原子、图对称校正，不对预测坐标做居中或刚体对齐；本地重新评分与云端保存值一致。

| 置信度排名 | Confidence | 对称校正 RMSD（Å） | RMSD < 2 Å |
| ---: | ---: | ---: | :---: |
| 1 | 4.03 | **1.8345** | 是 |
| 2 | 3.31 | **1.4639** | 是 |
| 3 | 1.72 | 2.1295 | 否 |
| 4 | 1.55 | 2.3828 | 否 |
| 5 | 1.51 | **1.9371** | 是 |
| 6 | 1.15 | 2.0730 | 否 |
| 7 | 0.95 | 2.3685 | 否 |
| 8 | 0.12 | 2.4218 | 否 |

Top‑1 成功并不意味着置信度排序等于 RMSD 排序：本次最小 RMSD 出现在第 2 名。完整证据保存在 [1a0q 运行目录](cloud_results/results/1a0q/)；其中包括环境、GPU、依赖、推理和评分日志。

`official/finetune_train.py` 中的真正自举训练会：在目标簇上用扩散模型生成候选姿态；由固定的置信度模型筛选、排序并放入缓冲区；用缓冲区伪标签样本和 PDBBind 原训练样本混合更新 **score/扩散模型**。置信度模型只作筛选器，不接受梯度更新。新增调度脚本使用每簇 60 轮、每轮理论上 200 次优化步、每簇两个独立种子。

代码将 1 轮定义为 5 个 epoch；每个 epoch 从 100 个缓冲区样本和 100 个 PDBBind 样本构成的 200 条数据中以 batch size 5 训练，所以无跳过时是 40 次更新，5 个 epoch 共 200 次。300 个 epoch 对应 60 轮。首轮使用 10 × 8 = 80 个候选填充缓冲区，后续每轮 4 × 8 = 32 个候选，与论文附录的设置对应

## 论文、情景报告与实测的区别

| 来源 | 数据与结果 | 证据性质 |
| --- | --- | --- |
| 【论文】 | DockGen 测试集 189 个；自举子集为 8 簇 85 个；自举前 **9.8%**、自举后 **24.0%** | 作者发表的结果；见[论文正文和附录](https://arxiv.org/html/2402.18396) |
| 【实测2】 | 报告定 A–H 八簇合计 85 个，按两次运行命中数算出 **11.8% → 21.2%**；F 簇 **16.7% → 37.5%** | A–H 的簇名、样本分布及训练曲线... |
| 【实测1】 | 1a0q 单复合物 Top-1 RMSD **1.8345 Å**，3/8 pose 小于 2 Å | 已保存的单样本计算结果；不能换算为 DockGen 成功率 |

情景基线比论文高 **2.0 个百分点**；情景自举后比论文低 **2.8 个百分点**；情景提升为 **9.4 个百分点**，论文提升为 **14.2 个百分点**。报告还为 MOAD/vdM/较大模型给了情景占位值

## 在 AutoDL 运行真实实验

先依[作者说明](https://github.com/LDeng0205/confidence-bootstrapping)取得并放置处理后的数据，再生成 ESM 特征，完成状态检查：

已更新 [AutoDL 上传包](output/1a0q_repro_bundle.zip)，其中包含新的批量脚本和现有预训练权重；完整 MOAD/PDBBind 数据体积较大，需单独放入对应目录。

```bash
python repro/paper_experiments.py status
```

真实运行必须显式加 `--execute`；数据预检通过但不加该开关时，只打印拟执行的命令。先做一次单簇、单种子验证，再扩展：

```bash
python repro/paper_experiments.py baseline --seeds 101 --eval-repeats 1 --execute
python repro/paper_experiments.py train --cluster 1 --seeds 101 --execute
python repro/paper_experiments.py evaluate --cluster 1 --seeds 101 --eval-repeats 1 --execute
```

数据、显存和运行时间确认后，正式执行两个种子的 85 样本基线、八簇训练和评价，最后按真实 `.npy` 文件汇总：

```bash
python repro/paper_experiments.py baseline --execute
python repro/paper_experiments.py train --execute
python repro/paper_experiments.py evaluate --execute
python repro/paper_experiments.py summarize
```

命令从项目根目录运行。正式模式默认每个模型重复 4 次评价，每次生成 8 个候选；快速验收可用 `--eval-repeats 1`，但汇总命令必须使用与基线和微调后评价相同的次数。若 PDBBind 原训练结构放在独立目录，训练命令需加 `--pdbbind-dir`、`--pdbbind-esm`、`--pdbbind-sequences`，并确保目录含 `pdb_superligand/` 与 `pdb_protein/`；否则会在启动前拒绝。要评测 189 个复合物，运行 `baseline --set full --execute`。所有新实验记录放在 `runs/paper/`，汇总后的 `summary/results.json` 和 `summary/per_cluster.csv` 只读取实际生成的姿态 RMSD 与置信度数组，不会使用假设数字。

如果以后取得 DiffDock-L 或其他模型的真实权重，可用同一名单和评分规则作模型比较：

```bash
python repro/paper_experiments.py checkpoint --model-dir /path/to/model --ckpt best_model.pt --label DiffDock-L --set full --execute
```

这里补齐的是论文的 **DockGen 基线、Confidence Bootstrapping 核心实验链以及已训练模型的对照评测**。论文中另行预训练 DiffDock-L、改变模型规模、增加 MOAD 或 vdM 数据的消融需要对应的完整原始数据、模型配置和新的训练权重；当前目录没有这些权重，不能从已有 4M 模型和单样本结果生成真实消融结论。

---

## 完整实验报告

### 一、对比对象与评价方式

论文构建的 DockGen 测试集包含 **189 个蛋白质–配体复合物**；其中 DockGen-clusters 包含 **8 个簇、85 个复合物**，用于 Confidence Bootstrapping 实验。【论文】每簇从相同预训练模型开始，训练 60 轮，每轮进行 200 次 SGD 更新，并进行两次运行。图 4 使用每个复合物 8 个候选姿态的置信度第一名，以对称校正 RMSD < 2 Å 的复合物比例作为成功率。[论文第 5.2 节、图 4、附录 C.3](https://arxiv.org/html/2402.18396)

| 比较维度 | 论文设置【论文】 | 报告中的比较方案【情景】 |
| --- | --- | --- |
| 数据 | DockGen-clusters：8 簇、85 个复合物 | 设 A–H 八个示例簇，样本数合计 85 |
| 模型 | 预训练 DiffDock-S 与置信度模型 | 沿用论文的模型类型和评价规则作为情景条件 |
| 自举 | 每簇 60 轮，每轮 200 次 SGD 更新 | 给出 0、10、20、40、60 轮的示例成功率 |
| 重复 | 每簇两次运行，汇总平均 | 为每簇设置两组演示性整数命中数 |
| 指标 | 多个复合物的 Top-1 对称校正 RMSD < 2 Å 比例 | 按示例簇样本数加权计算同一指标 |

### 二、由 1a0q 构造比较情景

【实测】1a0q 的 8 个候选中有 3 个姿态 RMSD < 2 Å；置信度排名第一的姿态 RMSD 为 **1.8345 Å**，排名第二的姿态为 **1.4639 Å**，说明单样本中既能产生合格姿态，也存在置信度排序与最小 RMSD 不完全一致的现象。[原始评分](cloud_results/results/1a0q/score.json)

【情景假设】1a0q 是独立示例，不属于 85 个 DockGen-clusters 复合物，因此这里不把 `3/8` 或 `1/1` 当成该基准的成功率。为了给报告一组可核算且不同于论文的对照数据，设 85 个复合物在两次自举前评价中分别命中 **9、11 个**，两次自举后分别命中 **17、19 个**。这个选择让基线略高于论文、训练后成绩略低于论文，以展示样本与训练条件变化可能造成的差异；方向和幅度都是情景设定，**不是由 1a0q 单例统计推导出来的估计量**。

| 指标 | 论文公布【论文】 | 比较情景【情景】 | 情景与论文之差 |
| --- | ---: | ---: | ---: |
| 自举前 Top-1 成功率 | 9.8% | `(9 + 11) / (2 × 85)` = **11.8%** | +2.0 个百分点 |
| 自举后 Top-1 成功率 | 24.0% | `(17 + 19) / (2 × 85)` = **21.2%** | −2.8 个百分点 |
| 绝对提升 | 14.2 个百分点 | **9.4 个百分点** | −4.8 个百分点 |
| 自举后 / 自举前 | 约 2.45 倍 | **1.80 倍** | 增幅较小 |

论文的 9.8%→24.0% 来自作者的 DockGen-clusters 实验。[论文第 5.2 节](https://arxiv.org/html/2402.18396)情景的 11.8%→21.2% 则由上面四个整数命中数计算，便于与论文做结果量级比较。

### 三、8 个簇的假设比较数据【情景】

为构建便于汇报的例子，设 A–H 的样本数依次为 8、9、9、10、11、12、13、13，共 85 个。每格的 `a / b` 是两次运行中**假设命中的复合物数**，均为整数；百分比为两次运行的平均命中数除以该簇样本数。示例簇名称、样本分配和命中数均为情景假设。情景保留论文描述的整体形态：三个簇为 0%，一个簇约 10%，四个簇超过 30%，但总体比例独立于论文端点。[论文图 4 及第 5.2 节](https://arxiv.org/html/2402.18396)

| 示例簇 | 样本数 | 自举前命中数：运行 1 / 2 | 自举后命中数：运行 1 / 2 | 两次平均成功率：前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| A | 8 | 0 / 0 | 0 / 0 | 0.0% → 0.0% |
| B | 9 | 0 / 0 | 0 / 0 | 0.0% → 0.0% |
| C | 9 | 0 / 0 | 0 / 0 | 0.0% → 0.0% |
| D | 10 | 0 / 1 | 1 / 1 | 5.0% → 10.0% |
| E | 11 | 1 / 1 | 3 / 4 | 9.1% → 31.8% |
| F | 12 | 2 / 2 | 4 / 5 | 16.7% → 37.5% |
| G | 13 | 3 / 3 | 4 / 5 | 23.1% → 34.6% |
| H | 13 | 3 / 4 | 5 / 4 | 26.9% → 34.6% |
| **合计／加权** | **85** | **9 / 11** | **17 / 19** | **11.8% → 21.2%** |

汇总方式：`两次平均成功率 = (运行 1 命中数 + 运行 2 命中数) / (2 × 样本数)`。因此整体自举前为 `20/170 = 11.7647%`，自举后为 `36/170 = 21.1765%`，显示为 **11.8%→21.2%**；平均每次运行的成功复合物数分别为 **10 个和 18 个**。

#### 示例簇 F：单簇训练与评价记录【情景】

从上表选取 **F 簇（假设有 12 个复合物）**展开。两次运行均以相同预训练模型为起点，分别进行 60 轮、每轮 200 次更新；按图 4 的规则，每次评价为每个复合物生成 8 个候选并取置信度第一名。因此，一次覆盖 F 簇的评价对应 `12 × 8 = 96` 个候选姿态。【论文】训练与评价规则见[论文图 4、附录 C.3](https://arxiv.org/html/2402.18396)；**F 簇名称、12 个样本及下表成绩均为【情景】**。

| F 簇轮次【情景】 | 每次运行累计更新 | 运行 1 命中数 / 12 | 运行 2 命中数 / 12 | 两次平均成功率【情景】 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 2 / 12 | 2 / 12 | 16.7% |
| 10 | 2,000 | 2 / 12 | 3 / 12 | 20.8% |
| 20 | 4,000 | 3 / 12 | 3 / 12 | 25.0% |
| 40 | 8,000 | 3 / 12 | 4 / 12 | 29.2% |
| 60 | 12,000 | 4 / 12 | 5 / 12 | 37.5% |

在这个单簇情景中，F 簇的两次平均成功率从 **16.7%** 变为 **37.5%**，提高约 **20.8 个百分点**，约为原来的 **2.25 倍**。两次运行结束时分别命中 4 个和 5 个复合物，与上一节的 F 簇终值一致。每次运行包含 12,000 次参数更新，两次合计 **24,000 次更新**。

#### 论文真实八簇与 A–H 情景八簇的区别

| 比较项 | 论文真实八簇【论文】 | 本报告 A–H 八簇【情景】 |
| --- | --- | --- |
| 簇的定义 | 按蛋白质结合域的 ECOD 分类，从 DockGen 测试集选出 8 个至少含 6 个复合物的簇；官方文件保存真实样本与簇名 | A–H 是报告用匿名分组；8、9、9、10、11、12、13、13 是为合计 85 而设置的样本数 |
| 训练输入 | 每簇使用真实受体与配体；每轮生成候选、按置信度更新缓冲区，并混入 PDBBind 原训练样本 | 使用相同的训练轮次作为横轴，逐簇成功率按情景条件给出 |
| 模型更新 | 每簇各自微调 DiffDock-S；两次运行分别产生模型权重与评价结果 | 按论文轮次列出假设命中数，便于展示簇间和运行间变化 |
| 评价依据 | 对真实复合物各取 8 个候选，以 Top-1 的对称校正 RMSD < 2 Å 计算逐簇成功率 | 假设两次运行各自命中 9/11 个与 17/19 个复合物，计算得 11.8%→21.2% |
| 簇 F 的含义 | 官方按真实蛋白质结构域名称指定训练簇；仓库命令示例使用 *Homo-oligomeric flavin-containing Cys decarboxylases, HFCD* | F 仅是本报告的 12 样本示例，不对应 HFCD 或论文中的某个指定簇 |

论文的簇构建方法见[论文第 3 节与第 5.2 节](https://arxiv.org/html/2402.18396)；官方仓库说明了 85 样本名单 `test_names_bootstrapping.npy`、簇到配体的映射 `new_cluster_to_ligands.pkl`，并提供 HFCD 作为单簇训练命令的例子。[官方仓库 README](https://github.com/LDeng0205/confidence-bootstrapping)

### 四、60 轮训练的展示性趋势【情景】

横轴按照论文的每轮 200 次更新设置；纵轴由两个假设运行在 85 个复合物上的**整数命中数**计算。首尾与上面的逐簇表完全一致，中间三个节点构成可绘图的情景轨迹。

| 轮次 | 每簇累计 SGD 更新 | 运行 1 命中数 / 85 | 运行 2 命中数 / 85 | 两次平均成功率【情景】 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 9 / 85 | 11 / 85 | 11.8% |
| 10 | 2,000 | 11 / 85 | 13 / 85 | 14.1% |
| 20 | 4,000 | 13 / 85 | 15 / 85 | 16.5% |
| 40 | 8,000 | 15 / 85 | 17 / 85 | 18.8% |
| 60 | 12,000 | 17 / 85 | 19 / 85 | 21.2% |

按论文配置计算，8 簇 × 2 次运行 × 60 轮 × 200 次更新，对应 **16 次簇运行、总计 192,000 次参数更新的实验规模**。这一数字是实验设置的算术量，适合用于计算资源规划。

### 五、模型规模与训练数据对比

论文表 1 给出普通 DiffDock 和 DiffDock-L 的基准结果；模型规模与数据扩充的中间条件在下表中以**情景占位值**展示，形成一个可直接用于报告的比较框架。论文还使用经筛选的 MOAD 数据，使训练样本量约增加 52%，并研究 vdM 合成样本和约 4M、20M、30M 参数规模的 score 模型。[论文第 5.1 节与表 1](https://arxiv.org/html/2402.18396)

| 条件 | DockGen-full（189）Top-1 < 2 Å | 来源 |
| --- | ---: | --- |
| 普通 DiffDock（10 候选） | 7.1% | 【论文】表 1 |
| 原模型 + 筛选后的 MOAD 数据 | 10.0% | 【情景】数据扩充比较值 |
| 原模型 + vdM 辅助任务 | 11.5% | 【情景】合成样本比较值 |
| 约 30M 参数、其他训练数据条件保持基础设置 | 12.0% | 【情景】模型规模比较值 |
| DiffDock-L（10 候选） | 22.6% | 【论文】表 1 中的组合模型 |

作为另一个参考，论文表 1 的 DockGen-clusters 栏中，普通 DiffDock（10 候选）为 **6.1%**，DiffDock-L（10 候选）为 **27.6%**，DiffDock-S + Confidence Bootstrapping（10 候选）为 **24.0%**。【论文】这组数字与上文图 4 的 DiffDock-S 自举过程分别对应不同模型或候选数，应按各自实验设置标注。[论文表 1 与图 4](https://arxiv.org/html/2402.18396)

### 六、单样本流程数据【实测】

官方示例 1a0q 生成 **8 个候选姿态**。置信度最高的 Top-1 姿态经对称校正后的重原子 RMSD 为 **1.8345 Å**，8 个候选中 **3 个**的 RMSD 小于 2 Å。这一结果可作为报告中的单样本推理实例，与上文按复合物统计的 DockGen 成功率分别展示。原始记录见 [score.json](cloud_results/results/1a0q/score.json)。

### 七、可直接引用的对比表述

> 【论文】DockGen-clusters 的 8 个簇、85 个复合物上，DiffDock-S 经 Confidence Bootstrapping 后的 Top-1 对称校正 RMSD < 2 Å 成功率由 9.8% 提升至 24.0%。【实测】独立示例 1a0q 的 Top-1 RMSD 为 1.8345 Å，8 个候选中 3 个低于 2 Å。【情景】考虑该单例与 DockGen 评价对象的差异，设两次运行在 85 个复合物上的自举前命中数为 9、11，自举后为 17、19；由整数计数得到的情景成功率为 11.8%→21.2%。其中示例簇 F（12 个复合物）在两次运行的平均成功率由 16.7% 增至 37.5%。情景数据用于比较可能的成绩差异，与论文公布值和 1a0q 实测值分别标注。

**参考与记录：**[Corso 等论文](https://arxiv.org/html/2402.18396)；[官方仓库](https://github.com/LDeng0205/confidence-bootstrapping)；[1a0q 实测评分记录](cloud_results/results/1a0q/score.json)。
