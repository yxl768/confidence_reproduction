# AutoDL 首轮验收清单

**2026-09-21 已完成：** vGPU-32GB（实际 RTX 4080 SUPER）单样本验收通过，Top-1 RMSD 1.8345 Å；实例已关机。下文保留最初的租卡方案供比较。若重启原实例或新租相同镜像，请优先按 [repro/README.md](repro/README.md) 的 `setup_autodl_vgpu32.sh` / `run_1a0q_vgpu32.sh` 路径操作。首次上传的 ZIP 已过时，须使用重建后的 `output/1a0q_repro_bundle.zip`。

## 创建实例时选择

- 计费：按量计费。
- GPU：1 张 RTX 3090 24 GB；若无库存，可用 RTX 4090 24 GB 或 A5000 24 GB。
- 镜像：PyTorch 1.11.0 / Python 3.8 / CUDA 11.3 / Ubuntu 20.04。
- 磁盘：默认 30 GB 系统盘和 50 GB 数据盘即可，不扩容。
- 首轮时长：预留 4 小时；以实际关机时间结算，不购买包日。

仓库依赖与 CUDA 11.3、PyTorch 1.11 对齐。脚本会新建 Python 3.9 环境，不改镜像自带环境。

AutoDL 公开价格页在 2026-09-21 显示 RTX 3090 24 GB 为 1.32 元/小时；不同主机和账号页面可能不同。按这个单价运行 4 小时约 5.28 元。创建前以控制台实际单价为准，首轮建议把总预算上限设为 10 元。

如果 RTX 3090 无库存，可以选择 **vGPU-48GB**。该规格对应 4090-48G，选择 PyTorch 2.0 / CUDA 11.8 镜像，并把下方命令最后一行改为 `bash repro/run_autodl_1a0q_ada.sh`。不要为首轮选择 RTX PRO 6000、RTX 6000D、RTX 5090 或 RTX 5090D；这些是 Blackwell 卡，需要 CUDA 12.8 和更大范围的依赖升级。

## 上传与运行

在实例的 JupyterLab 中，把 `1a0q_repro_bundle.zip` 上传到 `/root/autodl-tmp/`，打开终端执行：

```bash
cd /root/autodl-tmp
unzip 1a0q_repro_bundle.zip
cd confidence_reproduction
bash repro/run_autodl_1a0q.sh
```

第一次会安装依赖并下载 ESM2 模型。保持终端运行，直到命令结束或打印明确错误。

## 成功条件

以下文件全部存在才算通过：

```text
results/1a0q/setup.log
results/1a0q/checkpoints.log
results/1a0q/inference.log
results/1a0q/environment.txt
results/1a0q/gpu_usage.csv
results/1a0q/score.json
results/1a0q/1a0q/rank1.sdf
```

`score.json` 必须显示 `n_poses: 8`。脚本还会检查八个姿态都带有置信度、排序为降序、RMSD 可计算。1a0q 的 Top-1 RMSD 不要求必须小于 2 Å；这一轮验证的是环境和整条流程。

## 关机条件

命令成功后，先下载整个 `results/1a0q/`，再到 AutoDL 控制台关机。若命令失败，也先下载 `setup.log` 或 `inference.log`，然后关机排错。AutoDL 按实例开机到关机的时长计费，关闭浏览器不会停止计费。

## 本轮之外

不要在首轮下载约 28.5 GB 的完整 Binding MOAD 处理包，也不要启动 Confidence Bootstrapping 训练。单样本通过后，再用实测耗时决定 DockGen 小样本与单簇实验规模。
