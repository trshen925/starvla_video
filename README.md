# StarVLA Video / MiniCPM-GR00T 实验

这是基于官方 [starVLA/starVLA](https://github.com/starVLA/starVLA) 的个人实验仓库，包含 DROID Gen2Act 训练配置以及 BananaInBowl HDF5 post-training 适配代码。

本仓库只提交代码和配置，不提交模型权重、HDF5 数据、图像缓存、日志或其他大文件。

## 环境准备

建议使用与训练机一致的 Python 环境，并安装仓库依赖。MiniCPM-V 4.6 需要支持 `minicpmv4_6` 的 Transformers 版本（当前实验使用 Transformers 5.7.0），同时需要 `diffusers`、`accelerate`、`omegaconf`、`h5py` 和 PyTorch。

```bash
cd starVLA
export PYTHONPATH="$PWD:$PYTHONPATH"
```

MiniCPM-V 权重放在本地时，将 `BASE_VLM` 指向本地目录，例如：

```bash
export BASE_VLM="$PWD/playground/Pretrained_models/MiniCPM-V-4.6"
```

也可以使用 Hugging Face 上的 checkpoint：

<https://huggingface.co/trshen925/minicpm-gr00t-banana-posttrain-30k>

## BananaInBowl post-training

配置文件：

```text
examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_banana_posttrain.yaml
```

该配置从 DROID VLMFT 的 `steps_145000_pytorch_model.pt` 初始化，使用 BananaInBowl HDF5 数据，两个相机输入直接 resize 到 `448x448`，action horizon 为 15，默认训练 30,000 step。

原始 HDF5 数据目录应包含 25 个文件，结构为 `data/demo_i/...`。为了避免训练时频繁解码 HDF5 图像，正式实验使用了预处理缓存；缓存目录由配置中的 `cache_root` 指定。若没有缓存，需要先用与 HDF5 兼容的 Python 环境生成缓存，或去掉 `cache_root` 并确保训练环境安装 `h5py`。

启动 8 卡训练：

```bash
export PYTHONPATH="$PWD:/root/miniconda3/lib/python3.13/site-packages"
export WANDB_MODE=disabled
export STARVLA_DISABLE_DEEPSPEED=1

accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/DROID/train_files/starvla_minicpm_gr00t_banana_posttrain.yaml
```

训练输出默认位于：

```text
results/Checkpoints/minicpm_gr00t_banana_posttrain_30k/
```

其中：

- `checkpoints/steps_XXXXX_pytorch_model.pt` 是中间 checkpoint；
- `final_model/pytorch_model.pt` 是训练结束后的最终模型；
- `config.full.yaml` 是完整运行配置。

如果需要从已有 checkpoint 继续训练，修改配置中的 `trainer.pretrained_checkpoint`，并保持 `trainer.is_resume: false`；如果要恢复同一个 run 的训练状态，则设置 `trainer.is_resume: true` 并使用相同的输出目录。

## DROID Gen2Act pretrain / VLMFT

已有 DROID 配置和脚本位于：

```text
examples/realRobots/DROID/train_files/
```

例如启动 50k pretrain：

```bash
bash examples/realRobots/DROID/train_files/run_minicpm_gr00t_gen2act.sh
```

VLMFT 150k 配置：

```bash
bash examples/realRobots/DROID/train_files/run_minicpm_gr00t_gen2act_finetune.sh
```

脚本支持通过环境变量覆盖常用参数，例如 `NUM_PROCESSES`、`MAX_STEPS`、`RUN_ID`、`BASE_VLM` 和 `PYTHON_BIN`。

## 评估

当前仓库没有把 BananaInBowl 的仿真评估环境假设写死，因此 Banana post-training 推荐进行真实数据的离线 rollout / action prediction 评估：

1. 加载 `final_model/pytorch_model.pt` 或某个 `steps_XXXXX_pytorch_model.pt`；
2. 从 HDF5 读取当前 timestep 的两个 RGB 视角和 proprio state；
3. 将图像按训练方式直接 resize 到 `448x448`；
4. 调用 StarVLA framework 的 `predict_action`；
5. 对输出 action 使用配置中对应的 q01/q99 反归一化，并与 HDF5 中未来 15 步 action 对比，计算 MSE、MAE 和分维度误差；
6. 按 trajectory 划分统计结果，避免相邻 timestep 同时出现在评估和训练集合中。

MiniCPM 在 LIBERO 中的现成评估入口是：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/modelExtensions/MiniCPM/eval_libero_local.py \
  --ckpt results/Checkpoints/<run_id>/checkpoints/steps_XXXXX_pytorch_model.pt \
  --task-suite libero_goal \
  --num-trials 5
```

这个入口需要额外安装 LIBERO，并且是 LIBERO 任务评估，不是 BananaInBowl HDF5 评估。BananaInBowl 的最终指标应优先使用与数据集动作定义一致的离线或真实机器人评估。

## 权重位置

本实验最新的 30k post-training 权重不在 GitHub，而在 Hugging Face：

<https://huggingface.co/trshen925/minicpm-gr00t-banana-posttrain-30k>

GitHub 仓库只保存可复现实验所需的代码和配置，避免将数 GB 的 checkpoint、预训练模型和数据文件提交进 Git 历史。
