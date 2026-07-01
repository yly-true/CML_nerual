# CML 虚拟仿真器

这是一个 Neural CML 实验项目。模型学习一步动力学：

```text
当前观测 + 动作 -> 下一观测
```

当前结构：

```text
s = encoder(obs)
delta = action_encoder([s, action])
s_next = s + delta
obs_next = decoder(s_next)
```

支持任务：

```text
pendulum: Pendulum-v1
cartpole: 连续力输入的经典小车-单杆倒立摆
bipedalwalker: BipedalWalker-v3
```

观测特征：

```text
pendulum: [cos(theta), sin(theta), thetadot]
cartpole: [x, xdot, cos(theta), sin(theta), thetadot]
bipedalwalker: Gymnasium 原始 24 维观测
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 目录

```text
cml\        核心模型、任务、buffer、工具函数
train\      训练入口
evaluate\   MPC 评估入口
visualize\  一步预测可视化入口
runs\       checkpoint 和可视化结果，可提交到 git
```

## 训练

```powershell
python -m train.train_cml_pendulum --task pendulum --device cuda
python -m train.train_cml_pendulum --task cartpole --device cuda
python -m train.train_cml_pendulum --task bipedalwalker --device cuda
```

可以给 reconstruction loss 指定逐维权重：

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --recon-dim-weights 5.0 1.0 5.0 5.0 1.0
```

默认训练 300000 步数据、100000 次更新，`pred_weight=0.01`、`recon_weight=1.0`、`recon_weights=[5,1,5,5,1]`、`action_norm_weight=1e-4`、`latent_norm_weight=1e-4`。

checkpoint 保存到：

```text
runs\Pendulum_v1\<timestamp>\model_<updates>.pt
```

训练数据从目标平衡点开始采集：Pendulum 竖直向上；CartPole 为 `x=0`、杆竖直向上、速度为 0。之后通过随机动作扰动真实环境生成 transition。
两个任务的离散时间间隔均为 `0.05s`。
观测特征直接进入 replay buffer 和网络，不做均值方差标准化；checkpoint 也不保存 mean/std。

从已有 checkpoint 继续训练：

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --load-run runs\ContinuousCartPole_v0

$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m train.train_cml_pendulum --task cartpole --device cuda --load-run "$ckpt"
```

`--load-run` 可以传 `.pt` 文件，也可以传 run 目录；传目录时会自动选择最新的 `model_*.pt`。
续训时 `--updates` 表示在已有 checkpoint 后继续训练的更新次数。

## 评估

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task pendulum --checkpoint "$ckpt" --episodes 3 --device cuda --render

$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task cartpole --checkpoint "$ckpt" --episodes 3 --device cuda --render

$ckpt = Get-ChildItem runs\BipedalWalker_v3 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task bipedalwalker --checkpoint "$ckpt" --episodes 3 --device cuda --render
```

评估使用 MPC：每一步用模型预测未来 `horizon` 步，选择最优动作序列，只执行第一个动作。

```powershell
python -m evaluate.evaluate --task cartpole --checkpoint "$ckpt" --planner cem --horizon 10 --num-sequences 2048 --device cuda --render
python -m evaluate.evaluate --task cartpole --checkpoint runs\ContinuousCartPole_v0\20260629_155639\model_200000.pt --planner random --horizon 10 --num-sequences 4096 --device cuda --render --target-cart-position 0.0
```

## 可视化

查看最新 checkpoint 的真实环境和模型预测对比：

```powershell
python -m visualize.visualize_model --task pendulum --show
python -m visualize.visualize_model --task cartpole --show
```

指定 checkpoint 可视化：

```powershell
$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m visualize.visualize_model --task cartpole --checkpoint "$ckpt" --show
```

Pendulum 会显示角度、角速度和动作；CartPole 会显示角度、小车位置和动作。

保存 GIF：

```powershell
python -m visualize.visualize_model --task cartpole --checkpoint "$ckpt" --output runs\visualizations\cartpole_model.gif
```

## 网络结构

在 `cml\cml_model.py` 顶部修改：

```python
CML_LATENT_DIM = 16
CML_HIDDEN_DIMS = [64, 64]
CML_NETWORK_TYPE = "mlp"
CML_SNN_TIMESTEPS = 16
CML_SNN_TAU = 2.0
CML_SNN_THRESHOLD = 0.5
```

`CML_HIDDEN_DIMS` 中每个数字对应一层隐藏层宽度。
当前默认配置为 `latent_dim=16`、`hidden_dims=[64, 64]`、`network_type=mlp`。
MLP 每个隐藏层使用 `Linear -> ReLU`，输出层为 Linear。
训练时也可以通过命令行临时切到 SNN：

```powershell
python -m train.train_cml_pendulum --task cartpole --network-type snn --device cuda
```

CUDA 训练默认开启 AMP 混合精度以加速；如需关闭：

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --no-amp
```

CartPole 训练默认加入 kinematic continuity loss，约束 `x` 和杆角度按当前速度连续演化，减少模型预测里的位置闪跳。可通过 `--continuity-weight` 调整，设为 `0` 可关闭。
