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
```

观测特征：

```text
pendulum: [cos(theta), sin(theta), thetadot]
cartpole: [x, xdot, cos(theta), sin(theta), thetadot]
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 训练

```powershell
python train_cml_pendulum.py --task pendulum --device cuda
python train_cml_pendulum.py --task cartpole --device cuda
```

可以给 reconstruction loss 指定逐维权重：

```powershell
python train_cml_pendulum.py --task cartpole --device cuda --recon-dim-weights 5.0 1.0 5.0 5.0 1.0
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
python train_cml_pendulum.py --task cartpole --device cuda --load-run runs\ContinuousCartPole_v0

$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python train_cml_pendulum.py --task cartpole --device cuda --load-run "$ckpt"
```

`--load-run` 可以传 `.pt` 文件，也可以传 run 目录；传目录时会自动选择最新的 `model_*.pt`。
续训时 `--updates` 表示在已有 checkpoint 后继续训练的更新次数。

## 评估

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --task pendulum --checkpoint "$ckpt" --episodes 3 --device cuda --render

$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --task cartpole --checkpoint "$ckpt" --episodes 3 --device cuda --render
```

评估使用 MPC：每一步用模型预测未来 `horizon` 步，选择最优动作序列，只执行第一个动作。

```powershell
python evaluate.py --task cartpole --checkpoint "$ckpt" --planner cem --horizon 10 --num-sequences 2048 --device cuda --render
python evaluate.py --task cartpole --checkpoint "$ckpt" --planner random --horizon 10 --num-sequences 4096 --device cuda --render
```

## 可视化

查看最新 checkpoint 的真实环境和模型预测对比：

```powershell
python visualize_model.py --task pendulum --show
python visualize_model.py --task cartpole --show
```

指定 checkpoint 可视化：

```powershell
$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python visualize_model.py --task cartpole --checkpoint "$ckpt" --show
```

Pendulum 会显示角度、角速度和动作；CartPole 会显示角度、小车位置和动作。

保存 GIF：

```powershell
python visualize_model.py --task cartpole --checkpoint "$ckpt" --output runs\visualizations\cartpole_model.gif
```

## 网络结构

在 `cml_model.py` 顶部修改：

```python
CML_LATENT_DIM = 512
CML_HIDDEN_DIMS = [16, 64, 64]
```

`CML_HIDDEN_DIMS` 中每个数字对应一层隐藏层宽度。
每个隐藏层使用硬件友好的 `Linear -> ReLU`，没有 LayerNorm。
