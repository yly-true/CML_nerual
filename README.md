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

默认训练 300000 步数据、100000 次更新，checkpoint 保存到：

```text
runs\Pendulum_v1\<timestamp>\model_<updates>.pt
```

从已有 checkpoint 继续训练：

```powershell
python train_cml_pendulum.py --task cartpole --device cuda --load-run runs\ContinuousCartPole_v0
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

## 可视化

查看最新 checkpoint 的真实环境和模型预测对比：

```powershell
python visualize_model.py --task pendulum --show
python visualize_model.py --task cartpole --show
```

保存 GIF：

```powershell
python visualize_model.py --output runs\visualizations\pendulum_model.gif
```

## 网络结构

在 `cml_model.py` 顶部修改：

```python
CML_LATENT_DIM = 128
CML_HIDDEN_DIMS = [64, 64]
```

`CML_HIDDEN_DIMS` 中每个数字对应一层隐藏层宽度。
每个隐藏层使用硬件友好的 `Linear -> ReLU`，没有 LayerNorm。
