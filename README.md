# Pendulum CML 虚拟仿真器

这是一个只针对 Gymnasium `Pendulum-v1` 的 Neural CML 实验项目。模型学习一步动力学：

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

Pendulum 观测为：

```text
[cos(theta), sin(theta), thetadot]
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 训练

```powershell
python train_cml_pendulum.py --device cuda
```

默认训练 300000 步数据、50000 次更新，checkpoint 保存到：

```text
runs\Pendulum_v1\<timestamp>\model_<updates>.pt
```

## 评估

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_50000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint "$ckpt" --episodes 3 --device cuda --render
```

## 可视化

查看最新 checkpoint 的真实环境和模型预测对比：

```powershell
python visualize_model.py --show
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
