# CML 虚拟仿真器

这是一个 Neural CML 实验项目。模型学习一步动力学：

```text
当前观测 + 动作 -> 下一观测
```

当前结构：

```text
s = encoder(obs)
delta = action_encoder([obs, action])
s_next = s + delta
obs_next = decoder(s_next)
```

Mecanum 默认使用显式连续动力学：

```text
physical_dot = f(obs, action)
physical_next = physical + dt * physical_dot
```

其中 `physical=[body_vx, body_vy, body_yaw_rate, 4 个轮速]`，`dt=0.02`。

支持任务：

```text
pendulum: Pendulum-v1
cartpole: 连续力输入的经典小车-单杆倒立摆
mecanum: MuJoCo Summit XL 麦轮底盘
```

观测特征：

```text
pendulum: [cos(theta), sin(theta), thetadot]
cartpole: [x, xdot, cos(theta), sin(theta), thetadot]
mecanum: [body_vx, body_vy, body_yaw_rate, 4 个轮速]
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

麦轮底盘使用 `JunHeonYoon/mujoco_mecanum` 的 MuJoCo 模型。默认会查找：

```text
E:\mujoco_mecanum\robots\summit_xl_description\summit_xls.xml
```

如果 clone 到其他位置，设置：

```powershell
$env:MECANUM_MUJOCO_ROOT = "E:\mujoco_mecanum"
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
python -m train.train_cml_pendulum --task mecanum --device cuda
```

可以给 reconstruction loss 指定逐维权重：

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --recon-dim-weights 5.0 1.0 5.0 5.0 1.0
```

默认训练 300000 步数据、100000 次更新，`pred_weight=1.0`、`recon_weight=1.0`、`action_norm_weight=1e-4`、`latent_norm_weight=1e-4`。
CartPole 默认 `recon_weights=[5,1,5,5,1]`。
Mecanum 默认 `recon_weights=[10,10,10,2,2,2,2]`，训练数据会分段保持典型固定动作。

Mecanum 的状态为 7 维：

```text
[body_vx, body_vy, body_yaw_rate, front_right_wheel_speed, front_left_wheel_speed, back_right_wheel_speed, back_left_wheel_speed]
```

Mecanum 的前三维速度都在机体坐标系下表达，不使用世界坐标系速度。

Mecanum 默认使用连续动力学形式：

```text
s_dot = f(s, a)
s_next = s + dt * s_dot
dt = 0.02
```

对应损失为：

```text
step_loss = weighted_mse(s + dt * s_dot_hat, s_next)
s_dot_loss = weighted_mse(s_dot_hat, (s_next - s) / dt)
total = pred_weight * step_loss
      + recon_weight * s_dot_loss
      + action_norm_weight * act_reg
      + latent_norm_weight * state_reg
```

训练日志中，Mecanum 会打印 `total`、`step`、`s_dot`、`act_reg`、`state_reg`；其中 `step` 是一步积分后的状态误差，`s_dot` 是导数误差。

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
Mecanum 新训练默认使用 `s_dot=f(s,a)` 的 `obs_derivative` 模式；旧的 3 维、世界系 7 维、7 维 latent 或 11 维 mecanum checkpoint 不能直接继续使用，需要重新训练。

## 评估

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task pendulum --checkpoint "$ckpt" --episodes 3 --device cuda --render

$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task cartpole --checkpoint "$ckpt" --episodes 3 --device cuda --render

python -m evaluate.evaluate --task mecanum --horizon 10 --device cuda --render --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt
```

评估使用 MPC：每一步用模型预测未来 `horizon` 步，选择最优动作序列，只执行第一个动作。
Mecanum 可以使用多目标速度序列，每个目标默认持续 2 秒，序列默认循环：

```powershell
python -m evaluate.evaluate --task mecanum --horizon 10 --device cuda --render --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --mecanum-target-sequence "[[0.5,0,0],[0,0.5,0],[0,0,0.8],[-0.5,0,0]]"
```

每个目标可以写 3 维 `[body_vx, body_vy, body_yaw_rate]`，也可以写完整 7 维 `[body_vx, body_vy, body_yaw_rate, wheel0, wheel1, wheel2, wheel3]`。如果要改每个目标持续时间：

```powershell
python -m evaluate.evaluate --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --mecanum-target-sequence "[[2.0,0,0],[0,0.5,0]]" --mecanum-target-duration 3.0
```

```powershell
python -m evaluate.evaluate --task cartpole --checkpoint "$ckpt" --planner cem --horizon 10 --num-sequences 2048 --device cuda --render
python -m evaluate.evaluate --task cartpole --checkpoint runs\ContinuousCartPole_v0\20260629_155639\model_200000.pt --planner random --horizon 10 --num-sequences 4096 --device cuda --render --target-cart-position 0.0
```

`random` 和 `cem` 通过采样候选动作序列打分；`gradient` 直接优化整段动作序列，是针对神经网络非线性动力学的 MPC。代价函数为观测/latent 的二次跟踪误差，加动作二次惩罚和可选动作平滑惩罚。
Mecanum 评估日志会额外打印 `step_rmse` 和 `s_dot_rmse`：前者是一点积分后的状态 RMSE，后者是连续动力学导数 RMSE。

## 可视化

查看最新 checkpoint 的真实环境和模型预测对比：

```powershell
python -m visualize.visualize_model --task pendulum --show
python -m visualize.visualize_model --task cartpole --show
python -m visualize.visualize_model --task mecanum --show
python -m visualize.visualize_model --task mecanum --show --steps 300 --action-mode constant --constant-action 0.5 0.5 0.5 0.5
```

指定 checkpoint 可视化：

```powershell
$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m visualize.visualize_model --task cartpole --checkpoint "$ckpt" --show
```

Pendulum 会显示角度、角速度和动作；CartPole 会显示角度、小车位置和动作。
Mecanum 使用 MuJoCo 官方窗口渲染小车运动，同时显示 `[body_vx, body_vy, body_yaw_rate, 4 个轮速]` 的 one-step 预测曲线和 4 维动作曲线。动画左上角会显示 `one_step_error`、`step_rmse`，如果 checkpoint 是 `obs_derivative` 模式，还会显示 `s_dot_rmse`。

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
