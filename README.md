# CML Neural Planner for Gymnasium Control Tasks

这是一个用于 Gymnasium 连续控制任务的 CML-style neural planner。当前支持：

- `inverted_pendulum`: `InvertedPendulum-v5`
- `mountain_car_continuous`: `MountainCarContinuous-v0`
- `pendulum`: `Pendulum-v1`

模型把观测编码到 latent state，把动作编码成 latent 位移，并用残差项预测下一状态：

```text
s = encoder(obs)
s_next = s + action_encoder(action) + residual(s, action)
```

训练使用随机采集的 `(obs, action, next_obs)` 转移数据，不使用 reward。不同任务使用不同观测特征：

```text
inverted_pendulum:        [x, sin(theta), cos(theta), xdot, thetadot]
mountain_car_continuous:  [position, velocity]
pendulum:                 [cos(theta), sin(theta), thetadot]
```

不同任务的 checkpoint 不能混用。

## Install

`MountainCarContinuous-v0` 和 `Pendulum-v1` 只需要 Gymnasium classic-control 依赖；`InvertedPendulum-v5` 需要可用的 MuJoCo 依赖。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

InvertedPendulum-v5:

```powershell
python train_cml_inverted_pendulum_v5.py --task inverted_pendulum --total-env-steps 200000 --buffer-size 200000 --updates 30000 --save-every 5000 --random-cart-pos-range 2.0 --random-cart-vel-range 4.0 --random-pole-ang-vel-range 8.0 --device cuda
```

MountainCarContinuous-v0:

```powershell
python train_cml_inverted_pendulum_v5.py --task mountain_car_continuous --total-env-steps 100000 --buffer-size 100000 --updates 20000 --save-every 5000 --random-mountain-car-pos-low -1.2 --random-mountain-car-pos-high 0.6 --random-mountain-car-vel-range 0.07 --device cuda
```

Pendulum-v1:

```powershell
python train_cml_inverted_pendulum_v5.py --task pendulum --total-env-steps 100000 --buffer-size 100000 --updates 20000 --save-every 5000 --random-pendulum-theta-range 3.14159 --random-pendulum-ang-vel-range 8.0 --device cuda
```

默认模型参数：

```text
latent_dim = 64
hidden_dim = 16
depth = 2
updates = 30000
```

训练开始后会打印当前模型大小，例如参数量和参数内存。checkpoint 默认保存到：

```text
runs\<EnvName>\<timestamp>\model_<updates>.pt
```

## Evaluate

InvertedPendulum-v5:

```powershell
$ckpt = Get-ChildItem runs\InvertedPendulum_v5 -Recurse -Filter model_30000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint "$ckpt" --task inverted_pendulum --episodes 1 --device cuda --render --init-cart-pos 0.0 --init-pole-angle 3.14 --init-cart-vel 0.0 --init-pole-ang-vel 0.0 --inference-mode latentend --target-cart zero --action-std 2.0 --num-sequences 20000 --horizon 10 --obs-cost-weights 0.2 10.0 10.0 0.1 0.5
```

MountainCarContinuous-v0:

```powershell
$ckpt = Get-ChildItem runs\MountainCarContinuous_v0 -Recurse -Filter model_30000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint "$ckpt" --task mountain_car_continuous --episodes 3 --device cuda --render --inference-mode obsend --target-position 0.45 --target-velocity 0.0 --action-std 2.0 --num-sequences 20000 --horizon 40 --obs-cost-weights 10.0 0.0
```

Pendulum-v1:

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_20000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint "$ckpt" --task pendulum --episodes 3 --device cuda --render --inference-mode latentend --target-angular-velocity 0.0 --action-std 0.3 --num-sequences 40000 --horizon 10 --obs-cost-weights 10.0 10.0 0.0
```

评估加载 checkpoint 后也会打印模型大小，方便确认当前使用的是哪个结构。

## Key Options

- `--inference-mode`: `obsend`, `obstraj`, `latentend`, `latenttraj`
- `--task`: `inverted_pendulum`, `mountain_car_continuous`, `pendulum`，或 `auto`
- `--action-sampling`: `uniform` 或 `gaussian`
- `--action-std`: 高斯动作采样的标准差；可传 1 个数给所有动作维度，或传 `action_dim` 个数；不传时默认使用动作范围的 `1/4`
- `--num-sequences`: random shooting 候选动作序列数量
- `--horizon`: 每条候选动作序列长度
- `--obs-cost-weights`: 按当前任务的 feature 顺序设置观测代价权重
- `--target-cart`: 仅倒立摆使用；`sine` 让目标车位置按正弦随时间振荡，也可用 `zero` 或 `current`
- `--target-x-amplitude`: 正弦目标幅值，默认 `1.0`
- `--target-x-frequency`: 正弦目标频率，单位 Hz，默认 `0.1`
- `--target-x-phase`: 正弦目标相位，单位 rad，默认 `0.0`
- `--target-x-offset`: 正弦目标偏置，默认 `0.0`
- `--target-position`: 仅 MountainCar 使用，默认 `0.45`
- `--target-velocity`: 仅 MountainCar 使用，默认 `0.0`
- `--target-angular-velocity`: 仅 Pendulum 使用，默认 `0.0`
- `--random-mountain-car-pos-low/high`: MountainCar 训练初始位置随机范围，默认 `[-1.2, 0.6]`
- `--random-mountain-car-vel-range`: MountainCar 训练初始速度随机范围，默认 `[-0.07, 0.07]`
- `--random-pendulum-theta-range`: Pendulum 训练初始角度随机范围，默认 `[-pi, pi]`
- `--random-pendulum-ang-vel-range`: Pendulum 训练初始角速度随机范围，默认 `[-8.0, 8.0]`

倒立摆常用权重：

```powershell
--obs-cost-weights 0.2 10.0 10.0 0.1 0.5
```

如果小车跑太远，提高第一个权重；如果杆子到达竖直附近但不稳定，提高最后一个权重。

## Notes

这是实验性 baseline。控制效果依赖 checkpoint 质量、random shooting 参数和观测权重。
