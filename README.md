# CML Neural Planner for InvertedPendulum-v5

这是一个用于 MuJoCo `InvertedPendulum-v5` 的 CML-style neural planner。

模型把观测编码到 latent state，把动作编码成 latent 位移，并用残差项预测下一状态：

```text
s = encoder(obs)
s_next = s + action_encoder(action) + residual(s, action)
```

训练使用随机采集的 `(obs, action, next_obs)` 转移数据，不使用 reward。当前观测特征为：

```text
[x, sin(theta), cos(theta), xdot, thetadot]
```

旧的 4 维 checkpoint 不能直接用于当前评估脚本。

## Install

需要可用的 Gymnasium MuJoCo 环境。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python train_cml_inverted_pendulum_v5.py --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --total-env-steps 200000 --buffer-size 200000 --updates 30000 --save-every 5000 --random-cart-pos-range 2.0 --random-cart-vel-range 4.0 --random-pole-ang-vel-range 8.0 --device cuda
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
runs\InvertedPendulum_v5\<timestamp>\model_30000.pt
```

## Evaluate

Windows PowerShell：

```powershell
$ckpt = Get-ChildItem runs\InvertedPendulum_v5 -Recurse -Filter model_30000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint "$ckpt" --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-cart-pos 0.0 --init-pole-angle 0 --init-cart-vel 0.0 --init-pole-ang-vel 0.0 --inference-mode obsend --target-cart sine --target-x-amplitude 1.0 --target-x-frequency 0.6 --num-sequences 40000 --horizon 10 --obs-cost-weights 1.0 0.5 0.5 0.2 0.2
```

Linux/macOS bash：

```bash
ckpt=$(find runs/InvertedPendulum_v5 -name 'model_30000.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
python evaluate.py --checkpoint "$ckpt" --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-cart-pos 0.0 --init-pole-angle 3.14159 --init-cart-vel 0.0 --init-pole-ang-vel 0.0 --inference-mode obsend --target-cart sine --target-x-amplitude 1.0 --target-x-frequency 0.1 --num-sequences 20000 --horizon 10 --obs-cost-weights 0.2 10.0 10.0 0.1 0.5
```

评估加载 checkpoint 后也会打印模型大小，方便确认当前使用的是哪个结构。

## Key Options

- `--inference-mode`: `obsend`, `obstraj`, `latentend`, `latenttraj`
- `--action-sampling`: `uniform` 或 `gaussian`
- `--num-sequences`: random shooting 候选动作序列数量
- `--horizon`: 每条候选动作序列长度
- `--obs-cost-weights`: 按 `[x, sin(theta), cos(theta), xdot, thetadot]` 设置观测代价权重
- `--target-cart`: `sine` 让目标车位置按正弦随时间振荡；也可用 `zero` 固定为 0，或 `current` 使用当前车位置
- `--target-x-amplitude`: 正弦目标幅值，默认 `1.0`
- `--target-x-frequency`: 正弦目标频率，单位 Hz，默认 `0.1`
- `--target-x-phase`: 正弦目标相位，单位 rad，默认 `0.0`
- `--target-x-offset`: 正弦目标偏置，默认 `0.0`

常用权重：

```powershell
--obs-cost-weights 0.2 10.0 10.0 0.1 0.5
```

如果小车跑太远，提高第一个权重；如果杆子到达竖直附近但不稳定，提高最后一个权重。

## Notes

这是实验性 baseline。摆起控制效果依赖 checkpoint 质量、random shooting 参数和观测权重。
