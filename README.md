# CML-style neural planner for MuJoCo InvertedPendulum

This is a small PyTorch implementation of a neural extension of the Cognitive Map Learner (CML) idea:

- encode observations into a latent state `s = encoder(o)`
- encode actions as latent displacements `delta = action_encoder(a)`
- predict the next latent state with a residual CML transition:
  `s_next_hat = s + action_encoder(a) + residual(s, a)`
- plan online with random-shooting objectives selected by `--inference-mode`: `obsend`, `obstraj`, `latentend`, or `latenttraj`.

The model uses swing-up features `[x, sin(theta), cos(theta), xdot, thetadot]` rather than raw `[x, theta, xdot, thetadot]`, so old 4-D checkpoints are not compatible with the current evaluator.

The training is self-supervised from transitions `(obs, action, next_obs)` collected by full-angle random motor babbling. It does not train on rewards.

Checkpoints are saved under `runs\<task>\<timestamp>\` as `model_100.pt`, `model_200.pt`, ...

## Install

MuJoCo requires a working Gymnasium MuJoCo installation.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python train_cml_inverted_pendulum_v5.py --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --total-env-steps 200000 --buffer-size 200000 --updates 30000 --save-every 5000 --random-cart-pos-range 2.0 --random-cart-vel-range 4.0 --random-pole-ang-vel-range 8.0 --device cuda
```

The repo includes the full official `InvertedPendulum-v5` model file as [inverted_pendulum_v5.xml](E:/创新/cml_inverted_pendulum_pytorch/inverted_pendulum_v5.xml).
The training entrypoint is [train_cml_inverted_pendulum_v5.py](E:/创新/cml_inverted_pendulum_pytorch/train_cml_inverted_pendulum_v5.py).

The script writes checkpoints to `runs\<task>\<timestamp>\`.
For example: `runs\InvertedPendulum_v5\20260429_144020\model_100.pt`.

## Evaluate

Evaluate a swing-up from the downward position. This command finds the newest `model_30000.pt` automatically:

```powershell
$ckpt = Get-ChildItem runs\InvertedPendulum_v5 -Recurse -Filter model_30000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-cart-pos 0.0 --init-pole-angle 3.14159 --init-cart-vel 0.0 --init-pole-ang-vel 0.0 --inference-mode obstraj --target-cart zero --num-sequences 20000 --horizon 40 --action-sampling uniform --obs-cost-weights 0.2 5.0 5.0 0.1 0.5
```

`--render` uses Gymnasium's `human` render backend by default. Evaluation keeps four random-shooting inference modes:

- `obsend`: decoded observation terminal cost
- `obstraj`: accumulated decoded observation trajectory cost
- `latentend`: latent terminal cost
- `latenttraj`: accumulated latent trajectory cost

At every control step, the target feature vector is:

- `x`: `0` by default, or current cart position with `--target-cart current`
- `sin(theta)`: `0`
- `cos(theta)`: `1`
- `xdot`: `0`
- `thetadot`: `0`

You can tune the random-shooting planner:

```powershell
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --inference-mode obstraj --num-sequences 20000 --horizon 40 --action-sampling uniform --obs-cost-weights 0.2 5.0 5.0 0.1 0.5
```

You can still use the MuJoCo passive viewer path:

```powershell
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --render-backend passive --init-pole-angle 3.14159 --inference-mode obstraj --num-sequences 20000 --horizon 40 --action-sampling uniform --obs-cost-weights 0.2 5.0 5.0 0.1 0.5
```

You can switch action sampling:

```powershell
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-pole-angle 3.14159 --inference-mode obstraj --action-sampling gaussian
```

Available values:

- `--action-sampling gaussian`
- `--action-sampling uniform`

Optional observation weights follow the feature order `[x, sin(theta), cos(theta), xdot, thetadot]`:

```powershell
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-pole-angle 3.14159 --inference-mode obstraj --obs-cost-weights 0.2 5.0 5.0 0.1 0.5
```

You can also override the initial observation state from the command line:

```powershell
python evaluate.py --checkpoint $ckpt --env-id InvertedPendulum-v5 --xml-file inverted_pendulum_v5.xml --episodes 1 --device cuda --render --init-cart-pos 0.0 --init-pole-angle 3.14159 --init-cart-vel 0.0 --init-pole-ang-vel 0.0
```

Available initial-state arguments:

- `--init-cart-pos`
- `--init-pole-angle`
- `--init-cart-vel`
- `--init-pole-ang-vel`

## Notes

This is an experimental baseline, not a guaranteed SOTA controller. Swing-up requires a checkpoint trained with full-angle random states and 5-D periodic angle features.
