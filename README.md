# CML-style neural planner for MuJoCo InvertedPendulum

This is a small PyTorch implementation of a neural extension of the Cognitive Map Learner (CML) idea:

- encode observations into a latent state `s = encoder(o)`
- encode actions as latent displacements `delta = action_encoder(a)`
- predict the next latent state with a residual CML transition:
  `s_next_hat = s + action_encoder(a) + residual(s, a)`
- plan online by sampling candidate action sequences and choosing the one whose predicted latent state moves closest to a target latent state.

The training is self-supervised from transitions `(obs, action, next_obs)` collected by random motor babbling. It does not train on rewards.

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
python train_cml_inverted_pendulum.py --env-id InvertedPendulum-v4 --total-env-steps 50000 --updates 10000 --device cuda
```

If your Gymnasium version uses `InvertedPendulum-v5`, pass `--env-id InvertedPendulum-v5`.

The script writes checkpoints to `runs\<task>\<timestamp>\`.
For example: `runs\InvertedPendulum_v4\20260429_144020\model_100.pt`.

## Evaluate

```powershell
python evaluate.py --checkpoint runs\InvertedPendulum_v4\20260429_171228\model_10000.pt --env-id InvertedPendulum-v4 --episodes 5 --device cuda --render
```

## Notes

This is an experimental baseline, not a guaranteed SOTA controller. For InvertedPendulum, a one-step CML utility can be too myopic, so evaluation uses short-horizon latent MPC by default.
