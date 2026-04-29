# CML-style neural planner for MuJoCo InvertedPendulum

This is a small PyTorch implementation of a neural extension of the Cognitive Map Learner (CML) idea:

- encode observations into a latent state `s = encoder(o)`
- encode actions as latent displacements `delta = action_encoder(a)`
- predict the next latent state with a residual CML transition:
  `s_next_hat = s + action_encoder(a) + residual(s, a)`
- plan online by sampling candidate action sequences and choosing the one whose predicted latent state moves closest to a target latent state.

The training is self-supervised from transitions `(obs, action, next_obs)` collected by random motor babbling. It does not train on rewards.

## Install

MuJoCo requires a working Gymnasium MuJoCo installation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_cml_inverted_pendulum.py \
  --env-id InvertedPendulum-v4 \
  --total-env-steps 50000 \
  --updates 20000 \
  --device cuda
```

If your Gymnasium version uses `InvertedPendulum-v5`, pass `--env-id InvertedPendulum-v5`.

The script writes checkpoints to `runs/cml_inverted_pendulum/`.

## Evaluate

```bash
python evaluate.py \
  --checkpoint runs/cml_inverted_pendulum/model.pt \
  --env-id InvertedPendulum-v4 \
  --episodes 5 \
  --render
```

## Notes

This is an experimental baseline, not a guaranteed SOTA controller. For InvertedPendulum, a one-step CML utility can be too myopic, so evaluation uses short-horizon latent MPC by default.
