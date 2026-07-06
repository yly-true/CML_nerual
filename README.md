# CML Commands

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```powershell
$env:MECANUM_MUJOCO_ROOT = "E:\mujoco_mecanum"
```

## Train MLP

```powershell
python -m train.train_cml_pendulum --task pendulum --device cuda
python -m train.train_cml_pendulum --task cartpole --device cuda
python -m train.train_cml_pendulum --task mecanum --device cuda
```

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --recon-dim-weights 5.0 1.0 5.0 5.0 1.0
python -m train.train_cml_pendulum --task mecanum --device cuda --recon-dim-weights 10 10 20 2 2 2 2
```

## Train SNN

```powershell
python -m train.train_cml_pendulum --task pendulum --network-type snn --snn-timesteps 16 --device cuda
python -m train.train_cml_pendulum --task cartpole --network-type snn --snn-timesteps 16 --device cuda
python -m train.train_cml_pendulum --task mecanum --network-type snn --snn-timesteps 8 --updates 30000 --device cuda
python -m train.train_cml_pendulum --task mecanum --network-type snn --snn-timesteps 16 --updates 100000 --device cuda
```

## Continue Training

```powershell
python -m train.train_cml_pendulum --task cartpole --device cuda --load-run runs\ContinuousCartPole_v0
```

```powershell
$ckpt = Get-ChildItem runs\MecanumDrive_v0 -Recurse -Filter model_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m train.train_cml_pendulum --task mecanum --device cuda --load-run "$ckpt" --updates 50000
```

## Evaluate

```powershell
$ckpt = Get-ChildItem runs\Pendulum_v1 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task pendulum --checkpoint "$ckpt" --episodes 3 --device cuda --render
```

```powershell
$ckpt = Get-ChildItem runs\ContinuousCartPole_v0 -Recurse -Filter model_100000.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
python -m evaluate.evaluate --task cartpole --checkpoint "$ckpt" --episodes 3 --planner cem --horizon 10 --num-sequences 2048 --device cuda --render
```

```powershell
python -m evaluate.evaluate --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --planner gradient --inference-mode obstraj --horizon 10 --action-cost 0.01 --action-smooth-cost 0.05 --device cuda --render
```

```powershell
python -m evaluate.evaluate --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --horizon 10 --device cuda --render --mecanum-target-sequence "[[0.5,0,0],[0,0.5,0],[0,0,0.8],[-0.5,0,0]]"
python -m evaluate.evaluate --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --horizon 10 --device cuda --render --mecanum-target-sequence "[[2.0,0,0],[0,0.5,0]]" --mecanum-target-duration 3.0
```

## Visualize

```powershell
python -m visualize.visualize_model --task pendulum --show
python -m visualize.visualize_model --task cartpole --show
python -m visualize.visualize_model --task mecanum --show
```

```powershell
python -m visualize.visualize_model --task cartpole --checkpoint E:\CML_nerual\runs\ContinuousCartPole_v0\<run>\model_100000.pt --show
python -m visualize.visualize_model --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --show --steps 300 --action-mode constant --constant-action 0.5 0.5 0.5 0.5
python -m visualize.visualize_model --task mecanum --checkpoint E:\CML_nerual\runs\MecanumDrive_v0\<run>\model_100000.pt --show --steps 300 --action-mode constant --constant-torque 5 5 5 5
```

```powershell
python -m visualize.visualize_model --task cartpole --checkpoint E:\CML_nerual\runs\ContinuousCartPole_v0\<run>\model_100000.pt --output runs\visualizations\cartpole_model.gif
```

## Checkpoints

```powershell
Get-ChildItem runs\MecanumDrive_v0 -Recurse -Filter model_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName
```

## Git

```powershell
git status
git add README.md cml\cml_model.py cml\tasks.py train\train_cml_pendulum.py evaluate\evaluate.py
git commit -m "Use direct derivative dynamics"
git push
```
