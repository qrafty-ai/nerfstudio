# Using the Replica dataset

Nerfstudio supports the NICE-SLAM Replica archive through the `replica-data` dataparser.

## Installation
Follow Official installation or pixi:
```bash
direnv allow
pixi install
pixi run post-install
```

## Download and extract

Download the dataset manually, then extract it into `data/Replica`:

```bash
mkdir -p data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -O data/Replica.zip
# or using aria2c for faster download
aria2c -x 16 -s 16 -k 1M --check-certificate=false -o Replica.zip https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip

unzip -q data/Replica.zip -d data
```

After extraction, the dataset root should be `data/Replica/` and contain `cam_params.json` plus scene folders such as `office0/` and `office1/`. Each scene folder should contain `results/` and `traj.txt`.

## Train a scene

Train `office1` with the quality-oriented `splatfacto-big` command:

```bash
ns-train splatfacto-big --max-num-iterations 80000 --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.use_scale_regularization=True replica-data --data data/Replica/office1
```
