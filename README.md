# Edge-assisted Task-oriented Communication for Collaborative Embodied Multi-UAV Inference

**Exaud Tarimo, Huanlai Xing\*, Rong Qu, Zhiwen Xiao, Ali Kamran, and Weihua Ma**

School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, China ·
Research Institute, China United Network Communications Corporation, Beijing, China
(\* corresponding author: hxx@home.swjtu.edu.cn)


This repository contains the code for our manuscript under review.

## Getting Started

Tested with Python 3.10–3.13, PyTorch 2.x with CUDA, on Windows 11 and Linux with NVIDIA RTX 3080.

```bash
git clone https://github.com/exaud40/SPCRNet.git
cd SPCRNet
conda create -n uavsc python=3.11 -y
conda activate uavsc
# install PyTorch + torchvision for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -e .
python -m pytest              # all tests should pass and GPU tests run if PyTorch sees a GPU
```

## Inference and Evaluation

### Track A — Multi-UAV benchmark

```bash
python -m uavsc.data.synthetic --out caches/synth
python -m uavsc.eval.experiments.run_all --cache caches/synth --seeds 0,1,2,3,4 --parallel 6
python -m uavsc.eval.experiments.run4b_variance --cache caches/synth
```

`run_all` resumes finished runs, so it can be interrupted and restarted. `--parallel` is the number of training
processes and reduce it if memory is short.


### Track B — MDMT 

**Step 0 — dataset.** Download MDMT from the
[official repository](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking) and keep its
folder structure, e.g. `D:\Datasets\MDMT-VisDrone\{train,val,test}\{1,2}\...` with the CVAT XML annotations.

**Step 1 — prepare frames, labels, splits, boxes.**

```bash
python -m uavsc.data.mdmt stats   --root /path/to/MDMT-VisDrone
python -m uavsc.data.mdmt prepare --root /path/to/MDMT-VisDrone --width 1024 --height 576 --img_dir img1024
python -m uavsc.data.mdmt boxes   --root /path/to/MDMT-VisDrone
python -m uavsc.data.mdmt ceiling           
```

**Step 2 — split SPCRNet with scene cross-fitting.**

```bash
python -m pytest -m m6                                            # smoke tests of the perception code
python -m uavsc.perception.crossfit_split --smoke                 # end-to-end check
python -m uavsc.perception.crossfit_split --snr_grid=-15,-10,-7.5,-5,-2.5,0,5,10,20
```

**Step 3 — what each mode delivers, fidelity functions.**

```bash
python -m uavsc.eval.mode_value --cache caches/mdmt_cf --run runs_perception/crossfit
python -m uavsc.data.mdmt_fidelity --run runs_perception/crossfit --cache caches/mdmt_cf
python -m uavsc.eval.plot_fidelity --report caches/mdmt_cf/fidelity_report.json --out figures/mdmt_fidelity
```

**Step 4 — measured computation and payload → `configs/mdmt.yaml`.**

```bash
python -m uavsc.perception.cost                                  
python -m uavsc.eval.operating_point --config configs/mdmt.yaml  # latency / energy / SNR check
```

**Step 5 — end-to-end policies on MDMT.**

```bash
python -m uavsc.eval.experiments.run_all --config configs/mdmt.yaml --cache caches/mdmt_cf \
       --seeds 0,1,2,3,4 --parallel 6 --out_dir results_mdmt --fig_dir figures_mdmt
python -m uavsc.eval.experiments.run6_comparison --config configs/mdmt.yaml --cache caches/mdmt_cf \
       --skip_train --tune_split train --out_dir results_mdmt --fig_dir figures_mdmt
python -m uavsc.eval.diag_shift --config configs/mdmt.yaml --cache caches/mdmt_cf --out_dir results_mdmt
```

## Citation

```bibtex
@article{tarimo2026edge,
  title   = {Edge-assisted Task-oriented Communication for Collaborative Embodied UAV Inference},
  author  = {Tarimo, Exaud and Xing, Huanlai and Qu, Rong and Xiao, Zhiwen and Kamran, Ali and
             and Ma, Weihua},
  journal = {under review},
  year    = {2026}
}
```

## Acknowledgments

This work was partially supported by the Natural Science Foundation of Sichuan Province (No. 2026NSFSC0433), the Doctoral Innovation Fund Program of Southwest Jiaotong University (No. CX-2025ZD09), and the Fundamental Research Funds for the Central Universities, P. R. China, and the University of Nottingham, UK

