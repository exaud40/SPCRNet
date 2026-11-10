# Edge-assisted Task-oriented Communication for Collaborative Embodied UAV Inference

**Exaud Tarimo, Huanlai Xing\*, Lexi Xu, Zonghai Zhu, Li Feng, Farhan Hussein, and Ali Kamran**

School of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu, China ·
Research Institute, China United Network Communications Corporation, Beijing, China
(\* corresponding author: hxx@home.swjtu.edu.cn)

This repository contains the code for our manuscript (under review).
## Abstract

Embodied UAVs with on-board inference capabilities are enabling real-time task decisions at the edge, without
relying on continuous transmission of compressed visual data over bandwidth-limited, unreliable UAV links.
However, how to unleash the full potential of a multi-UAV network requires careful coordination among
distributed agents, catering to uneven inference accuracy and heterogeneous communication costs across UAVs and
the edge server. This paper presents a framework for collaborative embodied UAV inference that keeps semantic
latent compact and communication decisions adaptive, while achieving low inference latency and high task
accuracy. The key idea behind the framework is that local inference reliability should govern when and with whom
a UAV communicates, avoiding redundant transmission while uncertain observations are selectively resolved
through collaboration. The framework consists of three main components (i) a lightweight semantic perception
and collaborative reasoning network (SPCRNet) module to extract and compresses visual observations into
task-relevant tokens and inference confidence score, (ii) an adaptive collaboration selector that activates
agent-to-agent semantic exchange or agent-to-ground semantic offloading only when local confidence falls below a
learned threshold and (iii) a WGAE-PPO resource scheduler that jointly learns communication-mode and bandwidth
allocation decisions under a long-term communication-cost constraint. Our code is publicly available at
[github.com/exaud40/SPCRNet](https://github.com/exaud40/SPCRNet).


## Getting Started

Tested with Python 3.10–3.13, PyTorch 2.x with CUDA, on Windows 11 and Linux with one NVIDIA RTX 3080 (10 GB).

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

All commands below are run from the repository root. Outputs go to `caches/`, `runs/`, `runs_perception/`,
`results/`, and `figures/`.

## Inference and Evaluation

There are two independent tracks. **Track A** (controlled benchmark) needs no dataset and **Track B** (MDMT) needs the MDMT dataset.

Runtimes are those measured on our GPU.

### Track A — controlled benchmark

```bash
# 1. benchmark caches (seconds)
python -m uavsc.data.synthetic --out caches/synth

# 2. Runs 4-7: convergence, policy behavior, comparison + sweeps, ablation
python -m uavsc.eval.experiments.run_all --cache caches/synth --seeds 0,1,2,3,4 --parallel 6

# 3. policy-gradient statistics on frozen checkpoints (trains three short runs)
python -m uavsc.eval.experiments.run4b_variance --cache caches/synth
```

`run_all` resumes finished runs, so it can be interrupted and restarted. `--parallel` is the number of training
processes and reduce it if memory is short.



Key numbers to check against our manuscript. WGAE-PPO (κ = 2) reaches 95% of its final QoE in about 117 iterations
versus about 196 for GAE-PPO.At equal QoE, Greedy-Gate needs 21%–38% more energy than the learned policy
(`energy_saving_vs_greedy` in `results/run6.json`).

### Track B — MDMT 

**Step 0 — dataset.** Download MDMT from the
[official repository](https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking) and keep its
folder structure, e.g. `D:\Datasets\MDMT-VisDrone\{train,val,test}\{1,2}\...` with the CVAT XML annotations.

**Step 1 — prepare frames, labels, splits, boxes (minutes).**

```bash
python -m uavsc.data.mdmt stats   --root /path/to/MDMT-VisDrone
python -m uavsc.data.mdmt prepare --root /path/to/MDMT-VisDrone --width 1024 --height 576 --img_dir img1024
python -m uavsc.data.mdmt boxes   --root /path/to/MDMT-VisDrone
python -m uavsc.data.mdmt ceiling          # accuracy reachable from perfect counts 
```

The scene label is the number of distinct targets in the union of both views, quantized into four levels
(boundaries 45, 76, 108). Splits: MDMT train → train + held-out, MDMT val → calibration, MDMT test → test.

**Step 2 — split SPCRNet with scene cross-fitting (about 3.5 h).** Four folds over the 44 sequence pairs; every
scene is predicted by a model that never saw it; calibration of a fold uses only the other folds.

```bash
python -m pytest -m m6                                            # smoke tests of the perception code
python -m uavsc.perception.crossfit_split --smoke                 # end-to-end check
python -m uavsc.perception.crossfit_split --snr_grid=-15,-10,-7.5,-5,-2.5,0,5,10,20
```

Keep the `=` in `--snr_grid=-15,...`. Folds can be run separately (`--fold 0` … `--fold 3`) and pooled with
`--report_only`. Outputs: `runs_perception/crossfit/metrics.json` and the cross-fitted caches in
`caches/mdmt_cf/`.

**Step 3 — what each mode delivers, fidelity functions.**

```bash
python -m uavsc.eval.mode_value --cache caches/mdmt_cf --run runs_perception/crossfit
python -m uavsc.data.mdmt_fidelity --run runs_perception/crossfit --cache caches/mdmt_cf
python -m uavsc.eval.plot_fidelity --report caches/mdmt_cf/fidelity_report.json --out figures/mdmt_fidelity
```

**Step 4 — measured computation and payload → `configs/mdmt.yaml`.**

```bash
python -m uavsc.perception.cost                                  # chi_L, chi_G, k_s; E_bar = 6.2% of Always-A2G
python -m uavsc.eval.operating_point --config configs/mdmt.yaml  # latency / energy / SNR check
```

The onboard and edge throughputs (`--f_U 5e12`, `--f_G 3e13` FLOPS) are stated assumptions. The computation loads
and the payload size are measured on the trained model.

**Step 5 — end-to-end policies on MDMT.**

```bash
python -m uavsc.eval.experiments.run_all --config configs/mdmt.yaml --cache caches/mdmt_cf \
       --seeds 0,1,2,3,4 --parallel 6 --out_dir results_mdmt --fig_dir figures_mdmt
python -m uavsc.eval.experiments.run6_comparison --config configs/mdmt.yaml --cache caches/mdmt_cf \
       --skip_train --tune_split train --out_dir results_mdmt --fig_dir figures_mdmt
python -m uavsc.eval.diag_shift --config configs/mdmt.yaml --cache caches/mdmt_cf --out_dir results_mdmt
```

The second command tunes the threshold baselines on the training sequences, i.e. on the data the policy learns
from. The third evaluates every rule on each split and produces the per-split analysis.


## Citation

```bibtex
@article{tarimo2026edge,
  title   = {Edge-assisted Task-oriented Communication for Collaborative Embodied UAV Inference},
  author  = {Tarimo, Exaud and Xing, Huanlai and Xu, Lexi and Zhu, Zonghai and Feng, Li and
             Hussein, Farhan and Kamran, Ali},
  journal = {under review},
  year    = {2026}
}
```

## Acknowledgments

This work is supported by the Sichuan Science and Technology Program (2024NSFSC1472). The MDMT dataset is
provided by its authors through the VisDrone project; please cite their work when using it.

