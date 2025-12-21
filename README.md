<h2 align="center">
  <a href="https://github.com/PotatoBigRoom/CoherentGS">Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views</a>
</h2>
<h5 align="center">
If you find this work useful, please consider starring the repo ⭐ to stay updated.
</h5>
<p align="center">
<a href="https://arxiv.org/abs/2512.10369"><img src="https://img.shields.io/badge/Arxiv-2512.10369-b31b1b.svg?logo=arXiv" alt="arXiv"></a> -->
<a href="https://github.com/PotatoBigRoom/CoherentGS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT"></a>
<a href="https://github.com/PotatoBigRoom/CoherentGS/stargazers"><img src="https://img.shields.io/github/stars/PotatoBigRoom/CoherentGS?style=flat&logo=github&logoColor=whitesmoke&label=Stars" alt="GitHub stars"></a>
<a href="https://github.com/PotatoBigRoom/CoherentGS/network"><img src="https://img.shields.io/github/forks/PotatoBigRoom/CoherentGS?style=flat&logo=github&logoColor=whitesmoke&label=Forks" alt="GitHub forks"></a>
<a href="https://github.com/PotatoBigRoom/CoherentGS/watchers"><img src="https://img.shields.io/github/watchers/PotatoBigRoom/CoherentGS?style=flat&logo=github&logoColor=whitesmoke&label=Watchers" alt="GitHub watchers"></a>
</p>

CoherentGS tackles one of the hardest regimes for 3D Gaussian Splatting (3DGS): Sparse inputs with severe motion blur. We break the "vicious cycle" between missing viewpoints and degraded photometry by coupling a physics-aware deblurring prior with diffusion-driven geometry completion, enabling coherent, high-frequency reconstructions from as few as 3–9 views on both synthetic and real scenes.

<p align="center">
  <img src="docs/static/images/pipeline.jpg" alt="CoherentGS overview" width="90%">
</p>


## 🍭 News
- 2025-11-28: Project page released at https://github.com/PotatoBigRoom/CoherentGS.
- Initial code, data links, and training scripts are now public.

## 🚀 Getting Started
Tested with Python 3.10 and PyTorch 2.1.2 (CUDA 11.8). Adjust CUDA wheels as needed for your platform.

### 🦄 Installation
```bash
# (Optional) fresh conda env
conda create --name CoherentGS -y "python<3.11"
conda activate CoherentGS

# Install dependencies
pip install --upgrade pip setuptools
pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 🎥 Data
- DL3DV-Blur and related assets: https://huggingface.co/datasets/Passwerob/CoherentGS/tree/main  
- Additional samples and preprocessed splits: https://huggingface.co/datasets/Passwerob/CoherentGS/tree/main  
Place downloaded data under `datasets/` (or adjust paths in the provided scripts).

### 🌈 Training
Train on DL3DV-Blur (full resolution) with:
```bash
bash run_dl3dv.sh
```
For custom settings, start from `run.sh` and tweak dataset paths, resolution, and batch sizes.

## 🔗 Citation
If CoherentGS supports your research, please cite:
```bibtex
@article{feng2025coherentgs,
  author    = {Feng, Chaoran and Xu, Zhankuo and Li, Yingtao and Zhao, Jianbin and Yang, Jiashu and Yu, Wangbo and Yuan, Li and Tian, Yonghong},
  title     = {Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views},
  year      = {2025},
}
```

## 📍 Acknowledgements
This repo builds on outstanding open-source efforts, in particular [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main), [BAD-Gaussians](https://github.com/WU-CVGL/BAD-Gaussians), [DIFIX3D+](https://github.com/nv-tlabs/Difix3D). We thank the original authors for making their code available.

---

For issues or reproducibility questions, please open a GitHub issue. PRs that improve robustness, data loading, or evaluation scripts are welcome.
