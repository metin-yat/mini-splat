# 3DGS From-Scratch: Differentiable Rasterizer Development

This project implements a 3D Gaussian Splatting (3DGS) engine using Python and PyTorch. It moves beyond high-level abstractions to implement the differentiable rendering pipeline and volumetric accumulation from first principles. This implementation serves as a pedagogical deep-dive into the mechanics of radiance fields. By reconstructing the 3DGS pipeline from first principles, the objective was to move beyond the abstractions provided by production-ready frameworks and gain a granular, "under-the-hood" understanding of differentiable rendering, gradient-based parameter optimization, and the complex mathematical transformations required for screen-space projection.

---

## Installation

Quick steps to create a conda environment from the repository `environment.yml` and run a simple training example.

1) Create and activate the conda environment from `environment.yml`:

```bash
# run from the repository root
conda env create -f environment.yml 
conda activate gsplat
```

2) Start training:

```bash
python train.py
```

## Optimization & Loss Objectives

The system optimizes Gaussian parameters by minimizing the discrepancy between rendered views and ground-truth images. The training relies on a composite loss function:

$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}
$$

- **\(\mathcal{L}_1\) (Mean Absolute Error):** Ensures pixel-level color accuracy.  
- **\(\mathcal{L}_{D-SSIM}\):** Maintains structural integrity and high-frequency textures, preventing the model from producing "smeared" geometry.



## Training Intuition: Differentiable Feedback

The core of the learning process is the "Painter-Critic" loop, where every rendering decision is mathematically traceable:

- **Forward Pass:** The virtual camera snapshots the current Gaussian swarm. Each Gaussian is projected to 2D using a Jacobian matrix (\(J\)) to linearize the depth transformation.

- **Volumetric Integration:** Pixels are colored using alpha-blending, where each Gaussian's contribution is weighted by its density and the cumulative transparency (Transmittance) of the points in front of it.

- **Backpropagation:** Since the rasterizer is fully differentiable, the loss is backpropagated to adjust specific physical attributes (spatial alignment, morphological scaling, and photometric coloring).



## Performance & Severe Hardware Constraints

![Example Output](https://github.com/metin-yat/mini-splat/blob/main/final_result.png)

Due to significant hardware limitations and the high computational overhead of a non-optimized Python/PyTorch rasterizer, this demonstration utilizes a sparse set of only **500 Gaussians**. Consequently, the rendered images will appear significantly blurry and "cloudy." This is a direct result of under-parameterization and the inability to run the high-resolution, high-iteration training cycles required to resolve fine details. The system is currently capped by consumer-grade VRAM limits and the interpreter-bound nature of the custom rendering loop, preventing the dense point-cloud representation typical of production-level 3DGS.

---

## Future Roadmap

- **CUDA Kernels:** Transitioning from PyTorch loops to C++/CUDA for real-time rasterization and efficient memory management.

- **Spherical Harmonics:** Moving beyond fixed RGB to model view-dependent specularities. The current baseline utilizes 0th-degree Spherical Harmonics, effectively storing direct color values without accounting for directional lighting changes.

- **Adaptive Density Control:** Implementing automated splitting and pruning of Gaussians based on gradient magnitude to optimize VRAM usage.

---

> This project explores the intersection of traditional graphics and gradient-based scene reconstruction. This implementation served as a foundational      exploration and a "from-scratch" proof of concept. Moving forward, I will transition to utilizing state-of-the-art published repositories that leverage high-performance CUDA kernels and Dockerized environments to achieve professional-grade reconstruction quality and real-time performance.
