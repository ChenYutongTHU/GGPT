# GGPT: Geometry-grounded Point Transformer

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](#)
[![Arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2603.11174)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://chenyutongthu.github.io/research/ggpt)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/YutongGoose/GGPT)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green)](https://huggingface.co/datasets/YutongGoose/GGPT_eval)



Official PyTorch implementation of **Geometry-grounded Point Transformer (GGPT)** (CVPR 2026). GGPT is a method for high-quality 3D reconstruction from multiview images. For more details, please visit our [project webpage](https://chenyutongthu.github.io/research/ggpt).

---


## 🛠️ 1. Installation

### 1.1 Clone the repository
```bash
git clone --recursive https://github.com/chenyutongthu/GGPT.git
cd GGPT
```


### 1.2 Install dependencies

#### a. Create a virtual environment.
#### b. Install torch, torchvision for your CUDA version. (The environment does not require specific CUDA or pytorch version. It has been tested in CUDA 12.1/12.3 and torch 2.2.0/2.5.1.)

#### c. Install requirements for VGGT and SfM.
```
pip install -r requirements_sfm.txt
# Choose which matcher to use.
# For RoMaV2
cd RoMaV2/ && pip install -e .
# For RoMaV1
pip install fused-local-corr>=0.2.2
cd RoMa/ && pip install -e .
```

#### d. \[Optional\] If you need to run GGPT, the 3D point transformer, please follow [the script](ptv3_env.sh) install the following packages in the same virtual environement. You don't need to build another env for this.

#### e. Download our pretrained GGPT checkpoint directly [here](https://huggingface.co/YutongGoose/GGPT) and place it in [ckpts/].

---

## 2. 📖 Usage & Examples


```bash

python run_demo.py image_dir=/path/to/your/images

```

The outputs (including the feedforward points, SfM points, and the final GGPT points `ggpt_points.ply`) will be saved in the `outputs/demo/` directory by default.


Or you can add ```common_config.ggpt_refine=False``` disable GGPT refinement and run SfM to only obtain the sparse reconstruction.


### ⚙️ Configuration Settings (SfM)

You can adjust Structure-from-Motion (SfM) configuration blocks in the `.yaml` files (found under `configs/`) to better suit your data. 

```yaml
ba_config:
  shared_camera: True # Set it to False if images are captured with different camera intrinsics
  camera_type: SIMPLE_PINHOLE # Set it to PINHOLE if the images can have very different fx and fy.

dlt_config:
  # Adjust the filtering parameters if you need more accurate yet sparser SfM points. E.g.:
  score_thresh: 0.1       # Increase the matching confidence threshold to filter out noisy matchings
  cycle_err_thresh: 4     # Reduce the cycle error threshold to filter out noisy tracks
  max_epipolar_error: 4   # Reduce the epipolar error threshold to filter out noisy tracks
  min_tri_angle: 3        # Increase the triangulation angle threshold to filter out points with low parallax
  max_reproj_error: 4     # Reduce the reprojection error threshold to filter out noisy points
```


### 🎬 Large number of input views

When working with a large number of input views (e.g., > 50), use the following configuration to prevent out-of-memory (OOM) errors. Additionally, for denser view sets, we recommend applying stricter filtering thresholds during the SfM process. 

```
python run_demo.py \
    common_config.reduce_memory=True match_config.models=romav2-fast \
    ba_config.score_thresh=0.8 ba_config.cycle_err_thresh=1 ba_config.mintrack_per_view=512 \
    dlt_config.cycle_err_thresh=1 dlt_config.max_reproj_error=1 dlt_config.min_tri_angle=7 \
    hydra.run.dir=outputs/demo/ image_dir=/path
```

Note on Scalability: The current pipeline performs exhaustive matching, which leads to longer runtimes for large-scale or dense view reconstructions. For better efficiency in these scenarios, we recommend integrating sparse graph building.

### Trouble shooting.

* If you run into this issue when running RoMaV2 `raise RuntimeError("Float32 matmul precision must be set to highest")`,  manually add `torch.set_float32_matmul_precision("highest")` before the warning line.

* If you run into this issue when running DLT triangulation `torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE`, reduce `dlt_config.batch_size` to a smaller value, e.g. 50000. This is [a torch-version-specific issue](https://github.com/pytorch/pytorch/issues/166004).

---

## 3. 📊 Evaluation


### 3.1 Evaluation on our benchmark

Download the [preprocessed evaluation set](https://huggingface.co/datasets/YutongGoose/GGPT_eval) and place it at the root of this project as `GGPT_eval`. Run `sh benchmark_eval.sh`

### 3.2 Evaluation on your custom data

You can evaluate GGPT on your own datasets by organizing your data into a `custom_eval_set` or by modifying the dataloader.

**a. Organize your data directory**

Prepare your dataset following this directory structure:

```text
custom_eval_set/
└── seq0/
    ├── depths/                  # Ground truth depth maps
    │   ├── 000000.npy           # (H, W) float32 array
    │   └── ...
    ├── images/                  # Input images
    │   ├── 000000.jpg           # (H, W, 3) RGB image
    │   └── ... 
    ├── extrinsics.npy           # (N, 4, 4) world-to-camera matrices (OpenCV convention)
    └── intrinsics.npy           # (N, 3, 3) linear camera intrinsics (top-left pixel center is (0,0))
```

**b. Update dataset configurations**

Add your custom dataset configuration to the evaluation YAML file (e.g., `configs/benchmark_sfm.yaml`):

```yaml
valdataset_configs:
  - _target_: sfm.dataloader.extracted.ExtractedDataset
    name: custom_eval_set
    root: PATH/TO/custom_eval_set
```

**c. Run the evaluation**

First, run the Structure-from-Motion (SfM) pipeline to extract the sparse point clouds:

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv-endpoint=localhost:2026 \
    sfm/run_benchmark_sfm.py \
    match_config.models='romav2-base' \
    feedforward_config.model='vggt-point' \
    hydra.run.dir=outputs/custom_eval/sfm_vggt-point_romav2
```

Then, run the GGPT pipeline using the output from the SfM step:

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
    --rdzv-endpoint=localhost:2026 \
    ggpt/run_benchmark_ggpt.py \
    valdataset_configs.data_dict.custom_eval_set=outputs/custom_eval/sfm_vggt-point_romav2/save/custom_eval_set \
    hydra.run.dir=outputs/custom_eval/ggpt_vggt-point_romav2
```
## 💡 Limitations

While our work demonstrates that geometry optimization provides meaningful constraints that significantly improve feed-forward 3D reconstruction, our current method can still fail under specific conditions. For an in-depth discussion, please refer to the limitations section in our paper's appendix.

* **Ill-posed Geometry:**  extremely low overlap or parallax (distant tree in outdoor scenes), in-the-wild photos captured under significantly different lightings, ambiguous and misleading symmetric structures, and dynamic scenes.
* **GGPT Refinement:** GGPT currently faces challenges with long-range extrapolation in monocular regions and can occasionally produce patchify artifacts in large-scale scenes.

We invite you to share failure cases via GitHub Issues to help guide future research!


## License

We release the GGPT code and point transformer model weights under the MIT License.

GGPT builds on several third-party components, including RoMa (RoMaV2), VGGT, Pi3X, and DepthAnything3. These components are subject to their own licences, which may include additional restrictions (e.g. research-only use for certain model weights).

Users are responsible for complying with the respective licences of these dependencies when using GGPT. Please refer to the original repositories for detailed licensing terms.



## 📖 Citing

If you find our work useful, please cite:

```bibtex
@inproceedings{chen2026ggpt,
  title={GGPT: Geometry-Grounded Point Transformer},
  author={Chen, Yutong and Wang, Yiming and Zhang, Xucong and Prokudin, Sergey and Tang, Siyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
