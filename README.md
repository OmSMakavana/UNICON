
##  UNICON Thesis Implementation – Om Makavana (2025)

This repository contains all the code and configuration used for the Master's thesis titled:

**"Evaluating UNICON for Robust Learning under Noisy Labels with KMeans-Based Memory Selection"**  
*University of Otago, 2025*

> 🔬 Reproduces key experiments from the UNICON paper and proposes a KMeans-based alternative to entropy-based memory filtering and few abliations as well as multi seed evaluation

---


---

### 🎓 Paper Reference

This project is based on the paper:

📄 [UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Karim_UniCon_Combating_Label_Noise_Through_Uniform_Selection_and_Contrastive_Learning_CVPR_2022_paper.pdf)  
by Karim et al.

---

### 🔧 Installation Instructions

1. **Create and activate conda environment**
```bash
conda create -n unicon_project python=3.9 -y
conda activate unicon_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

### 📁 Dataset Setup

Manually download and place datasets under `./data/` directory as follows:

- `./data/cifar10`
- `./data/cifar100`
- `./data/tinyimagenet`

📦 Supported datasets:
- CIFAR-10 (Kaggle)
- CIFAR-100 (Kaggle)
- TinyImageNet (Kaggle)

---

###  Training Commands

#### ✅ Reproducing Baseline UNICON

Example (CIFAR-10, 50% symmetric noise):
```bash
python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode sym --r 0.5
```

Example (CIFAR-100, 50% symmetric noise):
```bash
python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode sym --r 0.5
```

---

### KMeans-Based Selection (My Proposed Modification)

To run with **KMeans clustering** for memory selection:

```bash
python Train_cifar.py   --dataset cifar10   --num_class 10   --data_path ./data/cifar10   --noise_mode sym   --r 0.5     --use_kmeans   --kmeans_clusters 100
```

---

### 📊 Logging & Evaluation

- All training logs are saved in `/logs/` and parsed using `wandb` or `.txt` readers.
- Evaluation includes:
  - Multi-seed accuracy plots
  - Boxplots of variability
  - Test/train loss tracking
  - Mean ± std results across seeds

---

### 📦 Software Package Contents (For Submission)

- `/UNICON_cifar`: Main project folder
- `/data/`: Contains datasets (if permitted)
- `/checkpoints/`: includes txt files
- `/scripts/`: SLURM job scripts for Otago Aoraki HPC
- `/wandb/`: Local run summaries
- `Train_cifar.py`: Main training script
- `dataloader_cifar.py`: Data loading and noise injection
- `memory_selector.py`: KMeans memory selection
- `PreResNet_cifar.py`: Model definition

---

### 📚 Citation

If you reference this work, please cite the original UNICON paper:

```bibtex
@InProceedings{Karim_2022_CVPR,
    author    = {Karim, Nazmul and Rizve, Mamshad Nayeem and Rahnavard, Nazanin and Mian, Ajmal and Shah, Mubarak},
    title     = {UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9676-9686}
}
```
