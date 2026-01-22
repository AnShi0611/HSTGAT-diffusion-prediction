# HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network for Multi-Scale Information Diffusion Prediction in Social Networks"**

## 📋 Abstract

HSTGAT is a novel deep learning framework for predicting information diffusion in social networks at multiple scales. The model jointly performs:
- **Macroscopic prediction**: Estimating final cascade size
- **Microscopic prediction**: Predicting the next influenced user

### Key Innovations

1. **Dual-Channel Graph Attention**: Separates local cascade dynamics from global social influence patterns
2. **Multi-Scale Temporal Hierarchy**: Uses dilated causal convolutions to capture patterns at different time scales
3. **Cross-Scale Attention Bridge**: Enables bidirectional knowledge transfer between macro and micro prediction tasks

## 📊 Results

HSTGAT achieves state-of-the-art performance on three benchmark datasets:

| Dataset | MSLE Improvement | Hits@10 Improvement |
|---------|------------------|---------------------|
| Weibo   | 8.7%            | 11.8%               |
| Twitter | 8.6%            | 11.8%               |
| APS     | 8.8%            | 11.8%               |

## 🗂️ Repository Structure

```
HSTGAT-repository/
├── src/
│   ├── model.py          # HSTGAT model architecture
│   ├── train.py          # Training script
│   ├── data_utils.py     # Data loading utilities
│   └── metrics.py        # Evaluation metrics
├── data/                  # Dataset directory (download separately)
├── results/
│   ├── table2_macroscopic_results.csv
│   ├── table3_microscopic_results.csv
│   ├── table4_ablation_study.csv
│   ├── table5_hits_at_k.csv
│   ├── table6_cascade_size_stratified.csv
│   ├── table7_efficiency.csv
│   └── table8_observation_window.csv
├── figures/
│   ├── figure2_parameter_sensitivity.csv
│   ├── figure3_hits_at_k_curves.csv
│   ├── figure4_convergence.csv
│   ├── figure5_attention_visualization.csv
│   ├── figure6_tsne_embeddings.csv
│   └── figure7_case_study.csv
├── models/                # Saved model checkpoints
├── scripts/               # Utility scripts
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/HSTGAT-diffusion-prediction.git
cd HSTGAT-diffusion-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

1. **Weibo Dataset**: Download from [Science Data Bank](https://cstr.cn/31253.11.sciencedb.08172)
2. **Twitter Dataset**: Available from [SNAP](https://snap.stanford.edu/data/)
3. **APS Citation Dataset**: Request from [APS](https://journals.aps.org/datasets)

Place downloaded data in the `data/` directory.

### Training

```bash
# Train on Weibo dataset
python src/train.py --dataset weibo --data_dir ./data/weibo

# Train on Twitter dataset
python src/train.py --dataset twitter --data_dir ./data/twitter

# Train on APS dataset
python src/train.py --dataset aps --data_dir ./data/aps

# Custom hyperparameters
python src/train.py \
    --dataset weibo \
    --embed_dim 128 \
    --hidden_dim 128 \
    --num_gat_layers 2 \
    --num_temporal_layers 4 \
    --num_heads 4 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 100
```

### Evaluation

```bash
python src/train.py --dataset weibo --eval_only --checkpoint ./models/best_model.pt
```

## 📈 Reproducing Paper Results

All experimental results reported in the paper are provided in CSV format:

### Tables
- `results/table2_macroscopic_results.csv`: Macroscopic prediction (MSLE, MAPE)
- `results/table3_microscopic_results.csv`: Microscopic prediction (Hits@10, MAP@10)
- `results/table4_ablation_study.csv`: Ablation study on Weibo
- `results/table5_hits_at_k.csv`: Hits@K for K={5,10,20,50,100}
- `results/table6_cascade_size_stratified.csv`: Performance by cascade size
- `results/table7_efficiency.csv`: Computational efficiency comparison
- `results/table8_observation_window.csv`: Observation window sensitivity

### Figures
- `figures/figure2_parameter_sensitivity.csv`: Hyperparameter sensitivity
- `figures/figure3_hits_at_k_curves.csv`: Hits@K curves
- `figures/figure4_convergence.csv`: Training convergence
- `figures/figure5_attention_visualization.csv`: Attention weight heatmaps
- `figures/figure6_tsne_embeddings.csv`: t-SNE embedding visualization
- `figures/figure7_case_study.csv`: Case study examples

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Embedding dimension |
| `hidden_dim` | 128 | Hidden layer dimension |
| `num_gat_layers` | 2 | Number of GAT layers |
| `num_temporal_layers` | 4 | Number of temporal conv layers |
| `num_heads` | 4 | Number of attention heads |
| `dropout` | 0.1 | Dropout rate |
| `lr` | 0.001 | Learning rate |
| `batch_size` | 64 | Batch size |
| `mu` | 0.5 | Task balance weight |
| `k_hop` | 2 | Ego network hops |

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@article{shi2026hstgat,
  title={HSTGAT: Hierarchical Spatial-Temporal Graph Attention Network for Multi-Scale Information Diffusion Prediction in Social Networks},
  author={Shi, An},
  journal={Cogent Engineering},
  year={2026}
}
```

## 📧 Contact

- **Author**: An Shi
- **Email**: ababa0611@163.com
- **Affiliation**: College of Media and Exhibition, Fujian Business University

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Weibo dataset provided by [Science Data Bank](https://cstr.cn/31253.11.sciencedb.08172)
- Twitter dataset from [Stanford SNAP](https://snap.stanford.edu/data/)
- APS citation data from [American Physical Society](https://journals.aps.org/datasets)
