# DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection (AAAI 2025)

This repository contains the official implementation for the paper **"DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection"**, accepted by the AAAI Conference on Artificial Intelligence (AAAI) 2025.

## Paper

**DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection**
*Authors: Weihai Lu, Yu Tong, Zhiqiu Ye*
*Conference: AAAI 2025*
*Link:* [https://ojs.aaai.org/index.php/AAAI/article/view/32036](https://ojs.aaai.org/index.php/AAAI/article/view/32036)

## Requirements

The required packages are listed in `requirements.txt`. The setup is based on the environment used for [MMDFND](https://github.com/yutchina/MMDFND/tree/main).

You can install the dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Note: Please ensure you have compatible versions of PyTorch, CUDA, etc., installed based on your system setup and the potential implicit dependencies from MMDFND's original requirements.)*

## Directory Structure

The project is organized as follows:

```text
.
├── Weibo_21                # Processed data for Weibo_21 dataset
│   ├── train_clip_loader.pkl
│   ├── train_loader.pkl
│   ├── train_origin.csv
│   └── ...
├── data                    # Processed data for the primary/weibo dataset 
│   ├── train_clip_loader.pkl
│   ├── train_loader.pkl
│   ├── train_origin.csv
│   └── ...
├── src                     # Source code directory
│   ├── clip_data_pre.py    # CLIP feature preprocessing script
│   ├── data_pre.py         # General data preprocessing script
│   ├── main.py             # Main training and evaluation script
│   ├── model               # Model definitions
│   │   ├── dammfnd.py      # DAMMFND model implementation
│   │   ├── layers.py       # Custom layers
│   │   └── pivot.py        # Pivot mechanism implementation
│   ├── models_mae.py       # MAE model components (if used)
│   ├── param_model         # Saved model parameters/checkpoints
│   │   └── dammfnd
│   ├── pretrained_model    # Directory for downloaded pretrained models
│   │   ├── chinese_roberta_wwm_base_ext_pytorch # Example: RoBERTa model
│   │   └── w2v             # Example: Word2Vec model
│   ├── run.py              # Potential helper script to run experiments
│   ├── util                # Utility functions (MAE related)
│   │   ├── __pycache__
│   │   ├── crop.py
│   │   ├── datasets.py
│   │   ├── lars.py
│   │   ├── lr_decay.py
│   │   ├── lr_sched.py
│   │   ├── misc.py
│   │   └── pos_embed.py
│   ├── utils               # General utility functions and data loaders
│   │   ├── __pycache__
│   │   ├── clip9.py
│   │   ├── clip_dataloader.py
│   │   ├── dataloader.py
│   │   ├── utils.py
│   │   └── weibo21_clip_dataloader.py
│   ├── weibo21_clip_data_pre.py # CLIP preprocessing specific to Weibo_21
│   └── weibo21_data_pre.py    # General preprocessing specific to Weibo_21
├── weibo                   # Raw data files for Weibo dataset (CSV format)
│   ├── test_2_domain.csv
│   ├── train_2_domain.csv
│   └── val_2_domain.csv
└── weibo21                 # Raw data files for Weibo_21 dataset (XLSX format)
    ├── test_2_domain.xlsx
    ├── train_2_domain.xlsx
    └── val_2_domain.xlsx
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

## Data Preparation

Please follow the specific instructions from the [MMDFND repository's Data Preparation section](https://github.com/yutchina/MMDFND/tree/main#data-preparation) for acquiring and initially placing the raw data.

**Note:** If the data linked or described by the MMDFND authors is unavailable, please contact `luweihai@pku.edu.cn` to request the complete dataset.

## Pretrained Models

This project utilizes pretrained models like RoBERTa and potentially Word2Vec or CLIP encoders.

1.  **Download Models:** Please refer to the [MMDFND repository's Pretrained Models section](https://github.com/yutchina/MMDFND/tree/main#pretrained-models) for instructions on downloading the required models.
2.  **Place Models:** Store the downloaded pretrained model files within the `src/pretrained_model/` directory, following the sub-directory structure shown (e.g., `src/pretrained_model/chinese_roberta_wwm_base_ext_pytorch/`).

## Training

To train the DAMMFND model:

1.  Navigate to the source code directory:
    ```bash
    cd src
    ```
2.  Run the main script:
    ```bash
    python main.py
    ```

You can switch between datasets (e.g., the default one using `data/` or the Weibo_21 dataset using `Weibo_21/`) by modifying the relevant dataset path configurations within the `main.py` script or potentially through command-line arguments if implemented. Please inspect `main.py` for details on dataset selection and other hyperparameters.

## Acknowledgements

This codebase is developed based on the implementation of [MMDFND](https://github.com/yutchina/MMDFND/tree/main). We sincerely thank the original authors for sharing their code and groundwork.

## Citation

If you find this work useful for your research, please consider citing our paper:

```
Lu W, Tong Y, Ye Z. DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(1): 559-567.
```

Or in BibTeX format:

```bibtex
@inproceedings{lu2025dammfnd,
  title={DAMMFND: Domain-Aware Multimodal Multi-view Fake News Detection},
  author={Lu, Weihai and Tong, Yu and Zhiqiu, Ye},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={559--567},
  year={2025},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/32036}
}
```
