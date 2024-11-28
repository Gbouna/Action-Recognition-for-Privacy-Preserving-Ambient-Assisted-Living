# Action Recognition for Privacy-Preserving Ambient Assisted Living (AIiH2024)
This is the official repository for **Action Recognition for Privacy-Preserving Ambient Assisted Living**, our paper published at the International Conference on AI in Healthcare (AIiH2024)

The paper can be found [here](https://doi.org/10.1007/978-3-031-67285-9_15)

# Prerequisites
Python >= 3.6

PyTorch >= 1.1.0

PyYAML, tqdm

Run `pip install -e torchlight`
`
# Data Preparation

## Download datasets.

### There are two datasets to download:

NTU RGB+D 60 Skeleton

NW-UCLA

1. **NTU RGB+D 60**: [Download the Skeleton dataset here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NW-UCLA**: [Download the dataset here](https://www.dropbox.com/scl/fi/6numm9wzu1cixw8nyzb91/all_sqe.zip?rlkey=it1ruxtsm4rggxldbbbr4w3yj&e=1&dl=0)

## Data Processing

### Directory Structure

Put downloaded data into the following directory structure.

```
- data/
  - NW-UCLA/
    - all_sqe
      ...
  - ntu/
    - nturgbd_raw/
	  - nturgb+d_skeletons
            ...
```
### Generating Data

**NW-UCLA dataset**

Move folder `all_sqe ` to `./data/NW-UCLA`

**NTU RGB+D 60 dataset**
```
First, extract all skeleton files to ./data/ntu/nturgbd_raw
 cd ./data/ntu
 # Get the skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the centre of the first frame
 python seq_transformation.py
```

# Training

### NTU RGB+D 60 dataset:

For cross-view, run `python main.py --device 0 1 --config ./config/nturgbd-cross-view/default.yaml`

For cross-subject, run `python main.py --device 0 1 --config ./config/nturgbd-cross-subject/default.yaml`

### NW-UCLA dataset:

Run `python main.py --device 0 1 --config ./config/ucla/nw-ucla.yaml`

# Testing

### NTU RGB+D 60 dataset:

For cross-view, run `python main.py --device 0 1 --config ./config/nturgbd-cross-view/default.yaml --phase test --weights path_to_model_weight`

For cross-subject, run `python main.py --device 0 1 --config ./config/nturgbd-cross-subject/default.yaml --phase test --weights path_to_model_weight`

### NW-UCLA dataset:

Run `python main.py --device 0 1 --config ./config/ucla/nw-ucla.yaml --phase test --weights path_to_model_weigh`

# Real-time action recognition in a real-world environment 

[![Action Recognition in a Structured Environment](https://img.youtube.com/vi/7vdGAu3zcCA/0.jpg)](https://www.youtube.com/watch?v=7vdGAu3zcCA)


# Citation
```
@InProceedings{10.1007/978-3-031-67285-9_15,
author="Zakka, Vincent Gbouna
and Dai, Zhuangzhuang
and Manso, Luis J.",
editor="Xie, Xianghua
and Styles, Iain
and Powathil, Gibin
and Ceccarelli, Marco",
title="Action Recognition for Privacy-Preserving Ambient Assisted Living",
booktitle="Artificial Intelligence in Healthcare",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="203--217",
isbn="978-3-031-67285-9"
}
```

