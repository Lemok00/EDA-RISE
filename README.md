# EDA-RISE

**An Enhanced Disentanglement Autoencoder for Robust Image Steganography without Embedding**

Xiyao Liu, Ziping Ma, Hui Fang, Jian Zhang, Gerald Schaefer, Kehua Guo, Yuesheng Zhu and Shichao Zhang

This repo provides pre-trained networks of EDA-RISE for evaluation.

## Requirements
+ Only Linux is supported.
+ Ninja >= 10.2, GCC/G++ >= 9.4.0
+ One high-end NVIDIA GPU with at least 11GB of memory. We have done all testing using a NVIDIA RTX 2080Ti.
+ Python >= 3.7 and PyTorch >= 1.6.0.
+ CUDA toolkit 10.2 or later.
+ Python libraries: `pip install lmdb imutils opencv-python pandas tqdm`. We use the Anaconda3 2020.11 distribution which installs most of these by default.

## Download pre-trained models
Please download the pre-trained models of EDA-RISE from Google Drive.

| [Bedroom N=1](https://drive.google.com/file/d/1vAz06Gux-38aefC6h1WgeH6UXkUAzaCv/view?usp=share_link)
| [Bedroom N=2](https://drive.google.com/file/d/1UfVQXp54hLnu-cq-RdnrCc2N9r8wWFZn/view?usp=share_link)
| [Church N=1](https://drive.google.com/file/d/1wcIEhPsYaMhGx9AyrPHD7IsyskRrAXrm/view?usp=share_link) 
| [Church N=2](https://drive.google.com/file/d/1SQOEG0PgHbDgyNEEe53L5PtmcZogu5_0/view?usp=share_link) 
| [FFHQ N=1](https://drive.google.com/file/d/1zaJOek5YRgXVVX_sSsinyvf-UXiU8IPM/view?usp=share_link)
| [FFHQ N=2](https://drive.google.com/file/d/194_kkf0XlGtwEQ9nHFlV4NCOchoOnyNy/view?usp=share_link) |

## Evaluating extraction accuracies
+ The extraction accuracies of EDA-RISE under various attacks can be evaluated using [test_accuracies_under_attacks.py](./test_accuracies_under_attacks.py).
```shell
python test_accuracies_under_attacks.py --checkpoint_path <path of a downloaded model>
```

+ The evaluating results will be saved in `./accuracy_results/` by default, in the form of .csv files like:

|     | Attack Name                | Accuracy     |
|-----|----------------------------|--------------|
| 0   | w/o Attack                 | 1.0          |
| 1   | Gaussian Noise sig. = 0.05 | 0.999709375  |
| ... | ...                        | ...          |
| 63  | Shearing r = 10            | 0.9699296875 |

+ Use `--save_result_dir` to change the directory for saving .csv files.
```shell
python test_accuracies_under_attacks.py --checkpoint_path <path of a downloaded model> --save_result_dir <your saving directory>
```

+ Use `--use_tqdm` to show the evaluating progress through a bar.
```shell
python test_accuracies_under_attacks.py --checkpoint_path <path of a downloaded model> --use_tqdm
```

+ Use `--test_num` to change the number of tests, which is `1000` by default.
```shell
python test_accuracies_under_attacks.py --checkpoint_path <path of a downloaded model> --test_num 10
```

## Evaluating FID scores

### 1. Synthesise container images
+ Synthesise container images using [synthesise_container_image_samples.py](./synthesise_container_image_samples.py).
```shell
python synthesise_container_image_samples.py checkpoint_path <path of a downloaded model>
```
+ The synthesised images are save in `./synthesised_samples/` by default, or you can change the saving directory using `--save_result_dir`.
```shell
python synthesise_container_image_samples.py checkpoint_path <path of a downloaded model> --save_result_dir <your saving directory>
```
+ You can also use `--use_tqdm` to show the synthesising progress.
```shell
python synthesise_container_image_samples.py checkpoint_path <path of a downloaded model> --use_tqdm
```

### 2. Calculate the FID score of synthesised images
+ Use [calculate_fid_score.py](./calculate_fid_score.py) to calculate the FID score for evaluation.
```shell
python calculate_fid_score.py <path of dataset statistics> <directory of synthesised images>
```
+ We provide prepared dataset statistics in `./dataset_statistics/` for calculation. For example, you can evaluate the FID score of the pre-trained Bedroom N=1 model by using the prepared statistics of LSUN Bedroom dataset.
```shell
python calculate_fid_score.py ./synthesised_samples/synthesised_Bedroon_N=1_checkpoint_samples ./dataset_statistics/bedroom.npz
```
+ Or you can prepare the dataset statistics by your self, using [prepare_dataset_statistical_features.py](./prepare_dataset_statistical_features.py).
```shell
python prepare_dataset_statistical_features.py --dataset_path <path of dataset> --dataset_type <type of dataset> --dataset_name <name of dataset>
```
+ Our code support two types of datasets, including `official_lmdb` for lmdb datasets downloaded from LSUN, and `image_folder` for datasets consisting of folders.
+ The prepared dataset statistics will be saved as `./dataset_statistics/<dataset_name>.npz`.


