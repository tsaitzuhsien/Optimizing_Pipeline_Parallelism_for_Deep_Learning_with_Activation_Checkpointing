# Optimizing Pipeline Parallelism for Deep Learning With Activation Checkpointing

This repository contains the DNN training programs that applies the model partitioning and activation checkpointing algorithms in the paper **Optimizing Pipeline Parallelism for Deep Learning With Activation Checkpointing**, published at *Concurrency and Computation: Practice and Experience*, 2026. [Paper](https://onlinelibrary.wiley.com/doi/10.1002/cpe.70975)

Please follow these steps to setup the experimental environment.

## Fetch the Programs
The code can be downloaded using git clone:
```
git clone git@github.com:tsaitzuhsien/Optimizing-Pipeline-Parallelism-for-Deep-Learning-With-Activation-Checkpointing.git
```

## Setup the Environment

1.  **Create a Conda Environment:**
    Create a new Conda environment named 'test' with Python version 3.11.11.
    ```bash
    conda create --name test python=3.11.11
    ```

2.  **Activate the Environment:**
    Activate the newly created environment.
    ```bash
    conda activate test
    ```

3.  **Install PyTorch:**
    Install PyTorch, TorchVision, and TorchAudio with CUDA support for version 12.8.
    ```bash
    pip3 install torch==2.7.0 torchvision==0.22.0+cu128 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    ```
    > Note: This command assumes you have an NVIDIA GPU and CUDA 12.8 installed.

4.  **Install Dependencies:**
    Install the remaining required libraries using Conda.
    ```bash
    conda install pybind11 pyyaml tqdm tiktoken datasets
    ```

5. **Prepare Data and Compile**
    Run the shell script `prepare_data_and_compile.sh` to fetch data and compile the algorithm.
    ```bash
    ./prepare_data_and_compile.sh
    ```
    
6. **Prepare ImageNet**
    The experiments train ResNet and ConvNeXt using ImageNet.
    The profiling and the training processes fetch data from the directory "/data/imagenet", so please put the ImageNet dataset over here.

## Reproduce the Experiments

The following scripts profile, partition, and train the models using our algorithms.
The throughput is printed to stdout.

| Script Name | Partitioning Algorithm | Model |
| :--- | :--- | :--- |
| `train_rn.sh` | Dynamic Programming | ResNet |
| `train_convnext.sh` | Dynamic Programming | ConvNeXt |
| `train_gpt2.sh` | Dynamic Programming | GPT-2 |

### Configuration
You can customize experimental settings by modifying the variables at the top of the scripts.
Their meanings are as follows.

- `world_size`: The number of GPUs to use for training.
- `device_memory`: The available memory per GPU in gigabytes (GB).
- `num_microbatches`: The number of microbatches the minibatch is split into.
- `batch_size`: The minibatch size.
- `model`: The name of the model to train.
	- ResNet: resnet50x8, resnet74x8, resnet101x8, resnet125x8, resnet152x8, and resnet176x8
	- ConvNeXt: convnext\_tiny, convnext\_mini, convnext\_small, convnext\_large
	- GPT2: gpt2 and gpt2\_xl

### Memory Usage
To record the memory usage during model training, please uncomment the instructions `torch.cuda.memory._record_memory_history()` and `torch.cuda.memory._dump_snapshot(f"mem_viz_log/{rank}.pickle")` in the `src/train_{model}.py`.

## Reference
```bibtex
@article{https://doi.org/10.1002/cpe.70975,
author = {Tsai, Tzu-Hsien and Chiang, Ming-Yen and Hong, Ding-Yong and Liu, Pangfeng and Wu, Jan-Jan},
title = {Optimizing Pipeline Parallelism for Deep Learning With Activation Checkpointing},
journal = {Concurrency and Computation: Practice and Experience},
volume = {38},
number = {19},
pages = {e70975},
doi = {https://doi.org/10.1002/cpe.70975},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.70975},
year = {2026}
}
```
