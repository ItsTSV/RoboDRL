# RoboDRL: Deep Reinforcement Learning for Continuous Control

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/ItsTSV/robodrl)

This repository contains the source code and results for my thesis. The goal was to implement modern Deep RL algorithms (PPO, TD3, SAC) from scratch and compare them against established libraries like **Stable-Baselines3** and **CleanRL**.

The main focus is on continuous control tasks in **MuJoCo** and **DeepMind Control Suite**.

## Algorithms & Implementation
Each algorithm is **self-contained** in its own directory; the code should be clean, easy to read and well documented.

- [x] **Proximal Policy Optimization (PPO)**
- [x] **Soft Actor-Critic (SAC)**
- [x] **Twin Delayed DDPG (TD3)**

All algorithms were tested on:
* **Gymnasium MuJoCo:** Swimmer, Hopper, HalfCheetah, Walker2D
* **DeepMind Control Suite:** Finger Spin, Reacher, Cartpole Swingup

## Because moving images are more fun than static text

| Hopper-v5  |           Swimmer-v5            |             HalfCheetah-v5               | Walker2D-v5 |
| :---: |:-------------------------------:|:----------------------------------------:| :---: |
| ![Hopper](outputs/hopper.gif) | ![Swimmer](outputs/swimmer.gif) | ![HalfCheetah](outputs/half_cheetah.gif) | ![Walker2D](outputs/walker2d.gif) |

|           Finger Spin            |   Reach    |                 Cartpole Swingup                  |
|:--------------------------------:|:----------:|:-------------------------------------------------:|
| ![Finger Spin](outputs/spin.gif) | ![Reach](outputs/reach.gif) | ![Cartpole Swingup](outputs/cartpole_swingup.gif) |

## Results
In thesis, the models were evaluated against **Stable-Baselines3** and **CleanRL**. Here, only the results of 
my implementations are shown. All models were trained for 1M steps on 3 seeds; evaluation was done using the best 
model across 50 episodes. The results are reported as mean reward ± standard deviation.

### Gymnasium MuJoCo
| Environment |       PPO       | TD3 | SAC |
| :--- |:---------------:| :---: | :---: |
| **Swimmer-v5** |  **344 ± 3.3**  | 37 ± 7.6 | 31 ± 9.9 |
| **Hopper-v5** | **2625 ± 65.1** | 2021 ± 691.1 | 1669 ± 366.0 |
| **HalfCheetah-v5** |  3105 ± 494.7   | **9802 ± 708.4** | 7250 ± 89.0 |
| **Walker2d-v5** |  3930 ± 1585.3  | **5030 ± 1711.6** | 3660 ± 581.1 |

### DeepMind Control Suite
| Environment | PPO | TD3 | SAC |
| :--- | :---: | :---: | :---: |
| **Finger Spin** | 738 ± 51.1 | 907 ± 10.1 | **988 ± 6.4** |
| **Reacher Hard** | 552 ± 469.8 | 905 ± 231.5 | **973 ± 17.9** |
| **Cartpole Swingup** | 341 ± 78.0 | **480 ± 0.3** | 475 ± 0.4 |

> Models, videos and configs can be found on [Hugging Face](https://huggingface.co/collections/ItsTSV/robodrl).

## How to run it?

### Dependencies
The project is written in **Python 3.11**.
It relies on PyTorch (tested with CUDA 12.8), Gymnasium and MuJoCo.

```bash
pip install -r requirements.txt
```
If you have a different CUDA version, you might need to install PyTorch manually.

### Training
To train an agent, run the main script with a config file:
```bash
python -m src.main --config <config-path>
```

### Playground
I included a simple terminal-based tool to visualize trained agents and benchmark their performance:
```bash
python -m src.playground
```
![Terminal App](outputs/terminalapp.png)

### Benchmarking
To train and benchmark SB3 agent, run:
```bash
python -m src.benchmark --env <env-name> --alg <TD3/SAC/PPO>
```

### Hugging Face Upload
To share trained models, use the included utility script. It automatically records a replay video, generates a model card with metadata, and uploads the model + config to the Hub.
You need to be logged in to Hugging Face CLI to use it. You can specify a regex to select which configs to upload.
```bash
python -m src.utils.upload_to_hf --username <your-username> --collection <collection-name> --select <config-regex> 
```

## Code Style
I used `black` and `pylint` to keep the code consistent. Sometimes, it looks weird, but it is what it is.
```bash
black src
pylint src
```
