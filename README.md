<h1 align="center">
Design and Evaluation of a Federated Learning-Based Approach for Anomaly Detection in Industrial Control Systems
</h1>

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
    - [Virtual Environment](#virtual-environment)
    - [Data Processing](#data-processing)
    - [Training](#training)
- [Contacts](#contacts)

## Overview

Industrial Control Systems face increasingly advanced cyber threats, yet many traditional intrusion detection methods depend on signatures or patterns that attackers can bypass. Modern anomaly detection approaches address this gap by learning what “normal” behaviour looks like and flagging deviations, but they often require large, high-quality datasets that individual organizations may not have. Data sharing could ease this limitation, although it conflicts with confidentiality requirements common in critical infrastructure.

This project introduces a decentralized variant of the [DAICS](papers/DAICS_A_Deep_Learning_Solution_for_Anomaly_Detection_in_Industrial.pdf) intrusion detection architecture that applies federated learning to support collaborative model training without sharing raw data. Each participant trains locally and contributes model updates to a central coordinator, allowing organizations to benefit from collective knowledge while retaining control over sensitive operational information.

To support real-world deployment scenarios, the framework includes adaptive coordination concepts inspired by [FLAD](papers/FLAD_Adaptive_Federated_Learning_for_DDoS_Attack_Detection.pdf). These mechanisms redistribute training workload across clients, helping stabilize and accelerate convergence in heterogeneous environments where data quantity, quality, and computational resources vary.

Experiments use the **SWaT dataset**, a widely adopted benchmark originating from a scaled-down water treatment plant. It provides realistic ICS sensor and actuator data that captures normal operations and a variety of cyber-physical attack scenarios.

Across the SWaT experiments, the federated version of DAICS reaches performance levels close to the centralized baseline, with only minor degradation despite data partitioning and heterogeneous client conditions.

## How It Works

### Virtual Environment
Set up the project environment using Conda with the following commands (run them from the project root):

```
conda env create -f setup/environment.yml
conda activate federated-daics
conda env update -f setup/environment.yml --prune
```

### Data Processing
Download the 2015 SWaT dataset from the official iTrust portal by submitting the access request form [here](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).  
Place the extracted files in `dataset/original`.

Run `processing.py` to transform the raw dataset into the format required by the training pipeline. This step is only needed once. After processing, the cached file is automatically used by the main script on subsequent runs.

### Training
Training behavior is controlled through `config.py`. The main options include:

- **WIDE_DEEP_NETWORK**: Enable training of the Wide & Deep network  
- **THRESHOLD_NETWORK**: Enable training of the Threshold Network  
- **GPU**: Use GPU acceleration when available. If set to `true` but no GPU is detected, the script defaults to CPU  
- **WANDB**: Enable logging through the Weights & Biases API. Ensure `ENTITY` and `PROJECT` are correctly set in your `.env` file  
- **VERBOSE**: Enable detailed logging output  

After configuring these parameters, start training by running:

```
python main.py
```

## Contacts

Christian Sassi - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/dark.png">
    <img alt="https://www.unitn.it/" src="assets/light.png" width="300px">
</picture>
