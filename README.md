# Visual Complexity and Object Hallucination in Vision-Language Models

This repository accompanies our study investigating the relationship between visual complexity and object hallucination in vision-language models (VLMs).

## Overview

This repository contains code for:
- Generating captions with LLaVA-1.5-7B
- Evaluating object hallucination with CHAIR
- Computing three visual complexity measures (handcrafted, visual-statistics, SAM-based)
- Training an interpretable hallucination detector (XGBoost)
- Running the mediation analysis (complexity → grounding → hallucination)
- Reproducing all figures and tables

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA (for LLaVA captioning, SAM, and CLIP grounding)

## Data

We use the MS COCO 2014 validation set:
- Images: [download from COCO](https://cocodataset.org/#download) (val2014, ~7 GB)
- Annotations: `instances_val2014.json` and `captions_val2014.json` from the same page

SAM checkpoint (ViT-H): [download from Meta](https://github.com/facebookresearch/segment-anything#model-checkpoints)

## Repository Status

This repository is currently being prepared.

The released code will include:

- Visual complexity computation
- Object hallucination evaluation using CHAIR metrics
- VLM inference pipelines
- Hallucination mitigation experiments
- Visual content intervention experiments
- Interpretable hallucination detector training and evaluation
- Grounding analysis and mediation analysis
- Statistical analysis and figure generation

## Dataset

Experiments are conducted on the MS COCO dataset.

The dataset is not redistributed in this repository. Instructions for downloading and preprocessing the required data will be provided.

## Citation

Citation information will be added upon publication.

## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA) for the base VLM
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for SAM
- [CHAIR](https://github.com/Maxlinn/CHAIR-metric) for hallucination evaluation
