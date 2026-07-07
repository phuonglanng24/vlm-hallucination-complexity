# Visual Complexity and Object Hallucination in Vision-Language Models

This repository accompanies our study investigating the relationship between visual complexity and object hallucination in vision-language models (VLMs).

## Overview

Vision-language models can generate descriptions containing objects that are not actually present in an image, a phenomenon known as object hallucination. While visual complexity is often considered an intuitive factor contributing to this issue, the relationship between scene complexity and hallucination remains unclear.

In this work, we conduct a complexity-aware analysis of object hallucination in a widely used open-source VLM evaluated on the MS COCO dataset. We investigate:

- The relationship between different visual complexity measures and hallucination rates
- The effect of visual content removal through image intervention experiments
- The impact of representative hallucination mitigation approaches, including decoding-time, training-based, and attention-based methods
- Whether hallucination can be predicted using interpretable visual and grounding features
- Whether grounding mediates the relationship between visual complexity and hallucination

## Key Findings

Our analysis shows that:

1. The relationship between scene-level complexity and hallucination is weak and depends strongly on the chosen complexity measure.
2. Reducing visual content does not consistently reduce hallucination; removing small regions can slightly decrease hallucination, while removing substantial content can increase it.
3. An interpretable hallucination detector relies primarily on object-level grounding features rather than complexity features.
4. Mediation analysis suggests that when complexity influences hallucination, the effect is largely explained through grounding.

These findings suggest that scene-level complexity alone is not a primary driver of object hallucination, while visual grounding provides a stronger explanatory signal.

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
