---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Cornell Birdcall Identification Competition"
subtitle: "My approach to achieve 1st place in the Kaggle Cornell Birdcall Identification Competition"
summary: ""
authors: [admin]
tags: []
categories: [kaggle]
date: 2020-09-17T07:45:08+01:00
lastmod: 2020-09-17T07:45:08+01:00
featured: false
draft: false
 
# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: Smart
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

An overview of the process I went through for the Cornell Birdcall Identification Competition hosted on Kaggle, where I achieved 1st place. The competition ran from mid June 2020 to mid September 2020.

{{< figure src="imgs/results.JPG" title="Final Competition Leaderboard" >}}

## Competition Overview

> Do you hear the birds chirping outside your window? Over 10,000 bird species occur in the world, and they can be found in nearly every environment, from untouched rainforests to suburbs and even cities. Birds play an essential role in nature. They are high up in the food chain and integrate changes occurring at lower levels. As such, birds are excellent indicators of deteriorating habitat quality and environmental pollution. However, it is often easier to hear birds than see them. With proper sound detection and classification, researchers could automatically intuit factors about an area’s quality of life based on a changing bird population.

...

> In this competition, you will identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!

## Data

The training data provided consisted of short recordings of individual bird calls uploaded to xenocanto.org.

The unseen test data was significantly different as it contained approximately 150 recordings which were roughly 10 minutes long. The recordings were taken at three separate sites in North America.

> Sites 1 and 2 were labeled in 5 second increments and need matching predictions, but due to the time consuming nature of the labeling process the site 3 files are only labeled at the file level. Accordingly, site 3 has relatively few rows in the test set and needs lower time resolution predictions.

## Models

My approach used a Sound Event Detection approach described in [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211). Kaggle user [hidehisaarai1213](hidehisaarai1213) provides a good explaination of how it works in the Kaggle kernel [introduction to sound event detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection).

### Main Differences

- Switched the CNN Feature extractor with a pretrained Densenet121 model (Previous papers in audio classification showed densenet-like architectures outperform other pretrained model approaches.)
- Replaced the `torch.clamp` method with `torch.tanh` in the attention layer.
- Reduced the AttBlock size to 1024.

The reasons for making the changes was to avoid overfitting because of the lack of data with less than or equal to 100 labelled samples per bird class.

## Training

Overall, I had over 300 experiment for hyperparameter searching and model training. The training took between 5-7 hours per model on GCP with a V100 GPU.

- mixup augmentation
- pink noise (very important)
- AdamW optimiser with weight decay
- lr warmup with cosine annealing
- Spec Augmentation
- 30 second clip length

It was difficult for me to get my CV to match the leaderboard. To find the best model to submit I selected models with the best f1 score which didn't sacrifice precision. When the valid_F_loss increased it generally meant that the models were starting to overfit and would perform poorly on the public leaderboard.

{{< figure src="imgs/validation.JPG" title="Validation of an Overfit Model" lightbox="true" >}}

## Inference

Model ensembling by voting and thresholds on both `clipwise_output` and `framewise_output` was key to reducing the number of false positives and maximising the f1-score.

- 4 fold models (without mixup)
- 5 fold models (without mixup)
- 4 fold models (with mixup)

{{< figure src="imgs/inference.jpg" title="Diagram of how threshold selection works" lightbox="true" >}}

2 submissions were allowed to be selected before the Private Leaderboard was revealed. My top ensemble was if 4 out of the 13 models predicted a bird with a threshold of `0.3` for both `clipwise_output` and `framewise_output` was within the audio snippet then it would be accepted as a valid prediction. This ensemble was not a the highest model on the public leaderboard but I selected it based on what I felt would be the most confident in as 3 votes felt too low for 13 models (highest on Public Leaderboard). With this ensemble I was able to jump from 7th on the Public Leaderboard to 1st on the Private Leaderboard.

My second selection was based on 9 models (the models without mixup). My top scoring ensemble with 9 models was with 3 votes and a threshold of `0.5` for the frame threshold and `0.3` for the clip threshold. I didn't try a threshold of `0.3` for the frame threshold since I ran out of submissions (max 2 submissions per day). The selected 9 model ensemble would have put me at 3rd position on the Private Leaderboard.

| **13 Models**        | Clip Threshold | Frame Threshold | Public Leaderboard | Private Leaderboard | Selected |
|---------|----------------|-----------------|--------------------|---------------------|----------|
| 4 votes | 0.3            | 0.3             | 0.616              | 0.681               | x        |
| 4 votes | 0.3            | 0.5             | 0.615              | 0.679               |         |
| 5 votes | 0.3            | 0.3             | 0.609              | 0.679               |         |
| 5 votes | 0.3            | 0.5             | 0.606              | 0.676               |         |
| 3 votes | 0.3            | 0.3             | 0.617              | 0.679               |         |
| 3 votes | 0.3            | 0.5             | 0.614              | 0.679               |         |

| **9 Models**        | Clip Threshold | Frame Threshold | Public Leaderboard | Private Leaderboard | Selected |
|---------|----------------|-----------------|--------------------|---------------------|----------|
| 3 votes | 0.3            | 0.5             | 0.613              | 0.676               | x        |
| 2 votes | 0.3            | 0.5             | 0.614              | 0.669               |         |
| 4 votes | 0.3            | 0.5             | 0.610              | 0.675               |         |

## Conclusions

I was extremely surprised to achieve 1st place in this competition as my main goal was to achieve gold so that I could be ranked as a Kaggle Competition Master. During this competition I was also able to complete my personal deep learning pytorch toolkit to reduce boilerplate code for future machine learning projects / competitions. I was also able to learn from many of my mistakes made in the DeepFake Detection Challenge and applied my learnings to this competition such the importance of ensembling, preprocessing for faster training and time management.
