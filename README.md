# Escaping-the-Big-Data-Paradigm-with-Compact-Transformers

## Overview:
#### Model Comparsion
Convolutional neural networks (CNNs): 
- the standard for computer vision, since convolutions are adept at vision based problems due to their invariance to spatial translations as well as having low relational inductive bias
- translational equivariance and invariance are properties of the convolutions and pooling layers, allowing CNNs to leverage natural image statistics and
subsequently allow models to have higher sampling efficiency

Transformers:
- originated in natural language processing, but also applied to other fileds like computer vision
- Vision Transformer (ViT) was the first major demonstration of a pure transformer backbone being applied to computer vision tasks. It highlights the power of such models, but also the large-scale training can trump inductive biases.
- lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore don't generlize well when trained on  insufficient amounts of data
- "data hungry": make training transformers from scratch seem intractable for many types of pressing problems where there are typically serveral orders of magnitude less data.

CNNs are still the go-to models for smaller datasets because they are more efficient, both computationlly and in terms of memory, when compared to transformers. However they don't enjoy the long range interdependence that attention mechanisms in transformers provide.

The long range interdependence of attention mechanisms refers to the ability of these mechanisms to capture relationships between distant parts of a sequence or input, as opposed to only capturing local or adjacent relationships. In other words, attention mechanisms allow for modeling long-range dependencies in data, which can be useful in tasks such as language translation, image captioning, and speech recognition, where understanding the relationship between different parts of the input is crucial for accurate prediction. This is achieved by allowing the model to focus on relevant parts of the input, regardless of their position or distance from each other, and weigh their importance in the prediction process.

#### Problem Statement
- reduce machine learning's dependence on large sums of data: as many domains, such as science and medicine, would hardly have datasets the size of ImageNet [10]. This is because events are far more rare and it would be more difficult to properly assign labels, let alone create a set of data which has low bias and is appropriate for conventional neural networks. In medical research, for instance, it may be difficult to compile positive samples of images for a rare disease without other correlating factors, such as medical equipment being attached to patients who are actively being treated. Additionally,
for a sufficiently rare disease there may only be a few thousand images for positive samples, which is typically not enough to train a network with good statistical prediction unless it can sufficiently be pre-trained on data with similar attributes.
- the requisite of large data results in a requisite of large computational resources and this prevents many researchers from being able to provide insight. This not only limits the ability to apply models in different domains, but also limits reproducibility.

#### Approach
- Vit-Lite, a smaller and more compact version on ViT, efficient in less data intensive domains with high accuracy
- Compact Vision Transformer (CVT), pools over output tokens and improves performance
- Compact Convolutional Transformer (CCT), increase performace and provide flexibility for input image size while also demonstrating that these variants don npt depend as much on Positional Embedding compared to the rest

[ Thus our focus is on an accessible model, with few parameters, that can quickly and efficiently be trained on smaller platforms while still maintaining SOTA results]

## Method & Datasets

<img width="784" alt="截屏2023-03-17 下午9 45 36" src="https://user-images.githubusercontent.com/82795673/226080043-bcb37ede-540c-4919-ba93-0f387ed46ee9.png">


## Code Demonstration

## Comparsion

## Question1:

## Question2:

## Critical Analysis








## Resource Links:
https://arxiv.org/abs/2104.05704
