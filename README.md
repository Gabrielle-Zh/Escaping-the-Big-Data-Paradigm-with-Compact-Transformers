# Escaping-the-Big-Data-Paradigm-with-Compact-Transformers
This paper shows that we can lightweight vision transformer models and achieve competitive performance on classification tasks while being trained from scratch on small datasets

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

#### Question1: several key components of a transformer model?

ViT is composed of several parts: Image Tokenization, Positional Embedding, Classification Token, the Transformer Encoder, and a Classification Head.

<img width="691" alt="截屏2023-03-19 下午9 52 35" src="https://user-images.githubusercontent.com/82795673/226235772-bc8bea43-c323-450b-a367-f9073116396a.png">

We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable “classification token” to the sequence.

<img width="784" alt="截屏2023-03-17 下午9 45 36" src="https://user-images.githubusercontent.com/82795673/226080043-bcb37ede-540c-4919-ba93-0f387ed46ee9.png">

- ViT-Lite, nearly identical to the original ViT in terms of architecture, but with a more suitable size and path size for small-scale learning
- CVT, using Sequence Pooling method (SeqPool) to pool the entire sequence of tokens produced by the transformer encoder
- CCT, generating richer tokens and preserving local information by a convolutional tokenizer which is better at encoding relationships between pathes compares to the original ViT

#### key components

1. Small and Compact Models:
transformer backbones in each variant:

<img width="370" alt="截屏2023-03-21 下午12 01 03" src="https://user-images.githubusercontent.com/82795673/226685716-7b733580-c637-403f-a8a2-e46cf3171ee9.png">

Tokenizers in each variant:

<img width="402" alt="截屏2023-03-21 下午12 01 35" src="https://user-images.githubusercontent.com/82795673/226685844-46fbf4c1-d643-44c6-8cfe-3cb795443f17.png">

2.SeqPool:
- an attention-based method which pools over the output sequence of tokens
- allowing the network to weight the sequential embeddings of the latent space produced by the transformer encoder and correlate data across the input data
(assign importance weights across the sequence of data, only after they have been processed by the encoder.)

We tested servel variations of this pooling method, including learnable and static methods, and found that the learnable pooling performs the best. Static methods have already been explored by ViT as well. But the learnable weighting is more efficient beacuse each embedded patch does not contain the same amount of entropy. It allows the model to apply weights to token with respect to the relevance of their information. Additionlly sequence pooling allows our model to better untilize information across spatially sparse data.

Our motivation is that the output sequence contains relevant information across different parts of the input image, therefore preserving this information can imprve performace, and at no additional parameters compared to the learnable token. Additionlly, the computation desceases slightly due one less token being forward.

3.Convolutional Tokenizer
In order to introduce an inductive bias into the model, we replace path and embedding in ViT-Lite and CVT with a simple convolutional block
- consist of a single convolution, ReLU activation, and a max pool
<img width="378" alt="截屏2023-03-22 下午9 27 44" src="https://user-images.githubusercontent.com/82795673/227084477-b3054bb7-9bee-4473-9a79-ff541b1d1e4e.png">

where the Conv2d operation has d filters, same number as the embedding dimension of the transformer backbone.
the convolution and max pool operations can be overlapping which could increase performace by injecting inductive biases. THis allow our model to maintain locally spatial information. By using this convolutional block, the models enjoy an added flexibility over models like ViT, by no longer being tied to the input resolution strictly divisible by the pre-set patch size.

We seek to use convolutions to embed the image into a latent representation, because we believe that it will be more efficient and produce richer tokens for the transformer. These blocks can be adjusted in terms of downsampling ratio (kernel size, stride and padding), and are repeatable for even
further downsampling. Since self-attention has a quadratic time and space complexity with respect to the number of tokens, and number of tokens is equal to the resolution of the input feature map, more downsampling results in fewer tokens which noticeably decreases computation (at the expense of performance). We found that on top of the added performance gains, this choice in tokenization also gives more flexibility toward removing the positional embedding in the model, as it manages to maintain a very good performance.

## Code Demonstration

## Question2:

## Critical Analysis
Strengths: the paper provides a detailed description of the architectures of vision transformers models, also deliver a thorough analysis of the models' performance, by comparing them with existing state-of-the-art methods, and finally demonstrate that they achieve competitive results with significantly fewer parameters.
Limitations: the paper lacks a detailed explanation of the training procedures, and the evaluation on various benchmarks behind the proposed models, although the authors justify their design choices based on empirical evidence; the paper's experimental evaluation could have been more thorough. Although the authors conduct a comprehensive analysis of the models' performance on various benchmarks, they do not investigate the models' robustness to adversarial attacks or their generalization to unseen data, which are important factors for evaluating the models' real-world applicability.

## Resource Links:
https://arxiv.org/abs/2104.05704
https://www.semanticscholar.org/paper/Learning-Multiple-Layers-of-Features-from-Tiny-Krizhevsky/5d90f06bb70a0a3dced62413346235c02b1aa086
