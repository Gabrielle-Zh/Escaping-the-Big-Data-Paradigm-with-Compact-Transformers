# Escaping-the-Big-Data-Paradigm-with-Compact-Transformers
This paper shows that transformers can be lightweights vision models and achieve competitive performance on classification tasks while being trained from scratch on small datasets

## Overview:

#### Problem Statement
1.  reliance of machine learning  on large sums of data in many domains (e.g. science and medicine) 
2. large computation. The requisite of large data results in a requisite of large computational resources and this prevents many researchers from being able to provide insight. This not only limits the ability to apply models in different domains, but also limits reproducibility.

#### Model Comparsion

Transformers:
- originated in natural language processing, but also applied to other fileds like computer vision (Vision Transformer (ViT))
- lack some of the inductive biases inherent to CNNs
- "data hungry" paradigm, make training transformers from scratch seem intractable for many types of pressing problems where there are typically serveral orders of magnitude less data.

Convolutional neural networks (CNNs): 
- the standard for computer vision, since convolutions are adept at vision based problems due to their invariance to spatial translations as well as having low relational inductive bias
- translational equivariance and invariance, allowing CNNs to leverage natural image statistics and subsequently allowing models to have higher sampling efficiency

#### *CNNs: need less training data due to their translational equivariance from scratch*
#### *Transformers: capture long-range dependencies*

#### Approach
- Vit-Lite, a smaller and more compact version on ViT, efficient in less data intensive domains with high accuracy
- Compact Vision Transformer (CVT), pools over output tokens and improves performance
- Compact Convolutional Transformer (CCT), increase performace and provide flexibility for input image size while also demonstrating that these variants don npt depend as much on Positional Embedding compared to the rest

[ Thus our focus is on an accessible model, with few parameters, that can quickly and efficiently be trained on smaller platforms while still maintaining SOTA results]

## Method & Datasets

#### Question1: several key components of a transformer model?

<img width="691" alt="截屏2023-03-19 下午9 52 35" src="https://user-images.githubusercontent.com/82795673/226235772-bc8bea43-c323-450b-a367-f9073116396a.png">
ViT is composed of several parts: Image Tokenization, Positional Embedding, Classification Token, the Transformer Encoder, and a Classification Head.

We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable “classification token” to the sequence.

<img width="784" alt="截屏2023-03-17 下午9 45 36" src="https://user-images.githubusercontent.com/82795673/226080043-bcb37ede-540c-4919-ba93-0f387ed46ee9.png">

- ViT-Lite, with a more suitable size and path size for small-scale learning
- CVT, using Sequence Pooling method (SeqPool) to pool the entire sequence of tokens produced by the transformer encoder
- CCT, generating richer tokens and preserving local information by a convolutional tokenizer which is better at encoding relationships between pathes compares to the original ViT

#### key components

1.SeqPool:
- an attention-based method which pools over the output sequence of tokens
- weigh the sequential embeddingsof the latent space produced by the transformerencoder and correlate data across the input data

```python
class TransformerClassifier(nn.Module):
  def __init__(self,
               *args,
               **kwargs,
              ):
    super().__init__()
    ...
    self.attention_pool = nn.Linear(self.embedding_dim, 1)
    ...
          
  def forward(self, x):
    ...
    x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
    ...
```

2.Convolutional Tokenizer
a convolutional block consists of a single convolution, ReLU activation, and a max pool
<img width="378" alt="截屏2023-03-22 下午9 27 44" src="https://user-images.githubusercontent.com/82795673/227084477-b3054bb7-9bee-4473-9a79-ff541b1d1e4e.png">

- where the Conv2d operation has d filters, same number as the embedding dimension of the transformer backbone
- the convolution and max pool operations can be overlapping which could increase performace by injecting inductive biases

```python
class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 ):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding, 
                          bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding)
            ) for i in range(n_conv_layers) ])

        self.flattener = nn.Flatten(2, 3)

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)
```

## Question2:

## Critical Analysis
1.Strengths: the paper provides a detailed description of the architectures of vision transformers models, also deliver a thorough analysis of the models' performance, by comparing them with existing state-of-the-art methods, and finally demonstrate that they achieve competitive results with significantly fewer parameters.

2.Limitations: the paper lacks a detailed explanation of the training procedures, and the evaluation on various benchmarks behind the proposed models, although the authors justify their design choices based on empirical evidence; the paper's experimental evaluation could have been more thorough. Although the authors conduct a comprehensive analysis of the models' performance on various benchmarks, they do not investigate the models' robustness to adversarial attacks or their generalization to unseen data, which are important factors for evaluating the models' real-world applicability.

## Resource Links:
https://arxiv.org/abs/2104.05704
https://www.semanticscholar.org/paper/Learning-Multiple-Layers-of-Features-from-Tiny-Krizhevsky/5d90f06bb70a0a3dced62413346235c02b1aa086
