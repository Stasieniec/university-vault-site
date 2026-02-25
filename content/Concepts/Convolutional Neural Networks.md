---
type: concept
aliases: [CNN, CNNs, convolutional neural networks, ConvNet]
course: [RL, IR]
tags: [deep-learning]
status: complete
---

# Convolutional Neural Networks (CNNs)

> [!definition] Convolutional Neural Network
> A [[Neural Networks|neural network]] architecture that uses **convolutional layers** with learnable filters to automatically extract local spatial features from grid-structured data (images, 1D sequences). Weight sharing across spatial positions makes CNNs translation-invariant and parameter-efficient.

## Core Components

1. **Convolutional layer**: Applies learnable filters (kernels) across input via sliding window → produces feature maps
2. **Pooling layer**: Downsamples feature maps (max-pool, average-pool) → reduces spatial dimensions, adds invariance
3. **Fully connected layer**: Final classification/regression after feature extraction

## Key Properties

- **Translation invariance**: same filter applied everywhere → detects features regardless of position
- **Weight sharing**: far fewer parameters than fully connected equivalent
- **Hierarchical features**: early layers detect edges/textures, deeper layers detect complex patterns
- **Local connectivity**: each neuron connects to a small receptive field, not the full input

## In RL Context

CNNs serve as the **feature extraction backbone** for [[Deep Reinforcement Learning]] on visual inputs:
- [[Deep Q-Network (DQN)]] uses CNNs to process raw Atari game frames
- The CNN maps pixels → learned state representation → fed to value/policy heads

## In IR Context

- Used in some early neural IR models for learning text representations from character/word n-grams
- Largely superseded by [[Transformers]] in modern [[Neural Reranking]] and [[Dense Retrieval]]

## Connections

- Component of [[Deep Q-Network (DQN)]] and [[Deep Reinforcement Learning]]
- [[Neural Network Function Approximation]] in RL
- Compared with [[Transformers]] (self-attention vs local convolution)

## Appears In

- [[RL-L08 - Deep RL Value-Based]] (DQN architecture)
- [[RL-Book Ch16 - Applications and Case Studies]] (Atari)
