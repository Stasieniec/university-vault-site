---
type: concept
aliases: [Artificial Neural Networks, MLP, Multi-Layer Perceptron]
course: [RL, IR]
tags: [foundations]
status: complete
---

# Neural Networks

## Definition

> [!definition] Neural Network
> A **Neural Network** is a parameterized function approximator composed of layers of linear transformations followed by non-linear activation functions. They are designed to learn complex, non-linear mappings from inputs to outputs using the principles of gradient-based optimization.

## Mathematical Formulation

A single hidden layer network can be represented as:
$$y = \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2)$$

where:
- $W_i, b_i$ — Weight matrix and bias vector of layer $i$
- $\sigma_i$ — Non-linear activation functions (e.g., ReLU, Sigmoid, Tanh)
- $x$ — Input vector

## Universal Approximation Theorem

> [!important] Universal Approximation
> A feedforward network with a single hidden layer and a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, provided the activation function is non-constant, bounded, and monotonically-increasing.

## Training (Backpropagation)

Neural networks are trained using **Stochastic Gradient Descent (SGD)** or variants like [[Adam]]. The gradients are calculated using **Backpropagation**, which is an application of the [[Chain Rule]] of calculus to compute the partial derivatives of a loss function $L$ with respect to every weight in the network.

## Key Concepts

- **Activation Functions**: Introduce non-linearity (e.g., $ReLU(z) = \max(0, z)$).
- **Layers**: Layers between input and output are "hidden layers." Deep networks have many.
- **Weights**: The parameters $\theta$ that the model "learns."

## Connections

- Used in: [[Deep Reinforcement Learning]]
- Optimization: [[Gradient Descent]], [[Adam]], [[RMSProp]]
- Prevention of Overfitting: [[Regularization]]
- Variant: [[Convolutional Neural Networks]] (for vision), Transformers (for sequence)

## Appears In

- [[RL-L08 - Deep Reinforcement Learning]]
- [[IR-L05 - Neural Information Retrieval]]
