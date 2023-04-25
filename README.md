## Abstract 

Super resolution is a fundamental problem in computer vision, which aims at generating high-resolution images from low-resolution inputs. In recent years, 
deep learning approaches have shown promising results in addressing this challenge. In this project, we present a comprehensive review of deep learning-based super 
resolution methods, with a particular focus on GAN-based architectures such as [SRGAN](https://arxiv.org/abs/1609.04802) and [ESRGAN](https://arxiv.org/abs/1809.00219). We discuss the key components of 
state-of-the-art super resolution networks, including feature extraction, feature fusion, and reconstruction. We also takcle the DIV2K dataset using these approaches
while exploring the different training strategies and loss functions that have been proposed to enhance the performance of super resolution models. Furthermore, we 
provide an overview of benchmark datasets and evaluation metrics for our super resolution models. 

## SRGAN Model : 

The primary objective is to train a generative function $G$ that can accurately estimate the corresponding high-resolution (HR) counterpart for any given low-resolution (LR) input image. \\
The authors use generative function $G$, which is a feed-forward convolutional neural network, specifically, a deep $ResNet$ (SRReset) parametrized by $\theta_G$. Where, $\theta_G = \{W_{1:L}; b_{1:L}\}$ are the weights and biases of an L-layer deep network. We optimize a super-resolution specific perceptual loss function $l_{SR}$ to obtain $\theta_G$. The generator function is trained on a set of training images ${I_{HR_n}, I_{LR_n}}$, where $n = 1, ..., N$ (N = 800 for DIV2K).\\
We aim to minimize the following objective function:
\begin{equation}
    \hat{\theta_G} = \arg \min_{\theta_G} \frac{1}{N}\sum_{n=1}^{N} l^{SR}(G_{\theta_G}(I^{LR}_n), I^{HR}_n)
\end{equation}
