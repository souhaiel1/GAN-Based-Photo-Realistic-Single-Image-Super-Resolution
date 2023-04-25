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
$$\hat{\theta_G} = \arg \min_{\theta_G} \frac{1}{N}\sum_{n=1}^{N} l^{SR}(G_{\theta_G}(I^{LR}_n), I^{HR}_n)$$

We introduce a discriminator network $D_{\theta_D}$, which is optimized in an alternating manner with $G_{\theta_G}$ to solve the adversarial min-max problem. This problem aims to find the optimal values of $\theta_G$ and $\theta_D$ that can generate super-resolved images that are indistinguishable from the high-resolution images in the training set. The objective function for this problem is defined as follows:

$$\min_{\theta_G}\max_{\theta_D} E_{I^{HR} \sim p_{train}(I^{HR})} [\log D_{\theta_D}(I^{HR})] + \\
                    E_{I^{LR} \sim p_{G}(I^{LR})} [\log (1 - D_{\theta_D}(G_{\theta_G}(I^{LR})))] \ \ \ \ \ \  \ $$


\noindent The main idea of using such a formulation is to train a generative model $G$ to produce super-resolved images that can fool a differentiable discriminator $D$, which is trained to distinguish between real images and super-resolved images. By adopting this approach, the generator network can learn to generate solutions that are photo realistic and difficult for the discriminator network to classify, leading to perceptually superior results that lie in the manifold of natural images. This is in contrast to traditional super-resolution methods that minimize pixel-wise error measurements such as the $MSE$, which often results in overly smooth and unrealistic super-resolved images.\\
The architecture of the SRGAN model consists of a  deep generator network $G$, which is composed of $16$ residual blocks with identical layout. Each residual block has two convolutional layers with $3x3$ kernels and $64$ feature maps, followed by $batch-normalization$ layers and $ParametricReLU$ as the activation function. The resolution of the input image is increased with two trained sub-pixel convolution layers ($PixelShuffle$ layers). We train the discriminator network to tell real $HR$ images from generated $SR$ samples. The discriminator network consist of eight convolutional layers with an increasing number of $3x3$ filters , increasing by a factor of 2 from 64 to 512 kernels, as in the $VGG$ network and strided convolutions are used to reduce the image resolution each time the number of features is doubled. The resulting 512 feature maps are followed by two dense layers and a final $sigmoid$ activation function to enforce a probability for sample classification. We use a $LeakyReLU$  ($\alpha$ = 0.2) is used while avoiding max-pooling throughout the network. 
