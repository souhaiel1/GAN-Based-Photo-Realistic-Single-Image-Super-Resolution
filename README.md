## Abstract 

Super resolution is a fundamental problem in computer vision, which aims at generating high-resolution images from low-resolution inputs. In recent years, 
deep learning approaches have shown promising results in addressing this challenge. In this project, we present a comprehensive review of deep learning-based super 
resolution methods, with a particular focus on GAN-based architectures such as [SRGAN](https://arxiv.org/abs/1609.04802) and [ESRGAN](https://arxiv.org/abs/1809.00219). We discuss the key components of 
state-of-the-art super resolution networks, including feature extraction, feature fusion, and reconstruction. We also takcle the DIV2K dataset using these approaches
while exploring the different training strategies and loss functions that have been proposed to enhance the performance of super resolution models. Furthermore, we 
provide an overview of benchmark datasets and evaluation metrics for our super resolution models. 

## SRGAN Model

The primary objective is to train a generative function $G$ that can accurately estimate the corresponding high-resolution (HR) counterpart for any given low-resolution (LR) input image. \\
The authors use generative function $G$, which is a feed-forward convolutional neural network, specifically, a deep $ResNet$ (SRReset) parametrized by $\theta_G$. Where, $\theta_G = \{W_{1:L}; b_{1:L}\}$ are the weights and biases of an L-layer deep network. We optimize a super-resolution specific perceptual loss function $l_{SR}$ to obtain $\theta_G$. The generator function is trained on a set of training images ${I_{HR_n}, I_{LR_n}}$, where $n = 1, ..., N$ (N = 800 for DIV2K).\\
We aim to minimize the following objective function:
$$\hat{\theta_G} = \arg \min_{\theta_G} \frac{1}{N}\sum_{n=1}^{N} l^{SR}(G_{\theta_G}(I^{LR}_n), I^{HR}_n)$$

We introduce a discriminator network $D_{\theta_D}$, which is optimized in an alternating manner with $G_{\theta_G}$ to solve the adversarial min-max problem. This problem aims to find the optimal values of $\theta_G$ and $\theta_D$ that can generate super-resolved images that are indistinguishable from the high-resolution images in the training set. The objective function for this problem is defined as follows:

$$\min_{\theta_G}\max_{\theta_D} E_{I^{HR} \sim p_{train}(I^{HR})} [\log D_{\theta_D}(I^{HR})] + \\
                    E_{I^{LR} \sim p_{G}(I^{LR})} [\log (1 - D_{\theta_D}(G_{\theta_G}(I^{LR})))] \ \ \ \ \ \  \ $$


\noindent The main idea of using such a formulation is to train a generative model $G$ to produce super-resolved images that can fool a differentiable discriminator $D$, which is trained to distinguish between real images and super-resolved images. By adopting this approach, the generator network can learn to generate solutions that are photo realistic and difficult for the discriminator network to classify, leading to perceptually superior results that lie in the manifold of natural images. This is in contrast to traditional super-resolution methods that minimize pixel-wise error measurements such as the $MSE$, which often results in overly smooth and unrealistic super-resolved images.\\
The architecture of the SRGAN model consists of a  deep generator network $G$, which is composed of $16$ residual blocks with identical layout. Each residual block has two convolutional layers with $3x3$ kernels and $64$ feature maps, followed by $batch-normalization$ layers and $ParametricReLU$ as the activation function. The resolution of the input image is increased with two trained sub-pixel convolution layers ($PixelShuffle$ layers). We train the discriminator network to tell real $HR$ images from generated $SR$ samples. The discriminator network consist of eight convolutional layers with an increasing number of $3x3$ filters , increasing by a factor of 2 from 64 to 512 kernels, as in the $VGG$ network and strided convolutions are used to reduce the image resolution each time the number of features is doubled. The resulting 512 feature maps are followed by two dense layers and a final $sigmoid$ activation function to enforce a probability for sample classification. We use a $LeakyReLU$  ($\alpha$ = 0.2) is used while avoiding max-pooling throughout the network. 

<p align="center">
<img src="https://github.com/souhaiel1/GAN-Based-Photo-Realistic-Single-Image-Super-Resolution/blob/main//images/srganarch.PNG" />
</p>

## Our Perceptual Loss and training process 
The main contribution introduced by authors of the SRGAN paper is the use of the GAN architecture and the introduction of a novel perceptual loss function that consists of an adversarial loss and a content loss based on VGG's feature maps. This loss function is proved to be much better than MSE at capturing perceptually relevant differences.

$$l^{SR}= \underbrace{6.10^{-3}l_{VGG/5,4}^{SR}}_{content \ loss} + \underbrace{10^{-3} l_{Gen}^{SR}}_{adversarial \ loss}$$ 


Where The generative loss $l_{SR}^{Gen}$ is defined based on the probabilities of the discriminator $D_{\theta_D}(G_{\theta_G}(I_{LR}))$ over all training samples as:

$$l^{SR}_{Gen} = -\sum_{n=1}^N \log D_{\theta_D}(G_{\theta_G}(I^{LR}))$$

Our approach consisted of implementing the SRGAN architecture proposed by the authors while making changes on the perceptual loss function and the training process. 

**Modified perceptual loss function for SRGAN:**

Through intuition and experimentation, we introduce the following perceptual loss function:

$$l^{SR}_{modified}= \underbrace{l^{MSE}_{SR}}_{pixel loss}+ \underbrace{6.10^{-3}l_{VGG/5,4}^{SR}}_{content loss} + \underbrace{10^{-3} l_{Gen}^{SR}}_{adversarial loss} + \underbrace{2.10^{-8}l_{TV}^{SR}}_{total variation}$$ 

**Intituition:**

- We introduce MSE loss to penalize the differences in pixel space which ultimately leads to more accurate color fidelity between  $I^{SR}$ and $I^{HR}$. By minimizing this difference, the super-resolved image can achieve accurate color fidelity with the ground truth image. In addition to the adversarial loss and perceptual loss, MSE loss can provide a more direct control of the image quality and is computationally efficient to compute. Therefore, the introduction of MSE loss in our model can lead to improved image quality and greater fidelity between the super-resolved image and the ground truth image in terms of pixel-wise similarity.
- Inspired by style transfer GANs, we introduce TV loss to reduce noise in $I^{SR}$. TV loss acts as a regularization term that measures the overall variation of intensities in the image, promoting spatial smoothness and reducing noise. In the context of SISR, the introduction of TV loss can improve the quality of the super-resolved image by reducing the noise that may arise from the low-resolution input image. However, since TV loss can also lead to smoothing of textures in the image, we used a very low weight  its loss component to avoid crashing textures during the super-resolution process. This allows for better preservation of the high-frequency details in the super-resolved image, while still benefiting from the noise reduction provided by the TV loss.


## Tweaking the pre-training process

Instead of pre-training the Generator network using MSE loss like the authors did, we experimented with a **weighted MSE** and $L1$  **loss**, which both yielded better results than MSE from a human viewer's perspective, and settled for the latter.
Intuitively, this is because MSE is  sensitive to outliers, which can result in overly-smoothed solutions and loss of fine details in the image. In contrast, MAE gives more weight to small differences between the predicted image and the ground truth image, allowing for better preservation of the high-frequency details in the $SR$ image. In addition, MAE is a more robust loss function for handling the non-linear mapping from the low-resolution input image to the high-resolution output image. By minimizing the absolute differences between the predicted and ground truth images, the generator is able to learn more effectively and produce higher quality super-resolved images.
For the most part, we follow the same training process described by the authors in the paper.
We trained our model for $850 \ epochs$ and used the following setting of hyperparameters:

- LR\_CROPPED\_SIZE = 24: The size of the low-resolution cropped input image used for training.
- UPSCALE = 4: The upscaling factor used to generate the high-resolution output image from the low-resolution input image.
- HR\_CROPPED\_SIZE = UPSCALE * LR\_CROPPED\_SIZE: The size of the high-resolution cropped output image used for training.
- BATCH\_SIZE = 16: The number of image samples in each mini-batch during training.
- EPOCHS = 50: The number of epochs (i.e., complete passes through the training dataset) used for training.
- LR = 0.0001: The learning rate used for training the model.
- BETAS = (0.5, 0.9): The values of the beta1 and beta2 hyperparameters used by the Adam optimizer during training.
- adversarial\_loss\_coef = 0.001: The coefficient for the adversarial loss used during training.
- vgg\_loss\_coef = 0.006: The coefficient for the VGG loss used during training.

## Datasets 

The models were trained on the DIV2K dataset that contains: 

- 1600 training images of different sizes divided into 800 high resolution images and their corresponding x4 down-scaled and bicubic-interpolated low resolution images.
- 200 test images divided into 100 high resolution images and their corresponding low resolution counterparts.

In addition to DIV2K's test set, we also evaluated our models on the Set5 and Set14 benchmark datasets.The performance of the models was evaluated using various quantitative metrics such as peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and visual quality.

## Results 

Using our own implementation of the discussed quality metrics, we evaluate and compare our SRGAN with the results of the original SRGAN paper and our ESRGAN.
From the qualitative results displayed in the table below, we can see that our SRGAN model compares fairly well with the results of the authors of the original SRGAN paper who used a much larger training set (350k images sampled from ImageNet) and trained their model for much longer.  We can also see the superiority of the ESRGAN model in each case. Indedd the enhanced architecture and the subtle but important technical modifications the authors introduced seem to be improve the results quantitatively. However, it is unclear to me if this improvement over previous methods is mostly thanks to the subtle architectural and theoretical changes or the sheer fact that the generator network of ESRGAN is much deeper and involves much more convolutions.\\
Note that for ground-truth HR images, $PSNR= \infty$ and $SSIM = 1$



|         | **SRGAN (ours)** | **SRGAN (Ledig et al)** | **ESRGAN** |
| :-----: | :--------------: | :---------------------: | :-------: |
| **DIV2K**  |        |             |         |
| PSNR    |     25.373       |             -            |   28.174  |
| SSIM    |     0.706        |             -            |   0.775   |
| **Set5**   |        |             |         |
| PSNR    |     24.945       |           29.40          |   30.474  |
| SSIM    |     0.718        |          0.8472          |   0.851   |
| **Set14**  |        |             |         |
| PSNR    |     23.770       |           26.02          |   26.614  |
| SSIM    |     0.636        |          0.7397          |   0.713   |



Qualitative results are in accordance with the results we got using the analytical quality measures. 
In fact, our  qualitative results demonstrate the ability of SRGAN in retrieving fine details and textures that were missing in the original low-resolution images. The high-resolution images produced by SRGAN show sharp and well-defined edges and feature relatively smooth transitions in color and texture. Overall, our results confirm the effectiveness of proposed perceptual loss and pre-training procedure. However, the results are not perfect or as good as ESRGAN's. This is expected since we only trained our model for 850 epochs on a relatively small dataset. \\
On the other hand, ESRGAN exhibits very good performance in producing super-resolved images with rich details and textures. The generated images have high perceptual quality (from a human's point of view) and surpass the ones produced by SRGAN. The enhanced performance of ESRGAN can be attributed to its more robust discriminator that utilizes a relativistic approach to evaluate the realism of the generated images, the subtle changes made in the loss function and architecture or the fact that it is a much more deeper network. 

<p align="center">
<img src="https://github.com/souhaiel1/GAN-Based-Photo-Realistic-Single-Image-Super-Resolution/blob/main//images/comparaison.jpg" />
</p>

## Discussion

we assessed the effectiveness of SRGAN and ESRGAN (mostly SRGAN) on the DIV2K dataset and other benchmarking sets. Both networks showed improvement over the traditional interpolation methods as well as previous state-of-the-art super-resolution methods.
The use of perceptual loss and deep residual learning in both SRGAN and ESRGAN have proven to be effective in super-resolving images, and the use of pre-trained VGG network in SRGAN has shown to be effective in capturing the high-level feature similarities between the super-resolved and ground truth images. However, the comparison between the two networks has shown that ESRGAN performs better than SRGAN in terms of both quantitative metrics (PSNR, SSIM) and visual quality.

Moreover, using a modified version of the perceptual loss function proposed by Ledig et al, our SRGAN model performs reasonably well on the benchmark datasets and so does ESRGAN, especially given the relatively low number of epochs that we used given that SR methods and  GAN-based models in generals require a lot of training time . We intend to continue with the training of ESRGAN on DIV2K and re-traing SRGAN for longer time and with a heavy augmentation pipeline as for this project, no augmentations were applied except a center crop. Allso exploring different datasets can prove to benificial.
We also intend to explore multi-frame super resolution methods like the one presented google's burt super-resolution work. These methods rely on fusing images of the same sceneg enerated by hand movements or intentional camera hardware movements, and merge their information to obtain a higher resolution image. 
