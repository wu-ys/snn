# Computer lab 3 

**This project accounts for 15% of final scores!**

**Group project; 3 students/group** 

**Exam form: presentation and defense only**

**Deadline for handing in presentation ppt: Dec.13, 23:59**

**Time for presentation: Project Discussion course, Dec. 14**

Deep learning, a machine learning branch, uses artificial neural networks to learn from large and complex data. However, conventional models are computationally expensive, energy-inefficient, and biologically unrealistic. Spiking neural networks (SNNs) mimic biological neural networks by using discrete spikes to encode and transmit information. These networks can incorporate time and temporal dynamics, making them suitable for processing spatio-temporal data and learning temporal patterns. SNNs also pose challenges for training and optimization, as their objective function is often non-smooth, non-convex, and non-differentiable. **In this homework, you are going to dive into the main streams of training spiking neural networks**.

### I. Surrogate gradient training of Spiking Neural Networks (15 points)

Surrogate gradient methods have been proposed to overcome the difficulties and improve SNN performance. Specifically, when describing the firing process of neurons, we usually use the Heaviside step function. Heaviside step functions are not differentiable. Therefore, direct training is not possible. In order to solve this problem, various surrogate gradient methods have been proposed, such as the choices offered in Fig. 3 in ref. [1]. The Heaviside step function is shown below:
$$
\Theta(x) = \begin{cases}1, & x>0 \\ 0, &x<0\end{cases}
$$
The principle of the gradient substitution method is that the spiking neuron uses the Heaviside function during forward propagation, while in backpropagation the derivative of a function is used to replace the pseudo gradient of Heaviside function. This function is also called the **surrogate function**. Typically, this function is shaped like a Heaviside, but has a smooth continuous function. For example, Sigmoid function can be used as a surrogate function:
$$
\sigma (x) = \frac{1}{1+\exp(-ax)}
$$
where a hyper-parameter $\alpha$ can be altered to change the smoothness. Fig. 1 shows the relationship between Heaviside function, Sigmoid surrogate function (a=5), and the derivative of surrogate function.

*SpikingJelly* is a framework based on PyTorch that uses SNN for deep learning. In *SpikingJelly*, some of the surrogate functions are predefined in *spikingjelly.activation_based.surrogate*, including *Sigmoid, Atan*, etc. For the installation of *SpikingJelly*, see *Installation of SpikingJelly.* For a detailed introduction and advanced usage of *SpikingJelly*, see https://spikingjelly.readthedocs.io/zh-cn/latest/index.html.

One example of using the Sigmoid surrogate function is:

```python
import torch
from spikingjelly.activation_based import surrogate
surrogate_function = surrogate.Sigmoid(alpha=5.)

#in spikingjelly.activation_based.surrogate.py
@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)
    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()
    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]

```

Neuromorphic datasets are collections of data that are highly compatible with spiking neural networks. The neuromorphic MNIST (N-MNIST) is the simplest neuromorphic dataset for classification. The neuromorphic MNIST dataset is a spiking version of the original frame-based MNIST dataset, which is a widely used benchmark for image classification and recognition. The MNIST dataset consists of 70,000 handwritten digits (60,000 for training and 10,000 for testing), each of size 28x28 pixels. The neuromorphic MNIST dataset was created by converting the static images of the MNIST dataset into dynamic event streams using an event-based camera (Asynchronous Time-based Image Sensor, ATIS). The ATIS camera only responds to changes in the visual scene and generates discrete events called spikes when the intensity of a pixel changes by a certain amount. The spikes are encoded with the pixel coordinates, the polarity (ON or OFF), and the timestamp. The camera performed three saccade-like motions (left-right, up-down, and diagonal) for each digit, resulting in three different recordings per digit. Each recording consists of a list of events, each occupying 40 bits as described below:

- bit 39 - 32: Xaddress (in pixels)

- bit 31 - 24: Yaddress (in pixels)

- bit 23: Polarity (0 for OFF, 1 for ON)

- bit 22 - 0: Timestamp (in microseconds)

The neuromorphic MNIST dataset preserves the same number and order of digits as the original MNIST dataset and is captured at the same visual scale (34x34 pixels) as the original MNIST dataset (28x28 pixels).

Your task is to train a two-layered SNN on the N-MNIST dataset using *SpikingJelly* and complete and discuss the following targets:

i. According to the design rules of surrogate gradient functions, we can also design many other gradient functions that can also train SNN. For example, based on a trigonometric function (S(x) below) or a power function (P(x) below),
$$
S(x) = \begin{cases}

1, &x>\alpha\\

\dfrac12 \sin(\dfrac{\pi x}{2\alpha}) + \dfrac12, & -\alpha\le x\le \alpha\\

0, & x<-\alpha

\end{cases}
$$

$$
P(x) = \begin{cases}
1, & x>1\\
\dfrac12\text{sgn}(x) |x|^\alpha + \dfrac12, & -1\le x \le 1 (0<\alpha<1)\\
0, & x<-1\end{cases}
$$

where $\alpha$ is a hyper-parameter. Please try to finish the code of these surrogate functions according to the example code and use your designed surrogate functions to train SNN on the N-MNIST dataset. Please set a=0.5 for P(x) and a=1 for S(x).

ii. The choice of surrogate functions and their corresponding parameters are important for training SNN. Please try various surrogate functions, such as the ones you implemented above (S(x), P(x)), or the ones provided in *SpikingJelly* (PiecewiseQuadratic, PiecewiseExp, SoftSign, ATan, Erf, etc.), and modify their parameters. Please report and analyze the performance of the spiking neural network under different surrogate functions and parameter settings in the presentation.

iii. For the PiecewiseQuadratic function in *spikingjelly.activation_based.surrogate*, there are many continuous intervals in which the gradient will reach 0, which means that the membrane potential of the neuros will not produce a gradient when it is taken in these intervals. In some literature, this sparse gradient is considered a learning advantage of spiking neural networks [2]. Please modify the parameters of this surrogate function and analyze the impact of this sparse gradient on the final learning effect. And try to answer the question: Do you think sparsity is important for a surrogate function?

 

### II Discussions (15 points)

There is another approach to obtaining deep SNNs, which is to convert them from pre-trained artificial neural networks (ANNs). This approach is easier to train and has achieved remarkable performance on various tasks. However, ANN-SNN conversion is not a trivial process, as it involves mapping the continuous activation values of ANNs to the discrete spike rates of SNNs. This mapping introduces conversion errors, which can degrade the performance of the converted SNNs. One paper titled “**Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks**” has made a deep analysis of the conversion errors [3] (Github link: [4]). In this paper, the authors propose a novel method to optimize the ANN-SNN conversion process (QCFS), which can reduce the conversion errors and the latency of the SNNs. The neurons they are using are integrate-and-fire neurons. For layer l, we have:
$$
\mathbf{m}^l(t) = \mathbf{v}^l(t-1) + \mathbf{W}^l \mathbf{x}^{l-1}(t),\\
\mathbf{s}^l(t) = H(\mathbf{m}^l(t) - \Theta^l),\\
\mathbf{v}^l(t) = \mathbf{m}^l(t) - \mathbf{s}^l(t) \theta^l.
$$
![img](D:\wys\classfile\2023-2024-1\neuralAI\lab3\readme.assets\clip_image002-1701136215655-11.png)

Here $\mathbf{s}^l(t)$ refers to the output spikes of all neurons in layer l at time t, the element of which equals 1 if there is a spike and 0 otherwise.$H(\cdot)$ is the Heaviside step function. $\Theta^l$ is the vector of the firing threshold $\theta^l$. Your task is:

 

i. According to ref. [3], there is a relationship between the average postsynaptic potential of neurons in adjacent layers:

![img](D:\wys\classfile\2023-2024-1\neuralAI\lab3\readme.assets\clip_image010.png)

Please use the above dynamic formula for integrate-and-fire neurons to deduce this conclusion.

**Discussion**: Why does the conversion error mentioned in ref. [3] occur?

 

ii. Unevenness errors are mentioned in ref. [3]. Unevenness errors are caused by the unevenness of input spikes. If the timing of arrival spikes changes, the output firing rates may change, which causes conversion errors. The paper concluded that **the unevenness error will degenerate into some common errors if** $\mathbf{v}^l(T)$ **is in the range of** $[0,\theta^l]$. Please discuss the unevenness error when $\mathbf{v}^l(T)$ is not in the range of $[0,\theta^l]$. If possible, provide case studies.

**Discussion**: What means can be used to eliminate unevenness errors?

 

iii. Please train an ANN of WideResNet [5] on the CIFAR10 dataset and convert it to an SNN using QCFS. 

**Discussion**: In the SNN version of WideResNet, is the distribution of unevenness errors the same in different layers? Please try to explain.

 

iv. (**Optional**) Currently, ANN-SNN conversion is mainly used in image classification tasks. Training SNN on neuromorphic datasets usually requires surrogate gradient training. But the cost of surrogate training is greater than the cost of ANN-SNN conversion. DVS-CIFAR10 is a neuromorphic dataset for object classification, generated by neuromorphic cameras. It is derived from the CIFAR-10 dataset in 10 classes. DVS-CIFAR10 converts the frame-based images into event streams using a dynamic vision sensor. The dataset is of intermediate difficulty and can be used to benchmark event-driven object classification algorithms. Using *SpikingJelly*, one can efficiently load the dataset [6].

**Discussion**: Please conduct experiments and discuss how to apply QCFS to the DVS-CIFAR10 neuromorphic dataset.



[1] Neftci, Emre O., Hesham Mostafa, and Friedemann Zenke. "Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks." IEEE Signal Processing Magazine 36.6 (2019): 51-63. https://arxiv.org/abs/1901.09948.

[2] Perez-Nieves, Nicolas, and Dan Goodman. "Sparse spiking gradient descent." Advances in Neural Information Processing Systems 34 (2021): 11795-11808. [proceedings.neurips.cc/paper_files/paper/2021/file/61f2585b0ebcf1f532c4d1ec9a7d51aa-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/61f2585b0ebcf1f532c4d1ec9a7d51aa-Paper.pdf)

[3] Bu, Tong, et al. "Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks." International Conference on Learning Representations. 2021. https://openreview.net/pdf?id=7B3IJMM1k_M.

[4] https://github.com/putshua/SNN_conversion_QCFS.

[5] https://github.com/DingJianhao/parseval-network-pytorch/blob/main/networks/wide_resnet.py.

[6] https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/datasets/cifar10_dvs.py.