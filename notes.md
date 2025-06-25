# Notes for AlexNet / CNN from scratch

These are raw learning notes, not polished

## AlexNet spefics:
- total 8 layers
- first 5 layers convalutions
    - Conv
    - RelU - non-linearity is applied to the output of every convolutional and fully-connected layer 
    - Max pooling layers - 3x3 kernal 2 pixels apart
- layer1 - The first convolutional layer filters the 224 × 224 × 3 input image with 96 kernels of size 11 × 11 × 3 with a stride of 4 pixels (This is the distance between the receptive field centers of neighboring neurons in a kernel map)
- layer2 - 256 kernels of size 5 × 5 × 48
- layer3 - 384 kernels of size 3 × 3 × 256 
- layer4 - 384 kernels of size 3 × 3 × 192 
- layer5 - 256 kernels of size 3 × 3 × 192.
- layers 3,4,5  are connected without any intervening pooling or normalization layers. layer5 output has normalization and Max pooling


- remaining 3 are fully connected layers with weights - shape of 4096
    -  use dropout in the first two fully-connected layers
    - FC  
        - Dropouts
        - ReLU
    - FC 1
        - Dropouts
        - ReLU
    - FC 2
        - The output of the last fully-connected layer is fed to a 1000-way softmax (?) which produces a distribution over the 1000 class labels (this is because the orginal paper had 1000 lables to classify)

## 1. ReLU Activation Function
I'm learning about ReLU activation function, I used to be intimidated by it because of the math symbols, but now it's just two if statements?!?!

> Return 0 if the input is negative otherwise return the input as it is. [^1]

```python
def relu(x):
    if x < 0:
        return 0
    else:
        return x
```

or simply:

```python
def relu(x):
    return max(0, x)
```

but in math, it's this scary looking thing:

$$ \text{ReLU}(x) = \max(0, x) = \begin{cases} 
0 & \text{if } x < 0 \\
x & \text{if } x \geq 0 
\end{cases} $$

Okay now that I actually wrote the formula, it's actually not so scary.

This helps keep the model non-linear and prevents eliminating vanishing gradient problem.

man, I'm never going to avoid looking under the hood - things are simpler then they seem most of the time

and I want to petition for all math to be writting in python code

## 2. Convolutions
What is a Convolution and why is it useful?

- A conv is a series of dot products between a kernel and a subset of the input image
- Filter kernel · input image = output feature map

So a CNN will typically have: [^2]
- Convolutional Layer
- ReLU Activation Function
- Pooling Layer
- Fully Connected Layer

now, the AlexNet code in [^4] is starting to make sense

things I need to learn about:
- [x] Dropout
- [ ] nn.linear
- [ ] nn.BatchNorm2d
- [ ] nn.Sequential
- [ ] nn.conv2d()
- [ ] super(AlexNet, self).__init__()
- [ ] what is a softmax?


## Dropout
is basically "killing" some neurons randomly, so the other neurons need to learn to perform without them

it's like if I go blind, my other senses need to work extra hard

in AlexNet paper, they mention having 0.5 prob of a setting a hidden layer nuron output to zero

> he recently-introduced technique, called “dropout”, consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in back- propagation.

from the paper:
>  It is, therefore, forced to learn more robust features that are useful in conjunction with many different random subsets of the other neurons

and dropouts also prevent overfitting

> We use dropout in the first two fully-connected layers of Figure 2. Without dropout, our network exhibits substantial overfitting. 

## nn.linear
it's pytorch's way of making dense connections 

> applies a linear transformation to input data using weights and biases [^3]

> This transformation is represented by the formula y = xA^T + b, where x is the input, A is the weight, b is the bias, and y is the output. [^3]

but why is no one talking about the `T`
thanks to gemini - the `T` turns out to stand for transposing and pytorch takes care of this for you

torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
Parameters
- in_features (int) – size of each input sample
- out_features (int) – size of each output sample
- bias (bool) – If set to False, the layer will not learn an additive bias. Default: True


you can also use ____ function

## nn.BatchNorm2d

This stands for 2D Batch Normalization. Think of it as an automatic "volume regulator" for your network that helps it learn faster and more reliably.

After a convolutional layer, the output numbers can sometimes get very large or very small. Batch Norm fixes this by rescaling the numbers so they have an average of 0 and a standard deviation of 1.

    What it does: It normalizes the output of a layer across all the images in the current batch.
    Why it's useful: This stabilization prevents the network's learning process from spiraling out of control and generally makes training much faster and more stable. It's a very powerful technique.
    Why "2d"? Because it's designed specifically to work on the 2D "feature maps" that come out of nn.Conv2d layers.


## nn.Sequential

you could do it like this:

```python
# this would go inside forward()
x = self.conv1(x)
x = self.batchnorm1(x)
x = self.relu1(x)
x = self.pool1(x)
```
using Sequential, it becomes a bit cleaner 

```python
# In __init__
self.layer1 = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4),
    nn.BatchNorm2d(96),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2)
)

# In forward() - much cleaner!
x = self.layer1(x)
```

## nn.conv2d()
This is the main engine of a CNN. Imagine you are in a dark room holding a small square flashlight (this is the kernel or filter). The wall in front of you is the image.

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

    in_channels (int): How "deep" is the wall you are looking at? For the very first layer looking at an RGB image, this is 3 (Red, Green, Blue). For later layers, it's the number of out_channels from the previous convolutional layer.
    out_channels (int): How many different flashlights are you using? Each flashlight (kernel) is built to find a specific pattern, like a vertical edge, a specific curve, or a patch of colour. This number determines the depth of the output "feature map".
    kernel_size (int): How big is your flashlight's beam? kernel_size=3 means a 3x3 pixel square.
    stride (int): How far do you move the flashlight in each step? stride=1 means you move it one pixel at a time. A larger stride makes the output smaller.
    padding (int): Adds a border of zeros around the wall. This lets your flashlight check the very edges of the image properly and gives you control over the output size.

## super(AlexNet, self).__init__()
I should have known this, but I'm not super femilear with OOP, so that's the thing I want to get good at next
I have forgottn it, since I last learned it in javascript, it seems like - haven't really used it after learning it

When you create a child class, you must first properly initialize the parent class it's based on.
super().__init__() does exactly that. It runs the original __init__ method from the parent nn.Module. This sets up all the critical background machinery that allows PyTorch to track your layers, manage parameters, move the model to a GPU, etc.

Rule of thumb: If your class inherits from another class, the first line inside __init__ should almost always be super().__init__().

## softmax 

given -> [1,2,1,2,1]

it does ->  1. Exponentiate every number (raise e to the power of the score)
[e^1, e^2, e^1, e^2, e^1] which is [2.718, 7.389, 2.718, 7.389, 2.718]

then does ->  2. Sum the exponentiated results to get a total
2.718 + 7.389 + 2.718 + 7.389 + 2.718 = 22.932

then -> 3. Divide each exponentiated number by the total
[2.718/22.932, 7.389/22.932, ...]

output -> [0.119, 0.322, 0.119, 0.322, 0.119]


## Aside

ugh I wanted to have a faithful rebuild of AlexNet, using the same methods from the paper
but I get this when I try to train on my M1 Mac:

> NotImplementedError: The operator 'aten::avg_pool3d.out' is not currently implemented for the MPS device. If you want this op to be considered for addition please comment on https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in missing op as well as commit hash e2d141dbde55c2a4370fac5165b0561b6af4798b. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.

## References
[^1]: https://www.digitalocean.com/community/tutorials/relu-function-in-python
[^2]: https://www.digitalocean.com/community/tutorials/how-does-cnn-views-the-images#why-do-we-need-a-dense-neural-network
[^3]: https://docs.kanaries.net/topics/Python/nn-linear
[^4]: https://www.digitalocean.com/community/tutorials/alexnet-pytorch