# Notes for AlexNet / CNN from scratch

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

but in math, it's this scary looking thing:

$$ \text{ReLU}(x) = \max(0, x) = \begin{cases} 
0 & \text{if } x < 0 \\
x & \text{if } x \geq 0 
\end{cases} $$

Okay now that I actually wrote the formula, it's actually not so scary.

This helps keep the model non-linear and prevents eliminating vanishing gradient problem.

## 2. Convolution
What is a Convolution and why is it useful?

- A conv is a series of dot products between a kernel and a subset of the input image
- Filter kernel · input image = output feature map

So a CNN will typically have: [^2]
- Convolutional Layer
- ReLU Activation Function
- Pooling Layer
- Fully Connected Layer

## References
[^1]: https://www.digitalocean.com/community/tutorials/relu-function-in-python
[^2]: https://www.digitalocean.com/community/tutorials/how-does-cnn-views-the-images#why-do-we-need-a-dense-neural-network





