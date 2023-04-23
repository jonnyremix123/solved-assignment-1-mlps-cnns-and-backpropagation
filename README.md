Download Link: https://assignmentchef.com/product/solved-assignment-1-mlps-cnns-and-backpropagation
<br>
In this assignment you will learn how to implement and train basic neural architectures like MLPs and CNNs for classification tasks. Modern deep learning libraries come with sophisticated functionalities like abstracted layer classes, automatic differentiation, optimizers, etc.

<ul>

 <li>To gain an in-depth understanding we will, however, first focus on a basic implementation of a MLP in numpy in exercise 1. This will require you to understand backpropagation in detail and to derive the necessary equations first.</li>

 <li>In exercise 2 you will implement a MLP in PyTorch and tune its performance by adding additional layers provided by the library.</li>

 <li>In order to learn how to implement custom operations in PyTorch you will reimplement a batch-normalization layer in exercise 3.</li>

 <li>Exercise 4 aims at implementing a simple CNN in PyTorch.</li>

</ul>

Python and PyTorch have a large community of people eager to help other people. If you have coding related questions: (1) read the documentation, (2) search on Google and StackOverflow, (3) ask your question on StackOverflow or Piazza and finally (4) ask the teaching assistants.

<h1>1           MLP backprop and NumPy implementation (50 <sub>points)</sub></h1>

Consider a generic MLP for classification tasks which is consisting of <em>N </em>layers<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. We want to train the MLP to learn a mapping from the input space R<em><sup>d</sup></em><sup>0 </sup>to a probability mass function (PMF) over <em>d<sub>N </sub></em>classes given a dataset consisting of <em>S </em>tuples of input vectors and targets. The superscript (0) is added to the input vectors <em>x</em><sup>(0) </sup>∈ R<em><sup>d</sup></em><sup>0 </sup>as notational convenience for identifying the input as the activation of a 0-th layer. The targets could be any pmf but we assume them to be one-hot

encoded in the following. Each layer <em>l </em>will first apply an affine mapping

LinearModule.forward: which is parameterized by weights <em>W</em><sup>(<em>l</em>) </sup>∈ R<em><sup>d</sup></em><em><sup>l</sup></em><sup>×<em>d</em></sup><em><sup>l</sup></em><sup>−1 </sup>and biases <em>b</em><sup>(<em>l</em>) </sup>∈ R<em><sup>d</sup></em><em><sup>l</sup></em><em>. </em>Subsequently, nonlinearities are applied to compute activations <em>x</em><sup>(<em>l</em>) </sup>from the pre-nonlinearity activations <em>x</em>˜<sup>(<em>l</em>)</sup><em>. </em>For all <em>hidden </em>layers we choose leaky rectified linear units (leaky ReLU), that is,

LeakyReLUModule.forward:

<em>x</em>˜<sup>(<em>l</em>) </sup>7→ <em>x</em><sup>(<em>l</em>) </sup>:= LeakyReLU

:= max    <em>.</em>

where <em>a </em>defines the slope in the negative part of the domain.

Since the desired output should be a valid pmf a softmax activation is applied in the <em>output </em>layer:

SoftmaxModule.forward: <em>x</em>˜<sup>(<em>N</em>) </sup>→7           <em>x</em><sup>(<em>N</em>) </sup>:= softmax

Note that both the maximum operation and the exponential function are applied elementwise when acting on a vector. A categorical cross entropy loss between the predicted and target distribution is applied,

CrossEntropyModule.forward:

<em>,</em>

where the last step holds for <em>t </em>being one-hot. Here  denotes the <em>t</em>-th component of the softmax output.

<h2>1.1           Analytical derivation of gradients</h2>

For optimizing the network via gradient descent it is necessary to compute the gradients w.r.t. all weights and biases. In the definition of the MLP above we split the forward computation into several <em>modules</em>. It turns out that, making use of the chain rule, the gradient calculations can be split in a similar way into <em>gradients of modules</em>. For each operation performed in the forward pass, a corresponding gradient computation can be performed to <em>propagate the gradients back </em>through the network. You will first compute the partial derivatives of each module w.r.t. its inputs and subsequently put these together to a chain to get the final backpropagation computations. Your answers will be the cornerstone of the MLP NumPy implementation that follows afterwards.

Note that  is the backpropagation equation of the last layer, that is, it corresponds to

CrossEntropyModule.backward :              <em>.</em>

Question 1.1 b)                                                                                                           (15 points)

Using the gradients of the modules calculate the gradients

<em>,</em>

and

by <em>propagating back </em>the gradients of the output of each module to the parameters and inputs of the module. The errors on the outputs of each operation occurring <em>on the right-hand sides </em>do not have to be expanded in the result since they were computed in the previous step of the backpropagation algorithm. Please give the final result in form of matrix (or in general tensor-) multiplications.

<em>Hint: In index notation the products on the right hand side can be written in components like e.g. </em><em>. </em><em>Make sure to not confuse the indices which might occur in a transposed form.</em>

The gradients calculated in the last exercise are the gradients occurring in the backpropagation equations:

SoftmaxModule.backward :

LeakyReLUModule.backward :        LinearModule.backward :

The backpropagation algorithm can be seen as a form of dynamic programming since it makes use of previously computed gradients to compute the current gradient. Note that it requires all activations <em>x</em><sup>(<em>l</em>) </sup>to be stored in order to propagate the gradients back from  to <em>. </em>In the case of a MLP, the memory cost of storing the weights exceeds the cost of storing the activations but for CNNs the latter typically make up the largest part of the memory consumption.

So far we only considered single samples being fed into the network. In practice we typically use batches of input samples which are processed by the network in parallel. The total loss

<em>L</em>total     individual<em>,</em>

is then defined as the mean value of the individual samples’ losses. Here <em>L</em><sub>individual </sub>is the cross entropy loss as used before which depends on <em>x</em><sup>(0)<em>,s </em></sup>via <em>x</em><sup>(<em>N</em>)<em>,s</em></sup>. In addition to major computational benefits when running on GPU, performing gradient descent in batches helps to reduce the variance of the gradients.

<h2>1.2           NumPy implementation</h2>

For those who are not familiar with Python and NumPy it is highly recommended to get through the <a href="https://docs.scipy.org/doc/numpy/user/quickstart.html">NumPy tutorial</a><a href="https://docs.scipy.org/doc/numpy/user/quickstart.html">.</a>

To simplify implementation and testing we have provided to you an interface to work with

<a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a> data in cifar10_utils.py. The CIFAR-10 dataset consists of 60000 32×32 color images in 10 classes, with 6000 images per class. The file cifar10_utils.py contains utility functions that you can use to read CIFAR-10 data. Read through this file to get familiar with the interface of the Dataset class. The main goal of this class is to sample new batches, so you don’t need to worry about it. To encode labels we are using an <a href="https://en.wikipedia.org/wiki/One-hot">one-hot encoding of labels</a><a href="https://en.wikipedia.org/wiki/One-hot">.</a>

<em>Please do not change anything in this file</em>. Usage examples:

<ul>

 <li>Prepare CIFAR10 data:</li>

</ul>

import cifar10_utils cifar10 = cifar10_utils . get_cifar10 ( ’ cifar10 / cifar −10−batches−py ’ )

<ul>

 <li>Get a new batch with the size of <em>batch_size </em>from the train set: x , y = cifar10 [ ’ train ’ ] . next_batch( batch_size )</li>

</ul>

Variables x and y are numpy arrays. The shape of x is [<em>batch</em>_<em>size,</em>3<em>,</em>32<em>,</em>32], the shape of y is [<em>batch</em>_<em>size,</em>10].

<ul>

 <li>Get test images and labels: x , y = cifar10 . test . images , cifar10 . test . labels</li>

</ul>

<em>Note</em>: For multi-layer perceptron you will need to reshape x that each sample is represented by a vector.

Question 1.2                                                                                                                (15 points)

Implement a multi-layer perceptron using purely NumPy routines. The network should consist of <em>N </em>linear layers with leaky ReLU activation functions followed by a final linear layer. The number of hidden layers and hidden units in each layer are specified through the command line argument dnn_hidden_units. As loss function, use the common cross-entropy loss for classification tasks. To optimize your network you will use the <a href="http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent">mini-batch stochastic gradient descent algorithm</a><a href="http://sebastianruder.com/optimizing-gradient-descent/index.html#minibatchgradientdescent">. </a>Implement all modules in the files modules.py and mlp_numpy.py.

Part of the success of neural networks is the high efficiency on graphical processing units (GPUs) through matrix multiplications. Therefore, all of your code should make use of matrix multiplications rather than iterating over samples in the batch or weight rows/columns. Implementing multiplications by iteration will result in a penalty.

Implement training and testing scripts for the MLP inside train_mlp_numpy.py. Using the default parameters provided in this file you should get an accuracy of around 0.46 for the entire <em>test </em>set for an MLP with one hidden layer of 100 units. Carefully go through all possible command line parameters and their possible values for running train_mlp_numpy.py. You will need to implement each of these into your code. Otherwise we can not test your code. Provide accuracy and loss curves in your report for the default values of parameters.

<h1>2           PyTorch MLP   (20 points)</h1>

The main goal of this part is to make you familiar with <a href="https://pytorch.org/">PyTorch</a><a href="https://pytorch.org/">.</a> PyTorch is a deep learning framework for fast, flexible experimentation. It provides two high-level features:

<ul>

 <li>Tensor computation (like NumPy) with strong GPU acceleration</li>

 <li>Deep Neural Networks built on a tape-based autodiff system</li>

</ul>

You can also reuse your favorite python packages such as NumPy, SciPy and Cython to extend PyTorch when needed.

There are several tutorials available for PyTorch:

<ul>

 <li><a href="https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">Deep Learning with PyTorch: A 60 Minute Blitz</a></li>

 <li><a href="https://pytorch.org/tutorials/beginner/pytorch_with_examples.html">Learning PyTorch with Examples</a></li>

 <li><a href="https://pytorch.org/tutorials/beginner/former_torchies_tutorial.html">PyTorch for former Torch users</a></li>

</ul>

Question 2                                                                                                                    (20 points)

Implement the MLP in mlp_pytorch.py file by following the instructions inside the file. The interface is similar to mlp_numpy.py. Implement training and testing procedures for your model in train_mlp_pytorch.py by following instructions inside the file. Using the same parameters as in Question 1.2, you should get similar accuracy on the test set.

Before proceeding with this question, convince yourself that your MLP implementation is correct. For this question you need to perform a number of experiments on your MLP to get familiar with several parameters and their effect on training and performance. For example you may want to try different regularization types, run your network for more iterations, add more layers, change the learning rate and other parameters as you like. Your goal is to get the best test accuracy you can. You should be able to get <em>at least 0.52 </em>accuracy on the test set but we challenge you to improve this. List modifications that you have tried in the report with the results that you got using them. Explain in the report how you are choosing new modifications to test. Study your best model by plotting accuracy and loss curves.

<h1>3           Custom Module: Batch Normalization   (20 <sub>points)</sub></h1>

Deep learning frameworks come with a big palette of preimplemented operations. In research it is, however, often necessary to experiment with new custom operations. As an example you will reimplement the <a href="https://arxiv.org/abs/1502.03167">Batch Normalization</a> module as a custom operations in PyTorch. This can be done by either relying on automatic differentiation (Sec. 3.1) or by a manual implementation of the backward pass of the operation (Sec. 3.2).

The batch normalization operation takes as input a minibatch  consisting of <em>B </em>samples in R<em><sup>C </sup></em>where <em>C </em>denotes the number of channels. It first normalizes each neuron’s<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>value over the batch dimension to zero mean and unit variance. In order to allow for different values it subsequently rescales and shifts the normalized values by learnable parameters <em>γ </em>∈ R<em><sup>C </sup></em>and <em>β </em>∈ R<em><sup>C</sup></em><em>. </em>Writing the neuron index out explicitly by a subscript, e.g. <em>x<sup>s</sup><sub>i</sub>, i </em>= 1<em>…C</em>, Batch Normalization can be defined by:

<ol>

 <li>compute mean:</li>

</ol>

=1

<ol start="2">

 <li>compute variance:</li>

 <li>normalize:<em>, </em>with a constant to avoid numerical instability.</li>

 <li>scale and shift:</li>

</ol>

Note that the notation differs from the one chosen in the original paper where the authors chose to not write out the channel index explicitly.

<h2>3.1           Automatic differentiation</h2>

The suggested way of joining a series of elementary operations to form a more complex computation in PyTorch is via <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module">nn.Modules</a><a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module">.</a> Modules implement a method forward which, when called, simply executes the elementary operations as specified in this function. The autograd functionality of PyTorch records these operations as usual such that the backpropagation works as expected. The advantage of using modules over standard objects or functions packing together these operations lies in the additional functionality which they provide. For example, modules can be associated with <a href="https://pytorch.org/docs/stable/nn.html#parameters">nn.Parameters</a><a href="https://pytorch.org/docs/stable/nn.html#parameters">.</a> All parameters of a module or whole network can be easily accessed via model.parameters(), types can be changed via e.g. model.float() or parameters can be pushed to the GPU via model.cuda(), see the documentation for more information.

Question 3.1                                                                                                                (10 points)

Implement the Batch Normalization operation as a nn.Module at the designated position in the file custom_batchnorm.py. To do this, register <em>γ </em>and <em>β </em>as nn.Parameters in the __init__ method. In the forward method, implement a check of the correctness of the input’s shape and perform the forward pass.

<h2>3.2           Manual implementation of backward pass</h2>

In some cases it is useful or even necessary to implement the backward pass of a custom operation manually. This is done in terms of <a href="https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function">torch.autograd.Functions</a><a href="https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function">.</a> Autograd function objects necessarily implement a forward and a backward method. A call of a function instance records its usage in the computational graph such that the corresponding gradient computation can be performed during backpropagation. Tensors which are passed as inputs to the function will automatically get their attribute requires_grad set to False inside the scope of forward. This guarantees that the operations performed inside the forward method are <em>not </em>recorded by the autograd system which is necessary to ensure that the gradient computation is not done twice. Autograd functions are automatically passed a

<em>context </em>object in the forward and backward method which can

<ul>

 <li>store tensors via ctx.save_for_backward in the forward method</li>

 <li>access stored tensors via ctx.saved_tensors in the backward method</li>

 <li>store non-tensorial constants as attributes, e.g. ctx.foo = bar</li>

 <li>keep track of which inputs require a gradients via ctx.needs_input_grad</li>

</ul>

The forward and backward methods of a torch.autograd.Function object are typically not called manually but via the apply method which keeps track of registering the use of the function and creates and passes the context object. For more information you can read <a href="https://pytorch.org/docs/0.3.1/notes/extending.html">Extending PyTorch</a> and <a href="https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html">Defining new autograd functions</a><a href="https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html">.</a>

Since we want to implement the backward pass of the Batch Norm operation manually we first need to compute its gradients.

Having calculated all necessary equations we can implement the forward- and backward pass as a torch.autograd.Function. It is very important to validate the correctness of the manually implemented backward computation. This can be easily done via <a href="https://pytorch.org/docs/stable/autograd.html?highlight=gradcheck#torch.autograd.gradcheck">torch.autograd.gradcheck</a> which compares the analytic solution with a finite differences approximation. These checks are recommended to be done in double precision.

Question 3.2 b)                                                                                                             (3 points)

Implement the Batch Norm operation as a torch.autograd.Function. Make use of the <em>context </em>object described above. To save memory do not store tensors which are not needed in the backward operation. Do not perform unnecessary computations, that is, if the gradient w.r.t. an input of the autograd function is not required, return None for it.

<em>Hint: If you choose to use torch.var for computing the variance be aware that this function uses Bessel’s correction by default. Since the variance of the Batch Norm operation is defined without this correction you have to set the option unbiased=False as otherwise your gradient check will fail.</em>

Since the Batch Norm operation involves learnable parameters, we need to create a nn.Module which registers these as nn.Parameters and calls the autograd function in its forward method.

<h1>4           PyTorch CNN   (10 points)</h1>

At this point you should have already noticed that the accuracy of MLP networks is far from being perfect. A more suitable type of architecture to process image data is the CNN. The main advantage of it is applying convolutional filters to the input images. In this part of the assignment you are going to implement a small version of the popular <a href="https://arxiv.org/abs/1409.1556">VGG network</a><a href="https://arxiv.org/abs/1409.1556">.</a>

<table width="360">

 <tbody>

  <tr>

   <td width="73">Name</td>

   <td width="63">Kernel</td>

   <td width="51">Stride</td>

   <td width="64">Padding</td>

   <td width="108">Channels In/Out</td>

  </tr>

  <tr>

   <td width="73">conv1</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">3<em>/</em>64</td>

  </tr>

  <tr>

   <td width="73">maxpool1</td>

   <td width="63">3×3</td>

   <td width="51">2</td>

   <td width="64">1</td>

   <td width="108">64<em>/</em>64</td>

  </tr>

  <tr>

   <td width="73">conv2</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">64<em>/</em>128</td>

  </tr>

  <tr>

   <td width="73">maxpool2</td>

   <td width="63">3×3</td>

   <td width="51">2</td>

   <td width="64">1</td>

   <td width="108">128<em>/</em>128</td>

  </tr>

  <tr>

   <td width="73">conv3_a</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">128<em>/</em>256</td>

  </tr>

  <tr>

   <td width="73">conv3_b</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">256<em>/</em>256</td>

  </tr>

  <tr>

   <td width="73">maxpool3</td>

   <td width="63">3×3</td>

   <td width="51">2</td>

   <td width="64">1</td>

   <td width="108">256<em>/</em>256</td>

  </tr>

  <tr>

   <td width="73">conv4_a</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">256<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">conv4_b</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">512<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">maxpool4</td>

   <td width="63">3×3</td>

   <td width="51">2</td>

   <td width="64">1</td>

   <td width="108">512<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">conv5_a</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">512<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">conv5_b</td>

   <td width="63">3×3</td>

   <td width="51">1</td>

   <td width="64">1</td>

   <td width="108">512<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">maxpool5</td>

   <td width="63">3×3</td>

   <td width="51">2</td>

   <td width="64">1</td>

   <td width="108">512<em>/</em>512</td>

  </tr>

  <tr>

   <td width="73">linear</td>

   <td width="63">−</td>

   <td width="51">–</td>

   <td width="64">–</td>

   <td width="108">512<em>/</em>10</td>

  </tr>

 </tbody>

</table>

Table 1. Specification of ConvNet architecture. All <em>conv </em>blocks consist of 2D-convolutional layer, followed by Batch Normalization layer and ReLU layer.

Question 4                                                                                                                    (10 points)

Implement the ConvNet specified in Table 1 inside convnet_pytorch.py file by following the instructions inside the file. Implement training and testing procedures for your model in train_convnet_pytorch.py by following instructions inside the file. Use <a href="https://arxiv.org/abs/1412.6980">Adam optimizer</a> with default learning rate. Use default PyTorch parameters to initialize convolutional and linear layers. With default parameters you should get around <em>0.75 </em>accuracy on the test set. Study the model by plotting accuracy and loss curves.




<h1>Deliverables</h1>

Create ZIP archive containing your report and all Python code. Please preserve the directory structure as provided in the Github repository for this assignment. Give the ZIP file the following name: lastname_assignment1.zip where you insert your lastname. Please submit your deliverable through Canvas. We cannot guarantee a grade for the assignment if the deliverables are not handed in according to these instructions.

<a href="#_ftnref1" name="_ftn1">[1]</a> We are counting the output as a layer but not the input.

<a href="#_ftnref2" name="_ftn2">[2]</a> In the case of CNNs, normalization is done for each channel individually with statistics computed over the batch- and spatial dimensions.