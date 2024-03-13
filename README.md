# DL-Assignments

## Assignment 1
Implemented a Feed-Forward neural network architecture using torh.nn.Linear featuring four hidden layers, each comprising minimum 32 neurons (excluding input and output layers) on the MNIST Dataset. Trained the model  with ReLU and Signmoid activation functions. Employed the Cross-Entropy loss function and Stochastic Gradient Descent (SGD) optimizer with default parameters, setting the learning rate to 0.0003 and achieved an accuracy of ~90% <br>
After that Implemented everything defined above completely from scratch. Also implemented the back-propagation algorithm from scratch using only PyTorch tensor operations. Used advanced Regularization techniques like Gradient clipping to boost the accuracy of the from-scratch version and make it comparable to the inbuilt version.<br><br>



## Assignment 2
This contains the implementation of myriad architectures of CNNs on two datasets: CIFAR-10 and SpeechCommand V0.02. The following are the descriptions on different architectures:-

### RESNET 
![image](https://github.com/arjit06/DL-Assignments/assets/108218688/e3b91100-ee75-408b-bcaa-dab00285e0af) 
<br>Here, (3x3 Conv(1D/2D)) denotes a single convolution layer with a filter size of 3×3 for 2D data and 1×3 for 1D data, (Batch norm(1D/2D)) represents a single Batch normalization layer for 1D or 2D data, and (ReLU ) represents the rectified linear unit activation function. The task was to create a network comprising 18 such blocks and train it on both datasets. For the image dataset, 2D Convolutions and Batch normalization were used, while for the audio dataset, 1D Convolutions and Batch normalization. Cross-Entropy as the loss function and Adam as the optimizer with default parameters as specified in the script (Batch size: 128, Learning Rate: 0.001 , No. of Epochs: 64) were used .<br><br>

### VGG NETWORK
![image](https://github.com/arjit06/DL-Assignments/assets/108218688/6e656f94-9e17-41fe-bc01-ee17de4d90ec) 
<br>The Figure outlines the modified VGG architecture. After each pooling layer, the number of channels is reduced by 35%, and the kernel size is increased by 25% (with ceil rounding for float calculations). Here, (Conv
n-m) denotes the nth block and mth layer within that block. The task was to create a VGG network and train it on both image and audio datasets using the specified loss function and optimizer from the previous question. <br><br>

### INCEPTION (GOOGLE NET) 
![image](https://github.com/arjit06/DL-Assignments/assets/108218688/9dece05c-950f-4c1a-8d59-94bf25b4227f) 
<br>The Figure illustrates the configuration of a single inception block, where (n×n (CNA)) denotes a sequence of convolution, batch normalization, and ReLU activation with an n×n convolution filter. The task was to construct a modified inception network comprising 4 such blocks and train it on both audio and image data, following similar configurations as in the previous two questions. <br><br>

### CUSTOM NETWORK 
A custom network was to be designed using the following combination of blocks and following the channel reduction and kernel size increase as in VGG. <br><br>
Network Architecture
<ol type='a'>
<li>Input Layer</li>
<li>Residual Block × 2</li>
<li>Inception Block × 2</li>
<li>Residual Block × 1</li>
<li>Inception Block × 1</li>
<li>Residual Block × 1</li>
<li>Inception Block × 1</li>
<li>Residual Block × 1</li>
<li>Inception Block × 1</li>
<li>Classification Network</li>
</ol>

<br>
The following were the accuracies obtained on the different architectures:-
<table>
<tr>
  <th>Architecture</th>
  <th>IMAGE DATA</th>
  <th>AUDIO DATA</th>
</tr>
<tr>
  <td>RESNET</td>
  <td>70%</td>
  <td>90%</td>
</tr>
  <tr>
  <td>VGG</td>
  <td>80%</td>
  <td>80%</td>
</tr>
  <tr>
  <td>INCEPTION</td>
  <td>70%</td>
  <td>75%</td>
</tr>
  <tr>
  <td>CUSTOM</td>
  <td>90%</td>
  <td>90%</td>
</tr>
</table>



