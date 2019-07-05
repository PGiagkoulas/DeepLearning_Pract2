# Human Activity Recognition with improved LSTM models
Performance comparison based on [CNNLSTM](https://arxiv.org/abs/1411.4389), [ConvolutionalLSTM](https://arxiv.org/abs/1506.04214), [Stacked LSTM](https://arxiv.org/abs/1303.5778), [Residual LSTM](https://arxiv.org/abs/1609.08144) on Human Acitivity Dataset [UCI_HAR_Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Environment Requirements 

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) 

[Anaconda](https://www.anaconda.com)

<ul style="list-style: none;">
  
  <li>python 3.6.7</li>
  <li>Tensorflow 1.12.0</li>
  <li>keras 2.2.4</li>

</ul>

For StackedLSTM and ResLSTM, the experiments must be executed on one GPU with CUDA support.  

| filename          | Functionality      |
| ------------- |---------------|
| SNN/snn_fashion_mnist.py | The original structure and hyperparameters |
| SNN/snn_fashion_mnist_lr_optimizer.py | Compare Momentum and Adam with different learning rates |
| SNN/snn_fashion_mnist_dropout.py | Compare different dropout rates |
| SNN/snn_fashion_mnist_optimal.py|           The final model with optimal settings|
