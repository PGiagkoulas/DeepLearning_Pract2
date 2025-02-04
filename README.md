# Human Activity Recognition with improved LSTM models
Performance comparison based on [CNNLSTM](https://arxiv.org/abs/1411.4389), [Convolutional LSTM](https://arxiv.org/abs/1506.04214), [Stacked LSTM](https://arxiv.org/abs/1303.5778), [Residual LSTM](https://arxiv.org/abs/1609.08144) on Human Acitivity Dataset [UCI_HAR_Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Environment Requirements 

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) 

[Anaconda](https://www.anaconda.com)

<ul style="list-style: none;">
  
  <li>python 3.6.7</li>
  <li>Tensorflow 1.12.0</li>
  <li>keras 2.2.4</li>

</ul>

For StackedLSTM and ResLSTM, the experiments must be executed on one GPU with CUDA support.  

## Explaination for every function file

| filename          | Functionality      |
| ------------- |---------------|
| allModels.py | The definition for 4 different models |
| all_utils.py | The different supplementary infrastructure functions we used |
| runExperiment.py| The execution of the experiments |

## Running the experiments

To run the program, activate proper environments first:

```python
source activate keras_env
```

Flags available to modify our experiment setting


| arguments          | setting    |
| ------------- |---------------|
| repeats| The number we repeat our experiments |
| grid | Weather execute grid search or not, set True or False  |
| arch | The choice of model architecture, chosen in ['conv_lstm', 'cnn_lstm', 'res_lstm', 'stacked_lstm'] |

start training and test one model with a command like this:

```bat
python runExperiment.py --grid True --arch conv_lstm --repeat 5
```


