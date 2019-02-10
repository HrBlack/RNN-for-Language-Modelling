# RNN-for-Language-Modelling
In this repo, I mainly implement the recurrent neural network (RNN) architecture using Python (without using deep learning frameworks) and explore how different hyperparameter settings influence the system's performance on two tasks, i.e., **language modelling** and **predicting subject-verb agreement**.  
Besides, based on the related work in [Linzen et al., 2016](https://arxiv.org/pdf/1611.01368.pdf), I also implement a long-short term memory (LSTM) model and a gated recurrent unit (GRU) with Pytorch, to evaluate how much these two models can improve the performance compared with vanilla RNN.

## Code

See [rnn.py](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/rnn.py) for code of vanilla RNN, and how the forward propagation and backpropagation through time (BPTT) are implemented. For task of language modelling, each input word will generate an output word, however, the subject-verb agreement will only predict one output once for a sentence. With regard to the implementation of LSTM and GRU, see the code [lstm_gru_trail.py](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/lstm_gru_trail.py).

## Data

The data I used here is a subset of the parsed Wikipedia corpus.
file| size
wiki-train.txt|50000 sentences
wiki-dev.txt|1000 sentences
wiki-test.txt|4000 sentences
vocab.wiki.txt|9954 words

## Language Modelling
After tuning hyperparameters on the 2000 sentences in [wiki-train.txt](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/data/wiki-train.txt), the mean loss on [wiki-dev.txt](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/data/wiki-dev.txt) are listed as below:
Lr   |Hidden | Lookback |Loss |Lr |Hidden |Lookback |Loss
0.05 | 25    | 0 |5.1366 |   0.1   |25 |0  |4.9994 
0.05 | 25    | 2 |5.1139 |   0.1   |25 |2  |4.9858 
0.05 | 25    | 5 |5.1216 |   0.1   |25 |5  |5.0011 
0.05 | 50    | 0 |5.0832 |   0.1   |50 |0  |4.9692 
0.05 | 50    | 2 |5.0677 |   0.1   |50 |2  |4.9595 
0.05 | 50    | 5 |5.0839 |   0.1   |50 |5  |4.9692 
0.5 | 25    | 0 |4.8626 |   1   |25 |0  |4.9301 (16) 
0.5 | 25    | 2 |4.8597 |   1   |25 |2  |4.8710 (17) 
0.5 | 25    | 5 |4.8576 |   1   |25 |5  |4.8548 (17) 
0.5 | 50    | 0 |4.8551 |   1   |50 |0  |4.9617 (16) 
0.5 | 50    | 2 |4.8537 |   1   |50 |2  |4.8791 (18)
0.5 | 50    | 5 |4.8428 |   **1**   |**50 |5  |4.8366 (18)** 
