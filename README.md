# RNN-for-Language-Modelling
In this repo, I mainly implement the recurrent neural network (RNN) architecture using Python (without using deep learning frameworks) and explore how different hyperparameter settings influence the system's performance on two tasks, i.e., **language modelling** and **predicting subject-verb agreement**.  
\
Besides, based on the related work in [Linzen et al., 2016](https://arxiv.org/pdf/1611.01368.pdf), I also implement a long-short term memory (LSTM) model and a gated recurrent unit (GRU) with Pytorch, to evaluate how much these two models can improve the performance compared with vanilla RNN.

## Code

See [rnn.py](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/rnn.py) for code of vanilla RNN, and how the forward propagation and backpropagation through time (BPTT) are implemented. For task of language modelling, each input word will generate an output word, however, the subject-verb agreement will only predict one output once for a sentence. With regard to the implementation of LSTM and GRU, see the code [lstm_gru_trail.py](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/lstm_gru_trail.py).

## Data

The data I used here is a subset of the parsed Wikipedia corpus.
file| size
-----|-----
wiki-train.txt|50000 sentences
wiki-dev.txt|1000 sentences
wiki-test.txt|4000 sentences
vocab.wiki.txt|9954 words

## Language Modelling
### RNN
After tuning hyperparameters on the 2000 sentences in [wiki-train.txt](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/data/wiki-train.txt), the mean loss on [wiki-dev.txt](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/data/wiki-dev.txt) are listed as below:

|Lr   |Hidden | Lookback |Loss |Lr  |Hidden |Lookback |Loss|
|----|-------|----------|-----|----|-------|---------|----|
|0.05 | 25    | 0 |5.1366 |   0.1   |25 |0  |4.9994 |
|0.05 | 25    | 2 |5.1139 |   0.1   |25 |2  |4.9858 |
|0.05 | 25    | 5 |5.1216 |   0.1   |25 |5  |5.0011 |
|0.05 | 50    | 0 |5.0832 |   0.1   |50 |0  |4.9692 |
|0.05 | 50    | 2 |5.0677 |   0.1   |50 |2  |4.9595 |
|0.05 | 50    | 5 |5.0839 |   0.1   |50 |5  |4.9692 |
|0.5 | 25    | 0 |4.8626 |   1   |25 |0  |4.9301 |
|0.5 | 25    | 2 |4.8597 |   1   |25 |2  |4.8710  |
|0.5 | 25    | 5 |4.8576 |   1   |25 |5  |4.8548  |
|0.5 | 50    | 0 |4.8551 |   1   |50 |0  |4.9617  |
|0.5 | 50    | 2 |4.8537 |   1   |50 |2  |4.8791 |
|0.5 | 50    | 5 |4.8428 |   **1**  |**50** |**5**  |**4.8366**| 

* From the results we can see that models with 50 hidden units always perform better than those with 25 hidden units, because the larger hidden layer can enable the network to fit more complex functions.
* When learning rate is not too small, models with larger look-back step always performbetter, because this will enable the RNN to encode longer history of an input sequence.
* For small learning rate, most of the model is not convergent yet (Since I set *annealing rate=20*).  
\
Then I train the model with the best hyper-parametercombinations (i.e., learning rate=1, hidden units=50, look back step=5, vocabulary size=2000, annealing=20), using the whole training set with 50000 sentences. The model gets the best mean loss equals 4.3126 on dev set (in the 18th epoch), and 4.4027 on test set. This is reasonable because the model is tuned on dev set, and the distribution of data in test set may be slightly different with that in the dev set. The adjusted and unadjusted perplexity of the model is 109.9101 and 157.9570 respectively. I also find the loss trained in the full training set is much lower than that trained using partial data, and it is because when training with the full set, the parameters will have more updates in each epoch, and the model can learn more information.
### LSTM & GRU
![alt text](https://github.com/HrBlack/RNN-for-Language-Modelling/blob/master/source/rnn_lstm_gru.jpg)
It can be from the picture that observed that LSTM and GRU outperform vanilla RNN in terms of both training loss and dev loss, which matches my expectation well. I also find that GRU converges much faster than LSTM, that is because there are less parameters in GRU model.

## Subject-verb Agreement

Firstly, I train the models using the same hyper-parameter settings above, with 1000 sentences in the wiki-train.txt and a vocabulary size of 2000. Then I evaluate the performance of the models on the wiki-dev.txt. The optimal hyper-parameters for this problem is: learning rate=1, look-back step=5, annealing=20, hidden units=25. The model can obtain a dev loss equals 0.5864 and accuracy equals 69.8%. Using the aforementioned hyper-parameter combination, I train the model on the full data set with 50000 sentences. Finally, I obtain the best performance at the 18th epoch: dev loss equals 0.2981 and devaccuracy equals 89.95% on dev set. This best model finally has the test loss equals 0.2849 and test accuracy equals 89.93% on the whole test set.  
\
For this task, I also use the RNN language model trained for language modelling to predict the subject-verb agreement. And the prediction accuracy that I is 67.7% for dev set and 66.5% for test set. Compared to the results (89.93%) above, we can see that the trained RNN language model performs quite worse than the one trained specifically for subject-verb agreement. That is because when training the general RNN language model, the network will capture all the information in the whole sentence, while the dedicate model for subject-verb agreement only focuses on the subject and verb relationships. Hence the dedicate model can more efficiently extract agreement information from the training corpus.
