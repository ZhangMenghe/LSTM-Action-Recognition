# This is the LSTM part of work for CSE 252C Project
## Abstract
This repo implements Nitish and Elman's work [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681) using Tensorflow framework.<br>
The work includes
- LSTM AutoEncoder
- LSTM future frames prediction

## Files
`testEncoder.py` provides single number autoEncode/Decode process. Simply run `python testEncoder.py` will get the results.
`update.py` provides this process on UCF101 dataset, the input array could be 
 -  patches: 112x112x3 images directly
 -  perceptons: 4096 dimensional fc6 features extracted from C3D model
 
`extract-fc6.py` provides the code to extract UCF101 fc6 features into bin files.
