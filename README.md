# This is the LSTM part of work for CSE 252C Project
## Abstract
This repo implements Nitish and Elman's work [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681) using Tensorflow framework.<br>
The work includes
- LSTM AutoEncoder
- LSTM future frames prediction

## Files
`extract-fc6.py` provides the code to extract UCF101 fc6 features into bin files.<br>
`extract-fc6-id3.py` extract ucf101 mixed-5c features into bin files<br>
`main.py` run lstm/gru classification training/testing<br>

`testEncoder.py` provides single number autoEncode/Decode process. Simply run `python testEncoder.py` will get the results.<br>
`update.py` (don't use) provides this process on UCF101 dataset, the input array could be <br>
 -  patches: 112x112x3 images directly<br>
 -  perceptons: 4096 dimensional fc6 features extracted from C3D model<br>
 

## Dataset and Reference
- __UCF-101__: [Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php)
- __Extract Frames__: Thanks to [This Repo](https://github.com/hx173149/C3D-tensorflow). Each single avi file is decoded with 5FPS in a single directory.
- __Model__: The pre-trained C3D model for extracting fc6 features is from [This work](https://github.com/wujinjun/C3D-tensorflow-UCF101-extrafc6)
- __Other Reference__: Here is another raw implementation by [Emansim](https://github.com/emansim/unsupervised-videos)

