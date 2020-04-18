# 360-images-VGG-based-Autoencoder-Pytorch
* PyTorch implementation of Autoencoder for 360 images , the encoder leverage vgg convolutions weight , in order to adapt 360 images characteristic
last maxpooling layer has removed ,third and fourth maxpooling layer are set to 4 pooling factor instead of 2 in order to have a receptive field of (580,580) which cover the whole input (576,288)   
to run this project just specify your root_dir in main.py .
