import torch
from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from troch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from Encoders import global_attention 

# create max pooling layer 
class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size, stride = 2):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.pool(x, kernel_size= self.kernel_size, stride= self.stride)
        return x

# create unpooling layer 
class Upsample(nn.Module):
    # specify the scale_factor for upsampling 
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
class Encoder(nn.Module):
    def  __init__(self):
        super(Encoder,self,pretainer = True).__init__()
        # Create encoder based on VGG16 architecture 
        # Change just 4,5 th maxpooling layer to 4 scale instead of 2 
        # select only convolutional layers first 5 conv blocks ,cahnge maxpooling=> enlarge receptive field
        # each neuron on bottelneck will see (580,580) all viewports  ,
        # input (576,288) , features numbers on bottelneck (9*4)*512, exclude last maxpooling
        encoder_list[
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size = 3)
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),            
            Downsample(kernel_size = 3)
            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),              
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(), 
            Downsample(kernel_size = 3 , stride = 4)
            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),              
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size = 3 , stride = 4)
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),              
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),                           
        ]
        self.encoder = torch.nn.Sequential(*Global_Attention_Encoder)
        print("encoder initialized")
        print("architecture len :",str(len(self.Autoencoder))) 

    def forward(self, input):
        x = self.encoder(input)
        return x
class Decoder(nn.Module):
    def  __init__(self):
        super(Decoder,self,pretainer = True).__init__()

        decoder_list=[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor= 4, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor= 4, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)
        self._initialize_weights()
        print("decoder initialized")
        print("architecture len :",str(len(self.Autoencoder))) 

    def forward(self, input):
        x = self.decoder(input)
        return x 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            
class Autoencoder(nn.Module):
    """
    In this model, we aggregate encoder and decoder
    """
    def  __init__(self , pretrained_encoder = True):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if pretrained_encoder:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth',progress=progress)
            self.encoder.load_state_dict(state_dict)
        print("Model initialized")
        print("architecture len :",str(len(self.Autoencoder)))

    def forward(self, input):
        x = self.encode(input)
        x = self.decoder = Decoder(x)
        return x 