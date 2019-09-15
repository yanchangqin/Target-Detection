import torch
import torch.nn as nn
import torch.nn.functional as f

class UpsampleLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        return f.interpolate(x,scale_factor=2,mode='nearest')

class ConverlutionalLayer(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super().__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.sub_module(x)

class ResidualLayer(nn.Module):

    def __init__(self,in_channels):
        super().__init__()

        self.sub_module = nn.Sequential(
            ConverlutionalLayer(in_channels,in_channels//2,1,1,0),
            ConverlutionalLayer(in_channels//2,in_channels,3,1,1)
        )
    def forward(self, x):

        return x+self.sub_module(x)

class DownsampleLayer(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.sub_module = nn.Sequential(
            ConverlutionalLayer(in_channels,out_channels,3,2,1)
        )
    def forward(self, x):
        return self.sub_module(x)

class ConvolutionalsetLayer(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.sub_module = nn.Sequential(
            ConverlutionalLayer(in_channels,out_channels,1,1,0),
            ConverlutionalLayer(out_channels,in_channels, 3, 1, 1),

            ConverlutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConverlutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConverlutionalLayer(in_channels, out_channels, 1, 1, 0),
        )
    def forward(self, x):
        return  self.sub_module(x)

class MainNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.trunk_52 = nn.Sequential(
            ConverlutionalLayer(3,32,3,1,1),
            ConverlutionalLayer(32,64,3,2,1),

            ResidualLayer(64),
            DownsampleLayer(64,128),

            ResidualLayer(128),
            ResidualLayer(128),
            DownsampleLayer(128,256),

            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            )

        self.trunk_26 = nn.Sequential(
            DownsampleLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512)
        )

        self.trunk_13 = nn.Sequential(

            DownsampleLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),

            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),

            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),

            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )
        self.convolutionalset_13 =nn.Sequential(
            ConvolutionalsetLayer(1024,512)
        )

        self.detective_13 = nn.Sequential(
            ConverlutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,45,1,1,0)
        )
        self.up_26 = nn.Sequential(
            ConverlutionalLayer(512,256,1,1,0),
            UpsampleLayer()

        )
        self.convolutionalset_26 = nn.Sequential(
            ConvolutionalsetLayer(768, 256)

        )
        self.detective_26 = nn.Sequential(
            ConverlutionalLayer(256,512,3,1,1),
            nn.Conv2d(512,45,1,1,0)
        )
        self.up_52 = nn.Sequential(
            ConverlutionalLayer(256,128,1,1,0),
            UpsampleLayer()
        )
        self.convolutionalset_52 = nn.Sequential(
            ConvolutionalsetLayer(384, 128)
        )
        self.detective_52 = nn.Sequential(
            ConverlutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 45, 1, 1, 0)
        )
    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convolution_out_13 = self.convolutionalset_13(h_13)
        detection_out_13 = self.detective_13(convolution_out_13)

        up_out_26 = self.up_26(convolution_out_13)
        route_out_26 = torch.cat((up_out_26,h_26),dim=1)
        convolution_out_26 = self.convolutionalset_26(route_out_26)
        detection_out_26 = self.detective_26(convolution_out_26)

        up_out_52 = self.up_52(convolution_out_26)
        route_out_52 = torch.cat((up_out_52,h_52),dim=1)
        convolution_out_52 = self.convolutionalset_52(route_out_52)
        detection_out_52 = self.detective_52(convolution_out_52)

        return detection_out_13,detection_out_26,detection_out_52

if __name__ == '__main__':
    trunk = MainNet()

    x = torch.Tensor(2, 3, 416, 416)

    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)