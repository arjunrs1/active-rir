import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class MapEncoder(nn.Module):
    def __init__(self, input_channels=2, crop_size=150, downsample_scale=2):
        super().__init__()
        self.crop_size = crop_size
        self.downsample_scale = downsample_scale

        # Initialize a ResNet-18 model
        self.cnn = models.resnet18(pretrained=False)

        # Adjust the first convolutional layer to accept two channels and not change the spatial resolution
        self.cnn.conv1 = nn.Conv2d(input_channels, self.cnn.conv1.out_channels,
                                   kernel_size=7, stride=1, padding=3, bias=False)

        # Initialize the weights for the new conv1 layer
        nn.init.kaiming_normal_(self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Replace the fully connected layer with global average pooling
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn.fc = nn.Sequential()  # Remove the final fully connected layer

        # Downsampling layer
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=self.downsample_scale, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

    @property
    def n_out_feats(self):
        """
        Get the number of output features from the encoder.
        :return: Number of output features from the encoder.
        """
        return 512

    def forward(self, observations,):

        # Cropping the global map by removing 150 pixels from each side
        #cropped_map = observations['occupancy_map'][:, :, self.crop_size:-self.crop_size, self.crop_size:-self.crop_size]

        # Downsample the map
        downsampled_map = self.downsample(observations['occupancy_map'])

        # Forward pass through the modified ResNet-18
        features = self.cnn(downsampled_map)
        return features



