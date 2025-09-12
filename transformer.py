# ==============================================================================
# Style Transfer Generator Network Architecture
# ==============================================================================
# This module defines the feed-forward generator network used for real-time
# style transfer, based on the architecture from Johnson et al.
#
# References:
# - Paper: https://arxiv.org/abs/1603.08155
# - Architecture Details: https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
# ==============================================================================

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["TransformerNetwork", "TransformerNetworkTanh"]

# --- Core Building Blocks ---

class ConvolutionalBlock(nn.Module):
    """
    A block consisting of ReflectionPad -> Conv2d -> Normalization.
    This is the primary building block for downsampling.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, norm_type: Optional[str] = "instance"):
        super().__init__()

        # Padding: Reflection padding is used to reduce border artifacts
        padding_size = kernel_size // 2
        
        layers = [nn.ReflectionPad2d(padding_size)]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        
        # Normalization
        if norm_type == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection.
    This allows for deeper networks while mitigating vanishing gradients.
    """
    def __init__(self, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, kernel_size, stride=1),
            nn.ReLU(),
            ConvolutionalBlock(channels, channels, kernel_size, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        return self.block(x) + residual

class UpsamplingBlock(nn.Module):
    """
    An upsampling block using ConvTranspose2d -> Normalization.
    This is the primary building block for decoding the feature map back into an image.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, output_padding: int, norm_type: Optional[str] = "instance"):
        super().__init__()

        # Transposed Convolution for upsampling
        padding_size = kernel_size // 2
        
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                     padding_size, output_padding)]
        
        # Normalization
        if norm_type == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm2d(out_channels, affine=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# --- Main Generator Architectures ---

class TransformerNetwork(nn.Module):
    """
    The primary feed-forward style transfer network.
    It processes an image in three stages:
    1. Downsampling: Encodes the input image into a compact feature map.
    2. Transformation: Applies a series of residual blocks to transform the features.
    3. Upsampling: Decodes the feature map back into a full-resolution stylized image.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Downsampling Blocks
        self.downsampling = nn.Sequential(
            ConvolutionalBlock(3, 32, kernel_size=9, stride=1),
            nn.ReLU(),
            ConvolutionalBlock(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            ConvolutionalBlock(64, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        # 2. Residual Transformation Blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # 3. Upsampling Blocks
        self.upsampling = nn.Sequential(
            UpsamplingBlock(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            UpsamplingBlock(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            # The final layer outputs a 3-channel image and uses no normalization.
            ConvolutionalBlock(32, 3, kernel_size=9, stride=1, norm_type=None)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: The stylized output image tensor.
        """
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        output = self.upsampling(x)
        return output

class TransformerNetworkTanh(TransformerNetwork):
    """
    A variant of the TransformerNetwork that adds a Tanh activation to the output.
    This scales the output pixel values to the range [-1, 1], which are then
    multiplied by a factor. This can produce images with higher contrast and a
    distinct, retro visual style.
    """
    def __init__(self, tanh_multiplier: float = 150.0):
        # First, initialize the parent class to build the base network
        super().__init__()
        
        # Now, append the Tanh layer to the existing upsampling sequence.
        # This avoids duplicating the entire nn.Sequential definition.
        self.upsampling.add_module("tanh", nn.Tanh())
        
        self.tanh_multiplier = tanh_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input through the network and applies the Tanh scaling.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: The stylized output tensor, scaled by the tanh_multiplier.
        """
        # Call the parent's forward pass, which now includes the Tanh layer
        base_output = super().forward(x)
        
        # Scale the Tanh output from [-1, 1] to [-multiplier, multiplier]
        return base_output * self.tanh_multiplier