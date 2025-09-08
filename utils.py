# Import necessary libraries
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List, Optional

## Core Tensor and Matrix Operations
def calculate_gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix for a given feature map tensor.
    The Gram matrix captures the style information by measuring feature correlations.

    Args:
        feature_map (torch.Tensor): 
        A 4D tensor of shape (B, C, H, W), where B is batch size, 
        C is channels,
        H is height, and W is width.

    Returns:
        torch.Tensor: The computed Gram matrix.
    """
    batch_size, num_channels, height, width = feature_map.shape
    # Reshape the tensor to combine spatial dimensions
    flattened_features = feature_map.view(batch_size, num_channels, height * width)
    flattened_features_transposed = flattened_features.transpose(1, 2)
    
    # Perform batch matrix multiplication
    gram = torch.bmm(flattened_features, flattened_features_transposed)
    
    # Normalize by the number of elements in the feature map
    normalization_factor = num_channels * height * width
    return gram / normalization_factor

## Image I/O and Display Utilities
def read_image_from_path(file_path: str) -> np.ndarray:
    """Loads an image from a file path using OpenCV.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.ndarray: The image loaded as a NumPy array in BGR format.
    """
    image = cv2.imread(file_path)
    return image

def save_image_to_path(image_array: np.ndarray, output_path: str):
    """Saves a NumPy array as an image file.

    Args:
        image_array (np.ndarray): The image data.
        output_path (str): The path where the image will be saved.
    """
    # Ensure pixel values are within the valid 0-255 range
    clipped_image = np.clip(image_array, 0, 255)
    cv2.imwrite(output_path, clipped_image)

def display_image(image_array: np.ndarray):
    """Displays an image using Matplotlib.

    Args:
        image_array (np.ndarray): The image data to display (assumed BGR).
    """
    # Convert from OpenCV's BGR to Matplotlib's RGB format
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to the [0, 1] range for display
    normalized_image = (image_rgb / 255.0).clip(0, 1)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(normalized_image)
    plt.axis('off') # Hide axes for a cleaner look
    plt.show()

## Pre-processing and Post-processing
def preprocess_image_to_tensor(image_array: np.ndarray, max_dim: Optional[int] = None) -> torch.Tensor:
    """
    Converts a NumPy image array to a PyTorch tensor and applies transformations.

    Args:
        image_array (np.ndarray): The input image in HWC format.
        max_dim (Optional[int]): The maximum size for the height or width. 
                                 If provided, the image is resized. Defaults to None.

    Returns:
        torch.Tensor: The processed image as a 4D tensor (B, C, H, W).
    """
    transforms_list = []
    
    if max_dim:
        original_height, original_width, _ = image_array.shape
        scale = float(max_dim) / max(original_height, original_width)
        new_size = (int(scale * original_height), int(scale * original_width))
        
        # transforms.Resize requires a PIL Image
        transforms_list.append(transforms.ToPILImage())
        transforms_list.append(transforms.Resize(new_size))
        
    # ToTensor scales pixels to [0, 1]
    transforms_list.append(transforms.ToTensor())
    # We multiply by 255 to bring it back to the [0, 255] range, 
    # which is expected by the VGG model.
    transforms_list.append(transforms.Lambda(lambda x: x.mul(255)))
    
    processor = transforms.Compose(transforms_list)
    
    # Apply the transformations and add a batch dimension
    processed_tensor = processor(image_array).unsqueeze(0)
    
    return processed_tensor

def convert_tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor back into a NumPy image array.

    Args:
        tensor (torch.Tensor): A 4D or 3D tensor.

    Returns:
        np.ndarray: The resulting image as a NumPy array in BGR format.
    """
    # Remove the batch dimension
    image_tensor = tensor.squeeze(0)
    
    # Move tensor to CPU and convert to NumPy array
    image_numpy = image_tensor.cpu().numpy()
    
    # Transpose from PyTorch's (C, H, W) to NumPy's (H, W, C)
    image_hwc = image_numpy.transpose(1, 2, 0)
    
    return image_hwc

## Advanced Image Manipulation
def transfer_color_profile(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """
    Transfers the color from a source image to a target image by matching luminance.
    This is useful for preserving the original colors of the content image in style transfer.
    It uses the YCrCb color space.

    Args:
        source_bgr (np.ndarray): The source image providing the color (e.g., content image).
        target_bgr (np.ndarray): The target image to apply the color to (e.g., stylized output).

    Returns:
        np.ndarray: The target image with the source's color profile.
    """
    # Ensure pixel values are in the valid range
    source_clipped = np.clip(source_bgr, 0, 255).astype(np.uint8)
    target_clipped = np.clip(target_bgr, 0, 255).astype(np.uint8)
    
    # Resize the source to match the target's dimensions
    height, width, _ = target_clipped.shape
    source_resized = cv2.resize(source_clipped, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # 1. Convert source to YCrCb and target to grayscale (luminance)
    source_ycrcb = cv2.cvtColor(source_resized, cv2.COLOR_BGR2YCrCb)
    target_luminance = cv2.cvtColor(target_clipped, cv2.COLOR_BGR2GRAY)
    
    # 2. Replace the luminance (Y channel) of the source with the target's luminance
    source_ycrcb[..., 0] = target_luminance
    
    # 3. Convert the modified YCrCb image back to BGR
    color_corrected_bgr = cv2.cvtColor(source_ycrcb, cv2.COLOR_YCrCb2BGR)
    
    return np.clip(color_corrected_bgr, 0, 255)

## Visualization and Data Handlin
def visualize_training_loss(content_losses: list, style_losses: list, total_losses: list, title: str = "Training Loss History"):
    """
    Plots the history of content, style, and total loss during training.

    Args:
        content_losses (list): A list of content loss values.
        style_losses (list): A list of style loss values.
        total_losses (list): A list of total loss values.
        title (str): The title of the plot.
    """
    iterations = range(len(total_losses))
    plt.figure(figsize=[10, 6])
    
    plt.plot(iterations, content_losses, label="Content Loss")
    plt.plot(iterations, style_losses, label="Style Loss")
    plt.plot(iterations, total_losses, label="Total Loss")
    
    plt.title(title)
    plt.xlabel('Logged Iterations')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

class CustomImageFolder(datasets.ImageFolder):
    """
    A custom data loader that extends torchvision.datasets.ImageFolder to also 
    return the path of each image.
    """
    def __getitem__(self, index: int) -> Tuple[Image.Image, int, str]:
        """
        Overrides the default __getitem__ to append the image file path.

        Args:
            index (int): The index of the data sample.

        Returns:
            Tuple[Image.Image, int, str]: A tuple containing the image, its label, and its file path.
        """
        # Get the original tuple (image, label)
        original_data = super(CustomImageFolder, self).__getitem__(index)
        
        # Get the path corresponding to this index
        image_path = self.imgs[index][0]
        
        # Combine them into a new tuple
        data_with_path = (*original_data, image_path)
        
        return data_with_path