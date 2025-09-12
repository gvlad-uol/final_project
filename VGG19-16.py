# --- Imports ---
import os
from typing import Dict, Optional, Callable, Type, Any

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights

# --- Public API ---
__all__ = ["VGG16FeatureExtractor", "VGG19FeatureExtractor"]


# --- Weight Loading Utility ---

def _load_and_adapt_checkpoint(target_module: nn.Module, checkpoint_path: str) -> None:
    """
    Loads weights from a potentially non-standard checkpoint file into a target module.

    This function is designed to handle various checkpoint formats commonly found in open-source projects:
    1.  Full nn.Module objects (requires `weights_only=False`).
    2.  Dictionary checkpoints containing a 'state_dict' key.
    3.  Raw state dictionary files.
    4.  State dictionaries with common prefixes (e.g., 'module.', 'features.') which are stripped automatically.

    Args:
        target_module (nn.Module): The module (e.g., vgg.features) to load weights into.
        checkpoint_path (str): Path to the saved weights file.

    Raises:
        FileNotFoundError: If the specified weights file does not exist.
        RuntimeError: If the checkpoint format is unrecognized.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {os.path.abspath(checkpoint_path)}")

    # Load checkpoint data. Setting weights_only=False can be a security risk if loading untrusted files.
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise IOError(f"Failed to load checkpoint file '{checkpoint_path}'. Error: {e}")

    # Extract state dictionary based on common save formats
    if isinstance(checkpoint_data, nn.Module):
        state_dict = checkpoint_data.state_dict()
    elif isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data
    else:
        raise RuntimeError(
            f"Unsupported checkpoint format in '{checkpoint_path}'. "
            "Expected nn.Module or a dictionary-like state_dict."
        )

    # --- Key Normalization ---
    def normalize_key(key: str) -> str:
        """Strips common prefixes from state dictionary keys."""
        prefixes_to_strip = ("module.features.", "features.", "module.", "vgg.", "cnn.")
        for prefix in prefixes_to_strip:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    normalized_state_dict = {normalize_key(k): v for k, v in state_dict.items()}

    # --- Filtering and Loading ---
    target_state_dict = target_module.state_dict()
    filtered_weights = {
        k: v for k, v in normalized_state_dict.items() if k in target_state_dict and target_state_dict[k].shape == v.shape
    }

    if not filtered_weights:
        raise RuntimeError(f"No matching keys found between checkpoint and model after normalization. Checkpoint keys: {list(normalized_state_dict.keys())[:5]}...")

    target_module.load_state_dict(filtered_weights, strict=False)
    print(
        f"[WeightLoader] Successfully loaded custom weights from '{os.path.abspath(checkpoint_path)}'. "
        f"Matched {len(filtered_weights)}/{len(target_state_dict)} keys."
    )


# --- Base Feature Extractor Class ---

class VggFeatureExtractor(nn.Module):
    """
    Abstract base class for VGG feature extraction.
    Handles weight loading, model freezing, and forward pass logic.
    Subclasses must define EXTRACTION_POINTS.
    """
    # Subclasses should override this dictionary: {layer_index: feature_name}
    EXTRACTION_POINTS: Dict[int, str] = {}

    def __init__(
        self,
        model_name: str,
        model_constructor: Callable[..., nn.Module],
        weights_enum: Type[Any],
        custom_weights_path: Optional[str] = None,
        use_default_on_failure: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        # --- Step 1: Initialize base model structure ---
        # Load standard weights only if no custom path is provided and default usage is enabled.
        # Otherwise, initialize with structure only (weights=None).
        if custom_weights_path is None and use_default_on_failure:
            base_model = model_constructor(weights=weights_enum.DEFAULT)
            print(f"[{self.model_name}] Initialized using standard torchvision pretrained weights.")
        else:
            base_model = model_constructor(weights=None)
            if custom_weights_path is None:
                print(f"[{self.model_name}] Initialized with random weights (no custom path or default fallback).")

        self.features = base_model.features

        # --- Step 2: Load custom weights if path provided ---
        if custom_weights_path:
            try:
                _load_and_adapt_checkpoint(self.features, custom_weights_path)
            except Exception as e:
                print(f"[{self.model_name}] WARNING: Failed to load custom weights from '{custom_weights_path}'. Error: {e}")
                if use_default_on_failure:
                    print(f"[{self.model_name}] Fallback: Loading standard torchvision weights.")
                    fallback_model = model_constructor(weights=weights_enum.DEFAULT)
                    self.features.load_state_dict(fallback_model.features.state_dict(), strict=True)
                else:
                    print(f"[{self.model_name}] Proceeding with uninitialized weights as fallback is disabled.")

        # --- Step 3: Freeze model parameters ---
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self._final_layer_index = max(self.EXTRACTION_POINTS.keys())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts features from specified layers."""
        extracted_features: Dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            feature_name = self.EXTRACTION_POINTS.get(index)
            if feature_name:
                extracted_features[feature_name] = x
            
            # Optimization: stop processing once all required features are collected.
            if index == self._final_layer_index:
                break
        return extracted_features


# --- Concrete Implementations ---

class VGG16FeatureExtractor(VggFeatureExtractor):
    """
    VGG16 wrapper that exposes intermediate feature maps:
    'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'

    The input image is expected to be pre-processed (e.g., BGR conversion and ImageNet mean subtraction).
    This module freezes all weights upon initialization.
    """
    # VGG16 layer mapping: conv1_1(0) relu1_1(1) conv1_2(2) relu1_2(3) pool1(4) ...
    EXTRACTION_POINTS = {
        3: "relu1_2",   # After conv1_2
        8: "relu2_2",   # After conv2_2
        15: "relu3_3",  # After conv3_3
        22: "relu4_3",  # After conv4_3
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_torchvision_pretrained_if_missing: bool = True,
    ):
        super().__init__(
            model_name="VGG16",
            model_constructor=vgg16,
            weights_enum=VGG16_Weights,
            custom_weights_path=weights_path,
            use_default_on_failure=use_torchvision_pretrained_if_missing,
        )


class VGG19FeatureExtractor(VggFeatureExtractor):
    """
    VGG19 wrapper that exposes intermediate feature maps:
    'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'

    The input image is expected to be pre-processed (e.g., BGR conversion and ImageNet mean subtraction).
    This module freezes all weights upon initialization.
    """
    # VGG19 layer mapping: ... relu3_4(17), relu4_4(26), relu5_4(35)
    EXTRACTION_POINTS = {
        3: "relu1_2",
        8: "relu2_2",
        17: "relu3_4",
        26: "relu4_4",
        # 35: "relu5_4",  original VGG16 implementation stopped at block 4.
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_torchvision_pretrained_if_missing: bool = True,
    ):
        super().__init__(
            model_name="VGG19",
            model_constructor=vgg19,
            weights_enum=VGG19_Weights,
            custom_weights_path=weights_path,
            use_default_on_failure=use_torchvision_pretrained_if_missing,
        )# --- Imports ---
import os
from typing import Dict, Optional, Callable, Type, Any

import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG16_Weights, VGG19_Weights

# --- Public API ---
__all__ = ["VGG16FeatureExtractor", "VGG19FeatureExtractor"]


# --- Weight Loading Utility ---

def _load_and_adapt_checkpoint(target_module: nn.Module, checkpoint_path: str) -> None:
    """
    Loads weights from a potentially non-standard checkpoint file into a target module.

    This function is designed to handle various checkpoint formats commonly found in open-source projects:
    1.  Full nn.Module objects (requires `weights_only=False`).
    2.  Dictionary checkpoints containing a 'state_dict' key.
    3.  Raw state dictionary files.
    4.  State dictionaries with common prefixes (e.g., 'module.', 'features.') which are stripped automatically.

    Args:
        target_module (nn.Module): The module (e.g., vgg.features) to load weights into.
        checkpoint_path (str): Path to the saved weights file.

    Raises:
        FileNotFoundError: If the specified weights file does not exist.
        RuntimeError: If the checkpoint format is unrecognized.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {os.path.abspath(checkpoint_path)}")

    # Load checkpoint data. Setting weights_only=False can be a security risk if loading untrusted files.
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise IOError(f"Failed to load checkpoint file '{checkpoint_path}'. Error: {e}")

    # Extract state dictionary based on common save formats
    if isinstance(checkpoint_data, nn.Module):
        state_dict = checkpoint_data.state_dict()
    elif isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict = checkpoint_data
    else:
        raise RuntimeError(
            f"Unsupported checkpoint format in '{checkpoint_path}'. "
            "Expected nn.Module or a dictionary-like state_dict."
        )

    # --- Key Normalization ---
    def normalize_key(key: str) -> str:
        """Strips common prefixes from state dictionary keys."""
        prefixes_to_strip = ("module.features.", "features.", "module.", "vgg.", "cnn.")
        for prefix in prefixes_to_strip:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    normalized_state_dict = {normalize_key(k): v for k, v in state_dict.items()}

    # --- Filtering and Loading ---
    target_state_dict = target_module.state_dict()
    filtered_weights = {
        k: v for k, v in normalized_state_dict.items() if k in target_state_dict and target_state_dict[k].shape == v.shape
    }

    if not filtered_weights:
        raise RuntimeError(f"No matching keys found between checkpoint and model after normalization. Checkpoint keys: {list(normalized_state_dict.keys())[:5]}...")

    target_module.load_state_dict(filtered_weights, strict=False)
    print(
        f"[WeightLoader] Successfully loaded custom weights from '{os.path.abspath(checkpoint_path)}'. "
        f"Matched {len(filtered_weights)}/{len(target_state_dict)} keys."
    )


# --- Base Feature Extractor Class ---
class VggFeatureExtractor(nn.Module):
    """
    Abstract base class for VGG feature extraction.
    Handles weight loading, model freezing, and forward pass logic.
    Subclasses must define EXTRACTION_POINTS.
    """
    # Subclasses should override this dictionary: {layer_index: feature_name}
    EXTRACTION_POINTS: Dict[int, str] = {}

    def __init__(
        self,
        model_name: str,
        model_constructor: Callable[..., nn.Module],
        weights_enum: Type[Any],
        custom_weights_path: Optional[str] = None,
        use_default_on_failure: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        # --- Step 1: Initialize base model structure ---
        # Load standard weights only if no custom path is provided and default usage is enabled.
        # Otherwise, initialize with structure only (weights=None).
        if custom_weights_path is None and use_default_on_failure:
            base_model = model_constructor(weights=weights_enum.DEFAULT)
            print(f"[{self.model_name}] Initialized using standard torchvision pretrained weights.")
        else:
            base_model = model_constructor(weights=None)
            if custom_weights_path is None:
                print(f"[{self.model_name}] Initialized with random weights (no custom path or default fallback).")

        self.features = base_model.features

        # --- Step 2: Load custom weights if path provided ---
        if custom_weights_path:
            try:
                _load_and_adapt_checkpoint(self.features, custom_weights_path)
            except Exception as e:
                print(f"[{self.model_name}] WARNING: Failed to load custom weights from '{custom_weights_path}'. Error: {e}")
                if use_default_on_failure:
                    print(f"[{self.model_name}] Fallback: Loading standard torchvision weights.")
                    fallback_model = model_constructor(weights=weights_enum.DEFAULT)
                    self.features.load_state_dict(fallback_model.features.state_dict(), strict=True)
                else:
                    print(f"[{self.model_name}] Proceeding with uninitialized weights as fallback is disabled.")

        # --- Step 3: Freeze model parameters ---
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self._final_layer_index = max(self.EXTRACTION_POINTS.keys())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts features from specified layers."""
        extracted_features: Dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.features):
            x = layer(x)
            feature_name = self.EXTRACTION_POINTS.get(index)
            if feature_name:
                extracted_features[feature_name] = x
            
            # Optimization: stop processing once all required features are collected.
            if index == self._final_layer_index:
                break
        return extracted_features


# --- Concrete Implementations ---
class VGG16FeatureExtractor(VggFeatureExtractor):
    """
    VGG16 wrapper that exposes intermediate feature maps:
    'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'

    The input image is expected to be pre-processed (e.g., BGR conversion and ImageNet mean subtraction).
    This module freezes all weights upon initialization.
    """
    # VGG16 layer mapping: conv1_1(0) relu1_1(1) conv1_2(2) relu1_2(3) pool1(4) ...
    EXTRACTION_POINTS = {
        3: "relu1_2",   # After conv1_2
        8: "relu2_2",   # After conv2_2
        15: "relu3_3",  # After conv3_3
        22: "relu4_3",  # After conv4_3
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_torchvision_pretrained_if_missing: bool = True,
    ):
        super().__init__(
            model_name="VGG16",
            model_constructor=vgg16,
            weights_enum=VGG16_Weights,
            custom_weights_path=weights_path,
            use_default_on_failure=use_torchvision_pretrained_if_missing,
        )


class VGG19FeatureExtractor(VggFeatureExtractor):
    """
    VGG19 wrapper that exposes intermediate feature maps:
    'relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'

    The input image is expected to be pre-processed (e.g., BGR conversion and ImageNet mean subtraction).
    This module freezes all weights upon initialization.
    """
    # VGG19 layer mapping: ... relu3_4(17), relu4_4(26), relu5_4(35)
    EXTRACTION_POINTS = {
        3: "relu1_2",
        8: "relu2_2",
        17: "relu3_4",
        26: "relu4_4",
        35: "relu5_4",
    }

    def __init__(
        self,
        weights_path: Optional[str] = None,
        use_torchvision_pretrained_if_missing: bool = True,
    ):
        super().__init__(
            model_name="VGG19",
            model_constructor=vgg19,
            weights_enum=VGG19_Weights,
            custom_weights_path=weights_path,
            use_default_on_failure=use_torchvision_pretrained_if_missing,
        )