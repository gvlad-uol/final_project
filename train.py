# ==============================================================================
# Fast Neural Style Transfer - Model Training
# ==============================================================================
# This script trains a style transfer network using a content dataset and a
# target style image. It relies on a pre-trained VGG perception network.
# ==============================================================================

import os
import sys
import time
import random
import traceback
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

# Import custom modules that initially was in my ipynb
import vgg as perception_network
import transformer as style_network
import utils as image_utils

# --- Configuration Class ---
class TrainingConfig:
    """Encapsulates all training hyperparameters and settings."""
    # I/O Paths
    CONTENT_DATA_PATH: str = "dataset"
    STYLE_IMAGE_PATH: str = "images/scream.jpg"
    MODEL_SAVE_DIR: str = "models/"
    IMAGE_SAVE_DIR: str = "images/out/"
    
    IMAGE_SIZE: int = 256
    NUM_EPOCHS: int = 1
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-3
    
    # Loss Weights
    CONTENT_WEIGHT: float = 17.0
    STYLE_WEIGHT: float = 50.0
    
    # Logistics
    SEED: int = 25
    SAVE_CHECKPOINT_EVERY: int = 1500 # Number of iterations between saves
    ENABLE_LOSS_PLOT: bool = True

# --- Main Trainer Class ---
class StyleTransferTrainer:
    """Manages the entire style transfer model training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cpu"
        self.transformer_net = None
        self.perception_net = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.style_gram_matrices = {}
        self.loss_history = {"content": [], "style": [], "total": []}
        
        # Automatic Mixed Precision (AMP) setup
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _setup_environment(self):
        """Initializes device, seeds, and prints a startup banner."""
        print("=" * 80)
        print("Initializing Fast Neural Style Training Environment")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
            try: # For newer PyTorch versions
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
        print(f"Using device: {self.device.upper()}")

        # Set seeds for reproducibility
        torch.manual_seed(self.config.SEED)
        torch.cuda.manual_seed_all(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        print(f"Random seed set to: {self.config.SEED}")
        
    def _validate_and_prepare_paths(self):
        """Ensures all necessary files and directories exist."""
        print("Validating project structure...")
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.IMAGE_SAVE_DIR, exist_ok=True)
        
        # Check for style image
        if not os.path.isfile(self.config.STYLE_IMAGE_PATH):
            raise FileNotFoundError(f"Style image not found at: {os.path.abspath(self.config.STYLE_IMAGE_PATH)}")
            
        # Check for dataset directory
        if not os.path.isdir(self.config.CONTENT_DATA_PATH) or not os.listdir(self.config.CONTENT_DATA_PATH):
            raise FileNotFoundError(f"Content dataset not found or is empty at: {os.path.abspath(self.config.CONTENT_DATA_PATH)}")
        print("All paths are valid.")
        print("-" * 80)

    def _prepare_dataloaders(self):
        """Prepares the dataset and DataLoader for content images."""
        # Top-level function for pickling with multiprocessing
        def scale_to_255(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.mul(255)
            
        transform = transforms.Compose([
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.CenterCrop(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(scale_to_255),
        ])
        
        dataset = datasets.ImageFolder(self.config.CONTENT_DATA_PATH, transform=transform)
        if not dataset:
            raise RuntimeError(f"No images found in '{self.config.CONTENT_DATA_PATH}'. Check subdirectories.")

        num_workers = min(8, os.cpu_count() or 1) if self.device == "cuda" else 0
        loader_params = {
            'batch_size': self.config.BATCH_SIZE, 
            'shuffle': True,
            'num_workers': num_workers, 
            'pin_memory': self.device == "cuda", 
            'drop_last': True
        }
        self.train_loader = torch.utils.data.DataLoader(dataset, **loader_params)
        
        print(f"Dataset: {len(dataset)} images found across {len(dataset.classes)} classes.")
        print(f"DataLoader: {len(self.train_loader)} batches per epoch.")

    def _initialize_models_and_optimizer(self):
        """Initializes the transformer and perception networks, and the optimizer."""
        print("Initializing networks...")
        # Style Network (the one we train)
        self.transformer_net = style_network.TransformerNetwork().to(self.device)
        
        # Perception Network (frozen, for loss calculation)
        self.perception_net = perception_network.VGG16FeatureExtractor().to(self.device).eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.transformer_net.parameters(), lr=self.config.LEARNING_RATE)
        
        # Loss Function
        self.loss_fn = nn.MSELoss().to(self.device)
        print("Models and optimizer are ready.")

    def _precompute_style_representation(self):
        """Loads the style image and computes the Gram matrices of its features."""
        print("Pre-computing style representation...")
        style_img = image_utils.read_image_from_path(self.config.STYLE_IMAGE_PATH)
        style_tensor = image_utils.preprocess_image_to_tensor(style_img).to(self.device)
        
        # VGG expects BGR images with ImageNet means subtracted
        imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68]).view(1, 3, 1, 1).to(self.device)
        style_tensor_vgg = style_tensor.add(imagenet_neg_mean)
        
        with torch.no_grad():
            style_features = self.perception_net(style_tensor_vgg)
            self.style_gram_matrices = {
                name: image_utils.calculate_gram_matrix(feat) for name, feat in style_features.items()
            }
        print("Style Gram matrices computed.")
        
    def _run_training_loop(self):
        """Executes the main training loop over epochs and batches."""
        print("=" * 80)
        print("Starting Training Loop...")
        start_time = time.time()
        total_iterations = len(self.train_loader) * self.config.NUM_EPOCHS
        current_iteration = 0
        
        imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68]).view(1, 3, 1, 1).to(self.device)

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{self.config.NUM_EPOCHS} ---")
            
            for content_batch, _ in self.train_loader:
                current_iteration += 1
                self.optimizer.zero_grad(set_to_none=True)
                
                # Prepare content batch (B, C, H, W) -> (B, 3, H, W) BGR
                content_batch_bgr = content_batch[:, [2, 1, 0]].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Generate stylized image
                    generated_output = self.transformer_net(content_batch_bgr)
                    
                    # Get VGG features for all three images
                    content_features   = self.perception_net(content_batch_bgr.add(imagenet_neg_mean))
                    generated_features = self.perception_net(generated_output.add(imagenet_neg_mean))
                    
                    # --- Calculate Content Loss ---
                    content_loss = self.config.CONTENT_WEIGHT * self.loss_fn(
                        generated_features['relu2_2'], content_features['relu2_2']
                    )

                    # --- Calculate Style Loss ---
                    current_batch_size = content_batch.shape[0]
                    style_loss_accumulator = 0.0
                    for name, gen_feat in generated_features.items():
                        gen_gram = image_utils.calculate_gram_matrix(gen_feat)
                        style_gram = self.style_gram_matrices[name].expand(current_batch_size, -1, -1)
                        style_loss_accumulator += self.loss_fn(gen_gram, style_gram)
                    
                    style_loss = self.config.STYLE_WEIGHT * style_loss_accumulator
                    
                    # --- Total Loss ---
                    total_loss = content_loss + style_loss

                # Backward pass and optimization step
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # --- Log progress ---
                if current_iteration % self.config.SAVE_CHECKPOINT_EVERY == 0 or current_iteration == total_iterations:
                    self._log_and_checkpoint(current_iteration, total_iterations, start_time,
                                            content_loss.item(), style_loss.item(), total_loss.item(), generated_output)

        # --- Finalize ---
        elapsed_time = time.time() - start_time
        print(f"\nTraining finished in {elapsed_time:.2f} seconds.")
        self._save_final_model()
        if self.config.ENABLE_LOSS_PLOT:
            self._plot_final_loss()

    def _log_and_checkpoint(self, iter_num, total_iters, start_time, c_loss, s_loss, t_loss, sample_output):
        """Logs metrics, saves model checkpoint, and saves a sample image."""
        self.loss_history["content"].append(c_loss)
        self.loss_history["style"].append(s_loss)
        self.loss_history["total"].append(t_loss)
        
        elapsed = time.time() - start_time
        print(f"\n[Iteration {iter_num}/{total_iters}] - Elapsed: {elapsed:.1f}s")
        print(f"  Losses -> Content: {c_loss:.4f} | Style: {s_loss:.4f} | Total: {t_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(self.config.MODEL_SAVE_DIR, f"checkpoint_iter_{iter_num}.pth")
        torch.save(self.transformer_net.state_dict(), checkpoint_path)
        print(f"  Saved model checkpoint to: {os.path.abspath(checkpoint_path)}")

        # Save image sample
        sample_img_tensor = sample_output[0].detach().unsqueeze(0).cpu()
        sample_img_np = image_utils.convert_tensor_to_image(sample_img_tensor)
        sample_path = os.path.join(self.config.IMAGE_SAVE_DIR, f"sample_iter_{iter_num}.png")
        image_utils.save_image_to_path(sample_img_np, sample_path)
        print(f"  Saved image sample to: {os.path.abspath(sample_path)}")

    def _save_final_model(self):
        """Saves the final trained model weights."""
        self.transformer_net.eval().cpu()
        final_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "final_style_model.pth")
        torch.save(self.transformer_net.state_dict(), final_model_path)
        print(f"Final model saved to: {os.path.abspath(final_model_path)}")
        
    def _plot_final_loss(self):
        """Plots the collected loss history."""
        print("Generating loss history plot...")
        try:
            image_utils.visualize_training_loss(
                self.loss_history["content"],
                self.loss_history["style"],
                self.loss_history["total"],
                title="Style Transfer Training Loss"
            )
        except Exception as e:
            print(f"Could not generate plot. Error: {e}")

    def execute(self):
        """Runs the entire training pipeline from start to finish."""
        self._setup_environment()
        self._validate_and_prepare_paths()
        self._prepare_dataloaders()
        self._initialize_models_and_optimizer()
        self._precompute_style_representation()
        self._run_training_loop()

# --- Entry Point ---
if __name__ == "__main__":
    try:
        print(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
        # 1. Create configuration object
        training_config = TrainingConfig()
        
        # 2. Instantiate and run the trainer
        trainer = StyleTransferTrainer(training_config)
        trainer.execute()
        
    except Exception:
        traceback.print_exc()
        input("\n[An error occurred. Press Enter to exit]")