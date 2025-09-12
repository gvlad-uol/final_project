import torch
from torchvision import transforms
from pathlib import Path
import time
from typing import Union
import utils
import transformer

SAVE_THE_COLOR = False

class NeuralStyler:
    """
    An object-oriented manager for applying neural style transfer.

    This class encapsulates the model loading, device management, and stylization
    logic, allowing for cleaner and more reusable code. The model is loaded only
    once upon instantiation.
    """

    def __init__(self, model_path: Union[str, Path]):
        """
        Initializes the styler and loads the specified model onto the best device.

        Args:
            model_path (Union[str, Path]): The file path to the pre-trained 
                                           style transfer model (.pth).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(Path(model_path))
        print(f"NeuralStyler initialized. Model loaded on device: '{self.device}'")

    def _load_model(self, model_path: Path) -> transformer.TransformerNetwork:
        """Loads the TransformerNetwork from disk and prepares it for inference."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        network = transformer.TransformerNetwork()
        # Using weights_only=True is a security best practice
        network.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        network.to(self.device)
        network.eval()  # Set the model to evaluation mode
        return network

    @torch.no_grad()
    def _apply_style(self, content_tensor: torch.Tensor) -> torch.Tensor:
        """Core inference logic to apply style to a tensor."""
        if content_tensor.dim() == 3:
            content_tensor = content_tensor.unsqueeze(0)
        content_tensor = content_tensor.to(self.device)
        return self.model(content_tensor).detach()

    def stylize_image(self, content_image_path: Union[str, Path], SAVE_THE_COLOR: bool = False):
        """
        Processes a single image file.

        Args:
            content_image_path (Union[str, Path]): Path to the source image.
            SAVE_THE_COLOR (bool): If True, applies the original image's color
                                   profile to the stylized output.

        Returns:
            The stylized image in a format suitable for saving or display.
        """
        content_image = utils.load_image(str(content_image_path))
        content_tensor = utils.itot(content_image)
        generated_tensor = self._apply_style(content_tensor)
        generated_image = utils.ttoi(generated_tensor.squeeze())

        if SAVE_THE_COLOR:
            generated_image = utils.transfer_color(content_image, generated_image)
        
        return generated_image

    def stylize_directory_batched(self, source_parent_dir: Union[str, Path], 
                                  output_dir: Union[str, Path], 
                                  batch_size: int, 
                                  SAVE_THE_COLOR: bool):
        """
        Stylizes a directory of images using efficient batch processing.

        Args:
            source_parent_dir (Union[str, Path]): The parent directory containing 
                                                  the content folder.
            output_dir (Union[str, Path]): Directory to save the stylized images.
            batch_size (int): The number of images to process per batch.
            SAVE_THE_COLOR (bool): Flag for color preservation.
        """
        source_path = Path(source_parent_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        
        dataset = utils.ImageFolderWithPaths(str(source_path), transform=transform_pipeline)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

        for content_batch, _, paths in loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            generated_batch = self._apply_style(content_batch)

            for i, original_path_str in enumerate(paths):
                original_path = Path(original_path_str)
                generated_image = utils.ttoi(generated_batch[i])

                if SAVE_THE_COLOR:
                    content_image = utils.load_image(str(original_path))
                    generated_image = utils.transfer_color(content_image, generated_image)
                
                save_destination = output_path / original_path.name
                utils.saveimg(generated_image, str(save_destination))

def stylize_folder(style_path: str, 
                   folder_containing_the_content_folder: str, 
                   save_folder: str, 
                   batch_size: int = 1):
    """
    A wrapper function to maintain 100% compatibility with external scripts
    like 'videoStyler1.py'. It instantiates and uses the NeuralStyler class.
    """
    print("--- Invoking Style Transfer via Compatibility Layer ---")
    try:
        styler = NeuralStyler(model_path=style_path)
        styler.stylize_directory_batched(
            source_parent_dir=folder_containing_the_content_folder,
            output_dir=save_folder,
            batch_size=batch_size,
            SAVE_THE_COLOR=SAVE_THE_COLOR
        )
        print("--- Style Transfer Complete ---")
    except Exception as e:
        print(f"\n[ERROR] An exception occurred during stylization: {e}")


def stylize_folder_single(style_path: str, content_folder: str, save_folder: str):
    """
    Rewritten version for stylizing images in a flat directory one by one.
    """
    styler = NeuralStyler(model_path=style_path)
    source_dir = Path(content_folder)
    output_dir = Path(save_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"]

    for image_file in image_files:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Stylizing: {image_file.name}")
        generated_image = styler.stylize_image(image_file, SAVE_THE_COLOR=SAVE_THE_COLOR)
        save_destination = output_dir / image_file.name
        utils.saveimg(generated_image, str(save_destination))

def stylize():
    """Rewritten interactive command-line stylization tool."""
    STYLE_TRANSFORM_PATH = "transforms/udnie_aggressive.pth"
    
    try:
        styler = NeuralStyler(model_path=STYLE_TRANSFORM_PATH)
        while True:
            try:
                image_path_str = input("Enter the image path (Press Ctrl+C to exit): ")
                image_path = Path(image_path_str)
                if not image_path.is_file():
                    print("Invalid path. Please try again.")
                    continue
                
                start_time = time.time()
                generated_image = styler.stylize_image(image_path, SAVE_THE_COLOR=SAVE_THE_COLOR)
                duration = time.time() - start_time
                print(f"Transfer Time: {duration:.2f} seconds")

                output_filename = image_path.parent / f"{image_path.stem}-stylized.jpg"
                utils.saveimg(generated_image, str(output_filename))
                print(f"Saved result to: {output_filename}")
                utils.show(generated_image)

            except KeyboardInterrupt:
                print("\nExiting.")
                break
    except Exception as e:
        print(f"\n[ERROR] A critical error occurred: {e}")