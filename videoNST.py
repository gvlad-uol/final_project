import cv2
import time
import os
import shutil
from dataclasses import dataclass

from stylize import stylize_folder


@dataclass(frozen=True)
class StyleTransferConfig:
    """Holds all the configuration settings for the video style transfer process."""
    # --- Input Files 
    input_video_path: str = "images/video_short.mp4"
    style_model_path: str = "transforms/starry.pth"

    # --- Output Configuration ---
    output_dir: str = "output"
    # The output filename will now be generated automatically ---
    
    # --- Temporary Directories (will be created and deleted automatically) ---
    temp_dir: str = "temp"
    raw_frames_dir: str = os.path.join(temp_dir, "raw_frames")
    styled_frames_dir: str = os.path.join(temp_dir, "styled_frames")
    
    # --- Processing Parameters ---
    style_batch_size: int = 20

    # Helper properties to derive names automatically ---
    @property
    def style_name(self) -> str:
        """Extracts the style name (e.g., 'wave') from the model path."""
        return os.path.splitext(os.path.basename(self.style_model_path))[0]

    @property
    def output_video_name(self) -> str:
        """Generates a descriptive output video name."""
        video_basename = os.path.splitext(os.path.basename(self.input_video_path))[0]
        return f"{video_basename}_{self.style_name}.mp4"


class VideoStyler:
    """Manages the end-to-end process of applying neural style transfer to a video."""

    def __init__(self, config: StyleTransferConfig):
        self.config = config
        self.height = 0
        self.width = 0
        self.fps = 0

    def process_video(self) -> None:
        """Executes the full style transfer pipeline from start to finish."""
        print(f"OpenCV Version: {cv2.__version__}")
        start_time = time.time()

        try:
            self._setup_directories()
            if not self._get_video_properties():
                return

            self._extract_frames_to_disk()
            self._apply_style_transfer()
            self._create_video_from_frames()

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            #  Cleanup is now enabled to remove temp files automatically ---
            self._cleanup_temp_dirs()
            elapsed_time = time.time() - start_time
            print(f"\nTotal Elapsed Time: {elapsed_time:.2f} seconds.")

    def _setup_directories(self) -> None:
        """Creates the necessary directories."""
        print("Setting up temporary directories...")
        
        #  Create a 'content' subfolder for the ImageFolder loader ---
        content_folder_path = os.path.join(self.config.raw_frames_dir, "content")
        os.makedirs(content_folder_path, exist_ok=True)
        
        os.makedirs(self.config.styled_frames_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _get_video_properties(self) -> bool:
        """Reads the resolution and FPS of the input video."""
        print(f"Reading video properties from '{self.config.input_video_path}'...")
        video_capture = cv2.VideoCapture(self.config.input_video_path)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file at '{self.config.input_video_path}'")
            return False

        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        
        print(f"Video Info: {self.width}x{self.height} at {self.fps:.2f} FPS")
        return True

    # In VideoStyler class

    def _extract_frames_to_disk(self) -> None:
        """Saves each video frame as a JPG image."""
        print("Extracting video frames...")
        video_capture = cv2.VideoCapture(self.config.input_video_path)
        frame_count = 0
        
        #  Set the save path to the 'content' subfolder ---
        save_path = os.path.join(self.config.raw_frames_dir, "content")

        while True:
            success, frame = video_capture.read()
            if not success: break
            frame_count += 1
            filename = os.path.join(save_path, f"frame{frame_count}.jpg")
            cv2.imwrite(filename, frame)
        
        video_capture.release()
        print(f"Successfully extracted {frame_count} frames.")

    def _apply_style_transfer(self) -> None:
        """Invokes the external style transfer function."""
        print("Performing style transfer on frames (this can take a long time)...")
        
        #  Pass the correct, simplified path to the stylize function ---
        stylize_folder(
            self.config.style_model_path,
            self.config.raw_frames_dir, 
            self.config.styled_frames_dir,
            batch_size=self.config.style_batch_size,
        )
        print("Style transfer function finished.")

    # In  VideoStyler class in videoStyler1.py

    def _create_video_from_frames(self) -> None:
        """Creates the final video file from the directory of styled frames."""
        print("Combining styled frames into the final video...")
        
        #  Look directly in the styled_frames directory, not a subfolder ---
        actual_styled_frames_path = self.config.styled_frames_dir

        if not os.path.exists(actual_styled_frames_path):
            print(f"Error: Styled frames directory not found at '{actual_styled_frames_path}'")
            return

        all_files = [f for f in os.listdir(actual_styled_frames_path) if f.endswith((".jpg", ".png"))]

        if not all_files:
            print(f"Error: No image files found in '{actual_styled_frames_path}'.")
            return

        # Sort files numerically to ensure correct order
        all_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        output_video_path = os.path.join(self.config.output_dir, self.config.output_video_name)
        
        video_codec = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_video_path,
            video_codec,
            self.fps,
            (self.width, self.height)
        )

        for frame_name in all_files:
            frame_path = os.path.join(actual_styled_frames_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ Successfully created video at '{output_video_path}'")

    def _cleanup_temp_dirs(self) -> None:
        """Removes the temporary directories used for frame storage."""
        print("Cleaning up temporary files...")
        if os.path.exists(self.config.temp_dir):
            shutil.rmtree(self.config.temp_dir)
        print("Cleanup complete.")


if __name__ == "__main__":
    config = StyleTransferConfig()
    styler = VideoStyler(config)
    styler.process_video()