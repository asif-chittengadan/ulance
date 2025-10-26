import cv2
import mediapipe as mp
import numpy as np
import logging
import torch
import torchvision.transforms as T

class BaseProcessor:
    """Base class for different processing modules."""
    def process_frame(self, frame, quality):
        raise NotImplementedError

class NormalProcessor(BaseProcessor):
    """
    The 'no-help' baseline processor. It does NOT adapt and ignores the quality setting.
    """
    def __init__(self):
        logging.info("NormalProcessor initialized: Non-Adaptive, Full Quality.")

    def process_frame(self, frame, quality='high'): # Now accepts 'quality' but ignores it
        if frame is None:
            return None
        # We draw the label to show it's receiving the quality state, even if unused.
        cv2.putText(frame, f"Normal Stream (Quality: {quality.upper()})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

class UlancProcessor(BaseProcessor):
    """
    Implements the ULANC concept with GPU acceleration.
    This version includes the final fix for the OpenCV data type error.
    """
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"UlancProcessor initialized. Using device: {self.device}")
        if not torch.cuda.is_available():
            logging.warning("CUDA not available. ULANC processor will run on CPU and may be slow.")

        self.blur_transform_poor = T.GaussianBlur(kernel_size=(55, 55), sigma=(10.0, 10.0))
        self.blur_transform_good = T.GaussianBlur(kernel_size=(11, 11), sigma=(2.0, 2.0))

    def process_frame(self, frame, quality='high'):
        if frame is None:
            return None
        
        original_h, original_w, _ = frame.shape

        try:
            # 1. AI Segmentation on CPU (on a small frame for speed)
            processing_scale = 4
            small_w, small_h = original_w // processing_scale, original_h // processing_scale
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = self.segmentation.process(rgb_small_frame)
            
            mask_small = results.segmentation_mask
            mask = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            # 2. Transfer Data to GPU
            frame_tensor = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask).to(self.device).unsqueeze(0)
            condition = mask_tensor.repeat(3, 1, 1) > 0.6

            # 3. All Graphics Processing on GPU, using the 'quality' parameter
            if quality == 'low' or quality == 'medium':
                background_tensor = self.blur_transform_poor(frame_tensor)
                label = f"ULANC - {quality.upper()} Network (GPU)"
            else: # 'high'
                background_tensor = self.blur_transform_good(frame_tensor)
                label = f"ULANC - {quality.upper()} Network (GPU)"

            output_tensor = torch.where(condition, frame_tensor, background_tensor)

            # 4. Transfer Final Image back to CPU for Encoding
            output_numpy = output_tensor.permute(1, 2, 0).cpu().numpy()
            
            # --- THIS IS THE FIX ---
            # Convert the float32 numpy array back to uint8 for OpenCV
            # np.ascontiguousarray ensures the memory layout is correct for OpenCV
            output_frame = np.ascontiguousarray(output_numpy, dtype=np.uint8)
            
            cv2.putText(output_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return output_frame
        except Exception as e:
            logging.error(f"Error in UlancProcessor: {e}")
            # On error, return the original frame to keep the stream alive
            return frame