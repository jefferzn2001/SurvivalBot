import torch
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)

class VisionEncoder:
    """
    Vision encoder using CLIP for processing image inputs.
    Uses frozen CLIP model for feature extraction.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Initializing CLIP vision encoder with {model_name}")
        
        # Initialize CLIP model with trust_remote_code to use safetensors when available
        self.clip_model = (
            CLIPModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=False
            )
            .to(device)
            .eval()
        )
        
        # Freeze model
        self.clip_model.requires_grad_(False)
        
        # Remove text model and logit scale since we only need vision
        self.clip_model.text_model = None
        self.clip_model.logit_scale = None
        
        # Initialize processor
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        logger.info("CLIP vision encoder initialized successfully")
    
    @staticmethod
    def get_image_path_from_row(row, raw_state_path):
        """
        Given a row from raw_state.csv and the path to raw_state.csv, return the full path to the image file.
        """
        # raw_state_path: .../data_xxx/raw_state.csv
        session_dir = os.path.dirname(os.path.abspath(raw_state_path))
        # Try images/ subfolder first, then session_dir
        image_file = row['image_file']
        image_path = os.path.join(session_dir, 'images', image_file)
        if not os.path.exists(image_path):
            image_path = os.path.join(session_dir, image_file)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return image_path
    
    def load_image_from_row(self, row, raw_state_path):
        """
        Load the PIL image corresponding to a row in raw_state.csv.
        """
        image_path = self.get_image_path_from_row(row, raw_state_path)
        return Image.open(image_path).convert('RGB')
    
    @torch.no_grad()
    def encode_images(self, batch_pil):
        """
        Encode a batch of PIL images using CLIP.
        
        Args:
            batch_pil: List of PIL.Image objects
            
        Returns:
            torch.Tensor: Normalized image features (B, 512)
        """
        # Process images
        inputs = self.processor(images=batch_pil, return_tensors="pt").to(self.device)
        
        # Get features
        feats = self.clip_model.get_image_features(**inputs)
        
        # Normalize
        return feats / feats.norm(dim=-1, keepdim=True)
    
    def __call__(self, images):
        """
        Callable interface for easy use.
        
        Args:
            images: List of PIL images or single PIL image
            
        Returns:
            torch.Tensor: Normalized image features
        """
        if isinstance(images, Image.Image):
            images = [images]
        return self.encode_images(images) 