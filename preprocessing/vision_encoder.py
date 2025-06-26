import torch
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
import logging

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
        
        # Initialize CLIP model
        self.clip_model = (
            CLIPModel.from_pretrained(model_name)
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