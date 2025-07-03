import torch
from transformers import AutoFeatureExtractor, ClapModel

class AudioEncoder:
    """
    Audio encoder using CLAP for processing audio inputs.
    Uses frozen CLAP model for feature extraction.
    """
    
    def __init__(self, model_name="laion/clap-htsat-unfused", device="cuda"):
        self.device = device
        self.model_name = model_name
        
        print(f"Initializing CLAP audio encoder with {model_name}")
        
        # Initialize CLAP model
        self.clap_model = (
            ClapModel.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        
        # Freeze model
        self.clap_model.requires_grad_(False)
        
        # Remove text model since we only need audio
        self.clap_model.text_model = None
        
        # Initialize feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        print("CLAP audio encoder initialized successfully")
    
    @torch.no_grad()
    def encode_audio(self, batch_audio):
        """
        Encode a batch of audio using CLAP.
        
        Args:
            batch_audio: List of audio arrays or single audio array
                        Expected format: torch tensors with shape (samples,) 
                        Expected sample rate: 16kHz (CLAP default)
            
        Returns:
            torch.Tensor: Normalized audio features (B, 512)
        """
        # Process audio
        inputs = self.feature_extractor(batch_audio, return_tensors="pt").to(self.device)
        
        # Get features
        feats = self.clap_model.get_audio_features(**inputs)
        
        # Normalize
        return feats / feats.norm(dim=-1, keepdim=True)
    
    def __call__(self, audio):
        """
        Callable interface for easy use.
        
        Args:
            audio: Audio input (torch tensor or list of torch tensors)
            
        Returns:
            torch.Tensor: Normalized audio features
        """
        if isinstance(audio, torch.Tensor):
            audio = [audio]
        return self.encode_audio(audio) 