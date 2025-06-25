import torch
from transformers import CLIPModel, CLIPImageProcessor
from transformers import CLIPImageProcessor
from PIL import Image


device = "cuda"

clip_model = (
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    .to(device)
    .eval()
)


# Frozen 
clip_model.requires_grad_(False)


# No Prompting
clip_model.text_model = None
clip_model.logit_scale = None



clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()                 # no autograd graph
def encode_images(batch_pil):
    inputs = clip_processor(images=batch_pil, return_tensors="pt").to(device)
    feats  = clip_model.get_image_features(**inputs)        # (B, 512)
    return feats / feats.norm(dim=-1, keepdim=True)


