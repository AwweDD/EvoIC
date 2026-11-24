from torch import nn
import torch
import clip


class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)
        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return x
