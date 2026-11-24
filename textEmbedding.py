from CLIP import CLIPModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import clip

batchSize = 2048
nouns = pd.read_csv("./data/1_text_embedding/WordNetNouns.csv").iloc[:, 0].values

templates = [
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}."
]

model = CLIPModel(model_name="ViT-B/32").cuda()
model.eval()


@torch.no_grad()
def get_features(texts):
    tokens = clip.tokenize(texts, truncate=True).to("cuda")
    return model.encode_text(tokens).cpu().numpy()


features = np.zeros((len(nouns), 512))
for template in tqdm(templates, desc="Processing templates"):
    prompts = [template(noun) for noun in nouns]
    for i in range(0, len(prompts), batchSize):
        batch = prompts[i:i + batchSize]
        features[i:i + batchSize] += get_features(batch)

np.save("./data/1_text_embedding/WordNetNouns_Embeddings.npy", features / len(templates))
