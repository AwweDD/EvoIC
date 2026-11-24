from open_clip import create_model_from_pretrained, get_tokenizer
from dataLoader import get_dataloader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


def image_encoder():
    #
    features = []
    labels = []
    print("Inferring image features and labels...")
    for iteration, (x, y) in enumerate(dataloader_train):
        x = x.cuda()
        with torch.no_grad():
            feature = clip_model.encode_image(x)
        features.append(feature.cpu().numpy())
        labels.append(y.numpy())
        if iteration % 10 == 0:
            print(f"[Iter {iteration}/{len(dataloader_train)}]")
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print("Feature shape:", features.shape, "Label shape:", labels.shape)
    #
    features_test = []
    labels_test = []
    print("Inferring test image features and labels...")
    for iteration, (x, y) in enumerate(dataloader_test):
        x = x.cuda()
        with torch.no_grad():
            feature = clip_model.encode_image(x)
        features_test.append(feature.cpu().numpy())
        labels_test.append(y.numpy())
        if iteration % 10 == 0:
            print(f"[Iter {iteration}/{len(dataloader_test)}]")
    features_test = np.concatenate(features_test, axis=0)
    labels_test = np.concatenate(labels_test, axis=0)
    print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)

    np.save("./data/0_image_embedding_BiomedCLIP/" + dataset + "_image_embedding_train.npy", features)
    np.save("./data/0_image_embedding_BiomedCLIP/" + dataset + "_image_embedding_test.npy", features_test)
    np.savetxt("./data/0_image_embedding_BiomedCLIP/" + dataset + "_labels_train.txt", labels)
    np.savetxt("./data/0_image_embedding_BiomedCLIP/" + dataset + "_labels_test.txt", labels_test)


def text_encoder():
    batchSize = 2048
    nouns = pd.read_csv("./data/1_text_embedding_BiomedCLIP/WordNetNouns.csv").iloc[:, 0].values

    templates = [
        lambda c: f"itap of a {c}.",
        lambda c: f"a bad photo of the {c}.",
        lambda c: f"a origami {c}.",
        lambda c: f"a photo of the large {c}.",
        lambda c: f"a {c} in a video game.",
        lambda c: f"art of the {c}.",
        lambda c: f"a photo of the small {c}."
    ]

    @torch.no_grad()
    def get_features(texts):
        tokens = tokenizer(texts).to("cuda")
        return clip_model.encode_text(tokens).cpu().numpy()

    features = np.zeros((len(nouns), 512))
    for template in tqdm(templates, desc="Processing templates"):
        prompts = [template(noun) for noun in nouns]
        for i in range(0, len(prompts), batchSize):
            batch = prompts[i:i + batchSize]
            features[i:i + batchSize] += get_features(batch)

    np.save("./data/1_text_embedding_BiomedCLIP/WordNetNouns_Embeddings.npy", features / len(templates))


#
model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
clip_model, preprocess = create_model_from_pretrained(model_name)
tokenizer = get_tokenizer(model_name)
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)
clip_model.eval()
#
dataset = "SARS_COV_2_CT_Scan"
#
dataloader_train, dataloader_test = get_dataloader(dataset=dataset, batch_size=1024)

if __name__ == '__main__':
    # image_encoder()
    text_encoder()
