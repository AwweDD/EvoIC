from dashscope import Generation
from CLIP import CLIPModel
import numpy as np
import dashscope
import pickle
import torch
import clip
import os

templates = [
    lambda c: f"itap of a {c} ",
    lambda c: f"a bad photo of the {c} ",
    lambda c: f"a origami {c} ",
    lambda c: f"a photo of the large {c} ",
    lambda c: f"a {c} in a video game ",
    lambda c: f"art of the {c} ",
    lambda c: f"a photo of the small {c} ",
]

dashscope.api_key = ""

model_name = "Qwen3.6–35B-A3B"


def generate_M_prompt_for_each_cluster(prompt_nouns, n_prompts, M):
    prompts_ = []
    batch_size = 8
    for i in range(0, n_prompts, batch_size):
        batch = prompt_nouns[i:i + batch_size]
        response = Generation.call(
            model=model_name,
            prompt=f"""You are an image prompt generation expert. Please generate {M} CLIP-friendly image prompts for each cluster based on the provided cluster keywords.
                               Requirements:
                               1. Use the template: {{class}} (synonyms), a type of {{domain}}, {{description}}.
                               2. {{class}} should be distilled from the keywords into the most commonly used noun.
                               3. {{domain}} is the broad category that {{class}} belongs to, described with the most commonly used noun.
                               4. For different prompts of a cluster, their {{class}} should be the same.
                               5. For different prompts of a cluster, their {{description}} should be diverse, each {{description}} should only list visual features that can uniquely identify this {{class}}.
                               6. Prompts must be concise and precise, without numbering or extraneous symbols.
                               7. Output the results directly, without additional explanations.
                               8. Generate exactly {M} prompts for each cluster.
                        Please generate exactly {M} prompts for each cluster based on the provided cluster keywords: {batch}.""",
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.output.text
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            print(p)
            prompts_.append(p)
    return prompts_


def generate_M_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, n_prompts, M):
    prompts_ = []
    batch_size = 8
    for i in range(0, n_prompts, batch_size):
        batch = prompt_nouns[i:i + batch_size]
        response = Generation.call(
            model=model_name,
            prompt=f"""You are an image prompt generation expert specialized in dog breed. Please generate {M} CLIP-friendly image prompts for each cluster based on the provided cluster keywords.
                               Requirements:
                               1. Use the template: {{breed}}, {{description}}.
                               2. {{breed}} should be distilled from the keywords into the most commonly used dog breed.
                               3. For different prompts of a cluster, their {{breed}} must be the same.
                               4. For different prompts of a cluster, their {{description}} should be diverse, each {{description}} should only list visual features that can uniquely identify this {{breed}}.
                               5. Prompts must be concise and precise, without numbering or extraneous symbols.
                               6. Output the results directly, must without any additional explanations.
                               7. Such as Golden Retriever, yellow coat, long hair, friendly expression, floppy ears.
                               8. Generate exactly {M} prompts for each cluster.
                        Please generate exactly {M} prompts for each cluster based on the provided cluster keywords: {batch}.""",
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.output.text
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            print(p)
            prompts_.append(p)
    return prompts_


if __name__ == "__main__":
    datasets = ["CIFAR-10"]
    for dataset in datasets:
        #
        with open("./data/5_related_nouns_for_each_center/related_nouns_for_each_center_" + dataset + ".pkl", "rb") as f:
            prompt_nouns = pickle.load(f)
        #
        M = 5
        if dataset == "ImageNet-Dogs":
            prompts_ = generate_M_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, len(prompt_nouns), M)
        else:
            prompts_ = generate_M_prompt_for_each_cluster(prompt_nouns, len(prompt_nouns), M)
        # np.save("./data/4_M_prompts/initialPrompts/Qwen3.5-27B/" + dataset + "_K_prompts_for_each_center.npy", prompts_)
