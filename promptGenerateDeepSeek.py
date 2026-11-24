from CLIP import CLIPModel
from openai import OpenAI
import numpy as np
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

api_key = ""
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# DeepSeek-V3.2
def generate_one_prompt_for_each_cluster(prompt_nouns, n_prompts):
    # one prompt for each class center
    prompts_ = []
    batch_size = 30
    for i in range(0, n_prompts, batch_size):
        batch = prompt_nouns[i:i + batch_size]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an image prompt generation expert. Please generate CLIP-friendly image prompts based on the provided cluster keywords."
                               "Requirements:"
                               "\n1. Use the template: {class} (synonyms), a type of {domain}, {description}."
                               "\n2. {class} should be distilled from the keywords into the most commonly used noun."
                               "\n3. {domain} is the broad category that {class} belongs to, described with the most commonly used noun."
                               "\n4. {description} should only list visual features that can uniquely identify this {class}."
                               "\n5. Prompts should be concise and precise, without numbering or extraneous symbols."
                               "\n6. Output the results directly, without additional explanations."
                },
                {
                    "role": "user",
                    "content": f"Please generate a prompt for each row of cluster keywords: {batch}."
                }
            ],
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.choices[0].message.content
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            p = p.replace("_", " ").replace("-", " ")
            print(p)
            prompts_.append(p)
    return prompts_


def generate_one_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, n_prompts):
    # one prompt for each class center
    prompts_ = []
    batch_size = 30
    for i in range(0, n_prompts, batch_size):
        batch = prompt_nouns[i:i + batch_size]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are an image prompt generation expert specialized in dog breeds. Please generate CLIP-friendly image prompts based on the provided cluster keywords."
                               "Requirements:"
                               "\n1. Use the template: {breed}, {description}."
                               "\n2. {breed} should be distilled from the keywords into the most commonly used breed name."
                               "\n3. {description} should only list visual features that can uniquely identify this {breed}."
                               "\n4. Prompts should be concise and precise, without numbering or extraneous symbols."
                               "\n5. Output the results directly, without additional explanations."
                               "\n6. For example: Golden Retriever, yellow coat, long hair, friendly expression, floppy ears."
                },
                {
                    "role": "user",
                    "content": f"Please generate a prompt for each row of cluster keywords: {batch}."
                }
            ],
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.choices[0].message.content
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            p = p.replace("_", " ").replace("-", " ")
            print(p)
            prompts_.append(p)
    return prompts_


def generate_M_prompt_for_each_cluster(prompt_nouns, n_prompts, M):
    # K prompts for each class center
    prompts_ = []
    batch_size = 8
    for i in range(0, n_prompts, batch_size):
        end_idx = min(i + batch_size, n_prompts)
        batch = prompt_nouns[i:end_idx]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an image prompt generation expert. Please generate {M} CLIP-friendly image prompts for each cluster based on the provided cluster keywords."
                               "Requirements:"
                               "\n1. Use the template: {class} (synonyms), a type of {domain}, {description}."
                               "\n2. {class} should be distilled from the keywords into the most commonly used noun."
                               "\n3. {domain} is the broad category that {class} belongs to, described with the most commonly used noun."
                               "\n4. For different prompts of a cluster, their {class} should be the same."
                               "\n5. For different prompts of a cluster, their {description} should be diverse, each {description} should only list visual features that can uniquely identify this {class}."
                               "\n6. Prompts should be concise and precise, without numbering or extraneous symbols."
                               "\n7. Output the results directly, without additional explanations."
                               f"\n8. Generate exactly {M} prompts for each cluster."
                },
                {
                    "role": "user",
                    "content": f"Please generate {M} prompts for each cluster based on the provided cluster keywords: {batch}."
                }
            ],
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.choices[0].message.content
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            p = p.replace("_", " ").replace("-", " ")
            print(p)
            prompts_.append(p)
    return prompts_


def generate_M_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, n_prompts, M):
    # K prompts for each class center
    prompts_ = []
    batch_size = 8
    for i in range(0, n_prompts, batch_size):
        end_idx = min(i + batch_size, n_prompts)
        batch = prompt_nouns[i:end_idx]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an image prompt generation expert specialized in dog breeds. Please generate {M} CLIP-friendly prompts for each cluster based on the provided cluster keywords."
                               "Requirements:"
                               "\n1. Use the template: {breed}, {description}."
                               "\n2. {breed} should be distilled from the keywords into the most commonly used dog breed."
                               "\n3. For different prompts of a cluster, their {breed} should be the same."
                               "\n4. For different prompts of a cluster, their {description} should be diverse, each {description} should only list visual features that can uniquely identify this {breed}."
                               "\n5. Prompts should be concise and precise, without numbering or extraneous symbols."
                               "\n6. Output the results directly, without additional explanations."
                               "\n7. For example: Golden Retriever, yellow coat, long hair, friendly expression, floppy ears."
                               f"\n8. Generate exactly {M} prompts for each cluster."
                },
                {
                    "role": "user",
                    "content": f"Please generate {M} prompts for each cluster based on the provided cluster keywords: {batch}."
                }
            ],
            temperature=0.5,
            max_tokens=3000,
            stream=False
        )
        content = response.choices[0].message.content
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            p = p.replace("_", " ").replace("-", " ")
            print(p)
            prompts_.append(p)
    return prompts_


if __name__ == "__main__":
    # datasets = ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"]
    datasets = ["DTD"]
    for dataset in datasets:
        with open("./data/5_related_nouns_for_each_center/related_nouns_for_each_center_" + dataset + ".pkl", "rb") as f:
            prompt_nouns = pickle.load(f)
        #
        flag = 'M_prompt'
        #
        if flag == 'one_prompt':
            if dataset == "ImageNet-Dogs":
                prompts_ = generate_one_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, len(prompt_nouns))
            else:
                prompts_ = generate_one_prompt_for_each_cluster(prompt_nouns, len(prompt_nouns))
            #
            model = CLIPModel(model_name="ViT-B/32").cuda()
            model.eval()
            #
            features = []
            for i_pro in prompts_:
                feature = torch.zeros(1, 512).to("cuda")
                for template in templates:
                    prompt = template(i_pro.split('(')[0]) + i_pro[i_pro.find('('):]
                    tokens = clip.tokenize(prompt, truncate=True).to("cuda")
                    feature = feature + model.encode_text(tokens)
                features.append((feature / len(templates)).cpu().detach().numpy().tolist())
            features = np.squeeze(np.array(features))
            np.save("./data/3_one_prompt/Deepseek/" + dataset + "_prompt_embedding_for_each_center.npy", features)
        else:
            M = 5
            if dataset == "ImageNet-Dogs":
                prompts_ = generate_M_prompt_for_each_cluster_ImageNet_Dogs(prompt_nouns, len(prompt_nouns), M)
            else:
                prompts_ = generate_M_prompt_for_each_cluster(prompt_nouns, len(prompt_nouns), M)
            np.save("./data/4_M_prompts/initialPrompts/Deepseek/" + dataset + "_K_prompts_for_each_center.npy", prompts_)
