from dashscope import Generation, MultiModalConversation
import numpy as np
import dashscope
import pickle
import os

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


def generate_M_prompt_for_each_cluster(prompt_nouns, n_prompts, M):
    prompts_ = []
    batch_size = 8
    for i in range(0, n_prompts, batch_size):
        end_idx = min(i + batch_size, n_prompts)
        batch = prompt_nouns[i:end_idx]
        messages = [
            {
                "role": "system",
                "content": f"You are an image prompt generation expert. Please generate {M} CLIP-friendly image prompts for each cluster based on the provided cluster keywords."
                           "Requirements:"
                           "\n1. Use the template: {class} (synonyms), a type of {domain}, {description}."
                           "\n2. {class} should be distilled from the keywords into the most commonly used noun. (synonyms) is its synonyms."
                           "\n3. {domain} is the broad category that {class} belongs to, described with the most commonly used noun."
                           "\n4. For different prompts of a cluster, their {class} must be the same."
                           "\n5. For different prompts of a cluster, their {description} should be diverse, each {description} should only list visual features that can uniquely identify this {class}."
                           "\n6. Prompts must be concise and precise, without numbering or extraneous symbols."
                           "\n7. Output the results directly, without additional explanations."
                           f"\n8. Generate exactly {M} prompts for each cluster."
                           """
                           Example:
                               aeroplane (aircraft, passenger plane), a type of aerospace vehicle, streamlined fuselage tapering gradually toward pointed nose and tail cone
                               aeroplane (aircraft, passenger plane), a type of aerospace vehicle, broad swept-back wings fixed midway along body with rounded wingtip pods
                               aeroplane (aircraft, passenger plane), a type of aerospace vehicle, compact underhung engine nacelles hanging beneath each wing surface
                               aeroplane (aircraft, passenger plane), a type of aerospace vehicle, vertical stabilizer paired with two small horizontal tail fins at rear fuselage
                               aeroplane (aircraft, passenger plane), a type of aerospace vehicle, smooth silver metallic outer skin with rows of tiny rectangular passenger windows along side
                           """
            },
            {
                "role": "user",
                "content": f"Please generate exactly {M} prompts for each cluster based on the provided cluster keywords: {batch}."
            }
        ]
        response = Generation.call(
            api_key=os.getenv("sk-d15c92dbd5794afd8ddff005e2145c66"),
            model="deepseek-v4-flash",
            messages=messages,
            result_format="message",
            temperature=0.5,
            max_tokens=3000,
            enable_thinking=False
        )
        content = response.output.choices[0].message.content
        prompts = [p.strip() for p in content.split('\n') if p.strip()]
        for p in prompts:
            p = p.replace("_", " ").replace("-", " ")
            print(p)
            prompts_.append(p)
    return prompts_


if __name__ == "__main__":
    datasets = ["CIFAR-10"]
    for dataset in datasets:
        with open("./data/5_related_nouns_for_each_center/related_nouns_for_each_center_" + dataset + ".pkl", "rb") as f:
            prompt_nouns = pickle.load(f)
        M = 5
        prompts_ = generate_M_prompt_for_each_cluster(prompt_nouns, len(prompt_nouns), M)
        np.save("./data/4_M_prompts/initialPrompts/deepseek-v4-pro/" + dataset + "_K_prompts_for_each_center.npy", prompts_)

# generate the prompt for the medical dataset

# from sklearn.metrics import pairwise_distances
# from dashscope import Generation
# import torch.nn.functional as F
# from openai import OpenAI
# import numpy as np
# import dashscope
# import requests
# import base64
# import faiss
# import torch
# import os
#
#
# def kmeans(data, cluster_num):
#     d = data.shape[1]
#     kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=30, nredo=20)
#     data = data.astype(np.float32)
#     kmeans.train(data)
#     #
#     centroids = kmeans.centroids
#     #
#     data = torch.from_numpy(data).cuda().half()
#     data = F.normalize(data, dim=1)
#     centroids = torch.from_numpy(centroids).cuda().half()
#     centroids = F.normalize(centroids, dim=1)
#     #
#     similarities = torch.matmul(data, centroids.T)
#     #
#     probabilities = torch.softmax(similarities, dim=1).cpu().numpy()
#     hard_labels = np.argmax(probabilities, axis=1)
#
#     return hard_labels
#
#
# dataset_name = "ISIC2018"
# n_clusters = 10
# top_m = 3
#
# img_embeddings = np.load("./data/0_image_embedding_BiomedCLIP/" + dataset_name + "_image_embedding_train.npy")
# img_embeddings = img_embeddings / np.linalg.norm(img_embeddings, axis=1, keepdims=True)
#
# img_paths = np.load("./data/0_image_embedding_BiomedCLIP/" + dataset_name + "_training_image_paths.npy")
#
# pre_lab = kmeans(img_embeddings, n_clusters)
#
# #
# image_centers = torch.zeros((n_clusters, 512), dtype=torch.float16).cuda()
# imagesEmbedding = torch.from_numpy(img_embeddings).cuda().half()
# for k in range(n_clusters):
#     image_centers[k] = imagesEmbedding[pre_lab == k].mean(dim=0)
# image_centers = F.normalize(image_centers, dim=1)
# #
# image_centers = image_centers.cpu().numpy()
# #
# cluster_paths = {}
# for c in range(n_clusters):
#     mask = pre_lab == c
#     c_features = img_embeddings[mask]
#     c_paths = img_paths[mask]
#     dists = pairwise_distances(c_features, image_centers[c].reshape(1, -1), metric='cosine').flatten()
#     top_idx = np.argsort(dists)[:top_m]
#     cluster_paths[c] = c_paths[top_idx].tolist()
#
#
# def img2base64(path):
#     with open(path, 'rb') as f:
#         return base64.b64encode(f.read()).decode('utf-8')
#
#
# # Qwen
# # API_KEY = "sk-d15c92dbd5794afd8ddff005e2145c66"
# # API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
# # #
# # headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
# #
# dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
# #
# N = 5
# meta_prompt = f"""
# You are a professional medical imaging expert. Analyze the given {top_m} medical images and generate exactly {N} distinctive and rich visual descriptions.
# Requirements:
# 1. **STRICTLY USE THIS TEMPLATE FOR EVERY LINE**: {{PATHOLOGY_CLASS}}, {{VISUAL_DISCRIMINATIVE_FEATURES}}.
# 2. **NO EXTRA CONTENT**: Do NOT add numbers, bullet points, colons, or explanations.
# 3. **PATHOLOGY_CLASS**: Must be the exact same {{PATHOLOGY_CLASS}} for these {N} visual descriptions.
# 4. **FEATURES RULE**: Each visual feature must be distinct, focusing on: location, shape, margin, attenuation/signal intensity, texture, internal architecture, homogeneity, enhancement pattern, contour, and boundary.
# 5. **FORMAT RULE**: All lines must follow exactly: no braces, no extra symbols, just plain text in this structure.
# """
#
# #
# model_name = "deepseek-v4-pro"
# cluster_descriptions = []
# prompts_ = []
# for c, paths in cluster_paths.items():
#     messages = [{"role": "user", "content": []}]
#     for p in paths:
#         messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2base64(p)}"}})
#     messages[0]["content"].append({"type": "text", "text": meta_prompt})
#
#     response = Generation.call(
#         api_key=os.getenv("sk-d15c92dbd5794afd8ddff005e2145c66"),
#         model=model_name,
#         messages=messages,
#         result_format="message",
#         temperature=0.5,
#         max_tokens=3000,
#         enable_thinking=False
#     )
#     content = response.output.choices[0].message.content
#     prompts = [p.strip() for p in content.split('\n') if p.strip()]
#     for p in prompts:
#         p = p.replace("_", " ").replace("-", " ")
#         print(p)
#         prompts_.append(p)

    # data = {"model": model_name, "messages": messages, "max_tokens": 2000}
    #
    # response = requests.post(API_URL, headers=headers, json=data, timeout=30)
    # res_json = response.json()
    # desc = res_json.get("choices", [{}])[0].get("message", {}).get("content", "生成失败")
    # desc = [p.strip() for p in desc.split('\n') if p.strip()]
    # for d in desc:
    #     print(d)
    #     cluster_descriptions.append(d)
    # print('==========================')

# np.save("./data/4_M_prompts/initialPrompts/" + model_name + "/" + dataset_name + "_K_prompts_for_each_center.npy", cluster_descriptions)
