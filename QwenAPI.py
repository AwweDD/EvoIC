from dashscope import Generation, MultiModalConversation
import numpy as np
import dashscope
import pickle
import os

dashscope.base_http_api_url = ''


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
            api_key=os.getenv(""),
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
