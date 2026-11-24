import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import torch
import faiss


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    X = X.astype(np.float32)
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=30, nredo=20)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I


"""
    ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
    [167, 167, 17, 43, 65, 141, 303, 4271]
"""
"""
    ours: 
    ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"]
    [100, 200, 100, 100, 100, 100]
    [3, 3, 3, 3, 3, 3]
"""
if __name__ == "__main__":
    model_name = "ViT-B-32"
    #
    datasets = ["ImageNet"]
    #
    cluster_num = 10000
    topK = 5
    #
    for dataset in datasets:

        nouns_embedding = np.load("./data/1_text_embedding/WordNetNouns_Embeddings.npy")
        nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)

        images_embedding = np.load("./data/0_image_embedding/" + model_name + "/" + dataset + "_image_embedding_train.npy")

        images_embedding = images_embedding / np.linalg.norm(images_embedding, axis=1, keepdims=True)

        nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
        nouns_num = nouns_embedding.shape[0]

        images_embedding = torch.from_numpy(images_embedding).cuda().half()
        image_num = images_embedding.shape[0]

        predLab = kmeans(images_embedding.cpu().numpy(), cluster_num)

        image_centers = torch.zeros((cluster_num, 512), dtype=torch.float16).cuda()

        for k in range(cluster_num):
            image_centers[k] = images_embedding[predLab == k].mean(dim=0)
        image_centers = F.normalize(image_centers, dim=1)

        similarity = torch.matmul(image_centers, nouns_embedding.T)
        softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()

        class_pred = torch.argmax(softmax_nouns, dim=0).long()
        selected_idx = torch.zeros_like(class_pred, dtype=torch.bool)
        related_nouns_for_each_center = []
        nouns = pd.read_csv("./data/1_text_embedding/WordNetNouns.csv").iloc[:, 0].values
        for k in range(cluster_num):
            if (class_pred == k).sum() == 0:
                continue
            class_index = torch.where(class_pred == k)[0]
            confidence = softmax_nouns[k, class_index]
            rank = torch.argsort(confidence, descending=True)
            selected_idx[class_index[rank[:topK]]] = True
            related_nouns_for_each_center.append(nouns[class_index[rank[:topK]].cpu().numpy()].tolist())
        print("selected done.")

        with open("./data/5_related_nouns_for_each_center/related_nouns_for_each_center_" + dataset + ".pkl", "wb") as f:
            pickle.dump(related_nouns_for_each_center, f)
