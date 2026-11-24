from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from evalMetric import cluster_metric
import torch.nn.functional as F
import numpy as np
import warnings
import torch
import faiss

warnings.simplefilter("ignore")


def align_cluster_probabilities(pred, true_label):
    cm = confusion_matrix(true_label, pred)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    label_mapping = dict(zip(col_ind, row_ind))
    aligned_labels = np.array([label_mapping[p] for p in pred])

    return aligned_labels


def kmeans(X, cluster_num):
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=30, nredo=20)
    X = X.astype(np.float32)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    return I


if __name__ == "__main__":
    """
        "CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"
    """
    model_name = "ViT-B-32"
    LLM = "Deepseek"
    #
    datasets = ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"]
    dataset_K = {"CIFAR-10": 10, "CIFAR-20": 10, "STL-10": 10, "DTD": 10, "ImageNet-10": 10, "ImageNet-Dogs": 2}
    #
    temperature = 0.02
    #
    for dataset in datasets:
        K = dataset_K[dataset]
        #
        if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
            cluster_num = 10
        elif dataset == "CIFAR-20":
            cluster_num = 20
        elif dataset == "ImageNet-Dogs":
            cluster_num = 15
        elif dataset == "DTD":
            cluster_num = 47
        elif dataset == "ImageNet":
            cluster_num = 1000
        else:
            raise NotImplementedError
        #
        images_embedding = np.load("./data/0_image_embedding/" + model_name + "/" + dataset + "_image_embedding_test.npy")
        images_embedding = images_embedding / np.linalg.norm(images_embedding, axis=1, keepdims=True)
        images_embedding = torch.from_numpy(images_embedding).cuda().half()
        image_num = images_embedding.shape[0]
        #
        labels = np.loadtxt("./data/0_image_embedding/" + model_name + "/" + dataset + "_labels_test.txt")
        #
        prompts_embedding = np.load("./data/3_one_prompt/" + LLM + "/" + dataset + "_prompt_embedding_for_each_center.npy")
        prompts_embedding = prompts_embedding / np.linalg.norm(prompts_embedding, axis=1, keepdims=True)
        prompts_embedding = torch.from_numpy(prompts_embedding).cuda().half()
        prompts_num = prompts_embedding.shape[0]
        #
        retrieval_embeddings = []
        batch_size = 8192
        #
        for i in range(image_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > image_num:
                end = image_num
            #
            similarity = torch.matmul(images_embedding[start:end], prompts_embedding.T)
            similarity = similarity / temperature
            similarity_probs = torch.softmax(similarity, dim=1)
            #
            topK_similarities, topK_indices = torch.topk(similarity_probs, k=K, dim=1)
            #
            topK_similarities = topK_similarities / topK_similarities.sum(dim=1, keepdim=True)
            #
            topK_prompts_embeddings = prompts_embedding[topK_indices]
            retrieval_embedding = (topK_similarities.unsqueeze(-1) * topK_prompts_embeddings).sum(dim=1)
            #
            retrieval_embeddings.append(retrieval_embedding)
        #
        retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
        retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
        #
        images_embedding = images_embedding.cpu().numpy()
        #
        concat_embedding = np.concatenate([images_embedding, retrieval_embedding], axis=1)
        predLab = kmeans(concat_embedding, cluster_num)
        predLab = align_cluster_probabilities(predLab, labels)
        print(dataset)
        cluster_metric(labels, predLab)
