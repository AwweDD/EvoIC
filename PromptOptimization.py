from open_clip import create_model_from_pretrained, get_tokenizer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from evalMetric import cluster_metric
import torch.nn.functional as F
from sklearn import metrics
from CLIP import CLIPModel
import numpy as np
import faiss
import torch
import clip

templates = [
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}."
]


def tsne_visualization(data, n_clusters, filename):
    from matplotlib.colors import ListedColormap
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    pro_, pred_ = kmeans(data, n_clusters)
    pro_, pred_ = align_cluster_probabilities(pro_, pred_, labels)
    ari = metrics.adjusted_rand_score(labels, pred_)
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    plot_data = data_2d

    _, cluster_labels = kmeans(data, n_clusters)

    base_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
                   '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ac']
    extend_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                     '#c7c7c7', '#dbdb8d', '#9edae5', '#f7b6d3', '#c49c94']
    full_colors = base_colors + extend_colors

    if n_clusters <= 10:
        cmap = ListedColormap(base_colors[:n_clusters])
    elif 10 < n_clusters <= 20:
        cmap = ListedColormap(full_colors[:n_clusters])
    else:
        cmap = plt.get_cmap('turbo', n_clusters)

    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], c=cluster_labels,
                          cmap=cmap, s=3.5, alpha=0.9)

    x_min, x_max = plot_data[:, 0].min(), plot_data[:, 0].max()
    y_min, y_max = plot_data[:, 1].min(), plot_data[:, 1].max()

    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(rotation=0)
    plt.yticks()
    plt.title("ARI: {:.1f}".format(ari * 100), fontsize=12, fontweight='bold')
    plt.tight_layout()

    ax = plt.gca()

    ax.spines['top'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['right'].set_linewidth(1.1)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)
    plt.savefig(filename, dpi=500, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.show()


def assign_nouns_to_each_cluster(imagesEmbedding, M):
    _, predLab = kmeans(imagesEmbedding, number_clustering)

    nouns_embedding = np.load("./data/1_text_embedding/WordNetNouns_Embeddings.npy")

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    imagesEmbedding = torch.from_numpy(imagesEmbedding).cuda().half()

    image_centers = torch.zeros((number_clustering, 512), dtype=torch.float16).cuda()
    for k in range(number_clustering):
        image_centers[k] = imagesEmbedding[predLab == k].mean(dim=0)
    image_centers = F.normalize(image_centers, dim=1)

    nouns_for_all_cluster = []

    for i in range(number_clustering):
        similarity = torch.matmul(nouns_embedding, image_centers[i].T)
        topK_similarity, topK_index = torch.topk(similarity, k=M)
        aa = nouns_embedding[topK_index]
        for ii in range(M):
            nouns_for_all_cluster.append(list(aa.cpu().numpy()[ii]))
    return np.array(nouns_for_all_cluster)


def replace_Ind(Ind, pro_sim):
    ce = np.apply_along_axis(lambda s: -np.sum((p := np.unique(s, return_counts=True)[1] / len(s)) * np.log2(p)), axis=0, arr=Ind)
    ce = (ce - np.min(ce) + 1e-9) / (np.max(ce) - np.min(ce) + 1e-9)
    P = ce * (1 - pro_sim[np.arange(number_clustering), Ind]) * 0.1
    N, M = Ind.shape
    R = np.random.rand(N, M)
    replace_mask = R <= P
    X_prime = Ind.copy()
    X_prime[replace_mask] = -1

    return X_prime


def align_cluster_probabilities(pro, predLab, true_label):
    #
    cm = confusion_matrix(true_label, predLab)
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    label_mapping = dict(zip(col_ind, row_ind))
    aligned_labels = np.array([label_mapping[p] for p in predLab])
    #
    aligned_probs = np.zeros_like(pro)
    for old_idx, new_idx in label_mapping.items():
        aligned_probs[:, new_idx] = pro[:, old_idx]

    return aligned_probs, aligned_labels


def fast_silhouette_gpu(X, labels, device="cuda"):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    unique_labs, lbl_indices, lbl_counts = torch.unique(labels, return_inverse=True, return_counts=True)
    n_clusters, n_samples = len(unique_labs), X.shape[0]

    a = torch.zeros(n_samples, device=device)
    for c in range(n_clusters):
        mask, cnt = (lbl_indices == c), lbl_counts[c]
        if cnt == 1:
            a[mask] = 0.
            continue
        cluster_X = X[mask]
        sq_sum = torch.sum(cluster_X ** 2, dim=1, keepdim=True)
        dist = torch.sqrt(torch.clamp(sq_sum + sq_sum.T - 2 * cluster_X @ cluster_X.T, min=1e-8))
        a[mask] = (dist.sum(1) - torch.diag(dist)) / (cnt - 1)

    centers = torch.stack([X[lbl_indices == c].mean(0) for c in range(n_clusters)])
    sample2center = torch.cdist(X, centers, p=2)
    sample2center[torch.arange(n_samples), lbl_indices] = float('inf')
    b = sample2center.min(1)[0]

    return torch.mean((b - a) / torch.maximum(a, b)).item()


def prompts_embedding(prompts):
    promptsEmbedding = []
    for i_pro in prompts:
        feature = torch.zeros(1, 512).to("cuda")
        for template in templates:
            prompt = template(i_pro.split('(')[0]) + i_pro[i_pro.find('('):]
            tokens = clip.tokenize(prompt, truncate=True).to("cuda")
            feature = feature + model.encode_text(tokens)
        promptsEmbedding.append((feature / len(templates)).cpu().detach().numpy().tolist())
    promptsEmbedding = np.squeeze(np.array(promptsEmbedding))
    promptsEmbedding = promptsEmbedding / np.linalg.norm(promptsEmbedding, axis=1, keepdims=True)
    return promptsEmbedding


def prompts_embedding_medical(prompts):
    @torch.no_grad()
    def get_features(texts):
        tokens = Bio_med_tokenizer(texts).to("cuda")
        return Bio_medCLIP_model_.encode_text(tokens).cpu().numpy()

    #
    promptsEmbedding = []
    for i_pro in prompts:
        feature = np.zeros((1, 512))
        for template in templates:
            prompt = template(i_pro.split('(')[0]) + i_pro[i_pro.find('('):]
            feature = feature + get_features(prompt)
        promptsEmbedding.append((feature / len(templates)).tolist())
    promptsEmbedding = np.squeeze(np.array(promptsEmbedding))
    promptsEmbedding = promptsEmbedding / np.linalg.norm(promptsEmbedding, axis=1, keepdims=True)
    return promptsEmbedding


def kmeans(data, cluster_num):
    d = data.shape[1]
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=30, nredo=20)
    data = data.astype(np.float32)
    kmeans.train(data)
    #
    centroids = kmeans.centroids
    #
    data = torch.from_numpy(data).cuda().half()
    data = F.normalize(data, dim=1)
    centroids = torch.from_numpy(centroids).cuda().half()
    centroids = F.normalize(centroids, dim=1)
    #
    similarities = torch.matmul(data, centroids.T)
    #
    probabilities = torch.softmax(similarities / temperature, dim=1).cpu().numpy()
    hard_labels = np.argmax(probabilities, axis=1)

    return probabilities, hard_labels


class fitness_function:
    def entropy(self, logit):
        logit = torch.from_numpy(logit).cuda().half()
        logit = logit.mean(dim=0)
        logit_ = torch.clamp(logit, min=1e-9)
        b = logit_ * torch.log(logit_)
        return -b.sum()

    def consistency_loss(self, logit1, logit2):
        logit1 = torch.from_numpy(logit1).cuda().half()
        logit2 = torch.from_numpy(logit2).cuda().half()
        b, n = logit1.size()
        similarity = torch.bmm(logit1.view(b, 1, n), logit2.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones)
        return consistency_loss


class crossover:
    def single_point_crossover_ind(self, p1, p2):
        cross_point = np.random.randint(1, len(p1))
        c1 = np.hstack((p1[:cross_point], p2[cross_point:]))
        c2 = np.hstack((p2[:cross_point], p1[cross_point:]))
        return c1, c2

    def two_point_crossover_ind(self, p1, p2):
        cp1, cp2 = sorted(np.random.randint(0, len(p1), 2))
        c1 = np.hstack((p1[:cp1], p2[cp1:cp2], p1[cp2:]))
        c2 = np.hstack((p2[:cp1], p1[cp1:cp2], p2[cp2:]))
        return c1, c2

    def uniform_crossover_ind(self, p1, p2, swap_prob=0.5):
        swap_mask = np.random.rand(len(p1)) < swap_prob
        c1 = np.where(swap_mask, p2, p1)
        c2 = np.where(swap_mask, p1, p2)
        return c1, c2


class mutation:
    def random_replace_mutation(self, individual, mut_prob):
        ind = individual.copy()
        mut_mask = np.random.rand(len(ind)) < mut_prob
        for idx in np.where(mut_mask)[0]:
            current_val = ind[idx]
            valid_vals = [v for v in range(M) if v != current_val]
            ind[idx] = np.random.choice(valid_vals)
        return ind

    def uniform_perturb_mutation(self, individual, mut_prob, perturb_range=1):
        ind = individual.copy()
        mut_mask = np.random.rand(len(ind)) < mut_prob
        perturbations = np.random.randint(
            -perturb_range, perturb_range + 1, len(ind)
        )
        ind[mut_mask] += perturbations[mut_mask]
        ind = np.clip(ind, -1, M - 1)
        return ind

    def targeted_swap_mutation(self, individual, mut_prob):
        ind = individual.copy()
        if np.random.rand() < mut_prob:
            idx1, idx2 = np.random.choice(len(ind), 2, replace=False)
            ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
        return ind


class evolutionary:
    def __init__(self, N, number_clustering):
        self.N = N
        self.number_clustering = number_clustering

    def initialization(self, imagesEmbedding, promptsEmbedding):
        _, predLab = kmeans(imagesEmbedding, self.number_clustering)
        #
        imagesEmbedding = torch.from_numpy(imagesEmbedding).cuda().half()
        promptsEmbedding = torch.from_numpy(promptsEmbedding).cuda().half()
        #
        image_centers = torch.zeros((self.number_clustering, 512), dtype=torch.float16).cuda()
        for k in range(self.number_clustering):
            image_centers[k] = imagesEmbedding[predLab == k].mean(dim=0)
        image_centers = F.normalize(image_centers, dim=1)
        #
        prompts_centers = promptsEmbedding.mean(dim=1)
        prompts_centers = F.normalize(prompts_centers, dim=1)
        #
        similarity = torch.matmul(prompts_centers, image_centers.T)
        softmax_similarity = torch.softmax(similarity, dim=1).cpu().float()
        max_values, max_indices = torch.max(softmax_similarity, dim=1)
        #
        promptsEmbedding = F.normalize(promptsEmbedding, dim=2)
        similarity_used_to_initial = (promptsEmbedding * image_centers[max_indices].unsqueeze(1)).sum(dim=-1)
        similarity_used_to_initial = F.softmax(similarity_used_to_initial / 0.001, dim=1)
        #
        ind = torch.zeros((self.N, self.number_clustering), dtype=torch.long)
        for i in range(self.N):
            ind[i] = torch.multinomial(similarity_used_to_initial, num_samples=1).squeeze(-1)
        return ind.cpu().numpy(), similarity_used_to_initial.cpu().numpy()

    def calFitness_version1(self, images_embedding, All_ind_proEmbedding):
        images_embedding = torch.from_numpy(images_embedding).cuda().half()
        All_ind_proEmbedding = torch.from_numpy(All_ind_proEmbedding).cuda().half()
        #
        batch_size = 8192
        #
        similarities = []
        for i in range(image_num // batch_size + 1):
            start = i * batch_size
            end = start + batch_size
            if end > image_num:
                end = image_num
            similarity = torch.einsum('md,nvd->nmv', images_embedding[start:end], All_ind_proEmbedding)
            similarities.append(similarity)
        similarities = torch.cat(similarities, dim=1).cuda().half()
        #
        similarities = similarities / temperature
        similarity_probs = torch.softmax(similarities, dim=2)
        topK_similarities, topK_indices = torch.topk(similarity_probs, k=K, dim=2)
        topK_similarities = topK_similarities / topK_similarities.sum(dim=2, keepdim=True)
        #
        cross_modal_cons_loss_ = []
        distribution_loss_ = []
        homogeneous_loss_ = []
        for i in range(self.N):
            topK_index = topK_indices[i]
            topK_similarity = topK_similarities[i]
            ind_proEmbedding = All_ind_proEmbedding[i]
            #
            topK_prompts_embeddings = ind_proEmbedding[topK_index]
            # weighted text embedding
            weighted_text_embedding = (topK_similarity.unsqueeze(-1) * topK_prompts_embeddings).sum(dim=1)
            weighted_text_embedding = F.normalize(weighted_text_embedding, dim=1)
            # multi modal embedding
            multiModal_embedding = np.concatenate(
                [images_embedding.cpu().numpy(), weighted_text_embedding.cpu().numpy()], axis=1)
            #
            pro_text, predLab_text = kmeans(weighted_text_embedding.cpu().numpy(), cluster_num)
            pro_multiModal, predLab_multiModal = kmeans(multiModal_embedding, cluster_num)
            pro_text, predLab_text = align_cluster_probabilities(pro_text, predLab_text, labels)
            pro_multiModal, predLab_multiModal = align_cluster_probabilities(pro_multiModal, predLab_multiModal, labels)
            #
            cluster_metric(labels, predLab_text)
            cluster_metric(labels, predLab_multiModal)
            #
            cross_modal_cons_loss = FF.consistency_loss(pro_multiModal, pro_text)
            distribution_loss = fast_silhouette_gpu(multiModal_embedding, predLab_multiModal)
            homogeneous_loss = FF.entropy(pro_multiModal)
            #
            cross_modal_cons_loss_.append(cross_modal_cons_loss.cpu().numpy())
            distribution_loss_.append(distribution_loss)
            homogeneous_loss_.append(homogeneous_loss.cpu().numpy())
            print(i, '==============================')
        #
        cross_modal_cons_loss_ = (cross_modal_cons_loss_ - np.min(cross_modal_cons_loss_) + 1e-9) / (np.max(cross_modal_cons_loss_) - np.min(cross_modal_cons_loss_) + 1e-9)
        distribution_loss_ = (distribution_loss_ - np.min(distribution_loss_) + 1e-9) / (np.max(distribution_loss_) - np.min(distribution_loss_) + 1e-9)
        homogeneous_loss_ = (homogeneous_loss_ - np.min(homogeneous_loss_) + 1e-9) / (np.max(homogeneous_loss_) - np.min(homogeneous_loss_) + 1e-9)
        #
        loss = (cross_modal_cons_loss_ * 0.4 - distribution_loss_ * 0.2 - homogeneous_loss_ * 0.4)
        loss = loss / np.linalg.norm(loss)
        #
        return 1 - loss

    def calFitness(self, images_embedding, All_ind_proEmbedding):
        images_embedding = torch.from_numpy(images_embedding).cuda().half()
        #
        cross_modal_cons_loss_ = []
        distribution_loss_ = []
        homogeneous_loss_ = []
        metrics = []
        #
        for i in range(self.N):
            Ind_proEmbedding = torch.from_numpy(All_ind_proEmbedding[i]).cuda().half()
            #
            batch_size = 8192
            #
            similarities = []
            for j in range(image_num // batch_size + 1):
                start = j * batch_size
                end = start + batch_size
                if end > image_num:
                    end = image_num
                similarity = torch.matmul(images_embedding[start:end], Ind_proEmbedding.T)
                similarities.append(similarity)
            similarities = torch.cat(similarities, dim=0).cuda().half()
            #
            similarities = similarities / temperature
            similarity_probs = torch.softmax(similarities, dim=1)
            topK_similarity, topK_index = torch.topk(similarity_probs, k=K, dim=1)
            topK_similarity = topK_similarity / topK_similarity.sum(dim=1, keepdim=True)
            #
            topK_prompts_embeddings = Ind_proEmbedding[topK_index]
            # weighted text embedding
            weighted_text_embedding = (topK_similarity.unsqueeze(-1) * topK_prompts_embeddings).sum(dim=1)
            weighted_text_embedding = F.normalize(weighted_text_embedding, dim=1)
            # multi modal embedding
            multiModal_embedding = np.concatenate([images_embedding.cpu().numpy(), weighted_text_embedding.cpu().numpy()], axis=1)
            #
            pro_text, predLab_text = kmeans(weighted_text_embedding.cpu().numpy(), cluster_num)
            pro_multiModal, predLab_multiModal = kmeans(multiModal_embedding, cluster_num)

            pro_text, predLab_text = align_cluster_probabilities(pro_text, predLab_text, labels)
            pro_multiModal, predLab_multiModal = align_cluster_probabilities(pro_multiModal, predLab_multiModal, labels)
            #
            metrics_ = cluster_metric(labels, predLab_multiModal)
            #
            cross_modal_cons_loss = FF.consistency_loss(pro_multiModal, pro_text)
            distribution_loss = fast_silhouette_gpu(multiModal_embedding, predLab_multiModal)
            homogeneous_loss = FF.entropy(pro_multiModal)
            #
            cross_modal_cons_loss_.append(cross_modal_cons_loss.cpu().numpy())
            distribution_loss_.append(distribution_loss)
            homogeneous_loss_.append(homogeneous_loss.cpu().numpy())
            metrics.append(metrics_)
            print(i, '==============================')
            #
        cross_modal_cons_loss_ = (cross_modal_cons_loss_ - np.min(cross_modal_cons_loss_) + 1e-9) / (np.max(cross_modal_cons_loss_) - np.min(cross_modal_cons_loss_) + 1e-9)
        distribution_loss_ = (distribution_loss_ - np.min(distribution_loss_) + 1e-9) / (np.max(distribution_loss_) - np.min(distribution_loss_) + 1e-9)
        homogeneous_loss_ = (homogeneous_loss_ - np.min(homogeneous_loss_) + 1e-9) / (np.max(homogeneous_loss_) - np.min(homogeneous_loss_) + 1e-9)
        #
        loss = (cross_modal_cons_loss_ * 0.4 - distribution_loss_ * 0.2 - homogeneous_loss_ * 0.4)
        loss = loss / np.linalg.norm(loss)
        #
        return 1 - loss, np.array(metrics)

    def TournamentSelection(self, fitness, k=5):
        sel_parents = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            candidate_idx = np.random.choice(self.N, size=k, replace=False)
            candidate_fitness = fitness[candidate_idx]
            best_in_candidate = np.argmax(candidate_fitness)
            sel_parents[i] = candidate_idx[best_in_candidate]
        return sel_parents

    def crossover(self, parents):
        offspring = []
        for i in range(0, self.N, 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            c1, c2 = CO.uniform_crossover_ind(p1, p2)
            offspring.append(c1)
            offspring.append(c2)
        return np.array(offspring)

    def mutation(self, parents, pro_sim):
        #
        offspring = []
        for i in range(self.N):
            c = MT.uniform_perturb_mutation(parents[i], 0.001)
            offspring.append(c)
        return np.array(offspring)

    def mutation_(self, parents, mut_pro=0.001):
        offspring = []
        for i in range(self.N):
            c = MT.uniform_perturb_mutation(parents[i], mut_pro)
            offspring.append(c)
        return np.array(offspring)

    def EnvironmentSelection(self, individual, fitness, metrics):
        sorted_indices = np.argsort(-fitness)
        individual_ = individual[sorted_indices[:self.N]]
        fitness_ = fitness[sorted_indices[:self.N]]
        metrics_ = metrics[sorted_indices[:self.N]]
        return individual_, fitness_, metrics_

    def calculate(self, images_embedding, ind):
        ind_ = ind.copy()
        All_ind_proEmbedding = []
        for i in range(N):
            valid_mask = ind_[i] != -1
            valid_first_dim = np.arange(number_clustering)[valid_mask]
            valid_second_dim = ind_[i][valid_mask]
            promptsEmbedding = initial_promptsEmbedding[valid_first_dim, valid_second_dim]
            All_ind_proEmbedding.append(promptsEmbedding)
        fitness, metrics = Evo.calFitness(images_embedding, All_ind_proEmbedding)
        return fitness, metrics


if __name__ == '__main__':
    """
        "CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"
    """
    LLM = "DeepSeek-V3.2"
    #
    datasets = ["DTD"]
    #
    datasets_K = {"CIFAR-10": 10, "CIFAR-20": 10, "STL-10": 10, "DTD": 10, "ImageNet-10": 10, "ImageNet-Dogs": 2, "ISIC2018": 2}
    temperatures = {"CIFAR-10": 0.02, "CIFAR-20": 0.02, "STL-10": 0.02, "DTD": 0.02, "ImageNet-10": 0.05, "ImageNet-Dogs": 0.05, "ISIC2018": 0.05}  # "ViT-B-32"
    # temperatures = {"CIFAR-10": 0.02, "CIFAR-20": 0.02, "STL-10": 0.05, "DTD": 0.05, "ImageNet-10": 0.05, "ImageNet-Dogs": 0.05}  # "ViT-B-16"
    #
    model_name = "ViT-B-32"
    model = CLIPModel(model_name="ViT-B/32").cuda()
    model.eval()
    #
    for dataset in datasets:
        print(dataset)
        #
        K = datasets_K[dataset]
        temperature = temperatures[dataset]
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
        elif dataset == "ISIC2018":
            cluster_num = 7
        else:
            raise NotImplementedError
        #
        if dataset == "ISIC2018":
            images_embedding = np.load("./data/0_image_embedding_BiomedCLIP/" + dataset + "_image_embedding_test.npy")
            labels = np.loadtxt("./data/0_image_embedding_BiomedCLIP/" + dataset + "_labels_test.txt")
        else:
            images_embedding = np.load("./data/0_image_embedding/" + model_name + "/" + dataset + "_image_embedding_test.npy")
            labels = np.loadtxt("./data/0_image_embedding/" + model_name + "/" + dataset + "_labels_test.txt")
        images_embedding = images_embedding / np.linalg.norm(images_embedding, axis=1, keepdims=True)
        image_num = images_embedding.shape[0]
        #
        M = 5  # number of the prompts of each class center
        initial_prompts = np.load("./data/4_M_prompts/initialPrompts/" + LLM + "/" + dataset + "_K_prompts_for_each_center.npy", allow_pickle=True)
        #
        try:
            initial_promptEmbeddings = np.load("./data/4_M_prompts/initialPrompts/" + LLM + "/" + dataset + "_initial_promptEmbeddings.npy")
            print("Multiple prompts for each center loaded.")
        except:
            print('perform prompt embedding...')
            if dataset == "ISIC2018":
                model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                Bio_medCLIP_model_, preprocess = create_model_from_pretrained(model_name)
                Bio_med_tokenizer = get_tokenizer(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                Bio_medCLIP_model_ = Bio_medCLIP_model_.to(device)
                Bio_medCLIP_model_.eval()
                #
                initial_promptEmbeddings = prompts_embedding_medical(initial_prompts)
            else:
                initial_promptEmbeddings = prompts_embedding(initial_prompts)
            np.save("./data/4_M_prompts/initialPrompts/" + LLM + "/" + dataset + "_initial_promptEmbeddings.npy", initial_promptEmbeddings)
            print('prompt embedding done.')
            exit()
        #
        number_clustering = initial_prompts.shape[0] // M
        initial_prompts = initial_prompts.reshape(number_clustering, M)
        # noinspection PyArgumentList
        initial_promptsEmbedding = initial_promptEmbeddings.reshape(number_clustering, M, -1)
        #
        """
            Optimization stage ==============================================================
        """
        Iter = 10
        N = 20
        #
        FF = fitness_function()
        CO = crossover()
        MT = mutation()
        Evo = evolutionary(N, number_clustering)
        #
        Ind, pro_sim = Evo.initialization(images_embedding, initial_promptsEmbedding)
        Ind = replace_Ind(Ind, pro_sim)
        #
        Fitness = np.zeros(N)
        Metrics = np.zeros(N)
        #
        for t in range(Iter):
            if t == 0:
                Fitness, Metrics = Evo.calculate(images_embedding, Ind)
            print("The", t + 1, "-th generation is evolving...")
            #
            Sel_parents = Evo.TournamentSelection(Fitness)
            Offspring = Evo.crossover(Ind[Sel_parents])
            Offspring = Evo.mutation(Offspring, pro_sim)
            #
            print('The performance of the offsprings......')
            Offspring_fitness, Offspring_metrics = Evo.calculate(images_embedding, Offspring)
            # Update the population
            Ind, Fitness, Metrics = Evo.EnvironmentSelection(np.concatenate((Ind, Offspring), axis=0), np.concatenate((Fitness, Offspring_fitness), axis=0),
                                                             np.concatenate((Metrics, Offspring_metrics), axis=0))
            print('The performance of the individuals......')
            for i in range(N):
                print("[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}".format(Metrics[i][0] * 100, Metrics[i][1] * 100, Metrics[i][2] * 100))
