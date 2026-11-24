from dataLoader import get_dataloader
from CLIP import CLIPModel
import numpy as np
import torch
import copy

"""
    "CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD"
"""

dataset = "ImageNet"

dataloader_train, dataloader_test = get_dataloader(dataset=dataset, batch_size=1024)
model_name = "ViT-B/32"
model = CLIPModel(model_name=model_name).cuda()
model.eval()
#
features = []
labels = []
print("Inferring image features and labels...")
for iteration, (x, y) in enumerate(dataloader_train):
    x = x.cuda()
    with torch.no_grad():
        feature = model.encode_image(x)
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
        feature = model.encode_image(x)
    features_test.append(feature.cpu().numpy())
    labels_test.append(y.numpy())
    if iteration % 10 == 0:
        print(f"[Iter {iteration}/{len(dataloader_test)}]")
features_test = np.concatenate(features_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)

if dataset == "CIFAR-20":
    coarse_label = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    labels_copy = copy.deepcopy(labels)
    labels_test_copy = copy.deepcopy(labels_test)
    for i in range(20):
        for j in coarse_label[i]:
            labels[labels_copy == j] = i
            labels_test[labels_test_copy == j] = i
if model_name == "ViT-B/32":
    model_name_ = "ViT-B-32"
elif model_name == "ViT-B/16":
    model_name_ = "ViT-B-16"
elif model_name == "ViT-L/14":
    model_name_ = "ViT-L-14"
else:
    model_name_ = ''
np.save("./data/0_image_embedding/" + model_name_ + "/" + dataset + "_image_embedding_train.npy", features)
np.save("./data/0_image_embedding/" + model_name_ + "/" + dataset + "_image_embedding_test.npy", features_test)
np.savetxt("./data/0_image_embedding/" + model_name_ + "/" + dataset + "_labels_train.txt", labels)
np.savetxt("./data/0_image_embedding/" + model_name_ + "/" + dataset + "_labels_test.txt", labels_test)
