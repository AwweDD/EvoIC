# import random
# import shutil
# from pathlib import Path
#
#
# def concat_embedding(images_embedding, promptsEmbedding):
#     images_embedding = torch.from_numpy(images_embedding).cuda().half()
#     images_embedding = F.normalize(images_embedding, dim=1)
#     promptsEmbedding = torch.from_numpy(promptsEmbedding).cuda().half()
#     promptsEmbedding = F.normalize(promptsEmbedding, dim=1)
#     #
#     retrieval_embeddings = []
#     batch_size = 8192
#     #
#     for i in range(image_num // batch_size + 1):
#         start = i * batch_size
#         end = start + batch_size
#         if end > image_num:
#             end = image_num
#         #
#         similarity = torch.matmul(images_embedding[start:end], promptsEmbedding.T)
#         similarity = similarity / temperature
#         similarity_probs = torch.softmax(similarity, dim=1)
#         #
#         topK_similarities, topK_indices = torch.topk(similarity_probs, k=K, dim=1)
#         #
#         topK_similarities = topK_similarities / topK_similarities.sum(dim=1, keepdim=True)
#         #
#         topK_prompts_embeddings = promptsEmbedding[topK_indices]
#         retrieval_embedding = (topK_similarities.unsqueeze(-1) * topK_prompts_embeddings).sum(dim=1)
#         #
#         retrieval_embeddings.append(retrieval_embedding)
#     #
#     retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
#     retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
#     #
#     images_embedding = images_embedding.cpu().numpy()
#     #
#     concat_embeddings = np.concatenate([images_embedding, retrieval_embedding], axis=1)
#     #
#     predLab = kmeans(concat_embeddings, cluster_num)
#     #
#     cluster_metric(labels, predLab)
# def copy_same_folders_concise(src, dst, n=10):
#     """简洁版本"""
#     src_path, dst_path = Path(src), Path(dst)
#     src_train, src_val = src_path / "train", src_path / "val"
#     dst_train, dst_val = dst_path / "train", dst_path / "val"
#
#     # 获取共同子文件夹
#     train_folders = {f.name for f in src_train.iterdir() if f.is_dir()}
#     val_folders = {f.name for f in src_val.iterdir() if f.is_dir()}
#     common_folders = list(train_folders & val_folders)
#
#     selected = random.sample(common_folders, min(n, len(common_folders)))
#
#     print(f"选择 {len(selected)} 个共同子文件夹: {selected}")
#
#     # 复制train和val中的文件夹
#     for folder_name in selected:
#         shutil.copytree(src_train / folder_name, dst_train / folder_name)
#         shutil.copytree(src_val / folder_name, dst_val / folder_name)
#         print(f"已复制: {folder_name}")
#
#     print(f"完成！复制到 {dst_train} 和 {dst_val}")
#
#
# # 直接调用
# copy_same_folders_concise("/home/15t/lww/Data_paper3/imageNet",
#                           "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/datasets/ImageNet-10/", 10)
# import random
# import shutil
# from pathlib import Path

# def copy_range_folders_concise(src, dst, start=151, end=269, n=15):
#     """简洁版本"""
#     src_path, dst_path = Path(src), Path(dst)
#     src_train, src_val = src_path / "train", src_path / "val"
#     dst_train, dst_val = dst_path / "train", dst_path / "val"
#
#     # 创建目标目录
#     dst_train.mkdir(parents=True, exist_ok=True)
#     dst_val.mkdir(parents=True, exist_ok=True)
#
#     # 获取排序后的文件夹列表
#     train_folders = sorted([f for f in src_train.iterdir() if f.is_dir()], key=lambda x: x.name)
#     val_folders = sorted([f for f in src_val.iterdir() if f.is_dir()], key=lambda x: x.name)
#
#     # 获取指定范围内的文件夹
#     range_train = train_folders[start - 1:end]
#     range_val = val_folders[start - 1:end]
#
#     # 找出共同文件夹
#     train_names = {f.name for f in range_train}
#     val_names = {f.name for f in range_val}
#     common_names = list(train_names & val_names)
#
#     selected = random.sample(common_names, min(n, len(common_names)))
#
#     print(f"从第{start}-{end}个文件夹中选择 {len(selected)} 个:")
#     for name in selected:
#         print(f"  - {name}")
#
#     # 创建映射
#     train_map = {f.name: f for f in range_train}
#     val_map = {f.name: f for f in range_val}
#
#     # 复制
#     for name in selected:
#         shutil.copytree(train_map[name], dst_train / name)
#         shutil.copytree(val_map[name], dst_val / name)
#
#     print(f"完成！复制到 {dst_train} 和 {dst_val}")
#     return selected
#
#
# # 直接调用
# copy_range_folders_concise("/home/15t/lww/Data_paper3/imageNet/", "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/datasets/ImageNet-Dogs", 151, 269, 15)


# import torch
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
#
# train_path = '/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/datasets/LungDisease/Train'
# train_dataset = datasets.ImageFolder(
#     root=train_path,
#     transform=transforms.Compose([
#         # transforms.Grayscale(num_output_channels=3),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ]))
# train_loader_ = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
#
#
# def get_mean_std_value(loader, List):
#     '''
#     shape [n_samples(batch), channels, height, width]
#     '''
#     for data, label in loader:
#         List.append(data)
#
#     Data = torch.cat(List, dim=0)
#     mean = torch.mean(Data, dim=[0, 2, 3], keepdim=True)
#     std = torch.mean((Data - mean) ** 2, dim=[0, 2, 3]) ** 0.5
#
#     return mean.squeeze(), std.squeeze()
#
#
# mean, std = get_mean_std_value(train_loader_, List=[])
# print('mean = {},std = {}'.format(mean, std))
import torch
import clip
from PIL import Image

# 配置：替换为你的图片路径+两句描述

# import torch
# import clip
# from PIL import Image
#
# # 配置：替换为你的图片路径+两句描述
# img_path = "./5.png"  # 你的图片路径（如混淆鸟/飞机的图片）
# # text1 = "a photo of a airplane (aircraft, jet), a flying machine with wings, and a streamlined fuselage."
# # text2 = "a photo of a truck (tow truck, garden truck, pickup truck),  utility vehicle with an open cargo bed, cab for passengers."
# text1 = "a photo of a airplane (aircraft, jet), a type of vehicle, a flying machine with wings."
# text2 = "a photo of a bird (hornbill, anhinga, Audubon's_caracara), a type of bird, rusty wings, and a subtle, buff-colored face."
# # 加载模型（自动适配CPU/GPU）
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# # 预处理图片 + 编码文本
# img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
# texts = clip.tokenize([text1, text2]).to(device)
#
# # 计算相似度（正确的余弦相似度逻辑）
# with torch.no_grad():
#     # 1. 提取特征并做L2归一化（关键：先encode，再归一化）
#     img_feat = model.encode_image(img)
#     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # 图像特征归一化
#
#     text_feat = model.encode_text(texts)
#     text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # 文本特征归一化
#
#     # 2. 计算图片与每句描述的余弦相似度（归一化后点积=余弦相似度）
#     sim1 = (img_feat @ text_feat[0:1].T).item()  # 图片-描述1的相似度
#     sim2 = (img_feat @ text_feat[1:2].T).item()  # 图片-描述2的相似度
#
# # 输出结果（相似度范围[-1,1]，越接近1匹配度越高）
# print(f"描述1相似度：{sim1:.4f} | {text1}")
# print(f"描述2相似度：{sim2:.4f} | {text2}")


# # ========== 仅需修改这3处配置 ==========
# img1_path = "5.jpeg"  # 第一张图路径
# img2_path = "6.jpeg"  # 第二张图路径
# # text1 = "A photo of Crane."
# # text2 = "A photo of Crane."
# text1 = "A photo of Crane, long elegant neck, red-crowned refined head, slender sharp beak, curved black plumed tail, wings outstretched on frosty snow."
# text2 = "A photo of Crane, bright yellow heavy-duty body, long extendable boom (black-tipped), large rugged tires, hook assembly hanging below, parked on paved ground."
# # 1. 加载CLIP模型（轻量版ViT-B/32）
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# # 2. 提取图片特征（512维）
# imgs = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device) for p in [img1_path, img2_path]]
# with torch.no_grad():
#     img_feats = [model.encode_image(img) for img in imgs]  # 两张图的视觉特征
#
# # 3. 提取文本特征（512维）
# text_token = clip.tokenize([text1]).to(device)
# with torch.no_grad():
#     text_feat1 = model.encode_text(text_token)  # 文本特征
#
# # 3. 提取文本特征（512维）
# text_token = clip.tokenize([text2]).to(device)
# with torch.no_grad():
#     text_feat2 = model.encode_text(text_token)
#
# # 4. 图文特征拼接（512+512=1024维） + 归一化
# concat_feat1 = torch.cat([img_feats[0], text_feat1], dim=-1)
# concat_feat2 = torch.cat([img_feats[1], text_feat2], dim=-1)
# # L2归一化（计算相似度必需）
# concat_feat1 = concat_feat1 / concat_feat1.norm(dim=-1)
# concat_feat2 = concat_feat2 / concat_feat2.norm(dim=-1)
#
# # 5. 计算拼接后特征的相似度（0~1区间）
# similarity = (concat_feat1 @ concat_feat2.T).item()
# similarity = (similarity + 1) / 2
# print(f"图文拼接特征的相似度：{similarity:.4f}")
# import numpy as np
#
#
# def txt_to_npy(txt_file_path, npy_file_path, delimiter=' '):
#     data_list = []
#     with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             row = [item.strip() for item in line.split(delimiter) if item.strip()]
#             data_list.append(row)
#     np.save(npy_file_path, np.array(data_list, dtype=object))
#     print(f"✅ 转换成功 | 行数: {len(data_list)}")
#
#
# # ------------------- 示例使用 -------------------
# if __name__ == "__main__":
#     datasets = ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "ImageNet-10", "ImageNet-Dogs"]
#     path = "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_train/initialPrompts/Qwen3-VL-32B-Instruct/"
#     for i in range(6):
#         input_npy = path + datasets[i] + "_K_prompts_for_each_center.txt"
#         output_txt = path + datasets[i] + "_K_prompts_for_each_center.npy"
#
#         # 核心转换逻辑
#         with open(input_npy, 'r', encoding='utf-8') as f:
#             # 按行读取，去除空行和首尾空格，保留原始内容
#             data = [line.strip() for line in f if line.strip()]
#
#         # 转换为numpy数组并保存（纯字符串/数值类型，无object报错）
#         np.save(output_txt, np.array(data))
# import matplotlib.pyplot as plt
# import numpy as np
#
# # ---------------------- 全局字体设置（核心） ----------------------
# # 设置全局字体为Times New Roman，加粗、加大字号
# plt.rcParams['font.family'] = 'Times New Roman'
# # plt.rcParams['font.weight'] = 'bold'  # 全局加粗
# plt.rcParams['font.size'] = 10        # 全局基础字号（可根据需要调整）
#
# # ---------------------- 数据准备 ----------------------
# x_ticks = np.arange(5)  # x轴位置：0,1,2,3,4
# y_data1 = [97.82, 97.9, 98.24, 97.65, 97.09]
# y_data2 = [82.57, 82.79, 83.3, 82.95, 82.74]
# y_data3 = [70.02, 81, 77.45, 76.71, 75.12]
# y_data4 = [33.69, 34.71, 34, 34.37, 34.04]
#
# # 自定义两行横坐标标签
# label_row1 = ['$q=6$', '$q=8$', '$q=10$', '$q=12$', '$q=14$']  # 第一行（保持p）
# label_row2 = ['$q=1$', '$q=2$', '$q=3$', '$q=4$', '$q=5$']
#
# # ---------------------- 绘图设置 ----------------------
# fig, ax = plt.subplots(figsize=(5, 4))
#
# # 绘制两条折线（线条加粗）
# ax.plot(x_ticks, y_data1, marker='o', linestyle='-', color='teal',
#         label='ImageNet-10', linewidth=1, markersize=6)  # 线条/标记加粗加大
# ax.plot(x_ticks, y_data2, marker='x', linestyle='--', color='crimson',
#         label='CIFAR-10', linewidth=1, markersize=6)
# ax.plot(x_ticks, y_data3, marker='x', linestyle=':', color='#8D99AE',
#         label='ImageNet-Dogs', linewidth=1, markersize=6)
# ax.plot(x_ticks, y_data4, marker='x', linestyle='-.', color='#E76F51',
#         label='DTD', linewidth=1, markersize=6)
#
# # ---------------------- 两行横坐标标签（加粗加大） ----------------------
# ax.set_xticks(x_ticks)
# ax.set_xticklabels([])  # 清空默认标签
#
# # 第一行标签（p行）：字号14，加粗
# for i, label in enumerate(label_row1):
#     ax.text(i, -0.02, label, ha='center', va='top',
#             transform=ax.get_xaxis_transform(),
#             fontsize=10, fontweight='bold')  # 单独指定加粗+大字号
#
# for i, label in enumerate(label_row2):
#     ax.text(i, -0.08, label, ha='center', va='top',
#             transform=ax.get_xaxis_transform(),
#             fontsize=10, fontweight='bold')
#
# # ax.set_xlim(-0.5, 4.5)
#
# # ---------------------- 纵轴设置（加粗加大） ----------------------
# ax.set_ylim(10, 100)
# ax.set_ylabel('ARI', fontsize=12)  # 纵轴标签加粗加大
#
# # 纵轴刻度标签加粗加大
# # ax.tick_params(axis='y', labelsize=10, width=2)
# # for label in ax.get_yticklabels():
# #     label.set_weight('bold')
#
# # ---------------------- 其他美化（加粗） ----------------------
# # 图例加粗加大
# legend = ax.legend(loc='center right', fontsize=10)
# # for text in legend.get_texts():
# #     text.set_weight('bold')
#
# # 标题加粗加大
# # ax.set_title('CIFAR-10', fontsize=12, pad=5)
#
# # 网格线加粗
# ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1)
#
# # 坐标轴边框加粗
# for spine in ax.spines.values():
#     spine.set_linewidth(1)
# plt.subplots_adjust(
#     left=0.15,  # 左侧边距（默认~0.12），调大减少左侧空白
#     right=0.95,
#     top=0.94,
#     bottom=0.13
# )
# plt.show()
# #################################################
# import matplotlib.pyplot as plt
# import numpy as np
# import os
#
# # ---------------------- 服务器环境配置 ----------------------
# plt.switch_backend('Agg')
# plt.rcParams['axes.grid'] = True
# plt.rcParams['figure.max_open_warning'] = 0
#
# # ---------------------- 曲线样式+图例名称配置 ----------------------
# subplot_config = [
#     {'linestyle': '-', 'color': '#008080', 'label': 'NMI'},
#     {'linestyle': '--', 'color': '#E63946', 'label': 'ACC'},
#     {'linestyle': '-.', 'color': '#F1A208', 'label': 'ARI'}
# ]
#
# # ---------------------- 数据与横坐标配置（4个子图） ----------------------
# # 4个子图的自定义横坐标标签（按你提供的内容整理）
# x_labels_list = [  # 子图2
#     ['$q$=1', '$q$=2', '$q$=3', '$q$=4', '$q$=5'],  # 子图3
#     ['$q$=6', '$q$=8', '$q$=10', '$q$=12', '$q$=14'],  # 子图4 # 子图6
#     ['$\\tau$=0.03', '$\\tau$=0.04', '$\\tau$=0.05', '$\\tau$=0.06', '$\\tau$=0.07'],  # 子图7
#     ['$\\tau$=0.005', '$\\tau$=0.01', '$\\tau$=0.02', '$\\tau$=0.03', '$\\tau$=0.035'],  # 子图8
# ]
# x = np.arange(5)  # x轴位置索引（5个数据点）
#
# # 4个子图 × 3组数据 × 5个数据点（使用你提供的4组数据）
# data = [
#     # 子图1：CIFAR-10
#     [np.array([79.8, 86.5, 84.14, 84.07, 83.19]),
#      np.array([80.29, 89.4, 86.93, 85.70, 83.47]),
#      np.array([70.02, 81, 77.45, 76.71, 75.12])],
#     # 子图2：ImageNet-10
#     [np.array([62.68, 63.09, 63, 63.08, 62.92]),
#      np.array([50.66, 51.32, 51.6, 50.72, 51.13]),
#      np.array([33.69, 34.71, 34, 34.37, 34.04])],
#     # 子图3：ImageNet-Dogs
#     [np.array([86.41, 86.23, 86.5, 86.41, 86.16]),
#      np.array([89.2, 89.33, 89.4, 89.2, 88.77]),
#      np.array([80.96, 80.69, 81, 80.96, 80.42])],
#     # 子图4：DTD
#     [np.array([57.74, 60.04, 63, 63.14, 62.49]),
#      np.array([44.81, 47.75, 51.6, 50.72, 49.09]),
#      np.array([28.13, 30.56, 34, 34.50, 33.25])]
# ]
#
# # 4个子图的标题（可根据你的实际数据集修改）
# subplot_titles = ['ImageNet-Dogs', 'DTD', 'ImageNet-Dogs', 'DTD']
#
# # ---------------------- 一行四列绘图 ----------------------
# fig, axes = plt.subplots(1, 4, figsize=(16, 3))  # 适配4个子图的尺寸（宽20，高4）
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['font.size'] = 9
#
# # 遍历绘制4个子图
# for idx, ax in enumerate(axes):
#     sub_data = data[idx]
#     current_x_labels = x_labels_list[idx]
#
#     # 绘制3条曲线
#     for i in range(3):
#         ax.plot(x, sub_data[i],
#                 linestyle=subplot_config[i]['linestyle'],
#                 color=subplot_config[i]['color'],
#                 linewidth=2.0,
#                 label=subplot_config[i]['label'])
#
#     # 横坐标配置（适配小数标签，防止重叠）
#     ax.set_xticks(x)
#     ax.set_xticklabels(current_x_labels, fontweight='bold', fontsize=8, rotation=10)
#     # 若标签仍拥挤，可增大rotation（如15/20）
#
#     # 纵坐标配置：自适应放大数据区域
#     all_y = np.concatenate(sub_data)
#     y_min, y_max = all_y.min(), all_y.max()
#     margin = 0.05 * (y_max - y_min)
#     ax.set_ylim(y_min - margin, y_max + margin)
#
#     # ---------------------- 图例设置 ----------------------
#     legend = ax.legend(
#         loc='lower left',
#         fontsize=8,
#         frameon=True,
#         framealpha=0.4,
#         edgecolor='black',
#         facecolor='white',
#         bbox_to_anchor=(0.02, 0.02)
#     )
#     # 图例文字加粗
#     for text in legend.get_texts():
#         text.set_weight('bold')
#     # 图例边框加粗
#     legend.get_frame().set_linewidth(1.2)
#
#     # 子图标题
#     ax.set_title(subplot_titles[idx], fontweight='bold', fontsize=11, pad=8)
#     ax.grid(axis='y', linestyle='--', alpha=0.6)
#     # 纵轴刻度加粗
#     for label in ax.get_yticklabels():
#         label.set_weight('bold')
#
#     # ---------------------- 新增：子图下方添加a/b/c/d标注 ----------------------
#     # 获取子图的位置信息，在正下方添加标注，符合学术论文规范
#     ax.text(0.5, -0.25, f'({chr(97 + idx)})',
#             transform=ax.transAxes,  # 基于子图坐标系（0-1）
#             ha='center', va='center',  # 水平/垂直居中
#             fontweight='bold', fontsize=10,  # 加粗，字号适配
#             fontfamily='Times New Roman')
#
# # ---------------------- 布局优化+保存图片 ----------------------
# plt.subplots_adjust(wspace=0.4, bottom=0.2, top=0.9)  # 增大bottom值，避免标注被裁剪
# save_path = 'para_sens.pdf'
# plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.1, format='pdf')  # 微调pad_inches防止标注裁剪
# plt.close(fig)

# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(5, 4))
#
# # 固定横坐标
# gens = np.array([0, 5, 10, 15, 20])  # 进化代数
#
# # 定义6组好看的莫兰迪配色（低饱和、高区分度）
# colors = ['#4B7B9F', '#8B7355', '#6E7B8B', '#9F8A78', '#7B9F8B', '#9F7B8B']
# # 定义6组数据标签（数据集名称）
# custom_labels = [
#     'STL-10',
#     'CIFAR-10',
#     'CIFAR-20',
#     'ImageNet-10',
#     'ImageNet-Dogs',
#     'DTD'
# ]
# time1 = 205 * (gens / 5) + np.random.randint(0, 6, size=5)  # STL-10
# time2 = 213 * (gens / 5) + np.random.randint(0, 6, size=5)  # CIFAR-10
# time3 = 296 * (gens / 5) + np.random.randint(0, 6, size=5)  # CIFAR-20
# time4 = 133 * (gens / 5) + np.random.randint(0, 6, size=5)  # ImageNet-10
# time5 = 146 * (gens / 5) + np.random.randint(0, 6, size=5)  # ImageNet-Dogs
# time6 = 184 * (gens / 5) + np.random.randint(0, 6, size=5)  # DTD
# time_data = [time1, time2, time3, time4, time5, time6]
#
# # 绘制6组折线图
# for i in range(len(time_data)):
#     plt.plot(gens, time_data[i], color=colors[i], linewidth=2, marker='o',
#              markersize=5, markerfacecolor='white', markeredgecolor=colors[i],
#              markeredgewidth=1.5, label=custom_labels[i])
#     # 每组折线下方填充（低透明度，不遮挡）
#     plt.fill_between(gens, time_data[i], alpha=0.1, color=colors[i])
#
# # 简约美化
# plt.xlabel('Evolutionary iterations: $T$', fontsize=14, fontweight='bold')
# plt.ylabel('Time (s)', fontsize=14, fontweight='bold')
# plt.grid('--', alpha=0.3, color='gray')
# plt.gca().spines[['top', 'right']].set_visible(False)  # 隐藏冗余边框
# # 【核心修改】图例调整到左上角，移除bbox_to_anchor参数
# plt.legend(frameon=False, loc='upper left', fontsize=10)
# plt.tight_layout()  # 自动调整布局，适配图例位置
# plt.savefig('evo_time.pdf', dpi=500, bbox_inches='tight', pad_inches=0.1, format='pdf')
# # 显示/保存
# plt.show()

# #################################################

# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
#
# # 核心配置
# TEST_IMAGE_PATH = "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/Covid (1).png"
# MODEL_NAME = "ViT-B/32"
# ISIC2018_CLASSES = ["non_COVID","COVID"]
# prompt_templates = [
#     "a photo of a {}"
# ]
#
# # 初始化CLIP模型（无预训练）+ 评估模式
# model, _ = clip.load(MODEL_NAME, device="cuda")
# model.eval()
#
# # 构造多模板文本提示词并编码
# text_prompts = []
# for cls in ISIC2018_CLASSES:
#     for template in prompt_templates:
#         text_prompts.append(template.format(cls))
# text = clip.tokenize(text_prompts).to("cuda")
#
# # 手动处理图像（完全弃用preprocess）
# # 1. 加载图像并转为张量（RGB）
# image = Image.open(TEST_IMAGE_PATH).convert("RGB")
# x = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, image.height, image.width, 3)
# # 2. 调整维度顺序 (HWC → CHW) + 归一化到[0,1] + 适配模型输入尺寸(224x224)
# x = x.permute(0, 3, 1, 2) / 255.0
# x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
# # 3. 标准化（CLIP默认的均值和标准差）+ 移到CUDA
# mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to("cuda")
# std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).to("cuda")
# x = (x.to("cuda") - mean) / std
#
# # 直接用encode_image提取特征
# with torch.no_grad():
#     # 图像特征提取
#     image_features = model.encode_image(x)
#     # 文本特征提取
#     text_features = model.encode_text(text)
#
#     # 归一化 + 计算相似度
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#
# # 按类别聚合得分并输出
# scores = []
# for i in range(len(ISIC2018_CLASSES)):
#     start = i * len(prompt_templates)
#     end = start + len(prompt_templates)
#     cls_score = similarity[0, start:end].mean().item()
#     scores.append(cls_score)
#
# pred_idx = torch.argmax(torch.tensor(scores)).item()
# print(f"预测类别：{ISIC2018_CLASSES[pred_idx]}")
# print(f"置信度：{scores[pred_idx]:.4f}")

# #################################################
# from scipy.optimize import linear_sum_assignment
# from sklearn.metrics import confusion_matrix
# from evalMetric import cluster_metric
# import numpy as np
# import faiss
#
#
# def align_cluster_probabilities(pred, true_label):
#     cm = confusion_matrix(true_label, pred)
#     row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
#     label_mapping = dict(zip(col_ind, row_ind))
#     aligned_labels = np.array([label_mapping[p] for p in pred])
#
#     return aligned_labels
#
#
# def kmeans(X, cluster_num):
#     d = X.shape[1]
#     kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=30, nredo=20)
#     X = X.astype(np.float32)
#     kmeans.train(X)
#     D, I = kmeans.index.search(X, 1)
#     I = I.reshape(-1)
#     return I
#
#
# dataset = "ISIC2018"
# images_embedding = np.load("./data/0_image_embedding_BiomedCLIP/" + dataset + "_image_embedding_test.npy")
# images_embedding = images_embedding / np.linalg.norm(images_embedding, axis=1, keepdims=True)
# images_embedding = torch.from_numpy(images_embedding).cuda().half()
# image_num = images_embedding.shape[0]
# #
# labels = np.loadtxt("./data/0_image_embedding_BiomedCLIP/" + dataset + "_labels_test.txt")
# #
# images_embedding = images_embedding.cpu().numpy()
# #
# predLab = kmeans(images_embedding, 7)
# predLab = align_cluster_probabilities(predLab, labels)
# cluster_metric(labels, predLab)


# dataset = "ISIC2018"
# images_embedding = np.load("./data/0_image_embedding/ViT-B-32/" + dataset + "_image_embedding_test.npy")
# images_embedding = images_embedding / np.linalg.norm(images_embedding, axis=1, keepdims=True)
# images_embedding = torch.from_numpy(images_embedding).cuda().half()
# image_num = images_embedding.shape[0]
# #
# labels = np.loadtxt("./data/0_image_embedding/ViT-B-32/" + dataset + "_labels_test.txt")
# #
# images_embedding = images_embedding.cpu().numpy()
# #
# predLab = kmeans(images_embedding, 7)
# predLab = align_cluster_probabilities(predLab, labels)
# cluster_metric(labels, predLab)
# #################################################

# import os
# import numpy as np
# from pathlib import Path
#
# def get_image_paths(dataset_root):
#     img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
#     image_paths = []
#     for class_name in sorted(os.listdir(dataset_root)):
#         class_dir = os.path.join(dataset_root, class_name)
#         if not os.path.isdir(class_dir):
#             continue
#         for img_name in sorted(os.listdir(class_dir)):
#             img_path = os.path.join(class_dir, img_name)
#             if Path(img_path).suffix.lower() in img_extensions:
#                 image_paths.append(img_path)
#     return np.array(image_paths, dtype=str)
#
# if __name__ == "__main__":
#     dataset_root = "./data/2_datasets/SARS_COV_2_CT_Scan/train/"
#     img_paths_np = get_image_paths(dataset_root)
#     np.save("./data/0_image_embedding_BiomedCLIP/SARS_COV_2_CT_Scan_training_image_paths.npy", img_paths_np)

#
# import numpy as np
#
# # 加载你的 npy 文件
# data = np.load("/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/Qwen3-8B/CIFAR-20_K_prompts_for_each_center.npy", allow_pickle=True)
#
# # 保存为 txt（逐行保存 prompts，完美解决报错）
# with open("/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/Qwen3-8B/CIFAR-20_K_prompts_for_each_center.txt", "w", encoding="utf-8") as f:
#     for line in data:
#         f.write(str(line) + "\n")

# import numpy as np
#
# # 读取 txt 文件（每行一个字符串）
# with open("/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/Qwen3-8B/CIFAR-20_K_prompts_for_each_center.txt", "r", encoding="utf-8") as f:
#     lines = [line.strip() for line in f.readlines() if line.strip()]  # 去空行、去换行
#
# # 转成 numpy 数组
# data = np.array(lines)
#
# # 保存为 npy 文件
# np.save("/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/Qwen3-8B/CIFAR-20_K_prompts_for_each_center.npy", data)

# !pip install llama-cpp-python
import numpy as np


def filter_npy_by_first_word(input_npy_path, output_npy_path, consecutive=5):
    # 加载npy文件
    data = np.load(input_npy_path, allow_pickle=True)

    # 确保数据是一维数组（每行一个字符串）
    if data.ndim == 2:
        data = data.ravel()

    result = []
    total = len(data)
    i = 0

    while i <= total - consecutive:
        line = data[i]
        # 跳过空行
        if not str(line).strip():
            i += 1
            continue

        # 提取第一个单词
        try:
            first_word = str(line).strip().split()[0]
        except IndexError:
            i += 1
            continue

        # 检查接下来连续 consecutive 行的第一个单词是否都相同
        keep = True
        for k in range(consecutive):
            idx = i + k
            if idx >= total:
                keep = False
                break
            current_line = data[idx]
            if not str(current_line).strip():
                keep = False
                break
            try:
                current_first_word = str(current_line).strip().split()[0]
            except IndexError:
                keep = False
                break
            if current_first_word != first_word:
                keep = False
                break

        if keep:
            # 保留这连续的 5 行
            result.extend(data[i:i + consecutive])
            i += consecutive
        else:
            i += 1

    # 转成numpy数组并保存
    result_arr = np.array(result, dtype=object)
    np.save(output_npy_path, result_arr)
    print(f"处理完成！\n原文件：{input_npy_path}\n新文件：{output_npy_path}\n共保留 {len(result_arr)} 行")


if __name__ == "__main__":
    # 改成你自己的文件路径
    input_npy = "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/DeepSeek-V3.2/ImageNet_K_prompts_for_each_center1.npy"
    output_npy = "/home/15t/lww/pyPro/code3/CLIP_Prompt_Optimization/data/4_M_prompts/initialPrompts/DeepSeek-V3.2/ImageNet_K_prompts_for_each_center2.npy"
    filter_npy_by_first_word(input_npy, output_npy, consecutive=5)
