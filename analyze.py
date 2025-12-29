"""Visualizing latent space and generating volumes for trained models."""

import os
import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from cryodata.data_preprocess import mrc
from Analyse import analysis, utils
from Model import models, decoder

from Model.configuration import AnalysisConfigurations, TrainingConfigurations
from Pose.lattice import Lattice

from sklearn.cluster import MiniBatchKMeans, Birch, AgglomerativeClustering
# from sklearn.cluster import MiniBatchKMeans, Birch
from munkres import Munkres
from sklearn import metrics
from sklearn.decomposition import PCA
import pickle
from umap import UMAP
from scipy.optimize import linear_sum_assignment
from cryosparc.dataset import Dataset
from torch.utils.data import DataLoader, Subset
# from Pose.cs_star_translate.cs2star import cs2star
from cryodata.cs_star_translate.cs2star import cs2star
import time
import warnings
import heapq
from multiprocessing import Pool
import collections.abc

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'templates')


class Clustering_tool:
    def __init__(self, data_num, n_clusters, labels_true=None, k_init=32, clustering_dim=4, cs_path=None, n_sample=400,
                 clustering_type='hierarchical'):
        if n_clusters is None or n_clusters <= 0:
            n_clusters = 1
        self.labels = torch.randint(0, n_clusters, (data_num,))
        self.n_clusters = n_clusters
        self.centers = 'k-means++'
        self.centers_merge = None
        self.data_num = data_num
        self.labels_true = labels_true
        self.k_init = k_init
        self.clustering_dim = clustering_dim
        self.labels_change_ratio = 1.0
        self.raw_centers_norm = None
        self.n_sample = n_sample
        self.clustering_type = clustering_type
        self.features_data_downsample = None
        self.current_inds = range(data_num)
        if cs_path is not None:
            if os.path.exists(os.path.join(os.path.dirname(cs_path), 'new_particles.cs')):
                cs_path = os.path.join(os.path.dirname(cs_path), 'new_particles.cs')
            elif cs_path.endswith('.cs'):
                cs_path = cs_path
            else:
                cs_path = None
                self.cs_data = None
            if cs_path is not None:
                cs_data = Dataset.load(cs_path)
                self.cs_data = cs_data
        else:
            self.cs_data = None

        # if labels_true is not None:
        #     self.labels_true = labels_true
        # else:
        #     self.labels_true = np.zeros((data_num,))

    def clustering(self, features_data, downsample_dim=None, n_jobs=8, sample_fraction=0.35,
                   downsample_type='UMAP'
                   # downsample_type='PCA'
                   ):
        features_data = features_data[self.current_inds]
        # features_data_norm = features_data / (np.linalg.norm(features_data, axis=1)[:, None] + 1e-10)
        if downsample_dim is None:
            downsample_dim = self.clustering_dim
        time_start = time.time()
        if downsample_dim is not None and features_data.shape[-1] > downsample_dim:
            # print('UMAP start')
            if downsample_type.lower() == 'pca':
                print('PCA start')
                pca = PCA(n_components=downsample_dim)
                features_data_downsample = pca.fit_transform(features_data)
            else:
                print('UMAP start')
                umap_tool = UMAP(n_components=downsample_dim, n_jobs=n_jobs)
                features_data_downsample = umap_tool.fit_transform(features_data)
            # features_data_umap = umap_tool.fit_transform(features_data_norm)
            time_end = time.time()
            print('(Clustering) Downsample time:', time_end - time_start)
        else:
            features_data_downsample = features_data
            # features_data_umap = features_data_norm
        if self.clustering_type.lower() == 'gmm':
            features_data_umap_norm = features_data_downsample
        else:
            features_data_umap_norm = features_data_downsample / (
                        np.linalg.norm(features_data_downsample, axis=1)[:, None] + 1e-10)

        self.features_data_downsample = features_data_downsample
        if self.clustering_type.lower() == 'birch':
            print('(Clustering) Birch start')
            clustering_method = Birch(n_clusters=self.n_clusters, threshold=0.3)
            clustering_method.fit(features_data_umap_norm)
            labels_temp = clustering_method.labels_
            labels, _ = labels_mapping(labels_temp=labels_temp, centers_temp=None, n_sample=self.n_sample,
                                       features_data=features_data_umap_norm, labels_old=self.labels, centers_old=None,
                                       current_inds=self.current_inds)
            centers = get_centers_from_labels_averages(labels, features_data_umap_norm)
            self.centers_merge = centers
            self.centers = centers
        elif self.clustering_type.lower() == 'hierarchical':
            print('(Clustering) AgglomerativeClustering start')
            clustering_method = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='single')
            # clustering_method = AgglomerativeClustering(n_clusters=self.n_clusters)
            clustering_method.fit(features_data_umap_norm)
            labels_temp = clustering_method.labels_
            # Z = fastcluster.linkage(features_data_norm_dr, method='average')
            # labels_temp = fcluster(Z, t=0.5, criterion='distance')
            labels, _ = labels_mapping(labels_temp=labels_temp, centers_temp=None, n_sample=self.n_sample,
                                       features_data=features_data_umap_norm, labels_old=self.labels, centers_old=None,
                                       current_inds=self.current_inds)
            centers = get_centers_from_labels_averages(labels, features_data_umap_norm)
            self.centers_merge = centers
            self.centers = centers
        elif self.clustering_type.lower() == 'gmm':
            n_samples = max(int(features_data_umap_norm.shape[0] * sample_fraction), 1000000)
            if n_samples < features_data_umap_norm.shape[0]:
                random_indices = np.random.permutation(features_data_umap_norm.shape[0])
                sample_indices = random_indices[:n_samples]
                training_data_sample = features_data_umap_norm[sample_indices, :]
            else:
                training_data_sample = features_data_umap_norm
            from sklearn.mixture import GaussianMixture
            print('(Clustering) GMM start')
            clustering_method = GaussianMixture(n_components=self.n_clusters,
                                                covariance_type='full',
                                                # covariance_type='spherical',
                                                # covariance_type='diag',
                                                # covariance_type='tied',
                                                n_init=20,
                                                max_iter=100, init_params='k-means++')
            clustering_method.fit(training_data_sample)
            labels_temp = clustering_method.predict(features_data_umap_norm)
            labels, _ = labels_mapping(labels_temp=labels_temp, centers_temp=None, n_sample=self.n_sample,
                                       features_data=features_data_umap_norm, labels_old=self.labels, centers_old=None,
                                       current_inds=self.current_inds)
            centers = get_centers_from_labels_averages(labels, features_data_umap_norm)
            self.centers_merge = centers
            self.centers = centers
        else:
            if self.k_init is not None and self.k_init > self.n_clusters and self.clustering_type.lower() == 'ak-means++':
                print('(Clustering) AK-means++ start')
                print('(Clustering) k-means++ with k_init:', self.k_init)
                n_clusters = self.k_init
            else:
                print('(Clustering) k-means++ start')
                n_clusters = self.n_clusters
            clustering_method = MiniBatchKMeans(n_clusters=n_clusters, init=self.centers, init_size=10000,
                                                n_init=10 if isinstance(self.centers, str) else 'auto', )
            clustering_method.fit(features_data_umap_norm, )
            labels = clustering_method.labels_
            centers = clustering_method.cluster_centers_
            if self.k_init is not None and self.k_init > self.n_clusters:
                if isinstance(self.centers, str):
                    centers_old = None
                    labels_old = None
                else:
                    centers_old = self.centers_merge
                    labels_old = self.labels
                labels_temp, centers_temp = merge_clusters(features_data=features_data_umap_norm, labels=labels,
                                                           centers=centers,
                                                           k_new=self.n_clusters, n_sample=self.n_sample, n_jobs=n_jobs
                                                           )
                labels, centers_merge = labels_mapping(labels_temp=labels_temp, centers_temp=centers_temp,
                                                       n_sample=self.n_sample, features_data=features_data_umap_norm,
                                                       labels_old=labels_old, centers_old=centers_old,
                                                       current_inds=self.current_inds)
                # centers_old=centers_old)
                self.centers_merge = centers_merge
            # clustering_method = MiniBatchKMeans(n_clusters=self.n_clusters,init=centers,n_init=10,init_size=10000)
            # clustering_method.fit(features_data_norm,)
            # labels=clustering_method.labels_
            # centers=clustering_method.cluster_centers_

        if self.labels is not None:
            current_labels = self.labels[self.current_inds]
            if isinstance(self.labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int64, device=self.labels.device)
                self.labels_change_ratio = torch.sum(current_labels != labels).item() / labels.numel()
            else:
                self.labels_change_ratio = np.sum(current_labels != labels) / len(labels)
        self.labels[self.current_inds] = labels

        # if centers is None:
        #     self.centers = clustering_method.cluster_centers_
        # else:
        #     self.centers = centers
        self.centers = centers
        if self.centers is not None and not isinstance(self.centers, str):
            self.centers_id = find_nearest_centers(
                self.centers_merge if self.centers_merge is not None else self.centers,
                features_data_umap_norm)
            self.raw_centers_norm = features_data_umap_norm[self.centers_id]
        time_end = time.time()
        print('(Clustering) clustering time:', time_end - time_start)
        return self.labels, self.centers

    def update_labels(self):
        self.labels = torch.randint(0, self.n_clusters, (self.data_num,))
        self.centers = 'k-means++'

    def update_current_inds(self, current_inds):
        self.current_inds = current_inds

    def get_clustering_acc(self):
        # 计算准确率和NMI
        acc, nmi = acc_nmi(self.labels, self.labels_true, self.current_inds)
        ari, ami = calculate_ari_ami(self.labels, self.labels_true, self.current_inds)
        return acc, nmi, ari, ami

    def get_class_num(self):
        # 计算每个类的数量
        num_class = []
        current_labels = self.labels[self.current_inds]
        for i in range(self.n_clusters):
            if i in current_labels:
                if isinstance(current_labels, torch.Tensor):
                    num_class.append(torch.sum(current_labels == i).item())
                else:
                    num_class.append(np.sum(current_labels == i))
            else:
                num_class.append(0)
            # num_class.append(np.sum(self.labels == i))
        return num_class

    def get_knn(self, features_data, device, sample_ratio=0.3):
        features_data = features_data[self.current_inds]
        features_norm = features_data / (np.linalg.norm(features_data, axis=1)[:, None] + 1e-10)
        labels_true_np = np.asarray(self.labels_true[self.current_inds])
        unique_labels = np.unique(labels_true_np)
        label_min = unique_labels.min()

        # 采样处理
        # if sample_ratio < 1.0:
        if sample_ratio is not None:
            sampled_indices = []
            for label in unique_labels:
                label_indices = np.where(labels_true_np == label)[0]
                if sample_ratio < 1.0:
                    sample_size = max(1, int(len(label_indices) * sample_ratio))
                else:
                    sample_size = min(sample_ratio, len(label_indices))

                sampled_indices.extend(np.random.choice(label_indices, sample_size, replace=False))
            features_norm = features_norm[sampled_indices]
            labels_true_np = labels_true_np[sampled_indices]

        # 转换为PyTorch张量
        train_features = torch.from_numpy(features_norm)
        train_labels = torch.from_numpy(labels_true_np - label_min).long()

        # 使用GPU加速
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        # 计算所有类别的KNN
        knn10_top1, knn10_top5, knn10_top10 = calculate_knn(
            train_features=train_features,
            test_features=train_features,
            train_labels=train_labels,
            test_labels=train_labels,
            num_classes=len(unique_labels),
            mask_start_index=0,
            k=10
            # mask_start_index=None
        )
        knn5_top1, knn5_top5, _ = calculate_knn(
            train_features=train_features,
            test_features=train_features,
            train_labels=train_labels,
            test_labels=train_labels,
            num_classes=len(unique_labels),
            mask_start_index=0,
            k=5
            # mask_start_index=None
        )

        return knn5_top1, knn5_top5, knn10_top1, knn10_top5, knn10_top10

    # def get_knn(self, features_data):
    #     features_norm = features_data / (np.linalg.norm(features_data, axis=1)[:, None] + 1e-10)
    #     id_set = set(self.labels_true)
    #     knn_top1 = []
    #     knn_top5 = []
    #     labels_true_np = np.array(self.labels_true)
    #     for id in id_set:
    #         id_i = np.where(labels_true_np == id)[0]
    #         # if resample_num < len(id_i):
    #         # resampled_id_i = np.random.choice(id_i, resample_num, replace=True)
    #         # else:
    #         # resampled_id_i = id_i
    #         # resampled_id.extend(resampled_id_i.tolist())
    #         label_min = min(labels_true_np)
    #         knn_acc_sample_top1, knn_acc_sample_top5 = calculate_knn(
    #             train_features=features_norm,
    #             test_features=features_norm[id_i],
    #             train_labels=labels_true_np - label_min,
    #             test_labels=labels_true_np[id_i] - label_min,
    #             num_classes=len(id_set),
    #             mask_start_index=int(id_i[0])
    #             # mask_start_index=None,
    #         )
    #
    #         knn_top1.append(knn_acc_sample_top1)
    #         knn_top5.append(knn_acc_sample_top5)
    #     # print(knn_top1)
    #     # print('knn_top1:', np.mean(knn_top1))
    #     # print(knn_top5)
    #     # print('knn_top3:', np.mean(knn_top5))
    #     return np.mean(knn_top1), np.mean(knn_top5)

    def generate_cs_from_labels(self, save_path, labels=None):
        if labels is None:
            labels = self.labels
        if self.cs_data is not None and len(self.cs_data) == len(labels):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(self.n_clusters):
                # print(f"cluster {i} num: {num_class[i]}")
                cs_sub_data = self.cs_data.take(labels == i)
                cs_save_path = save_path + '/cluster_' + str(i) + '.cs'
                cs_sub_data.save(cs_save_path)
                cs2star(cs_save_path, cs_save_path.replace('.cs', '.star'))
            print(f"save cs files to {save_path}")


def get_centers_from_labels_averages(labels, features_data):
    clusters_num = np.unique(labels)
    centers = np.zeros((len(clusters_num), features_data.shape[1]))
    for i in range(len(clusters_num)):
        cluster_i = np.where(labels == clusters_num[i])[0]
        if len(cluster_i) > 0:
            centers[i] = np.mean(features_data[cluster_i], axis=0)
        else:
            centers[i] = np.zeros((features_data.shape[1]))
    return centers


def best_map(L1, L2):
    # L1 should be the labels and L2 should be the my_clustering number we got
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.asarray(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def acc_nmi(labels_p, labels_t, current_inds=None):
    if isinstance(labels_p, torch.Tensor):
        labels_p = labels_p.cpu().numpy()
    if isinstance(labels_t, torch.Tensor):
        labels_t = labels_t.cpu().numpy()
    if current_inds is not None:
        labels_p = labels_p[current_inds]
        labels_t = labels_t[current_inds]
    label_same = best_map(labels_t, labels_p)
    count = np.sum(labels_t[:] == label_same[:])
    acc = count.astype(float) / (len(labels_t))
    nmi = metrics.normalized_mutual_info_score(labels_t, label_same)
    return acc, nmi


def calculate_knn(train_features=None, train_labels=None, test_features=None, test_labels=None,
                  num_classes=None, k=5, T=1, mask_start_index=None):
    # import torch
    # 检查输入
    if any(v is None for v in [train_features, train_labels, test_features, test_labels, num_classes]):
        raise ValueError("Essential parameters must be provided")

    # 转换为张量并移动到GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # train_features = torch.from_numpy(train_features).to(device)
    # train_labels = torch.from_numpy(np.asarray(train_labels)).long().to(device)
    # test_features = torch.from_numpy(test_features).to(device)
    # test_labels = torch.from_numpy(np.asarray(test_labels)).long().to(device)

    # 预计算转置矩阵
    train_features_t = train_features.T

    top1, top5, top10, total = 0.0, 0.0, 0.0, 0
    num_test = test_labels.shape[0]
    chunk_size = 4000  # 可以根据GPU内存调整

    with torch.no_grad():
        for idx in range(0, num_test, chunk_size):
            end_idx = min(idx + chunk_size, num_test)
            features = test_features[idx:end_idx]
            targets = test_labels[idx:end_idx]
            batch_size = targets.shape[0]

            # 计算相似度
            similarity = torch.mm(features, train_features_t)

            # 掩码处理
            if mask_start_index is not None:
                valid_start = mask_start_index + idx
                valid_end = mask_start_index + end_idx
                if valid_end > train_labels.shape[0]:
                    raise IndexError("Mask indices exceed training set size")

                mask = torch.ones_like(similarity, dtype=torch.bool)
                mask_indices = torch.arange(valid_start, valid_end, device=train_features.device).unsqueeze(1)
                mask.scatter_(1, mask_indices, False)
                similarity.masked_fill_(~mask, float('-inf'))

            # Top-k检索
            similarities, indices = similarity.topk(k, largest=True, sorted=True)
            retrieved_labels = train_labels[indices]

            # 加权投票
            weights = (similarities / T).exp().unsqueeze(-1)
            one_hot = torch.zeros(batch_size, k, num_classes, device=train_features.device)
            one_hot.scatter_(2, retrieved_labels.unsqueeze(-1), 1)
            probs = (one_hot * weights).sum(dim=1)

            # 计算准确率
            _, preds = probs.topk(min(k, num_classes), dim=1)
            correct = preds.eq(targets.view(-1, 1))
            top1 += correct[:, 0].sum().item()
            top5 += correct[:, :min(5, k, num_classes)].sum().item()
            top10 += correct[:, :min(10, k, num_classes)].sum().item()
            total += batch_size

    return (top1 / total) * 100, (top5 / total) * 100, (top10 / total) * 100


def calculate_ari_ami(cluster_labels, labels_t, current_inds=None):
    """
    计算给定特征和真实标签的ARI和AMI

    参数:
    features : ndarray, shape (N, dim)
        需要聚类的数据特征
    labels : ndarray, shape (N,)
        真实的类别标签

    返回:
    ari : float
        Adjusted Rand Index
    ami : float
        Adjusted Mutual Information
    """

    # from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    if current_inds is not None:
        cluster_labels = cluster_labels[current_inds]
        labels_t = labels_t[current_inds]

    # 确定聚类数量为真实标签中的类别数
    # n_clusters = len(np.unique(labels_t))

    # 使用KMeans进行聚类
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    # cluster_labels = kmeans.fit_predict(features)

    # 计算ARI和AMI
    ari = adjusted_rand_score(labels_t, cluster_labels)
    ami = adjusted_mutual_info_score(labels_t, cluster_labels)

    return ari, ami


# def calculate_knn(train_features=None, train_labels=None, test_features=None, test_labels=None,
#                   num_classes=None, k=5, T=1, mask_start_index=None):  # 修改默认值为None
#     import torch
#     # 添加空值检查
#     if any(v is None for v in [train_features, train_labels, test_features, test_labels, num_classes]):
#         raise ValueError("Essential parameters (train/test features/labels/num_classes) must be provided")
#
#     # 转换张量并确保设备一致
#     # train_features = torch.from_numpy(train_features).cuda()
#     # train_labels = torch.from_numpy(np.asarray(train_labels)).cuda().long()  # 确保标签为long类型
#     # test_features = torch.from_numpy(test_features).cuda()
#     # test_labels = torch.from_numpy(np.asarray(test_labels)).cuda().long()
#
#     train_features = torch.from_numpy(train_features)
#     train_labels = torch.from_numpy(np.asarray(train_labels)).long()  # 确保标签为long类型
#     test_features = torch.from_numpy(test_features)
#     test_labels = torch.from_numpy(np.asarray(test_labels)).long()
#
#     # 维度检查
#     assert train_features.shape[0] == train_labels.shape[0], "Train features/labels mismatch"
#     assert test_features.shape[0] == test_labels.shape[0], "Test features/labels mismatch"
#
#     top1, top5, total = 0.0, 0.0, 0
#     train_features = train_features.T  # 更直观的转置写法
#     num_test = test_labels.shape[0]
#     num_train = train_labels.shape[0]
#
#     # 分块处理
#     chunk_size = 4000
#     for idx in range(0, num_test, chunk_size):
#         # 当前分片
#         end_idx = min(idx + chunk_size, num_test)
#         features = test_features[idx:end_idx]
#         targets = test_labels[idx:end_idx]
#         batch_size = targets.shape[0]
#
#         # 相似度计算
#         similarity = torch.mm(features, train_features)  # [B, N]
#
#         # 掩码处理 (仅当需要排除特定样本时)
#         if mask_start_index is not None:
#             # 生成有效索引范围
#             valid_start = mask_start_index + idx
#             valid_end = mask_start_index + end_idx
#             # 索引越界检查
#             if valid_end > num_train:
#                 raise IndexError(f"Mask indices [{valid_start}:{valid_end}] exceed training set size {num_train}")
#
#             # 创建掩码
#             mask_indices = torch.arange(valid_start, valid_end, device=train_features.device).unsqueeze(1)
#             mask = torch.ones_like(similarity, dtype=torch.bool).scatter_(
#                 1, mask_indices, False
#             )
#             similarity = similarity.masked_fill(~mask, float('-inf'))  # 用负无穷确保不会被选为邻居
#
#         # 选取TopK
#         distances, indices = similarity.topk(k, largest=True, sorted=True)  # [B, k]
#
#         # 获取邻居标签
#         retrieved_labels = train_labels[indices]  # 直接索引代替gather
#
#         # 加权投票
#         weights = (distances / T).exp().unsqueeze(-1)  # [B, k, 1]
#         one_hot = torch.zeros(batch_size, k, num_classes, device=features.device)
#         one_hot.scatter_(2, retrieved_labels.unsqueeze(-1), 1)
#         probs = (one_hot * weights).sum(dim=1)
#
#         # 计算准确率
#         _, preds = probs.topk(2, dim=1)
#         correct = preds.eq(targets.view(-1, 1))
#         top1 += correct[:, 0].sum().item()
#         top5 += correct[:, :min(5, k)].sum().item()
#         total += batch_size
#
#     return (top1 / total) * 100, (top5 / total) * 100


class VolumeGenerator:
    """Helper class to call analysis.gen_volumes"""

    def __init__(self,
                 hypervolume, lattice, z_dim, invert, radius_mask, data_norm=(0, 1)):
        self.hypervolume = hypervolume
        self.lattice = lattice
        self.z_dim = z_dim
        self.invert = invert
        self.radius_mask = radius_mask
        self.data_norm = data_norm

    def gen_volumes(self, outdir, z_values, suffix=None, route_labels=None, intermediate_features=None):
        """
        z_values: [nz, z_dim]
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        zfile = f"{outdir}/z_values.txt"
        np.savetxt(zfile, z_values)

        for i, z in enumerate(z_values):
            if suffix is None:
                out_mrc = "{}/{}{:03d}.mrc".format(outdir, "vol_", i)
            else:
                out_mrc = "{}/{}{:03d}.mrc".format(outdir, "vol_", suffix)

            vol = models.eval_volume_method(self.hypervolume, self.lattice,
                                            self.z_dim, self.data_norm, zval=z,
                                            radius=self.radius_mask,
                                            route_labels=route_labels[i] if route_labels is not None else None,
                                            intermediate_features=[feature[i].unsqueeze(0) for feature in
                                                                   intermediate_features] if intermediate_features is not None else None
                                            )

            if self.invert:
                vol *= -1

            mrc.write(out_mrc, vol.astype(np.float32))


class ModelAnalyzer:
    """An engine for analyzing the output of a reconstruction model.

    Attributes
    ----------
    configs (AnalysisConfigurations):   Values of all parameters that can be
                                        set by the user.
    train_configs (TrainingConfigurations): Parameters that were used when
                                            the model was trained.

    epoch (int): Which epoch will be analyzed.

    skip_umap (bool):   UMAP clustering is relatively computationally intense
                        so sometimes we choose not to do it
    n_per_pc (int):     How many samples of the latent reconstruction space
                        will be taken along each principal component axis.
    """

    @classmethod
    def get_last_cached_epoch(cls, traindir: str) -> int:
        chkpnt_files = [fl for fl in os.listdir(traindir) if fl[:8] == "weights."]

        epoch = -2 if not chkpnt_files else max(
            int(fl.split('.')[1]) for fl in os.listdir(traindir)
            if fl[:8] == "weights."
        )

        return epoch

    def __init__(self,
                 traindir: str, config_vals: dict, train_config_vals: dict, encoder=None, dataset=None) -> None:
        self.logger = logging.getLogger(__name__)

        self.configs = AnalysisConfigurations(**config_vals)
        self.train_configs = TrainingConfigurations(**train_config_vals['training'])
        self.traindir = traindir
        self.encoder = encoder
        self.dataset = dataset

        # find how input data was normalized for training
        self.out_cfgs = {k: v for k, v in train_config_vals.items() if k != 'training'}
        if 'data_norm_mean' not in self.out_cfgs:
            self.out_cfgs['data_norm_mean'] = 0.
        if 'data_norm_std' not in self.out_cfgs:
            self.out_cfgs['data_norm_std'] = 1.

        # use last completed epoch if no epoch given
        if self.configs.epoch == -1:
            self.epoch = self.get_last_cached_epoch(traindir)
        else:
            self.epoch = self.configs.epoch

        if self.epoch == -2:
            raise ValueError(
                f"Cannot perform any analyses for output directory `{self.traindir}` "
                f"which does not contain any saved training checkpoints!"
            )

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.logger.info(f"Use cuda {self.use_cuda}")

        # load model
        checkpoint_path = os.path.join(self.traindir,
                                       f"weights.{self.epoch}.pkl")
        self.logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        hypervolume_params = checkpoint['hypervolume_params']
        hypervolume = decoder.HyperVolume(**hypervolume_params)
        hypervolume.load_state_dict(checkpoint['hypervolume_state_dict'])
        hypervolume.eval()
        hypervolume.to(self.device)

        lattice = Lattice(checkpoint['hypervolume_params']['resolution'],
                          extent=0.5, device=self.device)

        self.z_dim = checkpoint['hypervolume_params']['z_dim']
        radius_mask = (checkpoint['output_mask_radius']
                       if 'output_mask_radius' in checkpoint else None)
        self.vg = VolumeGenerator(
            hypervolume, lattice, self.z_dim, self.configs.invert, radius_mask,
            data_norm=(self.out_cfgs['data_norm_mean'], self.out_cfgs['data_norm_std'])
        )

        # load the conformations
        if self.train_configs.z_dim > 0:
            z = utils.load_pkl(
                os.path.join(self.traindir, f"conf.{self.epoch}.pkl"))

            ind_last_epoch_path = os.path.join(self.traindir, f"ind_epoch.{self.epoch}.pkl")
            if os.path.exists(ind_last_epoch_path):
                self.ind_last_epoch = sorted(utils.load_pkl(ind_last_epoch_path))
            else:
                self.ind_last_epoch = range(z.shape[0])

            self.z = z[self.ind_last_epoch]

            # self.z,self.keep_mask=remove_rows_by_sum(z,0.00001)
            self.n_samples = self.z.shape[0]
        else:
            self.z = None
            self.n_samples = None

        # create an output directory for these analyses
        self.outdir = os.path.join(self.traindir, f"analysis_{self.epoch}")
        os.makedirs(self.outdir, exist_ok=True)

        self.current_centers = checkpoint['current_centers'] if 'current_centers' in checkpoint else None
        if self.configs.data_resample is not None and self.configs.data_resample > 0 and self.n_samples > self.configs.data_resample:
            self.data_resample_id = np.sort(
                np.random.choice(range(self.n_samples), self.configs.data_resample, replace=False))
        else:
            self.data_resample_id = None

        if self.train_configs.labels_evaluate is not None or (
                self.train_configs.cluster_num_evaluate is not None and self.train_configs.cluster_num_evaluate > 0):
            if self.train_configs.labels_evaluate is not None and os.path.exists(self.train_configs.labels_evaluate):
                # labels_evaluate = np.array( utils.load_pkl(self.train_configs.labels_evaluate))[self.keep_mask]
                labels_evaluate = np.array(utils.load_pkl(self.train_configs.labels_evaluate))[self.ind_last_epoch]
            else:
                labels_evaluate = None
            self.clustering_tool_evaluate = Clustering_tool(data_num=self.n_samples, n_clusters=max(
                labels_evaluate) + 1 if labels_evaluate is not None else self.train_configs.cluster_num_evaluate,
                                                            # k_init=self.train_configs.k_init,
                                                            k_init=config_vals['k_init'],
                                                            # clustering_dim=self.train_configs.clustering_dim,
                                                            clustering_dim=config_vals['umap_dim'],
                                                            labels_true=labels_evaluate,
                                                            cs_path=self.train_configs.particles,
                                                            # clustering_type=self.train_configs.clustering_type,
                                                            clustering_type=config_vals['clustering_type'],
                                                            )
            # self.clustering_tool_evaluate.labels_true = labels_evaluate
            cluster_num_show = max(
                labels_evaluate) + 1 if labels_evaluate is not None else self.train_configs.cluster_num_evaluate
            if cluster_num_show > 1:
                self.logger.info('Evaluate cluster num: {}'.format(
                    cluster_num_show))
        else:
            self.labels_evaluate = None
            self.clustering_tool_evaluate = None

    @staticmethod
    def linear_interpolation(z_0, z_1, n, exclude_last=False):
        delta = 0 if not exclude_last else 1. / n
        t = np.linspace(0, 1 - delta, n)[..., None]

        return z_0[None] * (1. - t) + z_1[None] * t

    def analyze(self, data=None):
        if self.z_dim == 0:
            self.logger.info(
                "No analyses available for homogeneous reconstruction!")
            return

        if self.z_dim == 1:
            self.analyze_z1()
        else:
            self.analyze_zN()


        if self.configs.direct_traversal_txt is not None:
            dir_traversal_vertices_ind = np.loadtxt(
                self.configs.direct_traversal_txt)
            travdir = os.path.join(self.outdir, "direct_traversal")
            z_values = np.zeros((0, self.z_dim))

            for i, ind in enumerate(dir_traversal_vertices_ind[:-1]):
                z_0 = self.z[int(int)]
                z_1 = self.z[int(dir_traversal_vertices_ind[i + 1])]
                z_values = np.concatenate([
                    z_values,
                    self.linear_interpolation(z_0, z_1, 10, exclude_last=True)
                ], 0)

            # self.vg.gen_volumes(travdir, z_values, route_labels=self.get_route(z_values))
            self.generate_vols(travdir, z_values, route_labels=self.get_route(z_values))

        if self.configs.z_values_txt is not None:
            z_values = np.loadtxt(self.configs.z_values_txt)
            zvaldir = os.path.join(self.outdir, "trajectory")
            # self.vg.gen_volumes(zvaldir, z_values, route_labels=self.get_route(z_values))
            self.generate_vols(zvaldir, z_values, route_labels=self.get_route(z_values))

        if self.clustering_tool_evaluate is not None and self.clustering_tool_evaluate.n_clusters > 1:
            labels_predicted, _ = self.clustering_tool_evaluate.clustering(self.z)
            class_num_evaluate = self.clustering_tool_evaluate.get_class_num()
            self.clustering_tool_evaluate.generate_cs_from_labels(
                os.path.join(self.outdir, 'Clustering_Epoch_' + str(self.epoch)))

            if self.clustering_tool_evaluate.labels_true is not None:
                acc, nmi, ari, ami = self.clustering_tool_evaluate.get_clustering_acc()
                knn5_top1, knn5_top5, knn10_top1, knn10_top5, knn10_top10 = self.clustering_tool_evaluate.get_knn(
                    self.z, device=self.device, sample_ratio=1000)
                self.logger.info(
                    f"Epoch: {self.epoch} knn5 top1: {knn5_top1} knn5 top5: {knn5_top5} knn10 top1: {knn10_top1} knn10 top5: {knn10_top5} knn10 top10: {knn10_top10}")
                self.logger.info(f"Epoch: {self.epoch} clustering acc: {acc} nmi: {nmi}")
                self.logger.info(f"Epoch: {self.epoch} clustering ari: {ari} ami: {ami}")

            self.logger.info(f"Epoch: {self.epoch} class_num_evaluate: {class_num_evaluate}")

        self.logger.info('Done')

    def analyze_z1(self) -> None:
        """Plotting and volume generation for 1D z"""
        assert self.z.shape[1] == 1
        z = self.z.reshape(-1)
        n = len(z)

        plt.figure(1)
        plt.scatter(np.arange(n), z, alpha=0.1, s=2)
        plt.xlabel("particle")
        plt.ylabel("z")
        plt.savefig(os.path.join(self.outdir, "z.png"))
        plt.close()

        plt.figure(2)
        sns.distplot(z)
        plt.xlabel("z")
        plt.savefig(os.path.join(self.outdir, "z_hist.png"))
        plt.close()

        ztraj = np.percentile(z, np.linspace(5, 95, 10))
        # self.vg.gen_volumes(self.outdir, ztraj)
        self.generate_vols(self.outdir, ztraj)

        kmeans_labels, centers, features_data_norm_dr = analysis.analysis_clustering(
            z[..., None], self.k_num, reorder=False)
        centers, centers_ind = analysis.get_nearest_point(z[:, None], centers)

        volpath = os.path.join(self.outdir, f"clustering{self.configs.k_num}")
        # self.vg.gen_volumes(volpath, centers)
        self.generate_vols(volpath, centers,
                           # selected_id_list=centers_ind
                           )

    def analyze_zN(self) -> None:
        z_resample = self.z[self.data_resample_id] if self.data_resample_id is not None else self.z
        # zdim = self.z.shape[1]
        zdim = z_resample.shape[1]

        # Principal component analysis
        self.logger.info('Performing principal component analysis...')
        pc, pca = analysis.run_pca(z_resample)
        self.logger.info('Generating volumes...')

        for i in range(self.configs.pc):
            start, end = np.percentile(pc[:, i], (5, 95))
            z_pc = analysis.get_pc_traj(pca, z_resample.shape[1],
                                        self.configs.n_per_pc,
                                        i + 1, start, end)

            volpath = os.path.join(self.outdir,
                                   f"pc{i + 1}_{self.configs.n_per_pc}")
            # if self.train_configs.feature_take_indices is not None:
            #     intermidiate_features = self.get_intermediate_features(z_selected=z_pc)
            # else:
            #     intermidiate_features = None
            # self.vg.gen_volumes(volpath, z_pc, route_labels=self.get_route(z_pc),intermidiate_features=intermidiate_features)
            self.generate_vols(volpath, z_pc, route_labels=self.get_route(z_pc))

        # kmeans clustering
        self.logger.info(f'{self.configs.clustering_type} clustering...')
        k = min(self.configs.k_num, self.n_samples)
        if self.n_samples < self.configs.k_num:
            self.logger.warning(
                f'Changing k_num to # of samples: {self.n_samples}')

        kmean_path = os.path.join(self.outdir, f"clustering{k}")
        os.makedirs(kmean_path, exist_ok=True)
        kmeans_labels, centers, features_data_umap = analysis.analysis_clustering(z_resample, k,
                                                                                  clsutering_type=self.configs.clustering_type,
                                                                                  cs_dir_path=self.configs.cs_dir_path if self.configs.cs_dir_path is not None else self.train_configs.particles,
                                                                                  save_path=kmean_path if self.data_resample_id is None else None,
                                                                                  k_init=self.configs.k_init,
                                                                                  umap_dim=self.configs.umap_dim)
        centers, centers_ind = analysis.get_nearest_point(z_resample, centers)
        if self.data_resample_id is not None:
            centers_ind = [self.data_resample_id[i] for i in centers_ind]

        utils.save_pkl(kmeans_labels, os.path.join(kmean_path, "labels.pkl"))
        np.savetxt(os.path.join(kmean_path, "centers.txt"), centers)
        np.savetxt(os.path.join(kmean_path, "centers_ind.txt"),
                   centers_ind, fmt="%d")

        self.logger.info('Generating volumes...')
        # self.vg.gen_volumes(kmean_path, centers, route_labels=self.get_route(centers))
        self.generate_vols(kmean_path, centers,
                           selected_id_list=centers_ind,
                           route_labels=self.get_route(centers))

        # UMAP -- slow step
        umap_emb = None
        zdim = features_data_umap.shape[-1]
        if not self.configs.skip_umap:
            if os.path.exists(os.path.join(self.outdir, "umap.pkl")):
                umap_emb = utils.load_pkl(
                    os.path.join(self.outdir, "umap.pkl"))
            else:
                if zdim > 2:
                    self.logger.info('Running UMAP...')

                    if self.n_samples and self.n_samples < 15:
                        n_neighbours = self.n_samples - 1
                    else:
                        n_neighbours = 15

                    # umap_emb = analysis.run_umap(z_resample if features_data_umap is None else features_data_umap, n_neighbors=n_neighbours)
                    umap_emb = analysis.run_umap(z_resample, n_neighbors=n_neighbours)
                    # umap_emb = analysis.run_umap(self.z , n_neighbors=n_neighbours)
                else:
                    umap_emb = features_data_umap
                utils.save_pkl(umap_emb, os.path.join(self.outdir, "umap.pkl"))
            centers_ds_average = [np.mean(umap_emb[kmeans_labels == i], 0) for i in range(k)]
            centers_ds_average, centers_ind_ds_average = analysis.get_nearest_point(umap_emb,
                                                                                    np.array(centers_ds_average))

        # Make some plots
        self.logger.info('Generating plots...')

        def plt_pc_labels(pc1=0, pc2=1):
            plt.xlabel(f"PC{pc1 + 1} "
                       f"({pca.explained_variance_ratio_[pc1]:.2f})")
            plt.ylabel(f"PC{pc2 + 1} "
                       f"({pca.explained_variance_ratio_[pc2]:.2f})")

        def plt_pc_labels_jointplot(g, pc1=0, pc2=1):
            g.ax_joint.set_xlabel(
                f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]:.2f})")
            g.ax_joint.set_ylabel(
                f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]:.2f})")

        def plt_umap_labels():
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")

        def plt_umap_labels_jointplot(g):
            g.ax_joint.set_xlabel("UMAP1")
            g.ax_joint.set_ylabel("UMAP2")

        # PCA -- Style 1 -- Scatter
        if not os.path.exists(os.path.join(self.outdir, "z_pca.png")):
            plt.figure(figsize=(4, 4))
            plt.scatter(pc[:, 0], pc[:, 1], alpha=0.1, s=1, rasterized=True)
            plt_pc_labels()
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "z_pca.png"))
            plt.savefig(os.path.join(self.outdir, "z_pca.svg"))
            plt.close()

        # PCA -- Style 2 -- Scatter, with marginals
        if not os.path.exists(os.path.join(self.outdir, "z_pca_marginals.png")):
            g = sns.jointplot(x=pc[:, 0], y=pc[:, 1],
                              alpha=0.1, s=1, rasterized=True, height=4)
            plt_pc_labels_jointplot(g)
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "z_pca_marginals.png"))
            plt.savefig(os.path.join(self.outdir, "z_pca_marginals.svg"))
            plt.close()

        # PCA -- Style 3 -- Hexbin
        if not os.path.exists(os.path.join(self.outdir, "z_pca_hexbin.png")):
            g = sns.jointplot(x=pc[:, 0], y=pc[:, 1], height=4, kind="hex")
            plt_pc_labels_jointplot(g)
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "z_pca_hexbin.png"))
            plt.savefig(os.path.join(self.outdir, "z_pca_hexbin.svg"))
            plt.close()

        if umap_emb is not None:
            # Style 1 -- Scatter
            if not os.path.exists(os.path.join(self.outdir, "umap.png")):
                plt.figure(figsize=(4, 4))
                plt.scatter(umap_emb[:, 0], umap_emb[:, 1],
                            alpha=0.1, s=1, rasterized=True)
                plt_umap_labels()
                plt.tight_layout()
                plt.savefig(os.path.join(self.outdir, "umap.png"))
                plt.close()

            # Style 2 -- Scatter with marginal distributions
            if not os.path.exists(os.path.join(self.outdir, "umap_marginals.png")):
                g = sns.jointplot(x=umap_emb[:, 0], y=umap_emb[:, 1],
                                  alpha=0.1, s=1, rasterized=True, height=4)

                plt_umap_labels_jointplot(g)
                plt.tight_layout()
                plt.savefig(os.path.join(self.outdir, "umap_marginals.png"))
                plt.close()

            # Style 3 -- Hexbin / heatmap
            if not os.path.exists(os.path.join(self.outdir, "umap_hexbin.png")):
                g = sns.jointplot(x=umap_emb[:, 0], y=umap_emb[:, 1],
                                  kind="hex", height=4)
                plt_umap_labels_jointplot(g)
                plt.tight_layout()
                plt.savefig(os.path.join(self.outdir, "umap_hexbin.png"))
                plt.close()

        # Plot kmeans sample points

        colors = analysis._get_chimerax_colors(k + 1)[1:]  # avoid 'gray' for cluster 0

        g = analysis.scatter_annotate(
            pc[:, 0],
            pc[:, 1],
            centers_ind=centers_ind,
            annotate=True,
            colors=colors,
            clustering_labels=kmeans_labels,
        )
        plt_pc_labels()
        plt.tight_layout()
        plt.savefig(os.path.join(kmean_path, "z_pca.png"))
        plt.savefig(os.path.join(kmean_path, "z_pca.svg"))
        plt.close()

        g = analysis.scatter_annotate_hex(
            pc[:, 0],
            pc[:, 1],
            centers_ind=centers_ind,
            annotate=True,
            colors=colors,
            # clustering_labels=kmeans_labels
        )
        plt_umap_labels_jointplot(g)
        plt.tight_layout()
        plt.savefig(os.path.join(kmean_path, "z_pca_hex.png"))
        plt.savefig(os.path.join(kmean_path, "z_pca_hex.svg"))
        plt.close()

        if umap_emb is not None:
            g_original = analysis.scatter_annotate(
                x=umap_emb[:, 0],
                y=umap_emb[:, 1],
                centers_ind=centers_ind_ds_average,
                annotate=True,
                colors=colors,
                clustering_labels=kmeans_labels,
                # 默认所有参数都为 True
            )
            plt_umap_labels_jointplot(g_original)  # 您的自定义样式函数
            plt.tight_layout()
            plt.savefig(os.path.join(kmean_path, "umap_clusters.png"))
            plt.savefig(os.path.join(kmean_path, "umap_clusters.svg"))
            plt.close()


            g = analysis.scatter_annotate_hex(
                umap_emb[:, 0],
                umap_emb[:, 1],
                centers_ind=centers_ind,
                annotate=True,
                colors=colors,
                # clustering_labels=kmeans_labels
            )
            plt_umap_labels_jointplot(g)
            plt.tight_layout()
            plt.savefig(os.path.join(kmean_path, "umap_hex.png"))
            plt.close()

        # Plot PC trajectories
        for i in range(self.configs.pc):
            start, end = np.percentile(pc[:, i], (5, 95))
            pc_path = os.path.join(self.outdir,
                                   f"pc{i + 1}_{self.configs.n_per_pc}")
            z_pc = analysis.get_pc_traj(
                pca, z_resample.shape[1], 10, i + 1, start, end)

            if umap_emb is not None:
                # UMAP, colored by PCX
                analysis.scatter_color(
                    umap_emb[:, 0],
                    umap_emb[:, 1],
                    pc[:, i],
                    label=f"PC{i + 1}",
                )
                plt_umap_labels()
                plt.tight_layout()
                plt.savefig(os.path.join(pc_path, "umap.png"))
                plt.close()

                # UMAP, with PC traversal
                z_pc_on_data, pc_ind = analysis.get_nearest_point(z_resample, z_pc)
                dists = ((z_pc_on_data - z_pc) ** 2).sum(axis=1) ** 0.5

                if np.any(dists > 2):
                    self.logger.warning(f"Warning: PC{i + 1} point locations "
                                        "in UMAP plot may be inaccurate")

                plt.figure(figsize=(4, 4))
                plt.scatter(umap_emb[:, 0], umap_emb[:, 1],
                            alpha=0.05, s=1, rasterized=True)
                plt.scatter(umap_emb[pc_ind, 0], umap_emb[pc_ind, 1],
                            c="cornflowerblue", edgecolor="black", )
                plt_umap_labels()
                plt.tight_layout()
                plt.savefig(os.path.join(pc_path, "umap_traversal.png"))
                plt.close()

                # UMAP, with PC traversal, connected
                plt.figure(figsize=(4, 4))
                plt.scatter(umap_emb[:, 0], umap_emb[:, 1],
                            alpha=0.05, s=1, rasterized=True)

                plt.plot(umap_emb[pc_ind, 0], umap_emb[pc_ind, 1], "--", c="k")
                plt.scatter(umap_emb[pc_ind, 0], umap_emb[pc_ind, 1],
                            c="cornflowerblue", edgecolor="black")

                plt_umap_labels()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(pc_path, "umap_traversal_connected.png"))
                plt.close()

            # 10 points, from 5th to 95th percentile of PC1 values
            t = np.linspace(start, end, 10, endpoint=True)
            plt.figure(figsize=(4, 4))

            if i > 0 and i == self.configs.pc - 1:
                plt.scatter(pc[:, i - 1], pc[:, i],
                            alpha=0.1, s=1, rasterized=True)
                plt.scatter(np.zeros(10), t,
                            c="cornflowerblue", edgecolor="white")
                plt_pc_labels(i - 1, i)

            else:
                plt.scatter(pc[:, i], pc[:, i + 1],
                            alpha=0.1, s=1, rasterized=True)
                plt.scatter(t, np.zeros(10),
                            c="cornflowerblue", edgecolor="white")
                plt_pc_labels(i, i + 1)

            plt.tight_layout()
            plt.savefig(os.path.join(pc_path, "pca_traversal.png"))
            plt.savefig(os.path.join(pc_path, "pca_traversal.svg"))
            plt.close()

            if i > 0 and i == self.configs.pc - 1:
                g = sns.jointplot(x=pc[:, i - 1], y=pc[:, i],
                                  alpha=0.1, s=1, rasterized=True, height=4)
                g.ax_joint.scatter(np.zeros(10), t,
                                   c="cornflowerblue", edgecolor="white")
                plt_pc_labels_jointplot(g, i - 1, i)

            else:
                g = sns.jointplot(x=pc[:, i], y=pc[:, i + 1],
                                  alpha=0.1, s=1, rasterized=True, height=4)
                g.ax_joint.scatter(t, np.zeros(10),
                                   c="cornflowerblue", edgecolor="white")
                plt_pc_labels_jointplot(g)

            plt.tight_layout()
            plt.savefig(os.path.join(pc_path, "pca_traversal_hex.png"))
            plt.savefig(os.path.join(pc_path, "pca_traversal_hex.svg"))
            plt.close()

    def generate_vols(self, outdir, z_values, suffix=None, selected_id_list=None, route_labels=None):
        if self.train_configs.feature_take_indices is not None:
            intermidiate_features = self.get_intermediate_features(z_selected=z_values, id_list=selected_id_list)
        else:
            intermidiate_features = None
        self.vg.gen_volumes(outdir, z_values, suffix=suffix, route_labels=route_labels,
                            intermediate_features=intermidiate_features)

    def get_route(self, z):
        if self.current_centers is not None:
            dist_sq = np.sum((z[:, np.newaxis, :] - self.current_centers[np.newaxis, :, :]) ** 2, axis=2)
            route = np.argmin(dist_sq, axis=1)
            return torch.tensor(route[:, np.newaxis])
        else:
            return None

    def get_intermediate_features(self, z_selected=None, id_list=None):
        if id_list is None:
            id_list = find_nearest_centers(z_selected, self.z)
            id_list = [self.ind_last_epoch[id] for id in id_list]
        subset_dataset = Subset(self.dataset, id_list)
        dataloader_subset = DataLoader(
            subset_dataset,
            # batch_size=self.train_configs.batch_size,
            batch_size=8,
            shuffle=False,
            num_workers=self.train_configs.num_workers,
        )
        self.encoder.eval()
        with torch.no_grad():
            intermediate_features = []
            for batch in dataloader_subset:
                # batch = batch.to(self.device)
                batch['y_real_resized'] = batch['y_real_resized'].to(self.device)
                conf_dict = self.encoder(batch)
                intermediate_features.append(conf_dict['intermediates'])
        intermediate_features = [torch.cat(tensors, dim=0) for tensors in zip(*intermediate_features)]
        return intermediate_features


def cryosolver_clustering_inference(features_path, save_path, cluster_num=None, true_labels_path=None, umap_dim=None,
                                    cs_path=None,
                                    clustering_type='ak-means', k_init=32):
    features = pickle.load(open(features_path, 'rb'))
    true_labels = pickle.load(open(true_labels_path, 'rb')) if true_labels_path else None
    cluster_num = max(true_labels) + 1 if cluster_num is None else cluster_num
    clustering_tool = Clustering_tool(
        len(features),
        cluster_num,
        cs_path=cs_path,
        labels_true=true_labels,
        clustering_type=clustering_type,
        k_init=k_init)
    labels, centers = clustering_tool.clustering(features, downsample_dim=umap_dim)
    if true_labels is not None:
        acc, nmi, ari, ami = clustering_tool.get_clustering_acc()
        print('acc:', acc)
        print('nmi:', nmi)
        print('ari:', ari)
        print('ami:', ami)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    pickle.dump(labels, open(save_path, 'wb'))
    class_num = clustering_tool.get_class_num()
    print('class_num:', class_num)
    clustering_tool.generate_cs_from_labels(save_path=os.path.dirname(save_path))


def compute_min_distance(args):
    i, j, samples_dict, clusters = args
    c1, c2 = clusters[i], clusters[j]
    if c1 not in samples_dict or c2 not in samples_dict or c1 is None or c2 is None:
        return (i, j, np.inf)
    samples1 = samples_dict[c1]
    samples2 = samples_dict[c2]
    diff = samples1[:, np.newaxis, :] - samples2[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    min_dist = np.min(dist_matrix)
    return (i, j, min_dist)


# --- 距离计算函数 (无需修改) ---
def calculate_cluster_distance(samples1, samples2, distance_quantile):
    """
    根据指定的 quantile 或 quantile 范围计算两个样本集之间的距离。
    """
    if samples1 is None or samples2 is None or samples1.shape[0] == 0 or samples2.shape[0] == 0:
        return np.inf
    diff = samples1[:, np.newaxis, :] - samples2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    if isinstance(distance_quantile, collections.abc.Sequence) and len(distance_quantile) == 2:
        q_min, q_max = distance_quantile
        flat_dists = dist_matrix.flatten()
        if flat_dists.shape[0] == 0:
            return np.inf
        sorted_dists = np.sort(flat_dists)
        n_dists = len(sorted_dists)
        start_index = int(np.floor(n_dists * q_min))
        end_index = int(np.ceil(n_dists * q_max))
        if start_index >= end_index:
            mid_quantile = (q_min + q_max) / 2.0
            return np.quantile(dist_matrix, mid_quantile)
        selected_dists = sorted_dists[start_index:end_index]
        return np.mean(selected_dists)
    elif isinstance(distance_quantile, (float, int)):
        if distance_quantile == 0.0:
            return np.min(dist_matrix)
        else:
            return np.quantile(dist_matrix, distance_quantile)
    else:
        raise TypeError("distance_quantile 必须是 float 或包含两个 float 的 list/tuple。")


# --- 并行池包装函数 (无需修改) ---
def compute_dist_for_pool(args):
    """包装函数，用于从参数元组中解包并调用距离计算函数。"""
    i, j, s_dict, clus_list, distance_quantile = args
    c1, c2 = clus_list[i], clus_list[j]
    dist = calculate_cluster_distance(s_dict.get(c1), s_dict.get(c2), distance_quantile)
    return i, j, dist


# --- 主要修改的函数 ---
def merge_clusters(
        features_data,
        labels,
        centers,
        k_new,
        n_sample,
        # --- 新增：类规模阈值参数 ---
        min_cluster_size_threshold=300,
        max_merge_per_iter=2,
        n_jobs=8,
        distance_quantile=0.05,
):
    """
    通过迭代合并聚类来减少聚类数量。

    新增功能:
    min_cluster_size_threshold (int): 定义小规模类的阈值。
        - 任何样本数小于此阈值的类，在合并时将被赋予高优先级。
        - 默认值为 1，不产生特殊影响。若设置为例如 50，则样本数小于 50 的类会被优先合并。
    """
    # --- 参数校验 (无需修改) ---
    if isinstance(distance_quantile, collections.abc.Sequence) and len(distance_quantile) == 2:
        q_min, q_max = distance_quantile
        if not (isinstance(q_min, (float, int)) and isinstance(q_max, (float, int)) and
                0.0 <= q_min <= q_max <= 1.0):
            raise ValueError("distance_quantile 范围必须是 [min, max] 格式，且值在 0.0 到 1.0 之间。")
    elif isinstance(distance_quantile, (float, int)):
        if not 0.0 <= distance_quantile <= 1.0:
            raise ValueError("distance_quantile 作为单个值时必须在 0.0 到 1.0 之间。")
    else:
        raise TypeError("distance_quantile 必须是 float 或包含两个 float 的 list/tuple。")

    current_labels = labels.copy()
    current_centers_dict = {i: centers[i] for i in range(len(centers))}

    # --- 新增：计算并跟踪每个类的规模 ---
    unique_labels, counts = np.unique(current_labels, return_counts=True)
    cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}

    samples_dict = {}
    # --- 修改：确保只处理有实际样本的类 ---
    valid_clusters = list(cluster_sizes.keys())
    for c in valid_clusters:
        indices = np.where(current_labels == c)[0]
        # 由于我们已经从 cluster_sizes 开始，不再需要检查 len(indices) == 0
        sampled_indices = np.random.choice(indices, size=n_sample, replace=True)
        samples_dict[c] = features_data[sampled_indices]

    # 更新 centers_dict，移除没有样本的类
    current_centers_dict = {c: current_centers_dict[c] for c in valid_clusters}

    k_current = len(current_centers_dict)
    if k_current < k_new:
        raise ValueError(
            f"The initial number of clusters {k_current} is less than the target k_new={k_new}, cannot merge.")

    clusters = list(current_centers_dict.keys())
    n_clusters = len(clusters)
    cluster_to_idx = {cluster_label: i for i, cluster_label in enumerate(clusters)}

    # 并行计算初始距离 (无需修改)
    with Pool(n_jobs) as pool:
        args_list = [(i, j, samples_dict, clusters, distance_quantile) for i in range(n_clusters) for j in
                     range(i + 1, n_clusters)]
        results = pool.map(compute_dist_for_pool, args_list)

    # --- 修改：构建具有优先级的堆 ---
    heap = []
    for i, j, dist in results:
        if dist != np.inf:
            c1, c2 = clusters[i], clusters[j]
            size1, size2 = cluster_sizes.get(c1, 0), cluster_sizes.get(c2, 0)

            # 判断是否为高优先级合并
            is_priority_merge = (size1 < min_cluster_size_threshold or size2 < min_cluster_size_threshold)
            priority = 0 if is_priority_merge else 1

            # 堆元素格式为 ((priority, distance), index1, index2)
            heapq.heappush(heap, ((priority, dist), i, j))

    while k_current > k_new and heap:
        merged_in_iter = set()
        merge_operations = []
        remaining_merges = k_current - k_new
        max_merges = min(max_merge_per_iter, remaining_merges)

        temp_heap = []
        while len(merge_operations) < max_merges and heap:
            # --- 修改：从堆中弹出带优先级的元素 ---
            priority_dist_tuple, i, j = heapq.heappop(heap)

            c1, c2 = clusters[i], clusters[j]

            # 检查类是否已参与本轮合并或已被删除
            if (c1 in merged_in_iter or c2 in merged_in_iter or
                    c1 not in current_centers_dict or c2 not in current_centers_dict):
                continue

            # 暂存未处理的堆元素
            heapq.heappush(temp_heap, (priority_dist_tuple, i, j))

            if c1 < c2:
                target, source = c1, c2
                target_idx, source_idx = i, j
            else:
                target, source = c2, c1
                target_idx, source_idx = j, i

            merge_operations.append((target, source, target_idx, source_idx))
            merged_in_iter.add(target)
            merged_in_iter.add(source)

        # 将未使用的堆元素放回主堆
        heap.extend(temp_heap)
        heapq.heapify(heap)

        if not merge_operations:
            break

        for target, source, target_idx, source_idx in merge_operations:
            current_labels[current_labels == source] = target

            # --- 新增：更新合并后类的规模 ---
            if target in cluster_sizes and source in cluster_sizes:
                cluster_sizes[target] += cluster_sizes[source]
                del cluster_sizes[source]

            new_center = (current_centers_dict[target] + current_centers_dict[source]) / 2.0
            del current_centers_dict[source]
            current_centers_dict[target] = new_center

            # 将被合并的类标记为None，以便后续处理
            clusters[source_idx] = None

        # --- 修改：重建堆时，考虑所有受影响的类并重新计算优先级 ---
        new_heap = []
        # 从旧堆中移除所有与已合并类相关的条目
        # merged_indices 是本轮合并中所有被更改的类的索引集合 (target_idx 和 source_idx)
        merged_indices = {op[2] for op in merge_operations} | {op[3] for op in merge_operations}

        # 筛选出与本轮合并无关的旧条目
        heap = [item for item in heap if item[1] not in merged_indices and item[2] not in merged_indices]
        new_heap = heap  # 直接复用

        # 为所有被合并和更新过的类 (merged_in_iter) 重新计算与其它所有活动类的距离
        updated_clusters = {op[0] for op in merge_operations}  # 仅目标类 (target) 存活

        # 重新构建索引映射
        clusters_active = [c for c in clusters if c is not None]
        cluster_to_idx_active = {c: clusters.index(c) for c in clusters_active}

        # 重新计算距离
        for target_c in updated_clusters:
            target_idx = cluster_to_idx.get(target_c)
            if target_idx is None: continue

            for other_c in clusters_active:
                if target_c == other_c: continue

                other_idx = cluster_to_idx_active.get(other_c)
                if other_idx is None or target_idx >= other_idx: continue

                new_dist = calculate_cluster_distance(samples_dict.get(target_c), samples_dict.get(other_c),
                                                      distance_quantile)
                if new_dist != np.inf:
                    size1 = cluster_sizes.get(target_c, 0)
                    size2 = cluster_sizes.get(other_c, 0)
                    is_priority = (size1 < min_cluster_size_threshold or size2 < min_cluster_size_threshold)
                    priority = 0 if is_priority else 1
                    heapq.heappush(new_heap, ((priority, new_dist), target_idx, other_idx))

        heap = new_heap
        k_current -= len(merge_operations)

    if len(current_centers_dict) != k_new:
        raise RuntimeError(
            f"Merge failed, current cluster count is {len(current_centers_dict)}, target is {k_new}.")

    # --- 最终标签和中心重映射 (逻辑微调以适应None) ---
    final_clusters = sorted([c for c in clusters if c is not None])
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(final_clusters)}

    # 使用-1作为默认值，处理可能存在的未映射标签（理论上不应发生）
    final_labels = np.array([label_mapping.get(l, -1) for l in current_labels])
    final_centers = np.array([current_centers_dict[c] for c in final_clusters])

    return final_labels, final_centers


def labels_mapping(labels_temp, centers_temp, n_sample, features_data, centers_old=None, labels_old=None,
                   current_inds=None):
    if labels_old is not None:
        if torch.is_tensor(labels_old):
            labels_old = labels_old.cpu().numpy()
        # 确保labels_old的类数目与k2一致
        old_clusters = np.unique(labels_old)
        # assert len(old_clusters) == k_new, "length of old_clusters should be equal to k_new"

        # 采样旧标签的每个类的样本
        old_samples_dict = {}
        for c in old_clusters:
            indices = np.where(labels_old == c)[0]
            if current_inds is not None:
                indices = np.intersect1d(indices, current_inds)
            if len(indices) == 0:
                continue
            index_map = {val: i for i, val in enumerate(current_inds)}
            indices = np.array([index_map[val] for val in indices])
            sampled_indices = np.random.choice(indices, size=n_sample, replace=True)
            old_samples_dict[c] = features_data[sampled_indices]

        # 采样新标签的每个类的样本
        new_clusters = np.unique(labels_temp)
        new_samples_dict = {}
        for c in new_clusters:
            indices = np.where(labels_temp == c)[0]
            if len(indices) == 0:
                continue
            sampled_indices = np.random.choice(indices, size=n_sample, replace=True)
            new_samples_dict[c] = features_data[sampled_indices]

        # 构建新旧类距离矩阵（基于采样样本的平均距离）
        dist_matrix = np.zeros((len(new_clusters), len(old_clusters)))
        for i, c_new in enumerate(new_clusters):
            samples_new = new_samples_dict.get(c_new)
            if samples_new is None:
                dist_matrix[i, :] = np.inf
                continue
            for j, c_old in enumerate(old_clusters):
                samples_old = old_samples_dict.get(c_old)
                if samples_old is None:
                    dist_matrix[i, j] = np.inf
                    continue
                # 计算平均距离
                diff = samples_new[:, np.newaxis, :] - samples_old[np.newaxis, :, :]
                dist = np.sqrt((diff ** 2).sum(axis=2)).mean()
                dist_matrix[i, j] = dist

        # 使用匈牙利算法找到最小总距离的匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 重新映射标签
        label_mapping = {new_clusters[i]: old_clusters[col_ind[i]] for i in range(len(new_clusters))}
        labels_new = np.vectorize(label_mapping.get)(labels_temp)

        # # 将旧标签映射到连续整数 [0, k2)
        # unique_mapped = np.unique(labels_mapped)
        # final_mapping = {old: new for new, old in enumerate(unique_mapped)}
        # labels_new = np.vectorize(final_mapping.get)(labels_mapped)

        # 调整类中心顺序以匹配标签
        if centers_temp is not None:
            centers_new = np.zeros_like(centers_temp)
            for new_label in range(len(centers_new)):
                centers_new[new_label] = centers_temp[label_mapping[new_label]]
        else:
            centers_new = None


    elif centers_old is not None:
        # 根据新旧类中心匹配重新映射标签
        assert centers_old.shape == centers_temp.shape, "centers_old and centers_new have different shapes"

        # 计算新旧类中心距离矩阵
        dist_matrix = np.sqrt(((centers_old[:, np.newaxis, :] - centers_temp[np.newaxis, :, :]) ** 2).sum(axis=2))

        # 使用匈牙利算法找到最小总移动距离的匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 根据匹配结果重新排列centers_new并生成标签映射
        centers_new = centers_temp[col_ind]
        label_mapping = {original_label: new_label for new_label, original_label in enumerate(col_ind)}
        labels_new = np.vectorize(label_mapping.get)(labels_temp)
    else:
        labels_new = labels_temp
        centers_new = centers_temp
    return labels_new, centers_new


def find_nearest_centers(centers, candidates):
    """
    找到centers中每个元素在candidates中距离最近的元素的索引

    参数:
    centers -- numpy.ndarray, 形状为 (n1, dim)
    candidates -- numpy.ndarray, 形状为 (n2, dim)

    返回:
    indices -- numpy.ndarray, 形状为 (n1,), 包含每个center在candidates中的最近邻的索引
    """
    # 确保输入是二维数组
    assert centers.ndim == 2 and candidates.ndim == 2
    assert centers.shape[1] == candidates.shape[1], "维度不匹配"

    # 计算centers和candidates之间的平方距离矩阵
    # 使用广播机制计算所有点对之间的距离
    distances = np.sum((centers[:, np.newaxis, :] - candidates[np.newaxis, :, :]) ** 2, axis=2)

    # 找到每个center在candidates中的最近邻的索引
    nearest_indices = np.argmin(distances, axis=1)

    return nearest_indices


def remove_rows_by_sum(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    从一个2D NumPy数组中移除所有元素之和小于给定阈值的行。

    参数:
        arr (np.ndarray): 输入的 [n, dim] 大小的 NumPy 数组。
        threshold (float): 用于比较的阈值。如果一行的和小于此值，该行将被移除。

    返回:
        np.ndarray: 一个新的、经过筛选的数组，其中不包含和小于阈值的行。
    """
    # 1. 计算每一行的和 (axis=1 代表按行求和)
    row_sums = arr.sum(axis=1)

    # 2. 创建一个布尔掩码 (boolean mask)
    #    条件为：行的和大于或等于阈值。符合条件的行为 True，否则为 False。
    keep_mask = row_sums >= threshold

    # 3. 使用布尔掩码来索引原数组，即可筛选出所有符合条件的行
    return arr[keep_mask], keep_mask

