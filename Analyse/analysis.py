import os.path
import re
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, Axes
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Union, Tuple, List

# from .analyze import Clustering_tool

logger = logging.getLogger(__name__)


def parse_loss(f: str) -> np.ndarray:
    """Parse loss from run.log"""
    lines = open(f).readlines()
    lines = [x for x in lines if "====" in x]
    regex = "total\sloss\s=\s(\d.\d+)"  # type: ignore  # noqa: W605
    matches = [re.search(regex, x) for x in lines]
    loss = []
    for m in matches:
        assert m is not None
        loss.append(m.group(1))
    loss = np.asarray(loss).astype(np.float32)

    return loss


# Dimensionality reduction


def run_pca(z: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(z.shape[1])
    pca.fit(z)
    # logger.info("Explained variance ratio:")
    # logger.info(pca.explained_variance_ratio_)
    pc = pca.transform(z)
    return pc, pca


def get_pc_traj(
        pca: PCA,
        zdim: int,
        numpoints: int,
        dim: int,
        start: Optional[float],
        end: Optional[float],
        percentiles: Optional[np.ndarray] = None,
) -> npt.NDArray[np.float32]:
    """
    Create trajectory along specified principal component

    Inputs:
        pca: sklearn PCA object from run_pca
        zdim (int)
        numpoints (int): number of points between @start and @end
        dim (int): PC dimension for the trajectory (1-based index)
        start (float): Value of PC{dim} to start trajectory
        end (float): Value of PC{dim} to stop trajectory
        percentiles (np.array or None): Define percentile array instead of np.linspace(start,stop,numpoints)

    Returns:
        np.array (numpoints x zdim) of z values along PC
    """
    if percentiles is not None:
        assert len(percentiles) == numpoints
    traj_pca = np.zeros((numpoints, zdim))
    if percentiles is not None:
        traj_pca[:, dim - 1] = percentiles
    else:
        assert start is not None
        assert end is not None
        traj_pca[:, dim - 1] = np.linspace(start, end, numpoints)
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca


def run_tsne(
        z: np.ndarray, n_components: int = 2, perplexity: float = 1000
) -> np.ndarray:
    if len(z) > 10000:
        logger.warning(
            "WARNING: {} datapoints > {}. This may take awhile.".format(len(z), 10000)
        )
    z_embedded = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(z)
    return z_embedded


def run_umap(z: np.ndarray, **kwargs) -> np.ndarray:
    import umap  # CAN GET STUCK IN INFINITE IMPORT LOOP

    reducer = umap.UMAP(**kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded


# Clustering


def analysis_clustering(
        z: np.ndarray, K: int, on_data: bool = True, reorder: bool = True, clsutering_type: str = "k-means",
        cs_dir_path: Optional[str] = None, save_path: Optional[str] = None, k_init: int = 64,umap_dim=4
):
    """
    Cluster z by K means clustering
    Returns cluster labels, cluster centers
    If reorder=True, reorders clusters according to agglomerative clustering of cluster centers
    """
    from analyze import Clustering_tool
    clustering_tool = Clustering_tool(data_num=z.shape[0], n_clusters=K, clustering_type=clsutering_type,
                                      cs_path=cs_dir_path, k_init=k_init)
    labels, _ = clustering_tool.clustering(z, downsample_dim=umap_dim)
    # centers = z[clustering_tool.centers_id]
    centers=np.asarray([np.mean(z[labels==i], axis=0) for i in range(K)])
    import torch
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(centers):
        centers = centers.cpu().numpy()
    # kmeans = KMeans(n_clusters=K, n_init='auto', random_state=0, max_iter=10)
    # labels = kmeans.fit_predict(z)
    # centers = kmeans.cluster_centers_

    centers_ind = None
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)

    if reorder:
        g = sns.clustermap(centers)
        reordered = g.dendrogram_row.reordered_ind
        centers = centers[reordered]
        if centers_ind is not None:
            centers_ind = centers_ind[reordered]
        tmp = {k: i for i, k in enumerate(reordered)}
        labels = np.asarray([tmp[k] for k in labels])

    if save_path is not None:
        clustering_tool.generate_cs_from_labels(save_path=os.path.join(save_path, 'clustering_cs_star'), labels=labels)
    return labels, centers, clustering_tool.features_data_downsample



def get_nearest_point(
        data: np.ndarray, query: np.ndarray
) -> Tuple[npt.NDArray[np.float32], np.ndarray]:
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind


# HELPER FUNCTIONS FOR INDEX ARRAY MANIPULATION


def convert_original_indices(
        ind: np.ndarray, N_orig: int, orig_ind: np.ndarray
) -> np.ndarray:
    """
    Convert index array into indices into the original particle stack
    """  # todo -- finish docstring
    return np.arange(N_orig)[orig_ind][ind]


def combine_ind(
        N: int, sel1: np.ndarray, sel2: np.ndarray, kind: str = "intersection"
) -> Tuple[np.ndarray, np.ndarray]:
    # todo -- docstring
    if kind == "intersection":
        ind_selected = set(sel1) & set(sel2)
    elif kind == "union":
        ind_selected = set(sel1) | set(sel2)
    else:
        raise RuntimeError(
            f"Mode {kind} not recognized. Choose either 'intersection' or 'union'"
        )
    ind_selected_not = np.asarray(sorted(set(np.arange(N)) - ind_selected))
    ind_selected = np.asarray(sorted(ind_selected))
    return ind_selected, ind_selected_not


def get_ind_for_cluster(
        labels: np.ndarray, selected_clusters: np.ndarray
) -> np.ndarray:
    """Return index array of the selected clusters

    Inputs:
        labels: np.array of cluster labels for each particle
        selected_clusters: list of cluster labels to select

    Return:
        ind_selected: np.array of particle indices with the desired cluster labels

    Example usage:
        ind_keep = get_ind_for_cluster(kmeans_labels, [0,4,6,14])
    """
    ind_selected = np.array(
        [i for i, label in enumerate(labels) if label in selected_clusters]
    )
    return ind_selected


# PLOTTING


def _get_chimerax_colors(K: int) -> List:
    colors = [
        "#b2b2b2",
        "#ffffb2",
        "#b2ffff",
        "#b2b2ff",
        "#ffb2ff",
        "#ffb2b2",
        "#b2ffb2",
        "#e5bf99",
        "#99bfe5",
        "#cccc99",
    ]
    colors = [colors[i % len(colors)] for i in range(K)]
    return colors


def _get_colors(K: int, cmap: Optional[str] = None) -> List:
    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = [cm(i / float(K)) for i in range(K)]
    else:
        colors = ["C{}".format(i) for i in range(10)]
        colors = [colors[i % len(colors)] for i in range(K)]
    return colors


# def scatter_annotate(
#     x: np.ndarray,
#     y: np.ndarray,
#     centers: Optional[np.ndarray] = None,
#     centers_ind: Optional[np.ndarray] = None,
#     annotate: bool = True,
#     labels: Optional[np.ndarray] = None,
#     alpha: Union[float, np.ndarray, None] = 0.1,
#     s: Union[float, np.ndarray, None] = 1,
#     colors: Union[list, str, None] = None,
#     clustering_labels: Optional[np.ndarray] = None,
# ) -> Tuple[Figure, Axes]:
#     fig, ax = plt.subplots(figsize=(4, 4))
#     plt.scatter(x, y, alpha=alpha, s=s, rasterized=True)
#
#     # plot cluster centers
#     if centers_ind is not None:
#         assert centers is None
#         centers = np.array([[x[i], y[i]] for i in centers_ind])
#     if centers is not None:
#         if colors is None:
#             colors = "k"
#         plt.scatter(centers[:, 0], centers[:, 1], c=colors, edgecolor="black")
#     if annotate:
#         assert centers is not None
#         if labels is None:
#             labels = np.arange(len(centers))
#         assert labels is not None
#         for i in labels:
#             ax.annotate(str(i), centers[i, 0:2] + np.array([0.1, 0.1]))
#     return fig, ax


# def scatter_annotate(
#         x: np.ndarray,
#         y: np.ndarray,
#         centers: Optional[np.ndarray] = None,
#         centers_ind: Optional[np.ndarray] = None,
#         annotate: bool = True,
#         labels: Optional[np.ndarray] = None,
#         alpha: Union[float, np.ndarray, None] = 0.3,
#         s: Union[float, np.ndarray, None] = 3,
#         colors: Union[List[str], str, None] = None,
#         clustering_labels: Optional[np.ndarray] = None,
# ) -> Tuple[Figure, Axes]:
#     """
#     绘制散点图，并可选择性地标注聚类中心。
#
#     Args:
#         x (np.ndarray): x 坐标数组。
#         y (np.ndarray): y 坐标数组。
#         centers (Optional[np.ndarray], optional): 聚类中心的坐标。默认为 None。
#         centers_ind (Optional[np.ndarray], optional): 聚类中心在 x, y 中的索引。默认为 None。
#         annotate (bool, optional): 是否在中心点旁添加注释。默认为 True。
#         labels (Optional[np.ndarray], optional): 中心的标签。默认为 None。
#         alpha (Union[float, np.ndarray, None], optional): 透明度。默认为 0.1。
#         s (Union[float, np.ndarray, None], optional): 点的大小。默认为 1。
#         colors (Union[List[str], str, None], optional): 颜色。
#             - 当 clustering_labels 不为 None时, 应为一个颜色列表, 用于映射聚类标签。
#             - 当 clustering_labels 为 None 时, 可以是单个颜色字符串, 用于中心点。
#             默认为 None。
#         clustering_labels (Optional[np.ndarray], optional): 每个点的聚类标签 (整数)。
#             如果提供, 散点将根据此标签和 colors 列表进行着色。默认为 None。
#
#     Returns:
#         Tuple[Figure, Axes]: Matplotlib的Figure和Axes对象。
#     """
#     fig, ax = plt.subplots(figsize=(4, 4))
#
#     # --- 主要修改部分 ---
#     # 如果提供了聚类标签，则根据标签和颜色列表为每个点着色
#     if clustering_labels is not None:
#         assert colors is not None and isinstance(colors, list), \
#             "如果提供了 'clustering_labels'，'colors' 必须是一个颜色列表。"
#
#         # 根据 clustering_labels 中的每个标签值，从 colors 列表中取出对应索引的颜色
#         point_colors = [colors[i] for i in clustering_labels]
#         ax.scatter(x, y, c=point_colors, alpha=alpha, s=s, rasterized=True)
#     else:
#         # 否则，使用默认颜色绘制所有点
#         ax.scatter(x, y, alpha=alpha, s=s, rasterized=True)
#     # --- 修改结束 ---
#
#     # 绘制聚类中心
#     if centers_ind is not None:
#         assert centers is None, "不能同时提供 'centers' 和 'centers_ind'。"
#         centers = np.array([[x[i], y[i]] for i in centers_ind])
#
#     if centers is not None:
#         center_colors = colors  # 默认使用传入的colors
#         if center_colors is None:
#             center_colors = "k"  # 如果没有提供颜色，中心点默认为黑色
#
#         # 如果 'colors' 是一个列表 (在聚类模式下), 中心点也使用这些颜色
#         # 否则 (非聚类模式下), 'colors' 可能是一个字符串 (如 'r') 或 None
#         ax.scatter(centers[:, 0], centers[:, 1], c=center_colors, edgecolor="black", s=50, zorder=5)
#
#     # 添加注释
#     if annotate:
#         assert centers is not None, "必须提供 'centers' 或 'centers_ind' 才能添加注释。"
#         if labels is None:
#             labels = np.arange(len(centers))
#         assert labels is not None
#         for i, label in enumerate(labels):
#             ax.annotate(str(label), centers[i, 0:2] + np.array([0.1, 0.1]))
#
#     return fig, ax


def scatter_annotate(
        x: np.ndarray,
        y: np.ndarray,
        centers: Optional[np.ndarray] = None,
        centers_ind: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        colors: Union[list, str, None] = None,
        clustering_labels: Optional[np.ndarray] = None,
        # 新增参数以控制绘图元素
        plot_scatter: bool = True,
        plot_kde: bool = False,
        plot_centers: bool = True,
        annotate: bool = True,
) -> sns.JointGrid:
    """
    一个灵活的绘图函数，可以根据参数控制显示散点、密度图、聚类中心和标签。

    Args:
        x: X轴数据.
        y: Y轴数据.
        centers: 聚类中心的坐标.
        centers_ind: 聚类中心在原始数据中的索引.
        labels: 聚类中心的标签.
        colors: 聚类的颜色.
        clustering_labels: 每个数据点的聚类标签.
        plot_scatter: 如果为 True, 绘制散点.
        plot_kde: 如果为 True, 绘制密度等高线.
        plot_centers: 如果为 True, 绘制聚类中心.
        annotate: 如果为 True, 显示聚类中心的标签.

    Returns:
        A Seaborn JointGrid object.
    """
    # 创建基础图形
    g = sns.JointGrid(x=x, y=y, height=8)

    # 1. 绘制密度等高线
    if plot_kde:
        sns.kdeplot(x=x, y=y, ax=g.ax_joint, color='grey', alpha=0.3, levels=5)

    # 2. 绘制散点
    if plot_scatter:
        if clustering_labels is not None:
            unique_labels = np.unique(clustering_labels)
            if colors is None:
                colors = sns.color_palette("husl", len(unique_labels))
            elif isinstance(colors, str):
                colors = [colors] * len(unique_labels)

            for i, label in enumerate(unique_labels):
                mask = clustering_labels == label
                color = colors[i] if i < len(colors) else None
                g.ax_joint.scatter(x[mask], y[mask], c=color,
                                   # alpha=0.004,
                                   alpha=0.03,
                                   edgecolors='none',
                                   s=10, label=f'Cluster {label}')
        else:
            g.ax_joint.scatter(x, y,
                               alpha=0.03,
                               # alpha=0.004,
                               edgecolors='none', s=10)

    # 如果提供了中心点索引，则计算中心点坐标
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i], y[i]] for i in centers_ind])

    # 3. 绘制聚类中心
    if plot_centers and centers is not None:
        g.ax_joint.scatter(centers[:, 0], centers[:, 1], c=colors,
                           edgecolor="black", s=120, zorder=10)

    # 4. 添加中心点标签
    if annotate and centers is not None:
        if labels is None:
            labels = np.arange(len(centers))
        assert labels is not None
        for i in labels:
            g.ax_joint.annotate(
                str(i),
                centers[i, 0:2] + np.array([0.1, 0.1]),
                color="black",
                bbox=dict(boxstyle="square,pad=.1", ec="None", fc="1", alpha=0.8),
                zorder=11
            )
    return g


def scatter_annotate_hex(
        x: np.ndarray,
        y: np.ndarray,
        centers: Optional[np.ndarray] = None,
        centers_ind: Optional[np.ndarray] = None,
        annotate: bool = True,
        labels: Optional[np.ndarray] = None,
        colors: Union[list, str, None] = None,
) -> sns.JointGrid:
    g = sns.jointplot(x=x, y=y, kind="hex", height=8)

    # plot cluster centers
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i], y[i]] for i in centers_ind])
    if centers is not None:
        if colors is None:
            colors = "k"
        g.ax_joint.scatter(centers[:, 0], centers[:, 1],s=120, c=colors, edgecolor="black")
    if annotate:
        assert centers is not None
        if labels is None:
            labels = np.arange(len(centers))
        assert labels is not None
        for i in labels:
            g.ax_joint.annotate(
                str(i),
                centers[i, 0:2] + np.array([0.1, 0.1]),
                color="black",
                bbox=dict(boxstyle="square,pad=.1", ec="None", fc="1", alpha=0.5),
            )
    return g


def scatter_color(
        x: np.ndarray,
        y: np.ndarray,
        c: np.ndarray,
        cmap: str = "viridis",
        s=1,
        alpha: float = 0.1,
        label: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    assert len(x) == len(y) == len(c)
    sc = plt.scatter(x, y, s=s, alpha=alpha, rasterized=True, cmap=cmap, c=c)
    cbar = plt.colorbar(sc)
    cbar.set_alpha(1)
    # cbar.draw_all()
    if label:
        cbar.set_label(label)
    return fig, ax


def plot_by_cluster(
        x,
        y,
        K,
        labels,
        centers=None,
        centers_ind=None,
        annotate=False,
        s=2,
        alpha=0.1,
        colors=None,
        cmap=None,
        figsize=None,
        dpi=None
):
    if dpi is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if type(K) is int:
        K = list(range(K))

    if colors is None:
        colors = _get_colors(len(K), cmap)

    # scatter by cluster
    for i in K:
        ii = labels == i
        x_sub = x[ii]
        y_sub = y[ii]
        plt.scatter(
            x_sub,
            y_sub,
            s=s,
            alpha=alpha,
            label="cluster {}".format(i),
            color=colors[i],
            rasterized=True,
        )

    # plot cluster centers
    if centers_ind is not None:
        assert centers is None
        centers = np.array([[x[i], y[i]] for i in centers_ind])
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c="k")
    if annotate:
        assert centers is not None
        for i in K:
            ax.annotate(str(i), centers[i, 0:2])
    return fig, ax


def plot_by_cluster_subplot(
        x, y, K, labels, s=2, alpha=0.1, colors=None, cmap=None, figsize=None
):
    if type(K) is int:
        K = list(range(K))
    ncol = int(np.ceil(len(K) ** 0.5))
    nrow = int(np.ceil(len(K) / ncol))
    fig, ax = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(10, 10))
    if colors is None:
        colors = _get_colors(len(K), cmap)
    for i in K:
        ii = labels == i
        x_sub = x[ii]
        y_sub = y[ii]
        a = ax.ravel()[i]
        a.scatter(x_sub, y_sub, s=s, alpha=alpha, rasterized=True, color=colors[i])
        a.set_title(i)
    return fig, ax


def plot_euler(theta, phi, psi, plot_psi=True):
    sns.jointplot(
        x=theta, y=phi, kind="hex", xlim=(-180, 180), ylim=(0, 180)
    ).set_axis_labels("theta", "phi")
    if plot_psi:
        plt.figure()
        plt.hist(psi)
        plt.xlabel("psi")





def plot_projections(imgs, labels=None):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(min(len(imgs), 9)):
        axes[i].imshow(imgs[i], cmap="Greys_r")
        axes[i].axis("off")
        if labels is not None:
            axes[i].set_title(labels[i])
    return fig, axes




def load_dataframe(
        z=None, pc=None, euler=None, trans=None, labels=None, tsne=None, umap=None, **kwargs
):
    """Load results into a pandas dataframe for downstream analysis"""
    data = {}
    if umap is not None:
        data["UMAP1"] = umap[:, 0]
        data["UMAP2"] = umap[:, 1]
    if tsne is not None:
        data["TSNE1"] = tsne[:, 0]
        data["TSNE2"] = tsne[:, 1]
    if pc is not None:
        zD = pc.shape[1]
        for i in range(zD):
            data[f"PC{i + 1}"] = pc[:, i]
    if labels is not None:
        data["labels"] = labels
    if euler is not None:
        data["theta"] = euler[:, 0]
        data["phi"] = euler[:, 1]
        data["psi"] = euler[:, 2]
    if trans is not None:
        data["tx"] = trans[:, 0]
        data["ty"] = trans[:, 1]
    if z is not None:
        zD = z.shape[1]
        for i in range(zD):
            data[f"z{i}"] = z[:, i]
    for kk, vv in kwargs.items():
        data[kk] = vv
    df = pd.DataFrame(data=data)
    df["index"] = df.index
    return df
