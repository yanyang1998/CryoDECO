import os
import json
import math
import warnings
import numpy as np

from Analyse.utils import load_pkl
from sklearn.mixture import GaussianMixture


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_conf(traindir: str, epoch: int, apply_ind: bool = True):

    conf_path = os.path.join(traindir, f"conf.{epoch}.pkl")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"未找到 conf 文件: {conf_path}")

    z_full = load_pkl(conf_path)
    print(f"[load_conf] 加载 {conf_path}")
    print(f"[load_conf] conf 形状: {z_full.shape}, dtype: {z_full.dtype}")

    ind_path = os.path.join(traindir, f"ind_epoch.{epoch}.pkl")
    ind_last_epoch = None

    if os.path.exists(ind_path):
        ind_last_epoch = sorted(load_pkl(ind_path))
        print(f"[load_conf] 加载 {ind_path}")
        print(
            f"[load_conf] ind_epoch 长度: {len(ind_last_epoch)}, "
            f"min={min(ind_last_epoch)}, max={max(ind_last_epoch)}"
        )
        if apply_ind:
            z_mapped = z_full[ind_last_epoch]
            print(f"[load_conf] 应用 ind 后 z 形状: {z_mapped.shape}")
        else:
            z_mapped = z_full
            print("[load_conf] --no_ind 已被指定，跳过 ind 索引")
    else:
        print(f"[load_conf] {ind_path} 不存在，按全量粒子使用 conf")
        z_mapped = z_full

    return z_full, ind_last_epoch, z_mapped


# ---------------------------------------------------------------------------
# GMM helpers
# ---------------------------------------------------------------------------
def _build_gmm(n_components, random_state, cov_type="full", n_init=5):
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type=cov_type,
        n_init=n_init,
        max_iter=100,
        reg_covar=1e-4,
        random_state=random_state,
    )


def _fit_and_bic(data, n_components, random_state, cov_type="full", n_init=5):
    gmm = _build_gmm(n_components, random_state, cov_type=cov_type, n_init=n_init)
    gmm.fit(data)

    return float(gmm.bic(data))


def _resampled_bics(embeddings, k, rng, n_resamples, sample_ratio,
                    random_seed, cov_type="full", n_init=5,
                    min_resample_size=2, warn_tag=""):
    n = embeddings.shape[0]
    rs_size = max(min_resample_size, int(n * sample_ratio))
    out = []
    for i in range(n_resamples):
        try:
            rs_idx = rng.choice(n, size=rs_size, replace=False)
            out.append(_fit_and_bic(
                embeddings[rs_idx], k,
                random_state=random_seed + i * 100 + k,
                cov_type=cov_type, n_init=n_init))
        except ValueError as e:
            if warn_tag:
                print(f"    [warn] {warn_tag} rep {i} failed: {e}")

    return out


# ---------------------------------------------------------------------------
# Elbow / Convergence
# ---------------------------------------------------------------------------
def find_elbow_l_method(k_range, bic_scores):
    k_arr = np.array(k_range, dtype=float)
    y_arr = np.array(bic_scores, dtype=float)
    n = len(k_arr)

    best_c_idx = 1
    min_total_rmse = float('inf')

    for c_idx in range(1, n - 1):
        # 1. 左侧段线性拟合并计算 RMSE
        k_left = k_arr[:c_idx + 1]
        y_left = y_arr[:c_idx + 1]
        poly_left = np.polyfit(k_left, y_left, 1)
        y_left_pred = np.polyval(poly_left, k_left)
        rmse_left = np.sqrt(np.mean((y_left - y_left_pred) ** 2))

        # 2. 右侧段线性拟合并计算 RMSE
        k_right = k_arr[c_idx:]
        y_right = y_arr[c_idx:]
        poly_right = np.polyfit(k_right, y_right, 1)
        y_right_pred = np.polyval(poly_right, k_right)
        rmse_right = np.sqrt(np.mean((y_right - y_right_pred) ** 2))

        weight_left = c_idx / (n - 1)
        weight_right = (n - 1 - c_idx) / (n - 1)
        total_rmse = weight_left * rmse_left + weight_right * rmse_right

        if total_rmse < min_total_rmse:
            min_total_rmse = total_rmse
            best_c_idx = c_idx

    return int(k_arr[best_c_idx])


def compute_post_elbow_rate_ratio(bic_cache, elbow_k, bic_min_k, eps=1e-12):
    # if elbow_k <= 1 or bic_min_k <= elbow_k:
    #     return 0.0
    
    # if 1 not in bic_cache or elbow_k not in bic_cache or bic_min_k not in bic_cache:
    #     return 0.0
    pre_drop_per_k = (bic_cache[1] - bic_cache[elbow_k]) / (elbow_k - 1)
    post_drop_per_k = (bic_cache[elbow_k] - bic_cache[bic_min_k]) / (bic_min_k - elbow_k)

    if pre_drop_per_k <= eps:
        return float("inf")

    return max(0.0, post_drop_per_k) / (pre_drop_per_k + eps)


def detect_bic_rebound(k_values, bic_values, tail_fraction=0.2):
    n = len(k_values)
    start = int(n * (1 - tail_fraction))
    tail_bic = bic_values[start:]

    increases = 0
    total = 0

    for i in range(len(tail_bic) - 1):
        if tail_bic[i + 1] > tail_bic[i]:
            increases += 1
        total += 1

    if total == 0:
        return 0.0

    return increases / total


# ---------------------------------------------------------------------------
# Refine / Hierarchical split
# ---------------------------------------------------------------------------
def refine_window_grid(coarse_k, current_step, window_factor=1.0, fine_step=1):
    half_window = max(1, int(round(current_step * window_factor)))
    lower = max(2, coarse_k - half_window)
    upper = coarse_k + half_window

    return list(range(lower, upper + 1, fine_step))


def ensure_bic(bic_cache, k_values, embeddings, rng, n_resamples=2, n_init=5,
               sample_ratio=0.8, random_seed=42, cov_type="full", tag="K"):
    newly_computed = {}

    for k in sorted(set(k_values)):
        if k in bic_cache:
            continue

        k_bics = _resampled_bics(
            embeddings, k, rng, n_resamples, sample_ratio,
            random_seed, cov_type=cov_type, n_init=n_init,
            warn_tag=f"{tag} K={k}")

        if k_bics:
            bic_cache[k] = float(np.mean(k_bics))
            newly_computed[k] = bic_cache[k]
            print(f"[{tag}] K={k:3d} | 协方差: {cov_type:<4} | 均值 BIC = {bic_cache[k]:.1f}")
        else:
            print(f" [skip-{tag}] K={k} 全部重采样都失败,跳过")

    return newly_computed


def evaluate_hierarchical_split(embeddings, macro_k, bic_cache, n_init,
                                n_resamples, sample_ratio, random_seed,
                                support_threshold, min_relative_bic_gain,
                                min_global_gain_ratio, max_extra_classes=0):
    diagnostics = {
        "enabled": True,
        "macro_k": int(macro_k),
        "final_k": int(macro_k),
        "accepted": False,
        "selected_parent_cluster": None,
        "selected_parent_clusters": [],
        "n_splits": 0,
        "target_k": int(macro_k),
        "max_extra_classes": int(max_extra_classes),
        "global_gain_ratio": None,
        "global_gain_passed": False,
        "clusters": [],
    }

    print("\n[Hierarchical split check]")
    print(f"macro k = {macro_k}")
    print(
        f"max_extra_classes = {max_extra_classes} "
        f"({'<0: 无限制, 所有通过局部条件的簇都被分裂' if max_extra_classes <= 0 else '=正整数: 最多分裂这么多簇'})")

    # ---- 1) Macro fit on all embeddings ------------------------------
    try:
        macro_gmm = _build_gmm(macro_k, random_state=random_seed + int(macro_k), n_init=n_init)
        macro_gmm.fit(embeddings)
        macro_labels = macro_gmm.predict(embeddings)
    except Exception as e:
        print(f"    [warn] 宏观 GMM 拟合失败: {e}")
        return int(macro_k), diagnostics

    unique_labels = sorted(int(x) for x in np.unique(macro_labels).tolist())
    cluster_diags = []
    passing = []

    # ---- 2) Per-cluster stability test -------------------------------
    for cid in unique_labels:
        c_diag = {
            "cluster_id": int(cid),
            "size": 0,
            "support_fraction": None,
            "median_relative_gain": None,
            "local_passed": False}
        
        mask = (macro_labels == cid)
        idx = np.where(mask)[0]
        c_size = int(len(idx))
        c_diag["size"] = c_size

        cluster_points = embeddings[idx]
        rng_local = np.random.default_rng(random_seed + 1000 + cid * 7)

        deltas = []
        rel_gains = []
        for i in range(n_resamples):
            try:
                rs_size = max(2, int(c_size * sample_ratio))
                if rs_size >= c_size:
                    rs_idx = np.arange(c_size)
                else:
                    rs_idx = rng_local.choice(c_size, size=rs_size, replace=False)

                rs_data = cluster_points[rs_idx]
                bic_1 = _fit_and_bic(rs_data, 1,
                                     random_state=random_seed + i * 100 + cid * 10 + 1,
                                     n_init=n_init)
                bic_2 = _fit_and_bic(rs_data, 2,
                                     random_state=random_seed + i * 100 + cid * 10 + 2,
                                     n_init=n_init)
            except Exception:
                continue

            delta = bic_1 - bic_2
            deltas.append(delta)
            rel_gains.append(delta / (abs(bic_1) + 1e-12))

        if not deltas:
            print(f"  cluster {cid}: 所有 resample 失败, 跳过")
            cluster_diags.append(c_diag)
            continue

        support_fraction = float(np.mean([d > 0 for d in deltas]))
        median_relative_gain = float(np.median(rel_gains))

        local_passed = (
            support_fraction >= support_threshold
            and median_relative_gain >= min_relative_bic_gain)

        c_diag.update({
            "support_fraction": support_fraction,
            "median_relative_gain": median_relative_gain,
            "local_passed": bool(local_passed)})
        cluster_diags.append(c_diag)

        print(
            f"  cluster {cid}: size={c_size}, "
            f"support_fraction={support_fraction:.4f}, "
            f"median_relative_gain={median_relative_gain:.6f}, "
            f"local_passed={local_passed}"
        )

        if local_passed:
            passing.append({
                "cluster_id": int(cid),
                "size": c_size,
                "support_fraction": support_fraction,
                "median_relative_gain": median_relative_gain,
                "split_score": support_fraction * median_relative_gain})

    diagnostics["clusters"] = cluster_diags

    # ---- 3) Select up to max_extra_classes splits --------------------
    if passing:
        passing_sorted = sorted(passing, key=lambda x: -x["split_score"])
        if max_extra_classes > 0:
            selected = passing_sorted[:max_extra_classes]
        else:
            selected = passing_sorted
    else:
        selected = []

    n_splits = len(selected)
    target_k = int(macro_k) + n_splits

    diagnostics["n_splits"] = int(n_splits)
    diagnostics["target_k"] = int(target_k)
    diagnostics["selected_parent_clusters"] = [int(c["cluster_id"]) for c in selected]
    diagnostics["selected_parent_cluster"] = (int(selected[0]["cluster_id"]) if selected else None)

    cap = (max_extra_classes if max_extra_classes > 0 else len(passing))
    print(
        f"  -> {len(passing)} 个簇通过局部条件, "
        f"按 split_score 取 top {min(len(passing), cap)} = "
        f"{n_splits} 个作为分裂目标; "
        f"target_k = macro_k({macro_k}) + n_splits({n_splits}) = {target_k}"
    )

    # ---- 4) Global guard: BIC(k + n_splits) --------------------------
    if n_splits > 0 and target_k not in bic_cache:
        rng_global = np.random.default_rng(random_seed + 7000 + target_k)
        new_bics = _resampled_bics(
            embeddings, target_k, rng_global, n_resamples, sample_ratio,
            random_seed, n_init=n_init,
            warn_tag="")
        
        if new_bics:
            bic_cache[target_k] = float(np.mean(new_bics))
            print(
                f"    [info] BIC(k+{n_splits}={target_k}) = "
                f"{bic_cache[target_k]:.1f} (本次新计算并写入 cache)")

    bic_macro = bic_cache.get(int(macro_k))
    bic_target = bic_cache.get(target_k)

    if bic_macro is not None and bic_target is not None and 1 in bic_cache:
        baseline = abs(bic_cache[1] - min(bic_cache.values())) + 1e-12
        global_gain_ratio = (bic_macro - bic_target) / baseline
        global_passed = (
            bic_target < bic_macro
            and global_gain_ratio >= min_global_gain_ratio)
    else:
        global_gain_ratio = None
        global_passed = False

    diagnostics["global_gain_ratio"] = (float(global_gain_ratio) if global_gain_ratio is not None else None)
    diagnostics["global_gain_passed"] = bool(global_passed)

    print(
        f"  global: BIC(k={macro_k}) = {bic_macro}, "
        f"BIC(k+{n_splits}={target_k}) = {bic_target}, "
        f"global_gain_ratio = {global_gain_ratio}, "
        f"global_passed = {global_passed}"
    )

    # ---- 5) Build hierarchical split labels ------------------------
    labels_before = np.asarray(macro_labels).astype(int)
    labels_after = labels_before.copy()
    n_split_labels_built = 0

    if selected and global_passed:
        next_label_id = int(labels_before.max()) + 1
        for c in selected:
            parent_id = int(c["cluster_id"])
            mask = (labels_before == parent_id)
            sub_points = embeddings[mask]
            if sub_points.shape[0] < 2:
                print(
                    f"    [warn] parent cluster {parent_id} has < 2 points, "
                    f"skipping split label for it.")
                continue
            try:
                sub_gmm = _build_gmm(2, random_state=random_seed + 5000 + parent_id, n_init=n_init)
                sub_gmm.fit(sub_points)
                sub_labels = sub_gmm.predict(sub_points).astype(int)
            except Exception as e:
                print(f"    [warn] sub-GMM fit failed for parent cluster {parent_id}: {e}")
                continue

            # First sub-cluster keeps the original parent_id; second gets a fresh id.
            labels_after[mask] = np.where(sub_labels == 0, parent_id, next_label_id)
            next_label_id += 1
            n_split_labels_built += 1

    diagnostics["n_split_labels_built"] = int(n_split_labels_built)
    diagnostics["macro_labels"] = labels_before
    diagnostics["split_labels"] = labels_after

    # ---- 6) Decide ---------------------------------------------------
    if selected and global_passed:
        diagnostics["accepted"] = True
        diagnostics["final_k"] = int(target_k)
        print(
            f"  selected parents = "
            f"{[int(c['cluster_id']) for c in selected]}, "
            f"split accepted = True, final k = {target_k}, "
            f"hierarchical split labels built for {n_split_labels_built} "
            f"parent(s)"
        )
        return int(target_k), diagnostics
    else:
        print(
            f"  selected parents = "
            f"{[int(c['cluster_id']) for c in selected]}, "
            f"split accepted = False, final k = {macro_k}")
        
        return int(macro_k), diagnostics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def estimate_k(embeddings,
               initial_max_k=30,
               step=4,
               safety_threshold=1,
               absolute_max_k=math.inf,
               n_resamples=2,
               n_init=5,
               sample_ratio=0.8,
               random_seed=42,
               enable_refine=True,
               refine_window_factor=1.0,
               enable_split_check=False,
               split_n_resamples=10,
               split_sample_ratio=0.8,
               split_support_threshold=0.8,
               split_min_relative_bic_gain=0.08,
               split_min_global_gain_ratio=0.005,
               split_max_extra_classes=0):
    rng = np.random.default_rng(random_seed)
    n = embeddings.shape[0]
    embeddings_sample = embeddings

    print(f"[Adaptive-K] 数据集大小: {n}")
    print(f"[Adaptive-K] 主指标 = BIC")

    iteration = 1
    bic_cache = {}
    current_step = step
    current_max_k = initial_max_k
    remaining_drop_ratio = None
    coarse_scale_bic_minima = []

    # ------------------------------------------------------------------
    # Stage 1: coarse candidate grid
    # ------------------------------------------------------------------
    while True:
        print("\n" + "=" * 65)
        print(f" 迭代 #{iteration} | 范围上限 max_k = {current_max_k} | 当前步长 step = {current_step}")
        print("=" * 65)

        candidates = [1] + list(range(current_step, current_max_k, current_step))
        if current_max_k not in candidates:
            candidates.append(current_max_k)
        candidates = sorted(list(set(candidates)))

        active_k = [k for k in candidates if k <= current_max_k]
        missing_candidates = [k for k in candidates if k not in bic_cache]

        if missing_candidates:
            print(f" -> 检测到未计算的 K 节点: {missing_candidates}，开始计算...")
            ensure_bic(
                bic_cache, missing_candidates, embeddings_sample, rng,
                n_resamples=n_resamples, n_init=n_init,
                sample_ratio=sample_ratio, random_seed=random_seed,
                cov_type="full", tag="Coarse")
        else:
            print(" -> 历史区间候选点 100% 命中缓存，无新增计算！")

        active_k_safe = [k for k in active_k if k in bic_cache]
        active_bic = [bic_cache[k] for k in active_k_safe]

        # 1. L-method elbow
        best_k = find_elbow_l_method(active_k_safe, active_bic)
        print(f" -> 当前多尺度网格局部拐点: K = {best_k}")

        # 2. BIC min
        bic_min_k = min(active_k_safe, key=lambda k: bic_cache[k])
        bic_min = float(bic_cache[bic_min_k])
        print(f" -> 当前搜索尺度 BIC 最小点: K = {bic_min_k}, BIC = {bic_min:.1f}")

        # 3. 保存 diagnostic
        coarse_scale_bic_minima.append({
            "iteration": int(iteration),
            "max_k": int(current_max_k),
            "step": int(current_step),
            "bic_min_k": int(bic_min_k),
            "bic_elbow_k": int(best_k),
            "bic_min_value": bic_min,
            "bic_elbow_value": float(bic_cache[int(best_k)]),
        })

        # 4. remaining drop ratio
        remaining_drop_ratio = compute_post_elbow_rate_ratio(bic_cache, best_k, bic_min_k)
        print(f" -> 该点之后的剩余跌幅比例: {remaining_drop_ratio:.2%}")

        # 5. 收敛性判断
        is_converged = (remaining_drop_ratio <= 0.08)
        if is_converged and (best_k < safety_threshold * current_max_k):
            print(f"\n[Done] 安全终止！拐点锁定在 K = {best_k}")
            k_elbow = best_k
            break
        else:
            print(f"[Boundary Alert] 尾部仍有未收敛跌幅 ({remaining_drop_ratio:.2%})，判定为连续崖面。")

            next_max_k = min(current_max_k * 2, absolute_max_k)
            next_step = 4
            print(f"自适应双向翻倍: 范围 [{current_max_k} -> {next_max_k}] | 步长 [{current_step} -> {next_step}]")
            current_max_k = next_max_k
            current_step = next_step
            iteration += 1

            # 6. elbow-minimum consistency check
            rebound_ratio = detect_bic_rebound(active_k_safe, active_bic)
            print(f" -> tail rebound ratio={rebound_ratio:.2f}")
            if rebound_ratio > 0.6:
                print("检测到 BIC 触底反弹，修正 elbow")
                k_elbow = bic_min_k
                break

    results = {
        'k_estimated': k_elbow,
        'coarse_k_estimated': k_elbow,
        'all_evaluated_k': sorted(list(bic_cache.keys())),
        'all_evaluated_bic': [bic_cache[k] for k in sorted(list(bic_cache.keys()))],
        'bic_cache': bic_cache,
        'coarse_scale_bic_minima': coarse_scale_bic_minima,
        'refined': False,
        'refine_grid': sorted(list(bic_cache.keys())),
        'refine_local_window': [],
        'refine_newly_computed': {}
    }

    # ------------------------------------------------------------------
    # Stage 2: refine candidate grid
    # ------------------------------------------------------------------
    if enable_refine:
        print("\n" + "=" * 65)
        print(" Refine candidate grid ")
        print("=" * 65)
        print(f"粗网格拐点 K_coarse = {k_elbow} | 粗网格当前 step = {current_step} | 细化窗口因子 = {refine_window_factor}")

        local_window = refine_window_grid(
            coarse_k=k_elbow,
            current_step=current_step,
            window_factor=refine_window_factor,
            fine_step=1)
        print(f"[Refine Grid] 细化窗口 = [{local_window[0]}, {local_window[-1]}] | 候选点数 = {len(local_window)}")

        refine_newly = ensure_bic(
            bic_cache, local_window, embeddings, rng,
            n_resamples=n_resamples, n_init=n_init,
            sample_ratio=sample_ratio, random_seed=random_seed,
            cov_type="full", tag="Refine")

        local_window_available = [k for k in local_window if k in bic_cache]

        if len(local_window_available) >= 3:
            local_bics = [bic_cache[k] for k in local_window_available]
            final_k = find_elbow_l_method(local_window_available, local_bics)
            print(
                f" -> 精搜 window 拐点: K = {final_k} "
                f"(粗网格 K_coarse = {k_elbow}, window 长度 = {len(local_window_available)})")
            k_elbow = final_k
        else:
            print(
                f" -> 精搜 window 可用点数 < 3 ({len(local_window_available)}), "
                f"退回粗网格拐点 K = {k_elbow}")

        results.update({
            'refined': True,
            'k_estimated': k_elbow,
            'refine_grid': local_window_available,
            'refine_local_window': local_window,
            'refine_newly_computed': refine_newly,
        })
    else:
        print("\n[Adaptive-K] 已通过 --no_refine 禁用精搜阶段")

    # ------------------------------------------------------------------
    # Stage 3 hierarchical split check
    # ------------------------------------------------------------------
    k_macro = int(k_elbow)
    coarse_k_estimated = int(results.get("coarse_k_estimated", k_macro))
    split_eligible = (coarse_k_estimated != k_macro)

    if not enable_split_check:
        split_diagnostics = {"enabled": False}
    else:
        split_diagnostics = {
            "enabled": True,
            "split_eligible": bool(split_eligible),
            "coarse_k_estimated": coarse_k_estimated,
            "macro_k_estimated": k_macro,
            "accepted": False,
            "final_k": int(k_macro),
            "selected_parent_cluster": None,
            "global_gain_ratio": None,
            "global_gain_passed": False,
            "clusters": [],
            "skip_reason": None,
        }

        if split_eligible:
            k_elbow, split_diagnostics = evaluate_hierarchical_split(
                embeddings,
                macro_k=k_macro,
                bic_cache=bic_cache,
                n_init=n_init,
                n_resamples=split_n_resamples,
                sample_ratio=split_sample_ratio,
                random_seed=random_seed,
                support_threshold=split_support_threshold,
                min_relative_bic_gain=split_min_relative_bic_gain,
                min_global_gain_ratio=split_min_global_gain_ratio,
                max_extra_classes=split_max_extra_classes,
            )
        else:
            split_diagnostics["skip_reason"] = (f"coarse_k_estimated={coarse_k_estimated} == macro_k_estimated={k_macro}; 跳过层级分裂。")
            k_elbow = k_macro

    results['macro_k_estimated'] = int(k_macro)
    results['k_estimated'] = int(k_elbow)
    results['hierarchical_split'] = split_diagnostics

    results['all_evaluated_k'] = sorted(list(bic_cache.keys()))
    results['all_evaluated_bic'] = [bic_cache[k] for k in results['all_evaluated_k']]
    results['bic_cache'] = bic_cache

    print(f"\n[Adaptive-K] >>> 最终推断结果: 最优 K 估计值 = {k_elbow} <<<")

    return k_elbow, results


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------
class _NumpyJSONEncoder(json.JSONEncoder):
    """Handle NumPy types that the default JSON encoder can't serialize."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_auto_k_json(out_json_path, k_estimated, k_diag, user_k_num=None,
                      extra_meta=None):
    split = k_diag.get("hierarchical_split") or {}
    split_json = {}
    for k, v in split.items():
        if k in ("macro_labels", "split_labels"):
            continue
        split_json[k] = v

    payload = {
        "k_estimated": int(k_estimated),
        "coarse_k_estimated": int(k_diag.get("coarse_k_estimated", k_estimated)),
        "macro_k_estimated": int(k_diag.get("macro_k_estimated", k_estimated)),
        "refined": bool(k_diag.get("refined", False)),
        "k_range": [int(k) for k in k_diag.get("all_evaluated_k", [])],
        "bic_values": [float(v) for v in k_diag.get("all_evaluated_bic", [])],
        "bic_cache": {str(int(k)): float(v) for k, v in k_diag.get("bic_cache", {}).items()},
        "coarse_scale_bic_minima": k_diag.get("coarse_scale_bic_minima", []),
        "refine_grid": [int(k) for k in k_diag.get("refine_grid", [])],
        "refine_local_window": [int(k) for k in k_diag.get("refine_local_window", [])],
        "refine_newly_computed": {
            str(int(k)): float(v)
            for k, v in k_diag.get("refine_newly_computed", {}).items()
        },
        "hierarchical_split": split_json,
        "user_k_num": (int(user_k_num) if user_k_num is not None else None),
    }
    if extra_meta:
        payload["meta"] = dict(extra_meta)

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=_NumpyJSONEncoder, ensure_ascii=False)
    print(f"[k_estimator] auto-k summary saved → {out_json_path}")
