import os
import logging
import multiprocessing as mp
import numpy as np
import torch
import pickle
from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import random
from PIL import Image

from cryodata.data_preprocess import fft, mrc
from cryodata.data_preprocess.lmdb_preprocess import create_lmdb_dataset,process_one_dataset_paths
from cryodata.data_preprocess.mrc_preprocess import sample_and_evaluate, window_mask, to_int8
from cryodata.cryoemDataset import CryoMetaData
from Data import starfile
from Analyse import utils

import lmdb

logger = logging.getLogger(__name__)


def load_particles(mrcs_txt_star, lazy=False, datadir=None, relion31=False):
    """
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files,
    or a cryosparc particles.cs file

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    """
    if mrcs_txt_star.endswith('.txt'):
        particles = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith('.star'):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                               lazy=lazy)
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(mrcs_txt_star)  # assume .mrcs files are in the same director as the starfile
                particles = starfile.Starfile.load(mrcs_txt_star, relion31=relion31).get_particles(datadir=datadir,
                                                                                                   lazy=lazy)
            else:
                raise RuntimeError(e)
    elif mrcs_txt_star.endswith('.cs'):
        particles = starfile.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    else:
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy)
    return particles


def downsample(imgs, resolution_out, max_threads=1):
    """
    imgs: [..., resolution, resolution]
    resolution_out: int

    output: [..., resolution_out, resolution_out]
    """
    resolution = imgs.shape[-1]
    if resolution <= resolution_out:
        return imgs
    else:
        start = int(resolution / 2 - resolution_out / 2)
        stop = int(resolution / 2 + resolution_out / 2)
        with mp.Pool(min(max_threads, mp.cpu_count())) as p:
            oldft = np.asarray(p.map(fft.ht2_center, imgs))
            newft = oldft[..., start:stop, start:stop]
            new = np.asarray(p.map(fft.iht2_center, newft))
        return new


class MRCData(data.Dataset):
    """
    Class representing an .mrcs stack file
    """

    def __init__(self, mrcfile_path, outdir, accelerator, norm=None, invert_data=False, ind=None, window=True,
                 datadir=None,
                 relion31=False, max_threads=16, window_r=0.85, flog=None, lazy=False, poses_gt_pkl=None,
                 resolution_input=None, no_trans=False, use_pfm_encoder=False, use_generated_features=False,
                 augmentation_settings=None,
                 # use_lmdb=False,
                 use_lmdb=True,
                 processed_data=None,
                 use_gt_pose=False,
                 score_bar=None,
                 resize=224,
                 raw_resize=128,
                 resample_num=40000
                 ):
        self.lazy = lazy
        self.use_pfm_features = use_pfm_encoder
        self.use_generated_features = use_generated_features
        self.processed_data = processed_data
        self.use_gt_pose = use_gt_pose
        self.pose_id_map = None
        self.poses_gt = None
        self.use_lmdb = use_lmdb
        self.augmentation_transform = None
        self.labels_class =None

        if use_lmdb:
            lmdb_dir = os.path.join(processed_data, 'lmdb_data')
            if not os.path.exists(lmdb_dir):

                if accelerator.is_main_process:
                    if not os.path.exists(lmdb_dir):
                        rawdata_dir = os.path.dirname(mrcfile_path)
                        if mrcfile_path.endswith('.cs'):
                            mrc_dir_list, mrcs_names_list_process, num_resample_mrcs_per_dataset = process_one_dataset_paths(
                                rawdata_dir,num_resample_per_dataset=resample_num)
                        else:
                            if mrcfile_path.endswith('.txt'):

                                with open(mrcfile_path, 'r') as f:
                                    mrcs_names_list = f.readlines()
                                mrcs_names_list_process = [x.strip() for x in mrcs_names_list]
                            elif mrcfile_path.endswith('.mrcs') or mrcfile_path.endswith('.mrc'):
                                mrcs_names_list_process = [os.path.basename(mrcfile_path)]
                            else:
                                mrcs_names_list_process=os.listdir(rawdata_dir)
                            mrc_dir_list = [rawdata_dir] * len(mrcs_names_list_process)
                            num_resample_mrcs_per_dataset = [int(resample_num / len(mrcs_names_list_process))] * len(mrcs_names_list_process)
                        image_path_list = [os.path.join(mrc_dir, mrcs_name) for mrc_dir, mrcs_name in
                                           zip(mrc_dir_list, mrcs_names_list_process)]

                        mean_len = sample_and_evaluate(image_path_list, processed_data, window=window,
                                                       window_r=window_r, needs_FT=True, )

                        map_size_p = int(len(image_path_list) * mean_len * resize * resize * 1.2 * 1)
                        map_size_r = int(
                            len(image_path_list) * mean_len * (raw_resize or resize) * (raw_resize or resize) * 1.2 * 4)
                        map_size_f = int(
                            len(image_path_list) * mean_len * (raw_resize or resize) * (raw_resize or resize) * 1.2 * 4)
                        map_size_config = {'processed': map_size_p, 'raw': map_size_r, 'FT': map_size_f}
                        accelerator.print(
                            f"Estimated LMDB map size (lmdb_processed database): {map_size_config['processed'] / (1024 ** 3):.2f} GB")
                        accelerator.print(
                            f"Estimated LMDB map size (lmdb_raw database): {map_size_config['raw'] / (1024 ** 3):.2f} GB")
                        accelerator.print(
                            f"Estimated LMDB map size (lmdb_FT database): {map_size_config['FT'] / (1024 ** 3):.2f} GB")
                        # 创建 LMDB 数据库
                        create_lmdb_dataset(image_path_list, processed_data, num_processes=8, chunksize=1,
                                            map_size=map_size_config, window=window, window_r=window_r,
                                            generate_ft_data=True,
                                            raw_resize=raw_resize,
                                            resize=resize,
                                            save_raw_data=True,num_resample_mrcs_per_dataset=num_resample_mrcs_per_dataset)
                self.processed_data = processed_data
                accelerator.wait_for_everyone()

            if os.path.exists(lmdb_dir):
                self.use_lmdb = True

                example, n_particles = self.prepare_lmdb_env(lmdb_dir)

            else:
                self.use_lmdb = False
                self.raw_img_path_list = pickle.load(open(processed_data + '/output_tif_path.data', 'rb'))
                self.processed_img_path_list = pickle.load(
                    open(processed_data + '/output_processed_tif_path.data', 'rb'))
                example = pickle.load(open(self.raw_img_path_list[0], 'rb'))
                n_particles = len(self.raw_img_path_list)

            meta_data = CryoMetaData(processed_data_path=processed_data,
                                     )
            id_index_dict, _, id_scores_dict = meta_data.preprocess_trainset_index_pretrain(
                id_map_for_filtering=meta_data.pose_id_map2, score_bar=score_bar)
            self.labels_class = meta_data.labels_class
            if meta_data.pose_id_map2 is not None:
                self.pose_id_map = meta_data.pose_id_map2
                self.pose_id_map_reverse = {v: k for k, v in self.pose_id_map.items()}
                n_particles = len(self.pose_id_map)
            else:
                self.pose_id_map = {key: key for key in range(n_particles)}
                self.pose_id_map_reverse = self.pose_id_map

            self.id_index_dict = id_index_dict
            self.id_scores_dict = id_scores_dict

            if use_gt_pose:
                self.pose_list = pickle.load(open(processed_data + '/pose_list.data', 'rb'))
                self.shift_list = pickle.load(open(processed_data + '/shift_list.data', 'rb'))
            # norm= pickle.load(open(processed_data + '/means_stds_FT.data', 'rb'))
            self.mean_std_id_dict = pickle.load(open(processed_data + '/mean_std_id_dict.data', 'rb'))
            self.protein_id_list = pickle.load(open(processed_data + '/protein_id_list.data', 'rb'))
            ny, nx = example.shape
            self.particles = None
            self.N = n_particles
            self.D = ny + 1  # ny + 1 after symmetrizing HT
            self.imgs = None
            self.norm = self.mean_std_id_dict[0]['FT']
            self.norm_real = self.mean_std_id_dict[0]['raw']



        else:
            if lazy or ind is not None:
                particles_real = load_particles(mrcfile_path, lazy=True, datadir=datadir, relion31=relion31)
                if not lazy:
                    particles_real = np.array([particles_real[i].get() for i in ind])
                self.particles_real = particles_real
                self.ind = ind

                particles_real_sample = np.array([particles_real[i].get() for i in range(1000)])
                n_particles, ny, nx = particles_real_sample.shape
                assert ny == nx, "Images must be square"
                assert ny % 2 == 0, "Image size must be even"
                self.resolution_input = resolution_input

                accelerator.print(f"Lazy loaded {len(particles_real)} {ny}x{nx} images")

                self.window = window
                self.window_r = window_r
                if window:
                    accelerator.print(f"Windowing images with radius {window_r}")
                    particles_real_sample *= window_mask(ny, window_r, .99)

                max_threads = min(max_threads, mp.cpu_count())
                accelerator.print(f"Spawning {max_threads} processes")
                with mp.Pool(max_threads) as p:
                    particles_sample = np.asarray(p.map(
                        fft.ht2_center, particles_real_sample), dtype=np.float32)

                self.invert_data = invert_data
                if invert_data:
                    particles_sample *= -1

                particles_sample = fft.symmetrize_ht(particles_sample)

                if norm is None:
                    norm = [0, np.std(particles_sample)]
                norm_real = [np.mean(particles_real_sample), np.std(particles_real_sample)]
                D = particles_sample.shape[1]
                N = len(particles_real) if ind is None else len(ind)


            else:
                particles_real = load_particles(mrcfile_path, lazy=False, datadir=datadir, relion31=relion31)
                n_particles, ny, nx = particles_real.shape
                # Real space window
                if window:
                    accelerator.print(f"Windowing images with radius {window_r}")

                    particles_real *= window_mask(ny, window_r, .99)

                # compute HT

                accelerator.print("Computing FFT")

                max_threads = min(max_threads, mp.cpu_count())

                if max_threads > 1:

                    accelerator.print(f"Spawning {max_threads} processes")

                    with mp.Pool(max_threads) as p:
                        particles = np.asarray(p.map(
                            fft.ht2_center, particles_real), dtype=np.float32)

                else:
                    particles = []
                    for i, img in enumerate(particles_real):
                        if i % 10000 == 0:
                            accelerator.print(f"{i} FFT computed")
                        particles.append(fft.ht2_center(img))

                    particles = np.asarray(particles, dtype=np.float32)
                    accelerator.print("Converted to FFT")

                if invert_data:
                    particles *= -1

                accelerator.print("Symmetrizing image data")
                particles = fft.symmetrize_ht(particles)
                norm = [0, np.std(particles.astype(np.float64))]
                particles = (particles - norm[0]) / norm[1]
                imgs = particles_real.astype(np.float32)
                norm_real = [np.mean(imgs), np.std(imgs.astype(np.float64))]

                if resolution_input is not None:
                    imgs = downsample(imgs, resolution_input, max_threads=max_threads)
                    accelerator.print("Images downsampled to "
                                      f"{resolution_input}x{resolution_input}")
                D = particles.shape[1]
                N = n_particles
                self.particles = particles
                self.imgs = imgs.astype(np.float32)
            assert ny == nx, "Images must be square"
            assert ny % 2 == 0, "Image size must be even"

            accelerator.print(f"Loaded {n_particles} {ny}x{nx} images")

            self.N = N
            self.D = D  # ny + 1 after symmetrizing HT
            self.norm = norm
            self.norm_real = norm_real

            if poses_gt_pkl is not None:
                poses_gt = utils.load_pkl(poses_gt_pkl)
                if ind is not None:
                    if poses_gt[0].ndim == 3:
                        self.poses_gt = (
                            torch.tensor(poses_gt[0][np.array(ind)]).float(),
                            torch.tensor(poses_gt[1][np.array(ind)]).float() * self.D
                        )
                    else:
                        self.poses_gt = torch.tensor(poses_gt[np.array(ind)]).float()
                else:
                    if poses_gt[0].ndim == 3:
                        self.poses_gt = (
                            torch.tensor(poses_gt[0]).float(),
                            torch.tensor(poses_gt[1]).float() * self.D
                        )
                    else:
                        self.poses_gt = torch.tensor(poses_gt).float()


        if use_pfm_encoder:
            self.pfm_features = None
            if augmentation_settings is not None:
                augmentation_settings['dim'] = augmentation_settings['resize'] if (
                        self.use_lmdb or processed_data is not None) else nx
                self.augmentation_transform = get_train_transformations(augmentation_settings,
                                                                        mean_std=[0.549995056189533,
                                                                                  0.11784453744259035]
                                                                        # mean_std=norm_real
                                                                        )

        accelerator.print(f"Normalized HT by {self.norm[0]} +/- {self.norm[1]}")
        accelerator.print("Normalized real space images by "
                          f"{self.norm_real[0]} +/- {self.norm_real[1]}")

    def __len__(self):
        if self.pose_id_map is not None:
            return len(self.pose_id_map)
        else:
            return self.N

    def __getitem__(self, index):
        in_dict = {}
        index_p = index
        if self.lazy:
            if self.ind is not None:
                img_real = self.particles_real[self.ind[index]].get().astype(np.float32)
            else:
                img_real = self.particles_real[index].get().astype(np.float32)

            if self.window:
                img_real *= window_mask(img_real.shape[-1], self.window_r, .99)

            particle = fft.ht2_center(img_real)

            if self.invert_data:
                particle *= -1

            particle = fft.symmetrize_ht(particle)

            particle = (particle - self.norm[0]) / self.norm[1]

            y = particle.astype(np.float32)
            particle_real_n = (img_real - self.norm_real[0]) / self.norm_real[1]
            if self.resolution_input is not None:
                particle_real_n = downsample(particle_real_n, self.resolution_input)
            y_real = particle_real_n.astype(np.float32)
            img_processed = to_int8(img_real)
        elif self.processed_data is not None:
            img_id = self.protein_id_list[index]
            # index=self.pose_id_map_reverse[index]
            if self.use_gt_pose:
                index_p = self.pose_id_map[index]

                if self.use_lmdb:
                    img_raw, img_processed, particle = self._get_item_lmdb(index)
                else:
                    img_processed = pickle.load(open(self.processed_img_path_list[index], 'rb'))
                    particle = pickle.load(open(self.raw_img_path_list[index].replace('raw', 'FT'), 'rb'))
                    img_raw = pickle.load(open(self.raw_img_path_list[index], 'rb'))
                rotmat_gt = self.pose_list[index_p]
                trans_gt = np.asarray(self.shift_list[index_p]) * self.D
                in_dict['R'] = torch.tensor(rotmat_gt).float()
                in_dict['t'] = torch.tensor(trans_gt).float()
            else:

                img_raw, img_processed, particle = self._get_item_lmdb(index)
            img_raw_n = (img_raw - self.mean_std_id_dict[img_id]['raw'][0]) / self.mean_std_id_dict[img_id]['raw'][
                1]
            particle_n = particle / self.mean_std_id_dict[img_id]['FT'][1]
            y = particle_n.astype(np.float32)
            y_real = img_raw_n.astype(np.float32)
        else:
            img_raw = self.imgs[index]
            img_raw_n = (img_raw - self.norm_real[0]) / self.norm_real[1]
            y = self.particles[index].astype(np.float32)
            y_real = img_raw_n.astype(np.float32)
            img_processed = to_int8(img_raw)

        if self.poses_gt is not None:
            if self.poses_gt[0].ndim == 3:
                rotmat_gt = self.poses_gt[0][index]
                trans_gt = self.poses_gt[1][index]
                in_dict['R'] = rotmat_gt
                in_dict['t'] = trans_gt
            else:
                rotmat_gt = self.poses_gt[index]
                in_dict['R'] = rotmat_gt

        if self.augmentation_transform is not None:
            in_dict['y_real_resized'] = self.augmentation_transform(img_processed)
        in_dict['y'] = y
        in_dict['y_real'] = y_real
        in_dict['index'] = index
        in_dict['index_p'] = index_p
        return in_dict

    def _get_env(self, lmdb_path):
        """
        懒加载和缓存LMDB环境的辅助函数。
        """
        # 在PyTorch DataLoader的多进程模式下，每个worker是独立的进程。
        # 我们需要在每个worker中维持自己的环境缓存。
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else 0

        # 如果切换了worker，清空旧的缓存
        if self.worker_id != current_worker_id:
            self.worker_id = current_worker_id
            for env_raw, env_processed, env_FT in zip(self.env_raw.values(), self.env_processed.values(),
                                                      self.env_FT.values()):
                env_raw.close()
                env_processed.close()
                env_FT.close()
            self.env_raw.clear()
            self.env_processed.clear()
            self.env_FT.clear()

        # 检查缓存中是否已有此LMDB的环境
        if os.path.join(lmdb_path, 'lmdb_raw') not in self.env_raw:
            # 如果没有，就打开它并存入缓存
            # readonly=True, lock=False 对于多进程读取是安全且高效的
            env_raw = lmdb.open(os.path.join(lmdb_path, 'lmdb_raw'), readonly=True, lock=False, readahead=False,
                                meminit=False)
            env_processed = lmdb.open(os.path.join(lmdb_path, 'lmdb_processed'), readonly=True, lock=False,
                                      readahead=False, meminit=False)
            env_FT = lmdb.open(os.path.join(lmdb_path, 'lmdb_FT'), readonly=True, lock=False, readahead=False,
                               meminit=False)

            self.env_raw[lmdb_path] = env_raw
            self.env_processed[lmdb_path] = env_processed
            self.env_FT[lmdb_path] = env_FT

        # return self.open_envs[lmdb_path]
        return self.env_raw[lmdb_path], self.env_processed[lmdb_path], self.env_FT[lmdb_path]

    def prepare_lmdb_env(self, lmdb_dir):
        self.lmdb_dir = lmdb_dir

        self.lmdb_paths = sorted(
            [os.path.join(lmdb_dir, name) for name in os.listdir(lmdb_dir) if
             os.path.isdir(os.path.join(lmdb_dir, name))])
        if not self.lmdb_paths:
            raise ValueError(f"No LMDB directories found in {lmdb_dir}")

        self.metadata = []  # 存储每个LMDB的信息：(路径, 包含的样本数)
        self.cumulative_sizes = [0]  # 存储样本数量的累加和，用于快速定位全局索引

        print("Scanning LMDB files and building index...")
        # 1. 遍历所有LMDB路径，只为获取样本数量，然后立刻关闭
        for path in self.lmdb_paths:
            try:
                env = lmdb.open(os.path.join(path, 'lmdb_raw'), readonly=True, lock=False, readahead=False,
                                meminit=False)
                with env.begin() as txn:
                    num_samples = txn.stat()['entries']
                env.close()

                self.metadata.append((path, num_samples))
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + num_samples)
            except lmdb.Error as e:
                print(f"Warning: Could not read LMDB at {path}. Skipping. Error: {e}")

        # 移除起始的0
        self.cumulative_sizes.pop(0)

        total_samples = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        print(f"Found {len(self.lmdb_paths)} LMDBs with a total of {total_samples} samples.")

        # 2. 核心：不在这里打开任何env，只在需要时打开
        # self.open_envs = {}  # 用于缓存已打开的LMDB环境
        self.worker_id = None  # 用于多进程DataLoader

        self.env_raw = {}
        self.env_processed = {}
        self.env_FT = {}

        n_particles = total_samples
        key = f"{0}".encode()
        example_env = lmdb.open(os.path.join(self.lmdb_paths[0], 'lmdb_raw'),
                                readonly=True,
                                lock=False,
                                readahead=False, meminit=False)
        example_txn_raw = example_env.begin()
        value = example_txn_raw.get(key)
        # data = torch.load(BytesIO(value), weights_only=False)
        data = pickle.loads(value)
        example = data
        example_env.close()
        return example, n_particles

    def get(self, index):
        return self.particles[index]

    def _get_item_lmdb(self, index):
        # 1. 确定这个全局索引属于哪个LMDB
        #    通过二分查找或简单遍历cumulative_sizes即可找到
        lmdb_idx = 0
        while index >= self.cumulative_sizes[lmdb_idx]:
            lmdb_idx += 1

        lmdb_path, _ = self.metadata[lmdb_idx]

        # 2. 计算在该LMDB中的局部索引
        prev_size = self.cumulative_sizes[lmdb_idx - 1] if lmdb_idx > 0 else 0
        local_idx = index - prev_size

        # 3. 获取（可能需要懒加载）对应的LMDB环境
        env_raw, env_processed, env_FT = self._get_env(lmdb_path)

        # 4. 从LMDB中读取数据
        key = f"{local_idx}".encode('ascii')  # 假设key是从0开始的数字
        with env_raw.begin() as txn:
            value = txn.get(key)
            img_raw = pickle.loads(value)
        with env_processed.begin() as txn:
            value = txn.get(key)
            img_processed = pickle.loads(value)
        with env_FT.begin() as txn:
            value = txn.get(key)
            particle = pickle.loads(value)
        # img_raw_n = (img_raw - self.norm_real[0]) / self.norm_real[1]
        # particle_n = (particle - self.norm[0]) / self.norm[1]
        return img_raw, img_processed, particle


class ImagePoseDataset(data.Dataset):
    def __init__(self, mrcdata, indices, predicted_rot, predicted_trans):
        """
        mrcdata: MRCData
        indices: [n_imgs]
        predicted_rot: [n_imgs, 3, 3]
        predicted_trans: [n_imgs, 2] or None
        """
        self.mrcdata = mrcdata
        self.indices = indices
        self.predicted_rot = predicted_rot
        self.predicted_trans = predicted_trans

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        mrcdata_in_dict = self.mrcdata[self.indices[index]]
        in_dict = {
            'y': mrcdata_in_dict['y'],
            'y_real': mrcdata_in_dict['y_real'],
            'index': mrcdata_in_dict['index'],
            'R': torch.tensor(self.predicted_rot[index]).float()
        }
        if self.predicted_trans is not None:
            in_dict['t'] = torch.tensor(self.predicted_trans[index]).float()
        assert mrcdata_in_dict['index'] == self.indices[index]
        return in_dict


def get_train_transformations(p, mean_std=None):
    my_transform = transforms.Compose([])


    if 'random_circle_crop' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [RandomApply(
                random_circle_crop(
                    img_dim=p['dim'],
                    scale=p['random_circle_crop']['scale']),
                p=p['random_circle_crop']['p']

            ), ])



    if 'resize' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [transforms.Resize(p['resize'])])

    my_transform = transforms.Compose(
        my_transform.transforms + [transforms.ToTensor()])

    if mean_std is not None:
        my_transform = transforms.Compose(
            my_transform.transforms +
            [transforms.Normalize(mean=mean_std[0], std=mean_std[1])])

    return my_transform


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class random_circle_crop(object):
    def __init__(self, img_dim, scale=(0.5, 1.0), value_replace=None):
        self.scale_min = scale[0]
        self.scale_max = scale[1]
        self.value_replace = value_replace
        # Create coordinate grids
        self.center = (img_dim - 1) / 2  # Handles both even and odd dimensions
        x = np.arange(img_dim) - self.center
        y = np.arange(img_dim) - self.center
        xx, yy = np.meshgrid(x, y)

        # Calculate distance from center for each pixel
        self.distance = np.sqrt(xx ** 2 + yy ** 2)

    def __call__(self, mrcdata):
        mrcdata = np.array(mrcdata)
        if self.scale_max == self.scale_min:
            crop_size = int(self.center * self.scale_min)
        else:
            crop_size = random.randint(int(self.center * self.scale_min), int(self.center * self.scale_max))
        aug = circle_crop(mrcdata, crop_size, distance=self.distance, value_replace=self.value_replace)
        return Image.fromarray(aug)


def circle_crop(mrcdata, crop_size, distance, value_replace=None):
    """
    Crop a circular region from the center of the image, replacing outside pixels.

    Args:
        mrcdata: Input image as [1, dim, dim] numpy array
        crop_size: Radius of the circle to keep
        value_replace: Value to set for pixels outside the circle (None means 0)

    Returns:
        Processed image with same shape as input
    """
    if value_replace is None:
        value_replace = 0

    # Remove the first dimension (assuming it's 1) and work with 2D array
    img = mrcdata
    # dim = img.shape[0]

    # Create mask for pixels outside the circle
    mask = distance > crop_size

    # Apply the mask to replace values outside the circle
    img[mask] = value_replace

    # Restore the original shape [1, dim, dim]
    # return img[np.newaxis, :, :]
    return img


class rondom_pixel_lost(object):

    def __init__(self, p=0.5, ratio=(0.4, 1 / 0.4)):

        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        img = np.array(img)
        if random.random() < self.p:
            img_h, img_w = img.shape
            lost_pixel_num = int(img.size * self.ratio)
            mask = np.concatenate((np.zeros(lost_pixel_num), np.ones(img.size - lost_pixel_num)), axis=0)
            np.random.shuffle(mask)
            mask = mask.reshape((img_w, img_h))
            img = img * mask
        # return PIL.Image.fromarray(img)
        return img

