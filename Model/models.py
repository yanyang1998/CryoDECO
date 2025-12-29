"""
Models
"""
# import os.path

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import time

from cryodata.data_preprocess import fft
from Pose import lie_tools
from Data import mask
from Pose import pose_search
from Model import encoder
import math


# from . import decoder

# import random


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class CryoDECO(nn.Module):
    def __init__(self, lattice, output_mask, n_particles_dataset, n_tilts_dataset, cnn_params, conf_regressor_params,
                 hyper_volume_params,
                 device,
                 num_processes=1,
                 resolution_encoder=64, no_trans=False,
                 use_gt_poses=False, use_gt_trans=False, will_use_point_estimates=False,
                 ps_params=None, verbose_time=False, pretrain_with_gt_poses=False, use_pfm_encoder=False,
                 # finetune_vit_encoder=False,
                 n_tilts_pose_search=1,
                 # features_PFM=None,
                 finetune_strategy=None,
                 finetune_layer_num=1,
                 pretrained_model_path=None,
                 # hidden_dim=[1024,512]
                 hidden_dim=[],
                 gradient_checkpointing=False,
                 # hidden_dim=[2048,1024,512,512]
                 encoder_type='vit_small',
                 use_fused_encoder=False,
                 fuse_type='concat',
                 fuse_only_table=0.0,
                 min_fuse_only_table=0.0,
                 warmup_epochs=10,
                 feature_take_indices=None
                 ):
        super(CryoDECO, self).__init__()
        self.lattice = lattice
        self.D = lattice.D
        self.output_mask = output_mask
        self.verbose_time = verbose_time
        self.n_tilts_pose_search = n_tilts_pose_search
        self.device = device
        self.use_fused_encoder = use_fused_encoder
        self.fuse_type = fuse_type
        self.feature_take_indices = feature_take_indices
        self.trun_on_conf_table = False

        # will be moved to the local gpu of each replica
        self.coords = nn.Parameter(self.lattice.coords, requires_grad=False).to(device)  # [D * D, 3]
        self.freqs2d = nn.Parameter(self.lattice.freqs2d, requires_grad=False).to(device)
        if ps_params is not None:
            self.base_shifts = nn.Parameter(
                pose_search.get_base_shifts(ps_params), requires_grad=False).to(device)
            self.base_rot = nn.Parameter(
                pose_search.get_base_rot(ps_params), requires_grad=False).to(device)
            self.so3_base_quat = nn.Parameter(
                pose_search.get_so3_base_quat(ps_params), requires_grad=False).to(device)
            self.base_inplane = nn.Parameter(
                pose_search.get_base_inplane(ps_params), requires_grad=False).to(device)

        self.no_trans = no_trans
        self.z_dim = conf_regressor_params['z_dim']
        # self.pose_table_dim=conf_regressor_params['conf_table_dim']
        # self.feature_dim = conf_regressor_params['feature_dim']

        self.variational_conf = conf_regressor_params['variational']
        self.std_z_init = conf_regressor_params['std_z_init']

        self.pose_only = False
        self.use_point_estimates = False
        self.pretrain = False
        self.is_in_pose_search_step = False
        self.use_point_estimates_conf = False
        self.use_pfm_encoder = use_pfm_encoder
        if self.z_dim > 0:
            if cnn_params['conf']:
                if self.use_pfm_encoder:
                    if self.use_fused_encoder or conf_regressor_params['epochs_init_conf_table'] >= 0:
                        conftable = encoder.ConfTable(
                            n_imgs=n_particles_dataset,
                            conf_table_dim=conf_regressor_params['conf_table_dim'],
                            variational=conf_regressor_params['variational'],
                            std_z_init=conf_regressor_params['std_z_init'],
                            # conf_init=features_PFM
                        )
                        self.conftable = conftable
                    else:
                        self.conftable = None

                    self.encoder = encoder.CryosolverEncoder(encoder_type=encoder_type,
                                                             feature_dim=conf_regressor_params['feature_dim'],
                                                             conf_table_dim=conf_regressor_params['conf_table_dim'],
                                                             pretrained_model_path=pretrained_model_path,
                                                             fuse_only_table=fuse_only_table,
                                                             finetune_strategy=finetune_strategy,
                                                             finetune_layer_num=finetune_layer_num,
                                                             # conftable=conftable,
                                                             use_fused_encoder=use_fused_encoder, fuse_type=fuse_type,
                                                             gradient_checkpointing=gradient_checkpointing,
                                                             std_z_init=self.std_z_init, warmup_epochs=warmup_epochs,
                                                             min_fuse_only_table=min_fuse_only_table,
                                                             feature_take_indices=self.feature_take_indices)


                else:
                    self.encoder = encoder.SharedCNN(
                        resolution_encoder if resolution_encoder is not None else self.D - 1,
                        cnn_params['depth_cnn'],
                        cnn_params['channels_cnn'],
                        cnn_params['kernel_size_cnn'],
                        1,
                        conf_regressor_settings={'z_dim': conf_regressor_params['z_dim'],
                                                 'conf_table_dim': conf_regressor_params['conf_table_dim'],
                                                 'std_z_init': conf_regressor_params['std_z_init'],
                                                 'variational': conf_regressor_params['variational'],
                                                 },
                    )

            else:
                self.conftable = encoder.ConfTable(
                    n_particles_dataset, conf_regressor_params['conf_table_dim'], conf_regressor_params['variational'],
                    conf_regressor_params['std_z_init'],
                    # conf_init=features_PFM
                )

        self.use_gt_poses = use_gt_poses
        self.use_gt_trans = use_gt_trans
        self.pretrain_with_gt_poses = pretrain_with_gt_poses

        # pose search parameters
        self.ps_params = ps_params
        self.trans_search_factor = None
        if ps_params is not None and ps_params['no_trans_search_at_pose_search']:
            self.trans_search_factor = 0.0

        self.num_processes = num_processes

    def update_trans_search_factor(self, ratio):
        if self.trans_search_factor is not None:
            self.trans_search_factor = ratio

    def forward(self, in_dict, hypervolume, accelerator, pose_table=None):
        if self.verbose_time:
            # torch.cuda.synchronize(device)
            accelerator.wait_for_everyone()
        start_time_encoder = time.time()
        latent_variables_dict = self.encode(
            in_dict, hypervolume, accelerator=accelerator, ctf=in_dict['ctf'], pose_table=pose_table
        )
        in_dict['tilt_index'] = in_dict['tilt_index'].reshape(-1)
        if self.verbose_time:
            # torch.cuda.synchronize(device)
            accelerator.wait_for_everyone()
        start_time_decoder = time.time()
        y_pred, y_gt_processed, times, latent_variables_dict = self.decode(
            latent_variables_dict, in_dict['ctf'], in_dict['y'], hypervolume, route_labels=in_dict['route_labels']
        )
        if self.verbose_time:
            torch.cuda.synchronize()
        end_time = time.time()
        out_dict = {
            'y_pred': y_pred,
            'y_gt_processed': y_gt_processed
        }
        if self.verbose_time:
            out_dict['time_encoder'] = torch.tensor([start_time_decoder - start_time_encoder]).float().to(self.device)
            out_dict['time_decoder'] = torch.tensor([end_time - start_time_decoder]).float().to(self.device)
            out_dict['time_decoder_coords'] = torch.tensor([times['coords']]).float().to(self.device)
            out_dict['time_decoder_query'] = torch.tensor([times['query']]).float().to(self.device)
        for key in latent_variables_dict.keys():
            out_dict[key] = latent_variables_dict[key]
        return out_dict

    @staticmethod
    def process_y_real(in_dict):
        y_real = in_dict['y_real']
        return y_real[..., None, :, :]

    def encode(self, in_dict, hypervolume, accelerator, pose_table=None, ctf=None):

        latent_variables_dict = {}
        z = None
        batch_size = in_dict['y'].shape[0]
        if self.z_dim > 0:
            # pretrain and pose only
            if self.pose_only:
                if self.use_pfm_encoder:
                    conf_dict = self.encoder(in_dict, pose_only=self.pose_only)
                else:
                    z = self.std_z_init * torch.randn((batch_size, self.z_dim), dtype=torch.float32,
                                                      device=self.device)
                    conf_dict = {'z': z}
                if self.variational_conf:
                    logvar = torch.ones((batch_size, self.conf_table_dim), dtype=torch.float32, device=self.device)
                    conf_dict['z_logvar'] = logvar
            # amortized inference
            elif not self.use_point_estimates_conf and not self.trun_on_conf_table:
                if self.use_pfm_encoder:
                    if self.use_fused_encoder:

                        table_features = self.conftable(in_dict)['z']
                    else:
                        table_features = None
                    conf_dict = self.encoder(in_dict, table_features=table_features)
                else:
                    y_real = self.process_y_real(in_dict)
                    particles_real = y_real.mean(1) if y_real.ndim == 5 else y_real
                    conf_dict = self.encoder(particles_real)
            # latent optimization
            else:
                conf_dict = self.conftable(in_dict)
            z = conf_dict['z']
            for key in conf_dict:
                latent_variables_dict[key] = conf_dict[key]

        # use gt poses
        if self.use_gt_poses or (self.pretrain and self.pretrain_with_gt_poses):
            rots = in_dict['R']
            pose_dict = {'R': rots}
            if not self.no_trans:
                trans = in_dict['t']
                pose_dict['t'] = trans

        # random poses
        elif self.pretrain:
            in_dim = in_dict['y'].shape[:-2]
            # device = in_dict['y_real'].device
            pose_dict = {'R': lie_tools.random_rotmat(np.prod(in_dim), device=self.device)}
            pose_dict['R'] = pose_dict['R'].reshape(*in_dim, 3, 3)
            if not self.no_trans:
                pose_dict['t'] = torch.zeros((*in_dim, 2)).float().to(self.device)

        # use pose search
        elif self.is_in_pose_search_step:
            hypervolume.eval()
            rot, trans, _ = pose_search.opt_theta_trans(
                self, in_dict['y'], self.lattice, self.ps_params, hypervolume, z=z, ctf_i=ctf,
                gt_trans=in_dict['t'] if not self.no_trans and self.use_gt_trans else None,
                trans_search_factor=self.trans_search_factor,
                route_labels=in_dict['route_labels'] if 'route_labels' in in_dict else None,
                accelerator=accelerator,
            )
            pose_dict = {
                'R': rot,
                'index': in_dict['index']
            }
            if not self.no_trans:
                pose_dict['t'] = trans
            hypervolume.train()

        # use point estimates
        else:
            assert self.use_point_estimates
            pose_dict = pose_table(in_dict)

        for key in pose_dict:
            latent_variables_dict[key] = pose_dict[key]

        return latent_variables_dict

    def decode(self, latent_variables_dict, ctf_local, y_gt, hypervolume, route_labels=None):
        rots = latent_variables_dict['R']
        in_shape = latent_variables_dict['R'].shape[:-2]
        z = None

        # sample conformations
        if self.z_dim > 0:
            if self.variational_conf:
                z = sample_conf(latent_variables_dict['z'], latent_variables_dict['z_logvar'])
            else:
                z = latent_variables_dict['z']

        # generate slices
        if self.verbose_time:
            torch.cuda.synchronize(self.device)
        start_time_coords = time.time()
        x = self.coords[self.output_mask.binary_mask] @ rots  # batch_size(, n_tilts), n_pts, 3
        if self.verbose_time:
            torch.cuda.synchronize(self.device)
        start_time_query = time.time()
        y_pred = hypervolume(x, z, route_labels, intermediate_features=latent_variables_dict['intermediates'] if (
                self.use_pfm_encoder and not self.trun_on_conf_table) else None)  # batch_size(, n_tilts), n_pts
        if self.verbose_time:
            torch.cuda.synchronize(self.device)
        end_time_query = time.time()
        times = {
            'coords': start_time_query - start_time_coords,
            'query': end_time_query - start_time_query
        }

        # apply ctf
        y_pred = self.apply_ctf(y_pred, ctf_local)  # batch_size(, n_tilts), n_pts

        # apply translations (to gt)
        if not self.no_trans:
            trans = latent_variables_dict['t'][..., None, :].reshape(-1, 1, 2)
            y_gt_processed = self.lattice.translate_ht(y_gt.reshape(
                -1, self.lattice.D ** 2), trans, freqs2d=self.freqs2d).reshape(
                *in_shape, -1)
            y_gt_processed = y_gt_processed[..., self.output_mask.binary_mask]
        else:
            y_gt_processed = y_gt.reshape(*in_shape, -1)
            y_gt_processed = y_gt_processed[..., self.output_mask.binary_mask]

        return y_pred, y_gt_processed, times, latent_variables_dict

    def eval_on_slice(self, x, hypervolume, route_labels=None, z=None, intermediate_features=None):
        if x.dim() == 4:
            batch_size, nq, n_pts, _3 = x.shape
            x = x.reshape(batch_size, nq * n_pts, 3)
            y_pred = hypervolume(x, z, route_labels,
                                 intermediate_features=intermediate_features) if self.num_processes == 1 else hypervolume.module(
                x, z,
                route_labels)
            y_pred = y_pred.reshape(batch_size, nq, n_pts)
        else:
            y_pred = hypervolume(x, z, route_labels,
                                 intermediate_features=intermediate_features) if self.num_processes == 1 else hypervolume.module(
                x, z,
                route_labels)
        return y_pred

    def apply_ctf(self, y_pred, ctf_local):
        ctf_local = ctf_local.reshape(*ctf_local.shape[:-2], -1)[..., self.output_mask.binary_mask]
        y_pred = ctf_local * y_pred
        return y_pred

    def eval_volume(self, norm, hypervolume, zval=None, route_labels=None, intermediate_features=None):
        return eval_volume_method(hypervolume, self.lattice, self.z_dim, norm,
                                  zval=zval, radius=self.output_mask.current_radius, route_labels=route_labels,
                                  intermediate_features=intermediate_features)

    @classmethod
    def load(cls, config, weights=None, device=None):

        pass


def sample_conf(z_mu, z_logvar):
    std = torch.exp(0.5 * z_logvar)
    eps = torch.randn_like(std)
    z = eps * std + z_mu
    return z


def eval_volume_method(hypervolume, lattice, z_dim, norm, zval=None, radius=None, route_labels=None,
                       intermediate_features=None):
    coords = lattice.coords
    extent = lattice.extent
    resolution = lattice.D
    radius_normalized = extent * 2 * radius / resolution
    z = None
    if zval is not None:
        z = torch.tensor(zval, dtype=torch.float32, device=coords.device).reshape(1, z_dim)

    volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    assert not hypervolume.training
    with torch.no_grad():
        for i, dz in enumerate(np.linspace(-extent, extent, resolution, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz], device=coords.device)
            x = x.reshape(1, -1, 3)
            y = hypervolume(x, z, route_labels.to(coords.device) if route_labels is not None else None,
                            intermediate_features=intermediate_features) if not hasattr(
                hypervolume, 'module') else hypervolume.module(
                x, z, route_labels.to(coords.device) if route_labels is not None else None,
                intermediate_features=intermediate_features)
            slice_radius = int(np.sqrt(max(radius_normalized ** 2 - dz ** 2, 0.)) * resolution)
            slice_mask = mask.CircularMask(lattice, slice_radius).binary_mask
            y[0, ~slice_mask] = 0.0
            y = y.view(resolution, resolution).detach().cpu().numpy()
            volume[i] = y
        volume = volume * norm[1] + norm[0]
        volume_real = fft.ihtn_center(volume[0:-1, 0:-1, 0:-1])  # remove last +k freq for inverse FFT
    return volume_real



class PoseTable(nn.Module):
    def __init__(self, n_imgs, no_trans, resolution, use_gt_trans):
        super(PoseTable, self).__init__()
        s2s2_init = torch.tensor(np.array([1., 0., 0., 0., 1., 0.]).reshape(1, 6).repeat(n_imgs, axis=0)).float()
        self.table_s2s2 = nn.Parameter(s2s2_init, requires_grad=True)
        self.no_trans = no_trans
        self.resolution = resolution
        self.use_gt_trans = use_gt_trans
        if not self.no_trans and not self.use_gt_trans:
            trans_init = torch.tensor(np.zeros((n_imgs, 2))).float()
            self.table_trans = nn.Parameter(trans_init, requires_grad=True)

    def initialize(self, rots, trans):
        state_dict = self.state_dict()
        # rots must contain "corrected" rotations
        state_dict['table_s2s2'] = lie_tools.rotmat_to_s2s2(torch.tensor(rots).float())
        if 'table_trans' in state_dict:
            # trans must be order 1
            state_dict['table_trans'] = torch.tensor(trans).float()
        self.load_state_dict(state_dict)

    def forward(self, in_dict):
        rots_s2s2 = self.table_s2s2[in_dict['tilt_index']]
        rots_matrix = lie_tools.s2s2_to_rotmat(rots_s2s2)
        pose_dict = {'R': rots_matrix}
        if not self.no_trans:
            if not self.use_gt_trans:
                pose_dict['t'] = self.table_trans[in_dict['tilt_index']]
            else:
                pose_dict['t'] = in_dict['t']
        if in_dict['y'].ndim == 4:
            pose_dict['R'] = pose_dict['R'].reshape(*in_dict['y'].shape[:-2], 3, 3)
            if not self.no_trans:
                pose_dict['t'] = pose_dict['t'].reshape(*in_dict['y'].shape[:-2], 2)
        return pose_dict


def adjust_learning_rate(optimizer, lr, epoch, warmup_epochs_down, warmup_epochs_up=0, min_lr=0, type='cosine'):
    if type == 'cosine':
        """Decays the learning rate with half-cycle cosine after warmup"""

        if epoch < 0:
            lr = lr
        elif epoch < warmup_epochs_up:
            lr = lr * epoch / warmup_epochs_up
        elif epoch < warmup_epochs_down:
            lr = (lr - min_lr) * 0.5 * (
                    1. + math.cos(math.pi * (epoch - warmup_epochs_up) / warmup_epochs_down)) + min_lr
        else:
            lr = min_lr

    if type == 'linear':
        """Decays the learning rate linearly after warmup"""
        if epoch < 0:
            lr = lr
        elif epoch < warmup_epochs_down:
            lr = min_lr + lr * (warmup_epochs_down - epoch) / warmup_epochs_down
        else:
            lr = min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Optimizer_scheduler(object):
    def __init__(self, optimizer, epochs, warmup_epochs, lr, min_lr=0):
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.last_lr = lr
        self.min_lr = min_lr

    def step(self):
        pass

    def get_last_lr(self):
        return [self.last_lr]

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
