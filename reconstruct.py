"""Reconstructing volume(s) from picked cryoEM and cryoET particles."""

import os
import shutil
import pickle
import yaml
import logging
from datetime import datetime as dt
import numpy as np
import time
import re
from typing_extensions import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import GradScaler, autocast

from cryodata.data_preprocess import mrc
from cryodata.dataset_resample import MyResampleSampler_pretrain
# from .cryodata.cryoemDataset import CryoMetaData

from Data import dataset, ctf
from Analyse import summary, utils

from Model.configuration import TrainingConfigurations
from analyze import ModelAnalyzer, Clustering_tool
from Pose.lattice import Lattice
from Model.losses import kl_divergence_conf, l1_regularizer, l2_frequency_bias
from Model.models import CryoDECO, PoseTable, adjust_learning_rate
from Model.decoder import HyperVolume, VolumeExplicit
from Data.mask import CircularMask, FrequencyMarchingMask
from accelerate.utils import broadcast


class ModelTrainer:

    # options for optimizers to use
    optim_types = {'adam': torch.optim.Adam, 'lbfgs': torch.optim.LBFGS, 'adamw': torch.optim.AdamW}

    # placeholders for runtimes
    run_phases = [
        'dataloading',
        'to_gpu',
        'ctf',
        'encoder',
        'decoder',
        'decoder_coords',
        'decoder_query',
        'loss',
        'backward',
        'to_cpu'
    ]

    def make_dataloader(self, batch_size: int, my_sampler=None) -> DataLoader:
        if my_sampler is not None:
            sampler = my_sampler
        else:
            if self.configs.shuffle:
                generator = torch.Generator()
                generator = generator.manual_seed(self.configs.seed)
                sampler = RandomSampler(self.data, generator=generator)
            else:
                sampler = SequentialSampler(self.data)

        data_loader = DataLoader(
            self.data, batch_size=batch_size, sampler=sampler,
            num_workers=self.configs.num_workers,
        )

        return data_loader

    @classmethod
    def load_configs(cls, outdir: str) -> dict[str, Any]:
        """Get the configurations from different versions of output folders."""
        train_configs = dict()

        if not os.path.isdir(outdir):
            raise ValueError(f"Output folder {outdir} does not exist!")

        if os.path.exists(os.path.join(outdir, 'cryoDECO-configs.yaml')):
            with open(os.path.join(outdir, 'cryoDECO-configs.yaml'), 'r') as f:
                train_configs = yaml.safe_load(f)

        elif os.path.exists(os.path.join(outdir, 'train-configs.yaml')):
            with open(os.path.join(outdir, 'train-configs.yaml'), 'r') as f:
                train_configs = {'training': yaml.safe_load(f)}

        return train_configs


    def __init__(self,
                 outdir: str, config_vals: dict[str, Any], load: bool = False, accelerator=None) -> None:
        """Initialize model parameters and variables.

        Arguments
        ---------
        outdir:         Location on file where model results will be saved.
        config_vals:    Parsed model parameter values provided by the user.
        load:           Load model from last saved epoch found in this output folder?
        """

        self.logger = logging.getLogger(__name__)
        self.outdir = os.path.join(outdir, 'out')
        # self.accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000)),DistributedDataParallelKwargs(find_unused_parameters=True)])
        self.accelerator = accelerator
        # self.accelerator.print(json.dumps(config_vals, indent=4))

        # if we want to load the model from the last epoch saved in this directory...
        if load:
            if not os.path.isdir(self.outdir):
                raise ValueError(f"Cannot use --load with directory `{outdir}` which "
                                 f"has no default output folder `{self.outdir}`!")

            # ...find the last saved epoch, and tell the model to load the model from it
            last_epoch = ModelAnalyzer.get_last_cached_epoch(self.outdir)
            if last_epoch == -2:
                raise ValueError(
                    f"Cannot perform any analyses for output directory `{self.outdir}` "
                    f"which does not contain any saved training checkpoints!"
                )

            config_vals['load'] = os.path.join(self.outdir, f"weights.{last_epoch}.pkl")
            checkpoint = torch.load(config_vals['load'])
            config_vals['z_dim'] = checkpoint['hypervolume_params']['z_dim']
            if 'conf_encoder' in checkpoint['optimizers_state_dict']:
                config_vals['use_conf_encoder'] = True
            else:
                config_vals['use_conf_encoder'] = False

        else:
            checkpoint = None

        self.configs = TrainingConfigurations(**config_vals)

        # take care of existing output directories; if we are loading from a saved
        # checkpoint then we want to just use the existing `out/` folder...

        if self.accelerator.is_main_process:
            if os.path.exists(self.outdir):
                if ('load' in config_vals
                        and config_vals['load'] is not None
                        and os.path.dirname(config_vals['load']) == self.outdir):
                    self.logger.info("Reusing existing output directory "
                                     "containing loaded checkpoint.")

                else:
                    old_cfgs = self.load_configs(self.outdir)
                    if old_cfgs:
                        old_cfgs = TrainingConfigurations(**old_cfgs['training'])

                        outdirs = [p for p in os.listdir(outdir)
                                   if os.path.isdir(os.path.join(outdir, p))
                                   and re.match("^old-out_[0-9]+_", p)]

                        if not outdirs:
                            new_id = '000'
                        else:
                            new_id = str(max(int(os.path.basename(d).split('_')[1])
                                             for d in outdirs) + 1).rjust(3, '0')

                        train_lbl = (
                            'refine' if old_cfgs.refine_gt_poses
                            else 'fixed' if old_cfgs.use_gt_poses
                            else 'abinit'
                        )
                        train_lbl += (
                            '-homo' if old_cfgs.z_dim == 0
                            else f'-het{old_cfgs.z_dim}'
                        )

                        newdir = os.path.join(outdir, f"old-out_{new_id}_{train_lbl}")
                        shutil.move(self.outdir, newdir)
                        self.logger.warning(
                            f"Output directory `out/` already exists here!."
                            f"Renaming the old one to `{os.path.basename(newdir)}`."
                        )

                    elif os.listdir(self.outdir):
                        self.logger.info("Using existing output directory which does "
                                         "not yet contain any cryoDECO output!.")

            # create the output folder for model results and log file for model training
            os.makedirs(self.outdir, exist_ok=True)
            self.logger.addHandler(logging.FileHandler(
                os.path.join(self.outdir, "training.log")))
        self.accelerator.wait_for_everyone()

        self.batch_size_known_poses = (
            self.configs.batch_size_known_poses)
        self.batch_size_hps = self.configs.batch_size_hps
        self.batch_size_sgd = self.configs.batch_size_sgd

        np.random.seed(self.configs.seed)
        torch.manual_seed(self.configs.seed)

        # set the device
        self.use_cuda = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.accelerator.print(f"Use cuda {self.use_cuda}")

        # tensorboard writer
        if self.accelerator.is_main_process:
            self.summaries_dir = os.path.join(self.outdir, 'summaries')
            os.makedirs(self.summaries_dir, exist_ok=True)
            self.writer = SummaryWriter(self.summaries_dir)
            self.accelerator.print("Will write tensorboard summaries "
                                   f"in {self.summaries_dir}")

        # load the optional index used to filter particles
        if self.configs.ind is not None:
            if isinstance(self.configs.ind, int):
                self.accelerator.print(f"Keeping {self.configs.ind} particles")
                self.index = np.arange(self.configs.ind)

            elif isinstance(self.configs.ind, str):
                if not os.path.exists(self.configs.ind):
                    raise ValueError("Given subset index file "
                                     f"`{self.configs.ind}` does not exist!")

                self.accelerator.print(
                    f"Filtering dataset with {self.configs.ind}")
                self.index = pickle.load(open(self.configs.ind, 'rb'))

        else:
            self.index = None

        # load the particles
        self.accelerator.print("Creating dataset")
        self.data = dataset.MRCData(
            self.configs.particles, outdir=outdir, accelerator=self.accelerator,
            max_threads=self.configs.max_threads,
            ind=self.index, lazy=self.configs.lazy,
            relion31=self.configs.relion31, poses_gt_pkl=self.configs.pose,
            resolution_input=self.configs.resolution_encoder,
            window_r=self.configs.window_radius_gt_real,
            datadir=self.configs.datadir, no_trans=self.configs.no_trans,
            use_pfm_encoder=self.configs.use_pfm_encoder,
            augmentation_settings=self.configs.augmentation_settings,
            use_lmdb=self.configs.use_lmdb,
            processed_data=self.configs.processed_data,
            use_gt_pose=self.configs.use_gt_poses,
            score_bar=self.configs.score_bar,
        )

        self.n_particles_dataset = self.data.N
        self.n_tilts_dataset = (self.data.Nt
                                if self.configs.subtomogram_averaging
                                else self.data.N)
        self.resolution = self.data.D

        if self.data.pose_id_map is not None:
            self.pose_id_map = self.data.pose_id_map
        else:
            self.pose_id_map = None

        # load ctf
        if self.configs.ctf is not None:
            self.accelerator.print(f"Loading ctf params from {self.configs.ctf}")

            ctf_params = ctf.load_ctf_for_training(self.resolution - 1,
                                                   self.configs.ctf,
                                                   accelerator=self.accelerator)


        elif self.configs.processed_data is not None:
            self.accelerator.print(f"Loading ctf params from {self.configs.processed_data}")
            # ctf_params = np.asarray(utils.load_pkl(os.path.join(self.configs.processed_data, 'ctf_list.data')))
            ctf_params = np.asarray(utils.load_pkl(os.path.join(self.configs.processed_data, 'ctf.pkl')))

        else:
            ctf_params = None

        if ctf_params is not None:
            if self.index is not None:
                self.accelerator.print("Filtering dataset")
                ctf_params = ctf_params[self.index]

            # assert ctf_params.shape == (self.n_tilts_dataset, 8)
            if ctf_params.shape[1] != (self.n_tilts_dataset, 8):
                self.accelerator.print(
                    f"CTF parameters have shape {ctf_params.shape}, "
                    f"but expected {(self.n_tilts_dataset, 8)}"
                )

            if self.configs.subtomogram_averaging:
                ctf_params = np.concatenate(
                    (ctf_params, self.data.ctfscalefactor.reshape(-1, 1)),
                    axis=1  # type: ignore
                )
                self.data.voltage = float(ctf_params[0, 4])

            self.ctf_params = torch.tensor(ctf_params)
            # self.ctf_params = self.ctf_params.to(self.device)

            if self.configs.subtomogram_averaging:
                self.data.voltage = float(self.ctf_params[0, 4])

        # lattice
        self.accelerator.print("Building lattice")
        self.lattice = Lattice(self.resolution, extent=0.5)

        # output mask
        if self.configs.output_mask == 'circ':
            radius = (self.lattice.D // 2 if self.configs.max_freq is None
                      else self.configs.max_freq)
            self.output_mask = CircularMask(self.lattice, radius)

        elif self.configs.output_mask == 'frequency_marching':
            self.output_mask = FrequencyMarchingMask(
                self.lattice, self.lattice.D // 2,
                radius=self.configs.l_start_fm,
                add_one_every=self.configs.add_one_frequency_every
            )

        else:
            raise NotImplementedError

        # pose search
        ps_params = None
        self.epochs_pose_search = 0

        if self.configs.n_imgs_pose_search > 0:
            self.epochs_pose_search = max(2,
                                          self.configs.n_imgs_pose_search
                                          // self.n_particles_dataset + 1)

        if self.epochs_pose_search > 0:
            ps_params = {
                'l_min': self.configs.l_start,
                'l_max': self.configs.l_end,
                't_extent': self.configs.t_extent,
                't_n_grid': self.configs.t_n_grid,
                'niter': self.configs.n_iter,
                'nkeptposes': self.configs.n_kept_poses,
                'base_healpy': self.configs.base_healpy,
                't_xshift': self.configs.t_x_shift,
                't_yshift': self.configs.t_y_shift,
                'no_trans_search_at_pose_search': self \
                    .configs.no_trans_search_at_pose_search,
                'n_tilts_pose_search': self.configs.n_tilts_pose_search,
                'tilting_func': (self.data.get_tilting_func()
                                 if self.configs.subtomogram_averaging
                                 else None),
                'average_over_tilts': self.configs.average_over_tilts
            }

        # cnn
        cnn_params = {
            'conf': self.configs.use_conf_encoder,
            'depth_cnn': self.configs.depth_cnn,
            'channels_cnn': self.configs.channels_cnn,
            'kernel_size_cnn': self.configs.kernel_size_cnn
        }

        # conformational encoder
        if self.configs.z_dim > 0:
            self.accelerator.print("Heterogeneous reconstruction with "
                                   f"z_dim = {self.configs.z_dim}")
        else:
            self.accelerator.print("Homogeneous reconstruction")

        conf_regressor_params = {
            'z_dim': self.configs.z_dim,
            'std_z_init': self.configs.std_z_init,
            'variational': self.configs.variational_het,
            'feature_dim': self.configs.feature_dim,
            'conf_table_dim': self.configs.conf_table_dim,
            'epochs_init_conf_table': self.configs.epochs_init_conf_table,
        }

        # hypervolume
        hyper_volume_params = {
            'explicit_volume': self.configs.explicit_volume,
            'n_layers': self.configs.hypervolume_layers if checkpoint is None else checkpoint['hypervolume_params']['n_layers'],
            'hidden_dim': self.configs.hypervolume_dim if checkpoint is None else checkpoint['hypervolume_params']['hidden_dim'],
            'pe_type': self.configs.pe_type if checkpoint is None else checkpoint['hypervolume_params']['pe_type'],
            'pe_dim': self.configs.pe_dim if checkpoint is None else checkpoint['hypervolume_params']['pe_dim'],
            'feat_sigma': self.configs.feat_sigma if checkpoint is None else checkpoint['hypervolume_params']['feat_sigma'],
            'domain': self.configs.hypervolume_domain if checkpoint is None else checkpoint['hypervolume_params']['domain'],
            'extent': self.lattice.extent,
            'pe_type_conf': self.configs.pe_type_conf if checkpoint is None else checkpoint['hypervolume_params']['pe_type_conf'],
            'decoder_type': self.configs.decoder_type if checkpoint is None else checkpoint['hypervolume_params']['decoder_type'],
            'moe_num': self.configs.moe_num if checkpoint is None else checkpoint['hypervolume_params']['moe_num'],
            'num_shared_experts': self.configs.num_shared_experts if checkpoint is None else checkpoint['hypervolume_params']['num_shared_experts'],
            'use_clustering_route': self.configs.use_clustering_route,
            'feature_fuse_indices': self.configs.feature_fuse_indices if checkpoint is None else checkpoint['hypervolume_params']['feature_fuse_indices'],
            'decoder_ln': self.configs.decoder_ln if checkpoint is None else checkpoint['hypervolume_params']['decoder_ln'],
        }

        self.will_use_point_estimates = self.configs.epochs_sgd >= 1
        self.accelerator.print("Initializing model...")

        self.model = CryoDECO(
            self.lattice,
            self.output_mask,
            self.n_particles_dataset,
            self.n_tilts_dataset,
            cnn_params,
            conf_regressor_params,
            hyper_volume_params,
            device=self.accelerator.device,
            num_processes=self.accelerator.num_processes,
            resolution_encoder=self.configs.resolution_encoder,
            no_trans=self.configs.no_trans,
            use_gt_poses=self.configs.use_gt_poses,
            use_gt_trans=self.configs.use_gt_trans,
            will_use_point_estimates=self.will_use_point_estimates,
            ps_params=ps_params,
            verbose_time=self.configs.verbose_time,
            pretrain_with_gt_poses=self.configs.pretrain_with_gt_poses,
            n_tilts_pose_search=self.configs.n_tilts_pose_search,

            use_pfm_encoder=self.configs.use_pfm_encoder,
            # features_PFM=self.data.pfm_features,
            # use_generated_features=self.configs.use_generated_features,
            pretrained_model_path=self.configs.pretrained_model_path,
            finetune_strategy=self.configs.finetune_strategy,
            finetune_layer_num=self.configs.finetune_layer_num,
            gradient_checkpointing=self.configs.gradient_checkpointing,
            encoder_type=self.configs.encoder_type,
            use_fused_encoder=self.configs.use_fused_encoder,
            fuse_type=self.configs.fuse_type,
            fuse_only_table=self.configs.fuse_only_table,
            warmup_epochs=self.configs.warm_up_epochs,
            min_fuse_only_table=self.configs.min_fuse_only_table,
            feature_take_indices=self.configs.feature_take_indices
        )
        if not hyper_volume_params['explicit_volume']:
            self.hypervolume = HyperVolume(self.lattice.D, conf_regressor_params['z_dim'],
                                           hyper_volume_params['n_layers'],
                                           hyper_volume_params['hidden_dim'],
                                           hyper_volume_params['pe_type'], hyper_volume_params['pe_dim'],
                                           hyper_volume_params['feat_sigma'], hyper_volume_params['domain'],
                                           pe_type_conf=hyper_volume_params['pe_type_conf'],
                                           decoder_type=hyper_volume_params['decoder_type'],
                                           moe_num=hyper_volume_params['moe_num'],
                                           num_shared_experts=hyper_volume_params['num_shared_experts'],
                                           use_clustering_route=hyper_volume_params['use_clustering_route'],
                                           feature_fuse_indices=hyper_volume_params['feature_fuse_indices'],
                                           decoder_ln=hyper_volume_params['decoder_ln']
                                           )
        else:
            self.hypervolume = VolumeExplicit(self.lattice.D, hyper_volume_params['domain'],
                                              hyper_volume_params['extent'])

        if not self.configs.use_gt_poses and self.will_use_point_estimates:
            self.pose_table = PoseTable(self.n_tilts_dataset, self.model.no_trans, self.model.D,
                                        self.configs.use_gt_trans)
        else:
            self.pose_table = None

        # initialization from a previous checkpoint
        if self.configs.load:
            self.accelerator.print(f"Loading checkpoint from {self.configs.load}")

            state_dict = checkpoint['model_state_dict']

            if 'base_shifts' in state_dict:
                state_dict.pop('base_shifts')

            # self.accelerator.print(
            #     self.model.load_state_dict(state_dict, strict=False))
            if self.configs.use_conf_encoder:
                self.accelerator.print(
                    self.model.encoder.load_state_dict(state_dict, strict=False))
            self.accelerator.print(self.hypervolume.load_state_dict(
                checkpoint['hypervolume_state_dict'], strict=False))
            self.start_epoch = checkpoint['epoch'] + 1

            if 'output_mask_radius' in checkpoint:
                self.output_mask.update_radius(
                    checkpoint['output_mask_radius'])

        else:
            self.start_epoch = -1

        # move to gpu and parallelize
        # self.accelerator.print(self.model)
        # self.accelerator.print(self.hypervolume)
        parameter_count = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
        self.accelerator.print(f"{parameter_count} parameters in model")


        # if self.n_prcs > 1:
        #     self.model = MyDataParallel(self.model)

        self.accelerator.print("Model initialized. Moving to GPU...")
        # self.model.to(self.device)
        self.model.output_mask \
            .binary_mask = self.model.output_mask.binary_mask.cpu()

        self.optimizers = dict()
        self.optimizer_types = dict()

        # hypervolume
        hyper_volume_params = [{
            'params': list(self.hypervolume.parameters())}]
        self.optimizers['hypervolume'] = self.optim_types[
            self.configs.hypervolume_optimizer_type](hyper_volume_params,
                                                     lr=self.configs.lr)
        self.optimizer_types[
            'hypervolume'] = self.configs.hypervolume_optimizer_type

        # pose table
        if not self.configs.use_gt_poses:
            if self.configs.epochs_sgd > 0:
                pose_table_params = [{
                    'params': list(self.pose_table.parameters())}]

                self.optimizers['pose_table'] = self.optim_types[
                    self.configs.pose_table_optimizer_type](
                    pose_table_params, lr=self.configs.lr_pose_table)
                self.optimizer_types[
                    'pose_table'] = self.configs.pose_table_optimizer_type

        # conformations
        if self.configs.z_dim > 0:
            if self.configs.use_conf_encoder:
                # if self.configs.use_pfm_encoder:
                #     conf_encoder_params = [{
                #         'params': (list(self.model.encoder.parameters())
                #                    )
                #     }]
                conf_encoder_params = [{
                    'params': (list(self.model.encoder.parameters())
                               )
                }]

                self.optimizers['conf_encoder'] = self.optim_types[
                    self.configs.conf_encoder_optimizer_type](
                    conf_encoder_params, lr=self.configs.lr_conf_encoder,
                    weight_decay=self.configs.wd
                )
                self.optimizer_types[
                    'conf_encoder'] = self.configs.conf_encoder_optimizer_type
                if self.configs.use_fused_encoder or self.configs.epochs_init_conf_table>=0:
                    conf_table_params = [{
                        'params': list(self.model.conftable.parameters())}]
                    self.optimizers['conf_table'] = self.optim_types[
                        self.configs.conf_table_optimizer_type](
                        conf_table_params, lr=self.configs.lr_conf_table)
                    self.optimizer_types[
                        'conf_table'] = self.configs.conf_table_optimizer_type


            else:
                conf_table_params = [{
                    'params': list(self.model.conftable.parameters())}]

                self.optimizers['conf_table'] = self.optim_types[
                    self.configs.conf_table_optimizer_type](
                    conf_table_params, lr=self.configs.lr_conf_table)

                self.optimizer_types[
                    'conf_table'] = self.configs.conf_table_optimizer_type

        self.optimized_modules = []

        # initialization from a previous checkpoint
        if self.configs.load:
            # checkpoint = torch.load(self.configs.load)

            for key in self.optimizers:
                self.optimizers[key].load_state_dict(
                    checkpoint['optimizers_state_dict'][key])

        # dataloaders
        if self.configs.processed_data is not None:
            # meta_data = CryoMetaData(processed_data_path=self.configs.processed_data,
            #                          )
            # id_index_dict, _, id_scores_dict = meta_data.preprocess_trainset_index_pretrain(id_map_for_filtering=meta_data.pose_id_map2)
            id_index_dict = self.data.id_index_dict
            id_scores_dict = self.data.id_scores_dict
            my_sampler_ps = MyResampleSampler_pretrain(id_index_dict=id_index_dict,
                                                       batch_size_all=self.batch_size_hps * self.accelerator.num_processes,
                                                       shuffle_type=self.configs.shuffle_type,
                                                       max_number_per_sample=self.configs.resample_per_dataset,
                                                       shuffle_mix_up_ratio=0.0,
                                                       # scores_bar=self.configs.score_bar,
                                                       id_scores_dict=id_scores_dict
                                                       )
            my_sampler_lo = MyResampleSampler_pretrain(id_index_dict=id_index_dict,
                                                       batch_size_all=self.batch_size_sgd * self.accelerator.num_processes,
                                                       shuffle_type=self.configs.shuffle_type,
                                                       max_number_per_sample=self.configs.resample_per_dataset,
                                                       shuffle_mix_up_ratio=0.0,
                                                       # scores_bar=self.configs.score_bar,
                                                       id_scores_dict=id_scores_dict
                                                       )
            my_sampler_base = MyResampleSampler_pretrain(id_index_dict=id_index_dict,
                                                         batch_size_all=self.batch_size_known_poses * self.accelerator.num_processes,
                                                         shuffle_type=self.configs.shuffle_type,
                                                         max_number_per_sample=self.configs.resample_per_dataset,
                                                         shuffle_mix_up_ratio=0.0,
                                                         # scores_bar=self.configs.score_bar,
                                                         id_scores_dict=id_scores_dict
                                                         )
        else:
            my_sampler_ps = None
            my_sampler_lo = None
            my_sampler_base = None
        self.data_generator_pose_search = self.make_dataloader(
            batch_size=self.batch_size_hps, my_sampler=my_sampler_ps)
        self.data_generator = self.make_dataloader(
            batch_size=self.batch_size_known_poses, my_sampler=my_sampler_base)
        self.data_generator_latent_optimization = self.make_dataloader(
            batch_size=self.batch_size_sgd, my_sampler=my_sampler_lo)

        # save configurations
        self.configs.write(os.path.join(self.outdir, 'cryoDECO-configs.yaml'),
                           data_norm_mean=float(self.data.norm[0]),
                           data_norm_std=float(self.data.norm[1]))

        epsilon = 1e-8
        # booleans
        self.log_latents = False
        self.pose_only = True
        self.pretraining = False
        self.is_in_pose_search_step = False
        self.use_point_estimates = False
        self.first_switch_to_point_estimates = True
        self.first_switch_to_point_estimates_conf = True

        if self.configs.load is not None:
            if self.start_epoch >= self.epochs_pose_search:
                self.first_switch_to_point_estimates = False
            self.first_switch_to_point_estimates_conf = False

        self.use_kl_divergence = (not self.configs.z_dim == 0
                                  and self.configs.variational_het
                                  and self.configs.beta_conf >= epsilon)
        self.use_trans_l1_regularizer = (
                self.configs.trans_l1_regularizer >= epsilon
                and not self.configs.use_gt_trans and not self.configs.no_trans
        )
        self.use_l2_smoothness_regularizer = (
                self.configs.l2_smoothness_regularizer >= epsilon)

        if self.configs.load:
            self.num_epochs = self.start_epoch + self.configs.epochs_sgd
            self.num_epochs += max(self.epochs_pose_search - self.start_epoch, 0)
        else:
            self.num_epochs = self.epochs_pose_search + self.configs.epochs_sgd

        self.n_particles_pretrain = (self.configs.n_imgs_pretrain
                                     if self.configs.n_imgs_pretrain >= 0
                                     else self.n_particles_dataset)


        # placeholders for predicted latent variables,
        # last input/output batch, losses
        self.in_dict_last = None
        self.y_pred_last = None

        self.predicted_rots = np.empty((self.n_tilts_dataset, 3, 3))
        self.predicted_trans = (np.empty((self.n_tilts_dataset, 2))
                                if not self.configs.no_trans else None)
        self.predicted_conf = (np.empty((self.n_particles_dataset,
                                         self.configs.z_dim))
                               if self.configs.z_dim > 0 else None)
        if self.configs.use_fused_encoder and self.configs.use_pfm_encoder and self.configs.use_conf_encoder:
            self.predicted_conf_table = np.empty((self.n_particles_dataset,
                                                  self.configs.z_dim))
        else:
            self.predicted_conf_table = None

        self.predicted_logvar = (
            np.empty((self.n_particles_dataset, self.configs.z_dim))
            if self.configs.z_dim > 0 and self.configs.variational_het
            else None
        )

        self.mask_particles_seen_at_last_epoch = np.zeros(
            self.n_particles_dataset)
        self.mask_tilts_seen_at_last_epoch = np.zeros(self.n_tilts_dataset)

        # counters
        self.epoch = 0
        self.run_times = {phase: [] for phase in self.run_phases}
        self.current_epoch_particles_count = 0
        self.total_batch_count = 0
        self.total_particles_count = 0
        self.batch_idx = 0
        self.cur_loss = None
        if self.configs.moe_num > 1 and self.configs.decoder_type == 'moe':
            self.clustering_tool_moe = Clustering_tool(data_num=self.data.N, n_clusters=self.configs.moe_num,
                                                       k_init=self.configs.k_init,
                                                       clustering_dim=self.configs.clustering_dim,
                                                       clustering_type=self.configs.clustering_type, )
        else:
            self.clustering_tool_moe = None

        # self.labels_evaluate = None
        self.clustering_tool_evaluate = None
        if self.accelerator.is_main_process and (self.configs.labels_evaluate is not None or (
                self.configs.cluster_num_evaluate is not None and self.configs.cluster_num_evaluate > 0)):

            if self.configs.labels_evaluate is not None:
                labels_evaluate = np.asarray(utils.load_pkl(self.configs.labels_evaluate))
                # labels_evaluate = np.asarray(self.data.labels_class) if self.data.labels_class is not None else np.asarray(utils.load_pkl(self.configs.labels_evaluate))
            else:
                labels_evaluate = np.asarray(self.data.labels_class) if self.data.labels_class is not None else None
            cluster_num = max(labels_evaluate) + 1 if labels_evaluate is not None else self.configs.cluster_num_evaluate
            if cluster_num > 1:
                self.clustering_tool_evaluate = Clustering_tool(data_num=self.data.N, n_clusters=int(max(
                    labels_evaluate) + 1) if labels_evaluate is not None else self.configs.cluster_num_evaluate,
                                                                k_init=self.configs.k_init,
                                                                clustering_dim=self.configs.clustering_dim,
                                                                labels_true=labels_evaluate,
                                                                cs_path=self.configs.particles,
                                                                clustering_type=self.configs.clustering_type,
                                                                )
                # self.clustering_tool_evaluate.labels_true = labels_evaluate
                self.accelerator.print('Evaluate cluster num: {}'.format(
                    max(labels_evaluate) + 1 if labels_evaluate is not None else self.configs.cluster_num_evaluate))

        if 'conf_encoder' in self.optimizers:
            self.model.encoder, self.optimizers[
                'conf_encoder'], self.data_generator, self.data_generator_latent_optimization, self.data_generator_pose_search = self.accelerator.prepare(
                self.model.encoder, self.optimizers['conf_encoder'], self.data_generator,
                self.data_generator_latent_optimization, self.data_generator_pose_search)
        else:
            self.data_generator, self.data_generator_latent_optimization, self.data_generator_pose_search = self.accelerator.prepare(
                self.data_generator,
                self.data_generator_latent_optimization, self.data_generator_pose_search)

        self.hypervolume, self.optimizers['hypervolume'] = self.accelerator.prepare(self.hypervolume,
                                                                                    self.optimizers['hypervolume'])
        if not self.configs.use_gt_poses and self.will_use_point_estimates:
            self.pose_table, self.optimizers['pose_table'] = self.accelerator.prepare(self.pose_table,
                                                                                      self.optimizers['pose_table'])
        if 'conf_table' in self.optimizers:
            self.model.conftable, self.optimizers['conf_table'] = self.accelerator.prepare(
                self.model.conftable, self.optimizers['conf_table'])

        self.ctf_params = self.ctf_params.to(self.accelerator.device) if self.ctf_params is not None else None
        self.lattice.freqs2d = self.lattice.freqs2d.to(self.accelerator.device)
        self.lattice.coords = self.lattice.coords.to(self.accelerator.device)

    def train(self):
        self.accelerator.print("\n--- Training Starts Now ---")
        t_0 = dt.now()

        self.predicted_rots = np.eye(3).reshape(1, 3, 3).repeat(
            self.n_tilts_dataset, axis=0)
        self.predicted_trans = (np.zeros((self.n_tilts_dataset, 2))
                                if not self.configs.no_trans else None)
        self.predicted_conf = (np.zeros((self.n_particles_dataset,
                                         self.configs.z_dim))
                               if self.configs.z_dim > 0 else None)
        if self.configs.use_fused_encoder and self.configs.use_pfm_encoder and self.configs.use_conf_encoder:
            self.predicted_conf_table = np.empty((self.n_particles_dataset,
                                                  # self.configs.z_dim if not self.configs.fuse_type=='concat' else self.configs.z_dim//2
                                                  self.configs.conf_table_dim
                                                  ))

        self.total_batch_count = 0
        self.total_particles_count = 0

        self.epoch = self.start_epoch - 1

        # self.scaler = torch.amp.GradScaler(enabled=self.configs.use_amp)
        for epoch in range(self.start_epoch, self.num_epochs):
            te = dt.now()
            ind_epoch = []
            self.mask_particles_seen_at_last_epoch = np.zeros(
                self.n_particles_dataset)
            self.mask_tilts_seen_at_last_epoch = np.zeros(self.n_tilts_dataset)

            self.epoch += 1
            self.current_epoch_particles_count = 0
            self.optimized_modules = ['hypervolume']

            self.pose_only = (self.total_particles_count
                              < self.configs.pose_only_phase
                              or self.configs.z_dim == 0 or epoch < 0)
            self.pretraining = self.epoch < 0

            if not self.configs.use_gt_poses:
                self.is_in_pose_search_step = (
                        0 <= epoch < self.epochs_pose_search)
                self.use_point_estimates = (
                        epoch >= max(0, self.epochs_pose_search))

            data_generator = self.data_generator
            n_max_particles = self.n_particles_dataset if self.configs.processed_data is None else len(
                data_generator) * self.batch_size_known_poses * self.accelerator.num_processes

            # pre-training
            if self.pretraining:
                n_max_particles = self.n_particles_pretrain
                self.accelerator.print(
                    f"Will pretrain on {n_max_particles} particles")

            # HPS
            elif self.is_in_pose_search_step:
                # n_max_particles = self.n_particles_dataset
                data_generator = self.data_generator_pose_search
                n_max_particles = len(
                    data_generator) * self.batch_size_hps * self.accelerator.num_processes
                self.accelerator.print(
                    f"Will use pose search on {n_max_particles} particles")



            # SGD
            elif self.use_point_estimates:
                if self.first_switch_to_point_estimates:
                    self.first_switch_to_point_estimates = False
                    self.accelerator.print("Switched to autodecoding poses")

                    if self.configs.refine_gt_poses:
                        self.accelerator.print(
                            "Initializing pose table from ground truth")

                        poses_gt = utils.load_pkl(self.configs.pose)
                        if poses_gt[0].ndim == 3:
                            # contains translations
                            rotmat_gt = torch.tensor(poses_gt[0]).float()
                            trans_gt = torch.tensor(poses_gt[1]).float()
                            trans_gt *= self.resolution

                            if self.index is not None:
                                rotmat_gt = rotmat_gt[self.index]
                                trans_gt = trans_gt[self.index]

                        else:
                            rotmat_gt = torch.tensor(poses_gt).float()
                            trans_gt = None

                            if self.index is not None:
                                rotmat_gt = rotmat_gt[self.index]

                        self.pose_table.initialize(rotmat_gt, trans_gt)

                    else:
                        self.accelerator.print("Initializing pose table from "
                                               "hierarchical pose search")
                        if self.accelerator.num_processes > 1:
                            self.pose_table.module.initialize(self.predicted_rots,
                                                              self.predicted_trans)
                        else:
                            self.pose_table.initialize(self.predicted_rots,
                                                       self.predicted_trans)

                data_generator = self.data_generator_latent_optimization
                n_max_particles = len(
                    data_generator) * self.batch_size_sgd * self.accelerator.num_processes
                self.accelerator.print("Will use latent optimization on "
                                       f"{self.n_particles_dataset} particles")

                self.optimized_modules.append('pose_table')

            # GT poses
            else:
                assert self.configs.use_gt_poses
                n_max_particles = len(data_generator) * self.batch_size_known_poses * self.accelerator.num_processes
                self.accelerator.print("Will use ground truth poses on "
                                       f"{n_max_particles} particles")

            if epoch==self.configs.epochs_init_conf_table and self.configs.use_conf_encoder:
                self.model.trun_on_conf_table = True
                self.accelerator.print("Initializing conformation table from encoder's z at epoch {}".format(epoch))
                if self.accelerator.num_processes > 1:
                    self.model.conftable.module.initialize(self.predicted_conf)
                else:
                    self.model.conftable.initialize(self.predicted_conf)
            self.n_max_particles = n_max_particles

            # conformations
            if not self.pose_only:
                if self.model.trun_on_conf_table:
                    self.optimized_modules.append('conf_table')
                elif self.configs.use_conf_encoder:
                    self.optimized_modules.append('conf_encoder')
                    if self.configs.use_fused_encoder :
                        self.optimized_modules.append('conf_table')


                else:
                    if self.first_switch_to_point_estimates_conf:
                        self.first_switch_to_point_estimates_conf = False

                        if self.configs.initial_conf is not None:
                            self.accelerator.print("Initializing conformation table "
                                                   "from given z's")
                            self.model.encoder.initialize(utils.load_pkl(
                                self.configs.initial_conf))

                    self.optimized_modules.append('conf_table')

            will_make_summary = (
                    (self.configs.log_heavy_interval
                     and epoch % self.configs.log_heavy_interval == 0)
                    or self.is_in_pose_search_step or self.pretraining
                    or epoch == self.num_epochs - 1
            )
            self.log_latents = will_make_summary

            if will_make_summary:
                self.accelerator.print(
                    "Will make a full summary at the end of this epoch")

            for key in self.run_times.keys():
                self.run_times[key] = []

            end_time = time.time()
            self.cur_loss = 0
            torch.cuda.empty_cache()
            # inner loop
            for batch_idx, in_dict in enumerate(data_generator):
                self.batch_idx = batch_idx
                # print('1 batch_idx:', batch_idx)
                # with torch.autograd.detect_anomaly():
                if 'conf_table' in self.optimizers and 'conf_table' in self.optimized_modules:
                    conf_table_lr = adjust_learning_rate(self.optimizers['conf_table'],
                                                         epoch=self.epoch + batch_idx / len(data_generator),
                                                         # epochs=self.num_epochs,
                                                         lr=self.configs.lr_conf_table,
                                                         warmup_epochs_down=self.configs.warm_up_epochs,
                                                         min_lr=self.configs.min_lr_conf_table)
                else:
                    conf_table_lr = 0

                if 'conf_encoder' in self.optimizers and 'conf_encoder' in self.optimized_modules:
                    conf_encoder_lr = adjust_learning_rate(self.optimizers['conf_encoder'],
                                                           epoch=self.epoch + batch_idx / len(data_generator),
                                                           # epochs=self.num_epochs,
                                                           warmup_epochs_down=self.num_epochs,
                                                           lr=self.configs.lr_conf_encoder,
                                                           warmup_epochs_up=self.configs.warm_up_epochs_encoder,
                                                           min_lr=self.configs.min_lr_encoder)
                else:
                    conf_encoder_lr = 0



                if self.configs.use_pfm_encoder and self.configs.use_fused_encoder and self.configs.use_conf_encoder:
                    self.model.encoder.update_fuse_only_table(epoch=self.epoch + batch_idx / len(
                        data_generator)) if self.accelerator.num_processes == 1 else self.model.encoder.module.update_fuse_only_table(
                        epoch=self.epoch + batch_idx / len(data_generator))
                self.train_step(in_dict, end_time=end_time, accelerator=self.accelerator)

                if self.configs.verbose_time:
                    torch.cuda.synchronize()

                end_time = time.time()
                ind_epoch.extend(self.accelerator.gather_for_metrics(in_dict['index_p']).cpu().numpy().tolist())
                self.accelerator.wait_for_everyone()
                if self.current_epoch_particles_count > n_max_particles:
                    break

            total_loss = self.cur_loss / n_max_particles

            self.accelerator.print(
                f"# =====> SGD Epoch: {self.epoch} "
                f"finished in {dt.now() - te}; "
                f"total loss = {format(total_loss, '.6f')}"
                f" conf_table_lr: {conf_table_lr} "
                f"conf_encoder_lr: {conf_encoder_lr} "
            )

            # image and pose summary
            self.accelerator.wait_for_everyone()
            # if self.accelerator.is_main_process and will_make_summary and not self.pretraining:
            if self.accelerator.is_main_process:
                self.writer.add_scalar("Data Loss (epoch)", total_loss, self.epoch)
                self.writer.add_scalar("Learning Rate Conf-table (epoch)", conf_table_lr, self.epoch)
                self.writer.add_scalar("Learning Rate Conf-encoder (epoch)", conf_encoder_lr, self.epoch)
                if will_make_summary:
                    with torch.no_grad():
                        if self.clustering_tool_moe is not None:
                            if epoch >= self.configs.clustering_start_epoch:
                                self.clustering_tool_moe.update_current_inds(current_inds=ind_epoch)
                                _, _ = self.clustering_tool_moe.clustering(self.predicted_conf)
                                class_num = self.clustering_tool_moe.get_class_num()
                                self.accelerator.print(f"(Clustering) Epoch: {self.epoch} class_num: {class_num}")
                                self.accelerator.print(
                                    f"(Clustering) Epoch: {self.epoch} labels change ratio (moe): {self.clustering_tool_moe.labels_change_ratio}")
                                self.writer.add_scalar("labels change ratio (moe)",
                                                       self.clustering_tool_moe.labels_change_ratio,
                                                       self.epoch)
                            else:
                                self.clustering_tool_moe.update_labels()


                        if self.clustering_tool_evaluate is not None:
                            self.clustering_tool_evaluate.update_current_inds(current_inds=ind_epoch)
                            _, _ = self.clustering_tool_evaluate.clustering(self.predicted_conf)
                            class_num_evaluate = self.clustering_tool_evaluate.get_class_num()

                            if self.clustering_tool_evaluate.labels_true is not None:
                                acc, nmi, ari, ami = self.clustering_tool_evaluate.get_clustering_acc()
                                knn5_top1, knn5_top5,knn10_top1,knn10_top5, knn10_top10 = self.clustering_tool_evaluate.get_knn(self.predicted_conf,
                                                                                   device=self.accelerator.device,
                                                                                   sample_ratio=1000)
                                self.accelerator.print(f"(Clustering) Epoch: {self.epoch} knn5 top1: {knn5_top1} knn5 top5: {knn5_top5} knn10 top1: {knn10_top1} knn10 top5: {knn10_top5} knn10 top10: {knn10_top10}")
                                self.accelerator.print(
                                    f"(Clustering) Epoch: {self.epoch} clustering acc: {acc} nmi: {nmi}")
                                self.accelerator.print(f"(Clustering) Epoch: {self.epoch} ari: {ari} ami: {ami}")
                                self.writer.add_scalar("(Clustering) clustering acc", acc, self.epoch)
                                self.writer.add_scalar("(Clustering) clustering nmi", nmi, self.epoch)
                                self.writer.add_scalar("(Clustering) clustering ari", ari, self.epoch)
                                self.writer.add_scalar("(Clustering) clustering ami", ami, self.epoch)
                                self.writer.add_scalar("(Clustering) knn5 top1", knn5_top1, self.epoch)
                                self.writer.add_scalar("(Clustering) knn5 top5", knn5_top5, self.epoch)
                                self.writer.add_scalar("(Clustering) knn10 top1", knn10_top1, self.epoch)
                                self.writer.add_scalar("(Clustering) knn10 top5", knn10_top5, self.epoch)
                                self.writer.add_scalar("(Clustering) knn10 top10", knn10_top10, self.epoch)


                            self.accelerator.print(
                                f"(Clustering) Epoch: {self.epoch} labels change ratio: {self.clustering_tool_evaluate.labels_change_ratio}")
                            self.accelerator.print(
                                f"(Clustering) Epoch: {self.epoch} class_num_evaluate: {class_num_evaluate}")
                            self.writer.add_scalar("labels change ratio",
                                                   self.clustering_tool_evaluate.labels_change_ratio,
                                                   self.epoch)
                            self.clustering_tool_evaluate.generate_cs_from_labels(
                                os.path.join(self.outdir, 'Clustering_Epoch_' + str(self.epoch)))
                        self.make_heavy_summary()
                        self.save_latents()
                        self.save_volume()
                        self.save_model()
                        self.save_ind_epoch(ind_epoch)

            self.accelerator.wait_for_everyone()
            if self.clustering_tool_moe is not None:
                self.clustering_tool_moe.labels = self.clustering_tool_moe.labels.to(self.accelerator.device)
                self.clustering_tool_moe.labels = broadcast(self.clustering_tool_moe.labels, from_process=0)
            # print('device:'+str( self.accelerator.device)+', labels: '+str(self.clustering_tool_moe.labels[0:10]))

            # update output mask -- epoch-based scaling
            if (hasattr(self.output_mask, 'update_epoch')
                    and self.use_point_estimates):
                self.output_mask.update_epoch(
                    self.configs.n_frequencies_per_epoch)
            self.accelerator.print(f"\n")

        t_total = dt.now() - t_0
        self.accelerator.print(
            f"Finished in {t_total} ({t_total / self.num_epochs} per epoch)")

    def get_ctfs_at(self, index):
        batch_size = len(index)
        index_i = index
        ctf_params_local = (self.ctf_params[index_i]
                            if self.ctf_params is not None else None)

        if ctf_params_local is not None:
            freqs = self.lattice.freqs2d.unsqueeze(0).expand(
                batch_size, *self.lattice.freqs2d.shape) / ctf_params_local[:, 0].view(batch_size, 1, 1)

            ctf_local = ctf.compute_ctf(
                freqs, *torch.split(ctf_params_local[:, 1:], 1, 1)).view(
                batch_size, self.resolution, self.resolution)

        else:
            ctf_local = None

        return ctf_local

    def train_step(self, in_dict, end_time, accelerator):
        if self.configs.verbose_time:
            torch.cuda.synchronize()
            self.run_times['dataloading'].append(time.time() - end_time)

        # update output mask -- image-based scaling
        if hasattr(self.output_mask, 'update') and self.is_in_pose_search_step:
            self.output_mask.update(self.total_particles_count)

        if self.is_in_pose_search_step:
            self.model.ps_params['l_min'] = self.configs.l_start

            if self.configs.output_mask == 'circ':
                self.model.ps_params['l_max'] = self.configs.l_end
            else:
                self.model.ps_params['l_max'] = min(
                    self.output_mask.current_radius, self.configs.l_end)

        y_gt = in_dict['y']
        ind = in_dict['index_p']

        if not 'tilt_index' in in_dict:
            # in_dict['tilt_index'] = in_dict['index']
            in_dict['tilt_index'] = in_dict['index_p']
        else:
            in_dict['tilt_index'] = in_dict['tilt_index'].reshape(-1)

        ind_tilt = in_dict['tilt_index']
        self.total_batch_count += 1
        batch_size = len(y_gt) * self.accelerator.num_processes
        self.total_particles_count += batch_size
        self.current_epoch_particles_count += batch_size

        # move to gpu
        if self.configs.verbose_time:
            torch.cuda.synchronize()
        start_time_gpu = time.time()

        if self.configs.verbose_time:
            torch.cuda.synchronize()
            self.run_times['to_gpu'].append(time.time() - start_time_gpu)

        # zero grad
        for key in self.optimized_modules:
            self.optimizers[key].zero_grad()

        # forward pass
        latent_variables_dict, y_pred, y_gt_processed = self.forward_pass(
            in_dict)

        self.model.is_in_pose_search_step = False

        # loss
        if self.configs.verbose_time:
            torch.cuda.synchronize()

        start_time_loss = time.time()
        # with torch.autocast(dtype=torch.float16,device_type='cuda', enabled=self.configs.use_amp):
        total_loss, all_losses = self.loss(y_pred, y_gt_processed,
                                           latent_variables_dict)

        if self.configs.verbose_time:
            torch.cuda.synchronize()
            self.run_times['loss'].append(time.time() - start_time_loss)

        # backward pass
        if self.configs.verbose_time:
            torch.cuda.synchronize()
        start_time_backward = time.time()

        accelerator.backward(total_loss)
        self.cur_loss += total_loss.item() * len(ind) * self.accelerator.num_processes

        for key in self.optimized_modules:
            if self.optimizer_types[key] == 'adam' or self.optimizer_types[key] == 'adamw':
                self.optimizers[key].step()

            elif self.optimizer_types[key] == 'lbfgs':
                def closure():
                    self.optimizers[key].zero_grad()
                    _latent_variables_dict, _y_pred, _y_gt_processed = self.forward_pass(in_dict)
                    _loss, _ = self.loss(
                        _y_pred, _y_gt_processed, _latent_variables_dict
                    )
                    _loss.backward()
                    return _loss.item()

                self.optimizers[key].step(closure)

            else:
                raise NotImplementedError

        if self.configs.verbose_time:
            torch.cuda.synchronize()

            self.run_times['backward'].append(
                time.time() - start_time_backward)

        # detach
        if self.log_latents:
            self.in_dict_last = in_dict
            self.y_pred_last = y_pred

            if self.configs.verbose_time:
                torch.cuda.synchronize()

            start_time_cpu = time.time()
            rot_pred, trans_pred, conf_pred, conf_table_pred, logvar_pred = self.detach_latent_variables(
                latent_variables_dict)

            if self.configs.verbose_time:
                torch.cuda.synchronize()
                self.run_times['to_cpu'].append(time.time() - start_time_cpu)

            # log
            if self.use_cuda:
                ind = self.accelerator.gather_for_metrics(ind).cpu()
                ind_tilt = self.accelerator.gather_for_metrics(ind_tilt).cpu()

            self.mask_particles_seen_at_last_epoch[ind] = 1
            self.mask_tilts_seen_at_last_epoch[ind_tilt] = 1
            self.predicted_rots[ind_tilt] = rot_pred.reshape(-1, 3, 3)

            if not self.configs.no_trans:
                self.predicted_trans[ind_tilt] = trans_pred.reshape(-1, 2)

            if self.configs.z_dim > 0:
                self.predicted_conf[ind] = conf_pred

                if self.configs.variational_het:
                    self.predicted_logvar[ind] = logvar_pred
            if conf_table_pred is not None:
                self.predicted_conf_table[ind] = conf_table_pred
        else:
            self.run_times['to_cpu'].append(0.0)

        # scalar summary
        # if self.total_particles_count % self.configs.log_interval < batch_size:
        if self.accelerator.is_main_process and self.total_particles_count % self.configs.log_interval < batch_size:
            self.make_light_summary(all_losses)
        self.accelerator.wait_for_everyone()

    def detach_latent_variables(self, latent_variables_dict):
        rot_pred = self.accelerator.gather_for_metrics(latent_variables_dict['R']).detach().cpu().numpy()
        trans_pred = (self.accelerator.gather_for_metrics(latent_variables_dict['t']).detach().cpu().numpy()
                      if not self.configs.no_trans else None)

        conf_pred = (self.accelerator.gather_for_metrics(latent_variables_dict['z']).detach().cpu().numpy()
                     if self.configs.z_dim > 0 and 'z' in latent_variables_dict
                     else None)

        logvar_pred = (self.accelerator.gather_for_metrics(latent_variables_dict['z_logvar']).detach().cpu().numpy()
                       if self.configs.z_dim > 0
                          and 'z_logvar' in latent_variables_dict
                       else None)

        conf_table_pred = (self.accelerator.gather_for_metrics(latent_variables_dict['z_table']).detach().cpu().numpy()
                           if self.configs.use_fused_encoder and self.configs.use_pfm_encoder and self.configs.use_conf_encoder
                              and 'z_table' in latent_variables_dict
                           else None)

        return rot_pred, trans_pred, conf_pred, conf_table_pred, logvar_pred

    def forward_pass(self, in_dict):
        if self.configs.verbose_time:
            # torch.cuda.synchronize()
            self.accelerator.wait_for_everyone()

        start_time_ctf = time.time()
        # ctf_local = self.get_ctfs_at(in_dict['tilt_index'])
        ctf_local = self.get_ctfs_at(in_dict['tilt_index'])

        if self.configs.subtomogram_averaging:
            ctf_local = ctf_local.reshape(
                -1, self.configs.n_tilts, *ctf_local.shape[1:])

        if self.configs.verbose_time:
            # torch.cuda.synchronize()
            self.accelerator.wait_for_everyone()
            self.run_times['ctf'].append(time.time() - start_time_ctf)

        # forward pass
        if 'hypervolume' in self.optimized_modules:
            self.hypervolume.train()
        else:
            self.hypervolume.eval()

        if hasattr(self.model, 'encoder'):

            if 'conf_encoder' in self.optimized_modules or 'encoder' in self.optimized_modules:
                # self.model.encoder.train() if self.accelerator.num_processes==1 else self.model.module.encoder.train()
                self.model.encoder.train()
                # self.model.conf_regressor.train()
            else:
                # self.model.encoder.eval() if self.accelerator.num_processes==1 else self.model.module.encoder.eval()
                self.model.encoder.eval()
                # self.model.conf_regressor.eval()
        if hasattr(self.model,
                   'conftable') and self.configs.use_fused_encoder and self.configs.use_pfm_encoder and self.configs.use_conf_encoder:
            if ('conf_table' in self.optimized_modules) and (self.configs.use_fused_encoder or  self.model.trun_on_conf_table):
                self.model.conftable.train()
            else:
                self.model.conftable.eval()


        if hasattr(self, 'pose_table') and self.pose_table is not None:
            if 'pose_table' in self.optimized_modules:
                self.pose_table.train()
            else:
                self.pose_table.eval()


        in_dict["ctf"] = ctf_local

        self.model.pose_only = self.pose_only
        self.model.use_point_estimates = self.use_point_estimates
        self.model.pretrain = self.pretraining
        self.model.is_in_pose_search_step = self.is_in_pose_search_step
        self.model.use_point_estimates_conf = (
            not self.configs.use_conf_encoder)

        if self.configs.subtomogram_averaging:
            in_dict['tilt_index'] = in_dict['tilt_index'].reshape(
                *in_dict['y'].shape[0:2])

        if self.configs.decoder_type == 'moe':
            route_labels = self.clustering_tool_moe.labels.to(in_dict['index_p'].device)[in_dict['index_p']]
            in_dict['route_labels'] = route_labels
        else:
            in_dict['route_labels'] = None

        out_dict = self.model(in_dict, accelerator=self.accelerator, hypervolume=self.hypervolume,
                              pose_table=self.pose_table)

        self.run_times['encoder'].append(
            torch.mean(out_dict['time_encoder'].cpu())
            if self.configs.verbose_time else 0.
        )

        self.run_times['decoder'].append(
            torch.mean(out_dict['time_decoder'].cpu())
            if self.configs.verbose_time else 0.
        )

        self.run_times['decoder_coords'].append(
            torch.mean(out_dict['time_decoder_coords'].cpu())
            if self.configs.verbose_time else 0.
        )

        self.run_times['decoder_query'].append(
            torch.mean(out_dict['time_decoder_query'].cpu())
            if self.configs.verbose_time else 0.
        )

        latent_variables_dict = out_dict
        y_pred = out_dict['y_pred']
        y_gt_processed = out_dict['y_gt_processed']

        if (self.configs.subtomogram_averaging
                and self.configs.dose_exposure_correction):
            mask = self.output_mask.binary_mask
            a_pix = self.ctf_params[0, 0]

            dose_filters = self.data.get_dose_filters(
                in_dict['tilt_index'].reshape(-1),
                self.lattice,
                a_pix
            ).reshape(*y_pred.shape[:2], -1)

            y_pred *= dose_filters[..., mask]

        return latent_variables_dict, y_pred, y_gt_processed

    def loss(self, y_pred, y_gt, latent_variables_dict):
        """
        y_pred: [batch_size(, n_tilts), n_pts]
        y_gt: [batch_size(, n_tilts), n_pts]
        """
        all_losses = {}

        # data loss
        data_loss = F.mse_loss(y_pred, y_gt)
        all_losses['Data Loss'] = data_loss.item()
        total_loss = data_loss

        # KL divergence
        if self.use_kl_divergence:
            kld_conf = kl_divergence_conf(latent_variables_dict)
            total_loss += self.configs.beta_conf * kld_conf / self.resolution ** 2
            all_losses['KL Div. Conf.'] = kld_conf.item()

        # L1 regularization for translations
        if self.use_trans_l1_regularizer and self.use_point_estimates:
            trans_l1_loss = l1_regularizer(latent_variables_dict['t'])
            total_loss += self.configs.trans_l1_regularizer * trans_l1_loss
            all_losses['L1 Reg. Trans.'] = trans_l1_loss.item()

        # L2 smoothness prior
        if self.use_l2_smoothness_regularizer:
            smoothness_loss = l2_frequency_bias(y_pred, self.lattice.freqs2d,
                                                self.output_mask.binary_mask,
                                                self.resolution)
            total_loss += self.configs.l2_smoothness_regularizer * smoothness_loss
            all_losses['L2 Smoothness Loss'] = smoothness_loss.item()

        return total_loss, all_losses

    def make_heavy_summary(self):
        summary.make_img_summary(self.writer, self.in_dict_last,
                                 self.y_pred_last, self.output_mask,
                                 self.epoch)

        # conformation
        pca = None
        if self.configs.z_dim > 0:
            labels = None

            if self.configs.labels is not None:
                labels = utils.load_pkl(self.configs.labels)

                if self.index is not None:
                    labels = labels[self.index]

            if self.mask_particles_seen_at_last_epoch is not None:
                mask_idx = self.mask_particles_seen_at_last_epoch > 0.5
            else:
                mask_idx = np.ones((self.n_particles_dataset,), dtype=bool)

            predicted_conf = self.predicted_conf[mask_idx]
            labels = labels[mask_idx] if labels is not None else None
            logvar = (self.predicted_logvar[mask_idx]
                      if self.predicted_logvar is not None else None)

            pca = summary.make_conf_summary(
                self.writer, predicted_conf, self.epoch, labels,
                pca=None, logvar=logvar,
                palette_type=self.configs.color_palette
            )
            if self.configs.use_pfm_encoder and self.configs.use_fused_encoder and self.configs.use_conf_encoder:
                predicted_conf_table = self.predicted_conf_table[mask_idx]
                pca_conf_table = summary.make_conf_summary(
                    self.writer, predicted_conf_table, self.epoch, labels,
                    pca=None, logvar=logvar,
                    prefix='(conf_table) ',
                    palette_type=self.configs.color_palette
                )

        # pose
        rotmat_gt = None
        trans_gt = None
        shift = (not self.configs.no_trans)

        if self.mask_particles_seen_at_last_epoch is not None:
            mask_tilt_idx = self.mask_tilts_seen_at_last_epoch > 0.5
        else:
            mask_tilt_idx = np.ones((self.n_tilts_dataset,), dtype=bool)

        if self.configs.pose is not None:
            # poses_gt = utils.load_pkl(self.configs.pose)

            pose_path = os.path.join(self.configs.pose, "pose_list.data")
            trans_path = os.path.join(self.configs.pose, "shift_list.data")
            rotmat_gt=None
            trans_gt = None

            if os.path.exists(pose_path):
                rotmat_gt=torch.tensor(utils.load_pkl(pose_path)).float()
                if self.index is not None:
                    rotmat_gt = rotmat_gt[self.index]
            if os.path.exists(trans_path):
                trans_gt=torch.tensor(utils.load_pkl(trans_path)).float()* self.resolution
                if self.index is not None:
                    trans_gt = trans_gt[self.index]


            rotmat_gt = rotmat_gt[mask_tilt_idx]
            trans_gt = (trans_gt[mask_tilt_idx] if trans_gt is not None
                        else None)

        predicted_rots = self.predicted_rots[mask_tilt_idx]
        predicted_trans = (self.predicted_trans[mask_tilt_idx]
                           if self.predicted_trans is not None else None)

        summary.make_pose_summary(self.writer, predicted_rots, predicted_trans,
                                  rotmat_gt, trans_gt, self.epoch, shift=shift)

        return pca

    def make_light_summary(self, all_losses):
        self.accelerator.print(
            f"# [Train Epoch: {self.epoch}/{self.num_epochs - 1}] "
            f"[{self.current_epoch_particles_count}"
            # f"/{self.n_particles_dataset} particles]"
            f"/{self.n_max_particles} particles]"
        )

        if hasattr(self.output_mask, 'current_radius'):
            all_losses['Mask Radius'] = self.output_mask.current_radius

        # trans_search_factor=self.model.trans_search_factor if self.accelerator.num_processes==1 else self.model.module.trans_search_factor
        trans_search_factor = self.model.trans_search_factor
        if trans_search_factor is not None:
            # all_losses['Trans. Search Factor'] = self.model.trans_search_factor
            all_losses['Trans. Search Factor'] = trans_search_factor

        summary.make_scalar_summary(self.writer, all_losses,
                                    self.total_particles_count)

        if self.configs.verbose_time:
            for key in self.run_times.keys():
                self.accelerator.print(
                    f"{key} time: {np.mean(np.array(self.run_times[key]))}")

    def save_ind_epoch(self, ind_epoch):
        """Write current epoch's indices to file."""
        out_ind = os.path.join(self.outdir, f"ind_epoch.{self.epoch}.pkl")
        with open(out_ind, 'wb') as f:
            pickle.dump(ind_epoch, f)

        # self.accelerator.print(f"Saved indices for epoch {self.epoch} to {out_ind}")

    def save_latents(self):
        """Write model's latent variables to file."""
        out_pose = os.path.join(self.outdir, f"pose.{self.epoch}.pkl")

        if self.configs.no_trans:
            with open(out_pose, 'wb') as f:
                pickle.dump(self.predicted_rots, f)
        else:
            with open(out_pose, 'wb') as f:
                pickle.dump((self.predicted_rots, self.predicted_trans), f)

        if self.configs.z_dim > 0:
            out_conf = os.path.join(self.outdir, f"conf.{self.epoch}.pkl")
            with open(out_conf, 'wb') as f:
                pickle.dump(self.predicted_conf, f)

    def save_volume(self):
        """Write reconstructed volume to file."""
        out_mrc = os.path.join(self.outdir, f"reconstruct.{self.epoch}.mrc")

        self.hypervolume.eval()
        if hasattr(self.model, 'conf_cnn'):
            if hasattr(self.model, 'conf_regressor'):
                self.model.conf_cnn.eval()
                self.model.conf_regressor.eval()
        if hasattr(self.model, 'conf_linear'):
            if hasattr(self.model, 'conf_regressor'):
                self.model.conf_linear.eval()
                self.model.conf_regressor.eval()

        if hasattr(self.model, 'encoder'):
            # if hasattr(self.model, 'conf_regressor'):
            self.model.encoder.eval()
            # self.model.conf_regressor.eval()

        if hasattr(self, 'pose_table') and self.pose_table is not None:
            self.pose_table.eval()

        if hasattr(self.model, 'conf_table'):
            self.model.conftable.eval()

        if self.configs.z_dim > 0:
            zval = self.predicted_conf[0].reshape(-1)
        else:
            zval = None
        vol = -1. * self.model.eval_volume(self.data.norm, zval=zval, hypervolume=self.hypervolume,
                                           route_labels=self.clustering_tool_moe.labels[0].unsqueeze(
                                               -1) if self.clustering_tool_moe is not None else None) \
            # if self.accelerator.num_processes==1 else -1. * self.model.module.eval_volume(self.data.norm, zval=zval,hypervolume=self.hypervolume,route_labels=self.clustering_tool_moe.labels[0].unsqueeze(-1) if self.clustering_tool_moe is not None else None)
        mrc.write(out_mrc, vol.astype(np.float32))

    # TODO: weights -> model and reconstruct -> volume for output labels?
    def save_model(self):
        """Write model state to file."""
        out_weights = os.path.join(self.outdir, f"weights.{self.epoch}.pkl")

        optimizers_state_dict = {}
        for key in self.optimizers.keys():
            optimizers_state_dict[key] = self.optimizers[key].state_dict()

        saved_objects = {
            'epoch': self.epoch,

            'model_state_dict': (self.model.encoder.state_dict()
                                 if self.accelerator.num_processes == 1 else self.model.encoder.module.state_dict()
                                 ) if self.configs.use_conf_encoder else self.model.state_dict(),

            'hypervolume_state_dict': (
                self.hypervolume.state_dict() if self.accelerator.num_processes == 1 else self.hypervolume.module.state_dict()
            ),

            'hypervolume_params': self.hypervolume.get_building_params() if self.accelerator.num_processes == 1 else self.hypervolume.module.get_building_params(),
            'optimizers_state_dict': optimizers_state_dict,
            'current_centers': self.clustering_tool_moe.raw_centers_norm if self.clustering_tool_moe is not None else None,
        }

        if hasattr(self.output_mask, 'current_radius'):
            saved_objects[
                'output_mask_radius'] = self.output_mask.current_radius

        torch.save(saved_objects, out_weights)
