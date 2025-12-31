
from reconstruct import ModelTrainer
import argparse
import os
from Model.configuration import AnalysisConfigurations
from analyze import ModelAnalyzer
from argparse import ArgumentParser
from cryodata.data_preprocess.mrc_preprocess import raw_csdata_process_from_cryosparc_dir, sort_csdata
from Data import parse_ctf_csparc, parse_pose_csparc
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import torch
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def settings():

    settings = {
        'outdir': None,
        'load': False,

        'train_settings': {

            'particles': None,
            'processed_data': None,
            'ctf': None,

            'labels_evaluate': None,
            'use_lmdb': True,
            'shuffle': True,
            'use_gt_poses': False,
            'use_gt_trans': False,
            'pose_D': None,

            'datadir': None,

            'ind': None,
            'pose': None,

            'score_bar': 0.8,
            'n_imgs_pretrain': 10000,
            'n_imgs_pose_search': 500000,
            'epochs_sgd': 100,
            'epochs_init_conf_table': -2,

            'use_conf_encoder': True,
            'use_pfm_encoder': True,
            'use_fused_encoder': False,
            'fuse_type': 'gate',
            'fuse_only_table': 0.35,
            'min_fuse_only_table': 0.0,
            'feature_take_indices': None,
            'feature_fuse_indices': None,

            'encoder_type': 'vit_small',
            'finetune_strategy': 'all',
            'finetune_layer_num': 6, # for vit_block finetune
            'pretrained_model_path': None,
            'resolution_encoder': None,
            'feature_dim': 128,
            'conf_table_dim': 4,

            'seed': 1701,
            'conf_encoder_optimizer_type': 'adamw',
            'hypervolume_optimizer_type': 'adamw',

            # 'lr_conf_encoder': 1.0e-6,
            'lr_conf_encoder': 6.0e-5,
            'min_lr_encoder': 1.0e-5,
            'warm_up_epochs_encoder': 1,

            'lr': 1.0e-4,
            'lr_pose_table': 1.0e-4,
            'lr_conf_table': 1.0e-6,
            'warm_up_epochs': 30,
            'min_lr_conf_table': 0.0,

            'use_amp': True,
            'gradient_checkpointing': True,
            'batch_size_hps':22,
            'batch_size_known_poses': 64,
            'batch_size_sgd': 192,
            'lazy': False,

            # hypervolume
            'hypervolume_layers': 3,
            'hypervolume_dim': 256,
            'decoder_type': 'mlp',
            'decoder_ln': True,
            'moe_num': 4,
            'pe_dim': 64,
            'num_shared_experts': 1,
            'k_init': 48,
            'clustering_dim': 16,
            'cluster_num_evaluate': 0,
            'clustering_type': 'k-means++',
            'use_clustering_route': False,

            # pretrain with processed data
            'shuffle_type': 'all',
            'resample_per_dataset': None,

            'log_heavy_interval': 5,
            'quick_config': {'capture_setup': 'spa',
                             'conf_estimation': 'autodecoder',
                             'pose_estimation': 'abinit',
                             'reconstruction_type': 'het'}
        },

        'analysis': True,
        'analysis_settings': {
            'skip_train': False,
            'epoch': -1,
            'save_features': False,
            'k_num': 8,
            'k_init': 64,
            'umap_dim':16,
            'clustering_type': 'gmm',
            'cs_dir_path': None,
            'data_resample': None
        }

    }
    return settings


def main():

    '''get config'''
    parser = ArgumentParser()
    parser.add_argument('--outdir', default=None, type=str)
    parser.add_argument('--particles', default=None, type=str)
    parser.add_argument('--use_lmdb', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--pose', default=None, type=str)
    parser.add_argument('--ctf', default=None, type=str)

    parser.add_argument('--processed_data', default=None, type=str)

    parser.add_argument('--score_bar', default=None, type=float)
    parser.add_argument('--datadir', default=None, type=str)
    parser.add_argument('--feature_dim', default=None, type=int)
    parser.add_argument('--conf_table_dim', default=None, type=int)

    parser.add_argument('--ind', default=None, type=str)
    parser.add_argument('--epochs_sgd', default=None, type=int)
    parser.add_argument('--epochs_init_conf_table', default=None, type=int)
    parser.add_argument('--batch_size_sgd', default=None, type=int)
    parser.add_argument('--batch_size_known_poses', default=None, type=int)
    parser.add_argument('--batch_size_hps', default=None, type=int)
    parser.add_argument('--n_imgs_pose_search', default=None, type=int)
    parser.add_argument('--n_imgs_pretrain', default=None, type=int)
    parser.add_argument('--resample_per_dataset', default=None)
    parser.add_argument('--use_gt_poses', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--use_gt_trans', type=lambda x: x.lower() == 'true', default=None)

    parser.add_argument('--resolution_encoder', default=None)
    parser.add_argument('--use_conf_encoder', type=lambda x: x.lower() == 'true', default=None)

    parser.add_argument('--lazy', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--load', type=lambda x: x.lower() == 'true', default=None)
    # parser.add_argument('--finetune_vit_encoder', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--use_generated_features', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--skip_train', type=lambda x: x.lower() == 'true', default=None)

    parser.add_argument('--hypervolume_layers', default=None, type=int)
    parser.add_argument('--hypervolume_dim', default=None, type=int)
    parser.add_argument('--decoder_type', default=None, type=str)
    parser.add_argument('--use_clustering_route', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--decoder_ln', type=lambda x: x.lower() == 'true', default=None)
    # parser.add_argument('--attn_sample_num', default=None, type=int)

    parser.add_argument('--use_pfm_encoder', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--use_fused_encoder', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--encoder_type', default=None, type=str)
    parser.add_argument('--finetune_strategy', default=None, type=str)
    parser.add_argument('--finetune_layer_num', default=None, type=int)
    parser.add_argument('--pretrained_model_path', default=None, type=str)
    parser.add_argument('--conf_encoder_optimizer_type', default=None, type=str)

    parser.add_argument('--gradient_checkpointing', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--use_amp', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--lr_conf_encoder', type=float, default=None)
    parser.add_argument('--lr_conf_table', type=float, default=None)
    parser.add_argument('--lr_pose_table', type=float, default=None)
    parser.add_argument('--min_lr_encoder', type=float, default=None)
    parser.add_argument('--warm_up_epochs_encoder', type=int, default=None)
    parser.add_argument('--min_lr_conf_table', type=float, default=None)
    parser.add_argument('--warm_up_epochs', type=int, default=None)
    parser.add_argument('--hypervolume_optimizer_type', type=str, default=None)

    parser.add_argument('--analysis_save_features', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--labels_evaluate', default=None, type=str)
    parser.add_argument('--moe_num', default=None, type=int)
    parser.add_argument('--num_shared_experts', default=None, type=int)
    parser.add_argument('--k_init', default=None, type=int)
    parser.add_argument('--k_num', default=None, type=int)
    parser.add_argument('--clustering_dim', default=None, type=int)
    parser.add_argument('--cluster_num_evaluate', default=None, type=int)
    parser.add_argument('--clustering_type', default=None, type=str)
    parser.add_argument('--fuse_type', default=None, type=str)
    parser.add_argument('--fuse_only_table', type=float, default=None)
    parser.add_argument('--min_fuse_only_table', type=float, default=None)

    args = parser.parse_args()
    configs = settings()
    if args.outdir is not None:
        configs['outdir'] = args.outdir
    if args.particles is not None:
        if args.particles.lower() == 'none' or args.particles.lower() == 'null':
            configs['train_settings']['particles'] = None
        else:
            configs['train_settings']['particles'] = args.particles

    if args.use_lmdb is not None:
        configs['train_settings']['use_lmdb'] = args.use_lmdb

    if args.ctf is not None:
        if args.ctf.lower() == 'none' or args.ctf.lower() == 'null':
            configs['train_settings']['ctf'] = None
        else:
            configs['train_settings']['ctf'] = args.ctf

    if args.pose is not None:
        if args.pose.lower() == 'none' or args.pose.lower() == 'null':
            configs['train_settings']['pose'] = None
        else:
            configs['train_settings']['pose'] = args.pose

    if args.ind is not None:
        if args.ind.lower() == 'none' or args.ind.lower() == 'null':
            configs['train_settings']['ind'] = None
        else:
            configs['train_settings']['ind'] = args.ind
    if args.n_imgs_pose_search is not None:
        configs['train_settings']['n_imgs_pose_search'] = args.n_imgs_pose_search

    if args.processed_data is not None:
        if args.processed_data.lower() == 'none' or args.processed_data.lower() == 'null':
            configs['train_settings']['processed_data'] = None
        else:
            configs['train_settings']['processed_data'] = args.processed_data

    if args.n_imgs_pretrain is not None:
        configs['train_settings']['n_imgs_pretrain'] = args.n_imgs_pretrain

    if args.resample_per_dataset is not None:
        if args.resample_per_dataset.lower() == 'none' or args.resample_per_dataset.lower() == 'null':
            configs['train_settings']['resample_per_dataset'] = None
        else:
            configs['train_settings']['resample_per_dataset'] = int(args.resample_per_dataset)

    if args.use_gt_poses is not None:
        configs['train_settings']['use_gt_poses'] = args.use_gt_poses

    if args.use_gt_trans is not None:
        configs['train_settings']['use_gt_trans'] = args.use_gt_trans

    if args.epochs_sgd is not None:
        configs['train_settings']['epochs_sgd'] = args.epochs_sgd
    if args.resolution_encoder is not None:
        if args.resolution_encoder.lower() == 'none' or args.resolution_encoder.lower() == 'null':
            configs['train_settings']['resolution_encoder'] = None
        else:
            configs['train_settings']['resolution_encoder'] = args.resolution_encoder
    if args.use_conf_encoder is not None:
        configs['train_settings']['use_conf_encoder'] = args.use_conf_encoder
    if args.use_pfm_encoder is not None:
        configs['train_settings']['use_pfm_encoder'] = args.use_pfm_encoder
    if args.encoder_type is not None:
        configs['train_settings']['encoder_type'] = args.encoder_type
    if args.finetune_strategy is not None:
        configs['train_settings']['finetune_strategy'] = args.finetune_strategy
    if args.finetune_layer_num is not None:
        configs['train_settings']['finetune_layer_num'] = args.finetune_layer_num
    if args.pretrained_model_path is not None:
        configs['train_settings']['pretrained_model_path'] = args.pretrained_model_path

    if args.gradient_checkpointing is not None:
        configs['train_settings']['gradient_checkpointing'] = args.gradient_checkpointing

    if args.use_amp is not None:
        configs['train_settings']['use_amp'] = args.use_amp

    if args.feature_dim is not None:
        configs['train_settings']['feature_dim'] = args.feature_dim

    if args.conf_table_dim is not None:
        configs['train_settings']['conf_table_dim'] = args.conf_table_dim

    if args.analysis_save_features is not None:
        configs['analysis_settings']['save_features'] = args.analysis_save_features
    if args.batch_size_hps is not None:
        configs['train_settings']['batch_size_hps'] = args.batch_size_hps
    if args.batch_size_known_poses is not None:
        configs['train_settings']['batch_size_known_poses'] = args.batch_size_known_poses
    if args.batch_size_sgd is not None:
        configs['train_settings']['batch_size_sgd'] = args.batch_size_sgd

    if args.decoder_type is not None:
        configs['train_settings']['decoder_type'] = args.decoder_type
    if args.hypervolume_layers is not None:
        configs['train_settings']['hypervolume_layers'] = args.hypervolume_layers
    if args.hypervolume_dim is not None:
        configs['train_settings']['hypervolume_dim'] = args.hypervolume_dim
    if args.decoder_ln is not None:
        configs['train_settings']['decoder_ln'] = args.decoder_ln

    if args.skip_train is not None:
        configs['analysis_settings']['skip_train'] = args.skip_train
    if args.datadir is not None:
        if args.datadir.lower() == 'none' or args.datadir.lower() == 'null':
            configs['train_settings']['datadir'] = None
        else:
            configs['train_settings']['datadir'] = args.datadir

    if args.lr_conf_encoder is not None:
        configs['train_settings']['lr_conf_encoder'] = args.lr_conf_encoder
    if args.lr_conf_table is not None:
        configs['train_settings']['lr_conf_table'] = args.lr_conf_table
    if args.lr_pose_table is not None:
        configs['train_settings']['lr_pose_table'] = args.lr_pose_table
    if args.min_lr_encoder is not None:
        configs['train_settings']['min_lr_encoder'] = args.min_lr_encoder
    if args.warm_up_epochs_encoder is not None:
        configs['train_settings']['warm_up_epochs_encoder'] = args.warm_up_epochs_encoder
    if args.min_lr_conf_table is not None:
        configs['train_settings']['min_lr_conf_table'] = args.min_lr_conf_table
    if args.warm_up_epochs is not None:
        configs['train_settings']['warm_up_epochs'] = args.warm_up_epochs

    if args.conf_encoder_optimizer_type is not None:
        configs['train_settings']['conf_encoder_optimizer_type'] = args.conf_encoder_optimizer_type
    if args.lazy is not None:
        configs['train_settings']['lazy'] = args.lazy
    if args.load is not None:
        if not args.load:
            configs['load'] = None
        else:
            configs['load'] = args.load
    if args.labels_evaluate is not None:
        if args.labels_evaluate.lower() == 'none' or args.labels_evaluate.lower() == 'null':
            configs['train_settings']['labels_evaluate'] = None
        else:
            configs['train_settings']['labels_evaluate'] = args.labels_evaluate
    if args.moe_num is not None:
        configs['train_settings']['moe_num'] = args.moe_num
    if args.num_shared_experts is not None:
        configs['train_settings']['num_shared_experts'] = args.num_shared_experts
    if args.k_init is not None:
        configs['train_settings']['k_init'] = args.k_init

    if args.k_num is not None:
        configs['analysis_settings']['k_num'] = args.k_num
    if args.clustering_dim is not None:
        configs['train_settings']['clustering_dim'] = args.clustering_dim
    if args.cluster_num_evaluate is not None:
        if args.cluster_num_evaluate.lower() == 'none' or args.cluster_num_evaluate.lower() == 'null':
            configs['train_settings']['cluster_num_evaluate'] = None
        else:
            configs['train_settings']['cluster_num_evaluate'] = args.cluster_num_evaluate

    if args.clustering_type is not None:
        # configs['train_settings']['clustering_type'] = args.clustering_type
        configs['analysis_settings']['clustering_type'] = args.clustering_type

    if args.score_bar is not None:
        configs['train_settings']['score_bar'] = args.score_bar

    if args.use_fused_encoder is not None:
        configs['train_settings']['use_fused_encoder'] = args.use_fused_encoder

    if args.fuse_type is not None:
        configs['train_settings']['fuse_type'] = args.fuse_type

    if args.fuse_only_table is not None:
        configs['train_settings']['fuse_only_table'] = args.fuse_only_table

    if args.min_fuse_only_table is not None:
        configs['train_settings']['min_fuse_only_table'] = args.min_fuse_only_table

    if args.hypervolume_optimizer_type is not None:
        configs['train_settings']['hypervolume_optimizer_type'] = args.hypervolume_optimizer_type

    if args.epochs_init_conf_table is not None:
        configs['train_settings']['epochs_init_conf_table'] = args.epochs_init_conf_table

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=96000)),
                                               # DistributedDataParallelKwargs(find_unused_parameters=True)
                                               ]
                              )

    if "LOCAL_RANK" in os.environ:
        CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
        torch.distributed.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])

    if (configs['train_settings']['processed_data']  is None or not os.path.exists(configs['train_settings']['processed_data'])) and configs['train_settings']['use_lmdb']:
        configs['train_settings']['processed_data'] = os.path.join(configs['outdir'], 'tmp', 'preprocessed_data')
    if configs['train_settings']['particles'] is not None and os.path.isdir(configs['train_settings']['particles']):
        if accelerator.is_main_process:
            cs_data, _ = raw_csdata_process_from_cryosparc_dir(configs['train_settings']['particles'])
            sort_csdata(cs_data, save_path=os.path.join(
                configs['train_settings']['particles'], 'new_particles.cs'))
        accelerator.wait_for_everyone()
        configs['train_settings']['particles'] = os.path.join(
            configs['train_settings']['particles'], 'new_particles.cs')
    if configs['train_settings']['ctf'] is None or not os.path.exists(configs['train_settings']['ctf']):
        if configs['train_settings']['processed_data'] is not None and os.path.exists(os.path.join(configs['train_settings']['processed_data'],'ctf.pkl')):
            configs['train_settings']['ctf'] = os.path.join(configs['train_settings']['processed_data'], 'ctf.pkl')
        else:
            ctf_pkl_out = os.path.join(configs['train_settings']['processed_data'], "ctf.pkl")
            ctf_png_out = os.path.join(configs['train_settings']['processed_data'], "ctf.png")
            if not configs['train_settings']['particles'].endswith('.cs'):
                configs['train_settings']['particles'] = os.path.join(
                    os.path.dirname(configs['train_settings']['particles']),
                    'new_particles.cs')
            if accelerator.is_main_process:
                if not os.path.exists(os.path.dirname(ctf_pkl_out)):
                    os.makedirs(os.path.dirname(ctf_pkl_out))
                args = parse_ctf_csparc.add_args(argparse.ArgumentParser()).parse_args(
                    [os.path.join(configs['train_settings']['particles']), "-o", ctf_pkl_out, "--png", ctf_png_out]
                )
                parse_ctf_csparc.main(args)
            accelerator.wait_for_everyone()
            configs['train_settings']['ctf'] = ctf_pkl_out

    if configs['train_settings']['use_pfm_encoder'] and configs['train_settings']['use_conf_encoder']:

        if configs['train_settings']['use_fused_encoder']:
            if configs['train_settings']['fuse_type'] == 'gate':
                configs['train_settings']['z_dim'] = configs['train_settings']['feature_dim']
            elif configs['train_settings']['fuse_type'] == 'concat':
                configs['train_settings']['z_dim'] = configs['train_settings']['feature_dim'] + \
                                                     configs['train_settings']['conf_table_dim']
            else:
                raise ValueError("fuse_type must be 'gate' or 'concat'")
        else:
            configs['train_settings']['z_dim'] = configs['train_settings']['feature_dim']
    else:
        if configs['train_settings']['use_conf_encoder']:
            configs['train_settings']['z_dim'] = configs['train_settings']['feature_dim']
        else:
            configs['train_settings']['z_dim'] = configs['train_settings']['conf_table_dim']

    if configs['train_settings']['pose'] is not None:
        configs['train_settings']['use_gt_poses'] = True
        configs['train_settings']['use_gt_trans'] = True
    if ((configs['train_settings']['use_gt_poses'] == True or configs['train_settings']['use_gt_trans'] == True) and configs['train_settings']['pose'] is None):
        pose_pkl_out = configs['train_settings']['processed_data']
        if accelerator.is_main_process:

            if not configs['train_settings']['particles'].endswith('.cs'):
                configs['train_settings']['particles'] = os.path.join(
                    os.path.dirname(configs['train_settings']['particles']),
                    'new_particles.cs')
            if configs['train_settings']['pose_D'] is None:
                configs['train_settings']['pose_D'] = cs_data['blob/shape'].tolist()[0][0]
            if not os.path.exists(os.path.dirname(pose_pkl_out)):
                os.makedirs(os.path.dirname(pose_pkl_out))
            args = parse_pose_csparc.add_args(argparse.ArgumentParser()).parse_args(
                [configs['train_settings']['particles'], "-o", pose_pkl_out, "-D",
                 str(configs['train_settings']['pose_D'])]
            )
            parse_pose_csparc.main(args)
        accelerator.wait_for_everyone()
        configs['train_settings']['pose'] = pose_pkl_out


    configs['train_settings']['lr'] = configs['train_settings']['lr'] * accelerator.num_processes
    configs['train_settings']['lr_pose_table'] = configs['train_settings'][
                                                     'lr_pose_table'] * accelerator.num_processes
    configs['train_settings']['lr_conf_table'] = configs['train_settings'][
                                                     'lr_conf_table'] * accelerator.num_processes
    configs['train_settings']['lr_conf_encoder'] = configs['train_settings'][
                                                       'lr_conf_encoder'] * accelerator.num_processes
    configs['train_settings']['min_lr_conf_table'] = configs['train_settings']['min_lr_conf_table'] * accelerator.num_processes
    configs['train_settings']['min_lr_encoder'] = configs['train_settings'][
                                                      'min_lr_encoder'] * accelerator.num_processes
    accelerator.print(json.dumps(configs, indent=4))
    trainer = ModelTrainer(configs['outdir'], configs['train_settings'],
                           load=configs['load'] if not configs['analysis_settings']['skip_train'] else True,
                           accelerator=accelerator)
    if not configs['analysis_settings']['skip_train']:
        trainer.train()

    if configs['analysis'] and accelerator.is_main_process:
        accelerator.print("\n--- Analysis Starts Now ---")
        train_configs = ModelTrainer.load_configs(os.path.join(configs['outdir'], 'out'))
        anlz_args = argparse.Namespace(**configs['analysis_settings'])
        anlz_configs = {
            fld.name: (getattr(anlz_args, fld.name) if hasattr(anlz_args, fld.name) else fld.default)
            for fld in AnalysisConfigurations.fields() if fld.name != 'quick_configs'
        }
        analyzer = ModelAnalyzer(os.path.join(configs['outdir'], 'out'), anlz_configs, train_configs,
                                 encoder=trainer.model.encoder if configs['train_settings'][
                                     'use_conf_encoder'] else None, dataset=trainer.data)
        analyzer.analyze()

    accelerator.wait_for_everyone()
    accelerator.state.destroy_process_group()


if __name__ == '__main__':
    main()
