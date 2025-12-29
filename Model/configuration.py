
import os
import sys
import yaml
import numpy as np
import inspect
from collections import OrderedDict
from abc import ABC
from dataclasses import dataclass, field, fields, MISSING, asdict, Field
from typing import Any, ClassVar
import difflib


@dataclass
class _BaseConfigurations(ABC):
    """Base class for sets of model configuration parameters."""

    # a parameter belongs to this set if and only if it has a default value
    # defined in this set of data class attributes
    # note that ordering makes e.g. printing easier for user viewing
    verbose: int = 0
    seed: int = None
    quick_config: OrderedDict = field(default_factory=OrderedDict)
    test_installation: bool = False

    quick_configs: ClassVar[dict[str, dict[str, Any]]] = OrderedDict()

    def __post_init__(self) -> None:
        """Checking and parsing given parameter values.

        This is a special method used in data classes for processing the given field
        values. Here, and across our config children classes, we use it to verify that
        the parameter values specified by the user are valid, as well as doing any
        necessary parsing of these values.

        """
        for this_field in fields(self):
            assert (
                this_field.name == "quick_config" or this_field.default is not MISSING
            ), (
                f"`{self.__class__.__name__}` class has no default value defined "
                f"for parameter `{this_field.name}`"
            )

        # special parameter used to test whether the package was installed correctly
        if self.test_installation:
            print("Installation was successful!")
            sys.exit()

        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ValueError(
                f"Given verbosity `{self.verbose}` is not a positive integer!"
            )

        # if the user didn't pick a random seed we will pick one for them
        # to ensure reproducibility
        if self.seed is None:
            self.seed = np.random.randint(0, 10000)

        if not isinstance(self.seed, int):
            raise ValueError(
                "Configuration `seed` must be given as an integer, "
                f"given `{self.seed}` instead!"
            )

        # process the quick_config parameter
        if self.quick_config is not MISSING:
            for cfg_k, cfg_val in self.quick_config.items():
                if cfg_k not in self.quick_configs:
                    raise ValueError(
                        f"Unrecognized quick_config shortcut field `{cfg_k}`!")

                if cfg_val not in self.quick_configs[cfg_k]:
                    raise ValueError(
                        f"Unrecognized quick_config shortcut value `{cfg_val}` "
                        f"for field `{cfg_k}`, "
                        f"choose from: {','.join(list(self.quick_configs[cfg_k]))}"
                    )

                for par_k, par_value in self.quick_configs[cfg_k][cfg_val].items():
                    if par_k not in self:
                        raise ValueError(
                            f"Unrecognized configuration parameter `{par_k}` found "
                            f"in this classes quick_config entry `{cfg_k}:{cfg_val}`!"
                        )

                    # parameters given elsewhere in configs have priority
                    if getattr(self, par_k) == getattr(type(self), par_k):
                        setattr(self, par_k, par_value)

    def __iter__(self):
        return iter(asdict(self).items())

    def __str__(self):
        return "\n".join([f"{par}{str(val):>20}" for par, val in self])

    def __contains__(self, val) -> bool:
        return val in {k for k, _ in self}

    def write(self, fl: str, **add_cfgs) -> None:
        """Saving configurations to file using the original order."""
        with open(fl, "w") as f:
            yaml.dump(dict(training=asdict(self), **add_cfgs), f,
                      default_flow_style=False, sort_keys=False)

    @classmethod
    def fields(cls) -> list[Field]:
        """Returning all fields defined for this class without needing an instance.

        The default Python dataclass `fields` method does not have a counterpart for
        classes, which we need in cases like `parse_cfg_keys()` which we want to call
        without using an instance of the data class!

        """
        members = inspect.getmembers(cls)
        return list(list(filter(
            lambda x: x[0] == '__dataclass_fields__', members))[0][1].values())

    @classmethod
    def parse_cfg_keys(cls, cfg_keys: list[str]) -> dict[str, Any]:
        cfgs = dict()

        for cfg_str in cfg_keys:
            if cfg_str.count("=") != 1:
                raise ValueError("--cfgs entries must have exactly one equals sign "
                                 "and be in the form 'CFG_KEY=CFG_VAL'!")
            cfg_key, cfg_val = cfg_str.split('=')

            if cfg_val is None or cfg_val == 'None':
                cfgs[cfg_key] = None

            else:
                for fld in cls.fields():
                    if cfg_key == fld.name:
                        if fld.type is str:
                            cfgs[cfg_key] = str(cfg_val)
                        else:
                            cfgs[cfg_key] = fld.type(eval(cfg_val))

                        # accounting for parameters like `ind` which can be paths
                        # to files as well as integers
                        if isinstance(cfgs[cfg_key], str) and cfgs[cfg_key].isnumeric():
                            cfgs[cfg_key] = int(cfgs[cfg_key])

                        break

                else:
                    close_keys = difflib.get_close_matches(
                        cfg_key, [fld.name for fld in cls.fields()])

                    if close_keys:
                        close_str = f"\nDid you mean one of:\n{', '.join(close_keys)}"
                    else:
                        close_str = ""

                    raise ValueError(f"--cfgs parameter `{cfg_key}` is not a "
                                     f"valid configuration parameter!{close_str}")

        return cfgs


@dataclass
class TrainingConfigurations(_BaseConfigurations):
    # input datasets
    particles: str = None
    ctf: str = None
    pose: str = None
    dataset: str = None
    server: str = None
    datadir: str = None
    ind: str = None
    labels: str = None
    relion31: bool = False
    no_trans: bool = False
    labels_evaluate: str = None

    # pretrain with processed data
    processed_data: str = None
    shuffle_type: str='all'
    resample_per_dataset: int= 10000

    # initialization
    pose_D: int=None
    use_gt_poses: bool = False
    refine_gt_poses: bool = False
    use_gt_trans: bool = False
    load: str = None
    initial_conf: str = None

    # logging and verbosity
    log_interval: int = 10000
    log_heavy_interval: int = 5
    verbose_time: bool = False

    #data loading
    shuffle: bool = True
    # shuffle: bool = False
    lazy: bool = False
    num_workers: int = 2
    max_threads: int = 16
    # fast_dataloading: bool = False
    shuffler_size: int = 32768
    batch_size_known_poses: int = 32
    batch_size_hps: int = 8
    batch_size_sgd: int = 256
    use_lmdb: bool = True
    score_bar:float=0.0

    # optimizers
    hypervolume_optimizer_type: str = "adam"
    pose_table_optimizer_type: str = "adam"
    conf_table_optimizer_type: str = "adam"
    conf_encoder_optimizer_type: str = "adamw"
    lr: float = 1.0e-4
    lr_pose_table: float = 1.0e-3
    lr_conf_table: float = 1.0e-2
    min_lr_conf_table: float = 1.0e-5
    warm_up_epochs: int = 5
    lr_conf_encoder: float = 1.0e-4
    min_lr_encoder: float = 1.0e-5
    warm_up_epochs_encoder: int = 1


    wd: float = 0.0
    use_amp: bool=False
    gradient_checkpointing: bool=False

    # scheduling
    n_imgs_pose_search: int = 500000
    epochs_sgd: int = 100
    pose_only_phase: int = 0
    epochs_init_conf_table: int=-2

    # masking
    output_mask: str = "circ"
    add_one_frequency_every: int = 100000
    n_frequencies_per_epoch: int = 10
    max_freq: int = None
    window_radius_gt_real: float = 0.85
    l_start_fm: int = 12

    # loss
    beta_conf: float = 0.0
    trans_l1_regularizer: float = 0.0
    l2_smoothness_regularizer: float = 0.0

    # conformations
    variational_het: bool = False
    z_dim: int = 4
    feature_dim: int = 4
    conf_table_dim: int = 4
    std_z_init: float = 0.1

    use_conf_encoder: bool = False
    depth_cnn: int = 5
    channels_cnn: int = 32
    kernel_size_cnn: int = 3
    resolution_encoder: str = None

    # cryo PFM
    # finetune_vit_encoder: bool = False
    use_pfm_encoder: bool = False

    finetune_strategy: str=''
    finetune_layer_num: int=1
    pretrained_model_path: str=''
    encoder_type: str='vit_small'
    use_fused_encoder: bool = False
    fuse_type: str = 'concat'
    fuse_only_table: float=0.0
    min_fuse_only_table: float=0.0
    feature_take_indices: list=None
    feature_fuse_indices: list = None

    # use_pfm_features: bool = False
    # use_generated_features: bool = False

    # hypervolume
    decoder_type: str = "mlp",
    # decoder_type: str = "moe",
    explicit_volume: bool = False
    hypervolume_layers: int = 8
    hypervolume_dim: int = 512
    decoder_ln: bool = True

    pe_type: str = "gaussian"
    pe_dim: int = 32
    feat_sigma: float = 0.5
    hypervolume_domain: str = "hartley"
    pe_type_conf: str = None
    num_shared_experts: int = 1
    k_init : int=32
    clustering_dim: int=4
    cluster_num_evaluate: int=0
    use_clustering_route: bool = False

    # pre-training
    n_imgs_pretrain: int = 10000
    pretrain_with_gt_poses: bool = False

    # pose search
    l_start: int = 12
    l_end: int = 32
    n_iter: int = 4
    t_extent: float = 20.0
    t_n_grid: int = 7
    t_x_shift: float = 0.0
    t_y_shift: float = 0.0
    no_trans_search_at_pose_search: bool = False
    n_kept_poses: int = 8
    base_healpy: int = 2

    # subtomogram averaging
    subtomogram_averaging: bool = False
    n_tilts: int = 11
    dose_per_tilt: float = None
    angle_per_tilt: float = None
    n_tilts_pose_search: int = None
    average_over_tilts: bool = False
    tilt_axis_angle: float = 0.0
    dose_exposure_correction: bool = True

    # others
    color_palette: str = None
    test_installation: bool = False

    # clustering
    moe_num: int = 2
    clustering_start_epoch: int = 0
    clustering_type: str = 'hierarchical'

    # Data Augmentation for Encoder Input
    augmentation_settings = OrderedDict(
        {
            'resize': 224,
            'random_circle_crop': {'p': 1.0, 'scale': [0.85,0.85]},
        }
    )

    quick_configs = OrderedDict(
        {
            'capture_setup': {
                'spa': dict(),
                'et': {'subtomogram_averaging': True,
                       # 'fast_dataloading': True,
                       'shuffler_size': 0, 'num_workers': 0, 't_extent': 0.0,
                       'batch_size_known_poses': 8, 'batch_size_sgd': 32,
                       'n_imgs_pose_search': 150000, 'pose_only_phase': 50000,
                       'lr_pose_table': 1.0e-5}
                },

            'reconstruction_type': {
                'homo': {'z_dim': 0}, 'het': dict()},

            'pose_estimation': {
                'abinit': dict(),
                'refine': {'refine_gt_poses': True, 'pretrain_with_gt_poses': True,
                           'lr_pose_table': 1.0e-4},
                'fixed': {'use_gt_poses': True, 'n_imgs_pose_search': 0}
                },

            'conf_estimation': {
                None: dict(),
                'autodecoder': dict(),
                'refine': dict(),
                'encoder': {'use_conf_encoder': True}
                }
            }
        )

    def __post_init__(self):
        super().__post_init__()

        if self.explicit_volume and self.z_dim >= 1:
            raise ValueError("Explicit volumes do not support "
                             "heterogeneous reconstruction.")

        if self.dataset is None and self.processed_data is None:
            if self.particles is None:
                raise ValueError("As dataset was not specified, please "
                                 "specify particles!")

            if self.ctf is None:
                raise ValueError("As dataset was not specified, please "
                                 "specify ctf!")

        if self.hypervolume_optimizer_type not in {'adam','adamw'}:
            raise ValueError("Invalid value "
                             f"`{self.hypervolume_optimizer_type}` "
                             "for hypervolume_optimizer_type!")

        if self.pose_table_optimizer_type not in {'adam', 'lbfgs','adamw'}:
            raise ValueError("Invalid value "
                             f"`{self.pose_table_optimizer_type}` "
                             "for pose_table_optimizer_type!")

        if self.conf_table_optimizer_type not in {'adam', 'lbfgs'}:
            raise ValueError("Invalid value "
                             f"`{self.conf_table_optimizer_type}` "
                             "for conf_table_optimizer_type!")

        if self.conf_encoder_optimizer_type not in {'adam','adamw'}:
            raise ValueError("Invalid value "
                             f"`{self.conf_encoder_optimizer_type}` "
                             "for conf_encoder_optimizer_type!")

        if self.output_mask not in {'circ', 'frequency_marching'}:
            raise ValueError("Invalid value "
                             f"{self.output_mask} for output_mask!")

        if self.pe_type not in {'gaussian'}:
            raise ValueError(f"Invalid value {self.pe_type} for pe_type!")

        if self.pe_type_conf not in {None, 'geom'}:
            raise ValueError(f"Invalid value {self.pe_type_conf} "
                             "for pe_type_conf!")

        if self.hypervolume_domain not in {'hartley'}:
            raise ValueError(f"Invalid value {self.hypervolume_domain} "
                             "for hypervolume_domain.")

        if self.n_imgs_pose_search < 0:
            raise ValueError("n_imgs_pose_search must be greater than 0!")

        if self.use_conf_encoder and self.initial_conf:
            raise ValueError("Conformations cannot be initialized "
                             "when using an encoder!")

        if self.use_gt_trans and self.pose is None and self.processed_data is None:
            raise ValueError("Poses must be specified to use GT translations!")

        if self.refine_gt_poses:
            self.n_imgs_pose_search = 0

            if self.pose is None:
                raise ValueError("Initial poses must be specified "
                                 "to be refined!")

        # subtomogram averaging is not supported yet in cryoDECO
        if self.subtomogram_averaging:
            # self.fast_dataloading = True

            # TODO: Implement conformation encoder for subtomogram averaging.
            if self.use_conf_encoder:
                raise ValueError("Conformation encoder is not implemented "
                                 "for subtomogram averaging!")

            # TODO: Implement translation search for subtomogram averaging.
            if not (self.use_gt_poses or self.use_gt_trans
                    or self.t_extent == 0.):
                raise ValueError("Translation search is not implemented "
                                 "for subtomogram averaging!")

            if self.average_over_tilts and self.n_tilts_pose_search % 2 == 0:
                raise ValueError("n_tilts_pose_search must be odd "
                                 "to use average_over_tilts!")

            if self.n_tilts_pose_search is None:
                self.n_tilts_pose_search = self.n_tilts
            if self.n_tilts_pose_search > self.n_tilts:
                raise ValueError("n_tilts_pose_search must be "
                                 "smaller than n_tilts!")

        if self.use_gt_poses:
            # "poses" include translations
            self.use_gt_trans = True

            if self.pose is None and self.processed_data is None:
                raise ValueError("Ground truth poses must be specified!")

        if self.no_trans:
            self.t_extent = 0.
        if self.t_extent == 0.:
            self.t_n_grid = 1

        if self.dataset:
            with open(os.environ.get("cryoDECO_DATASETS"), 'r') as f:
                paths = yaml.safe_load(f)

            self.particles = paths[self.dataset]['particles']
            self.ctf = paths[self.dataset]['ctf']

            if self.pose is None and 'pose' in paths[self.dataset]:
                self.pose = paths[self.dataset]['pose']
            if 'datadir' in paths[self.dataset]:
                self.datadir = paths[self.dataset]['datadir']
            if 'labels' in paths[self.dataset]:
                self.labels = paths[self.dataset]['labels']
            if self.ind is None and 'ind' in paths[self.dataset]:
                self.ind = paths[self.dataset]['ind']
            if 'dose_per_tilt' in paths[self.dataset]:
                self.dose_per_tilt = paths[self.dataset]['dose_per_tilt']


@dataclass
class AnalysisConfigurations(_BaseConfigurations):

    epoch: int = -1
    skip_umap: bool = False
    save_features: bool = False
    pc: int = 2
    n_per_pc: int = 10
    k_num: int = 20
    clustering_type: str='k-means'
    invert: bool = True
    sample_z_idx: int = None
    trajectory_1d: int = None
    direct_traversal_txt: str = None
    z_values_txt: str = None
    cs_dir_path: str = None
    data_resample: int=None
    k_init: int=32
    umap_dim: int=4
