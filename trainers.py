import collections
import inspect
from tqdm.auto import tqdm
from torch.distributed import get_rank
import math
import optuna
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from torch.distributed.fsdp import FullyShardedDataParallel as ShardedDDP
import importlib.util
from packaging import version
import torch.nn.functional as F
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.utils import logging
# from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from .regularizers import istar 
from .ValidationConstrativeLoss import ValidationContrastiveLoss,CrossDeviceContrastiveLoss,DistributedIsoRegular
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import *
from transformers.integrations import (
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)
if is_sagemaker_mp_enabled():# type: ignore
    import smdistributed.modelparallel.torch as smp  # type: ignore
    from smdistributed.modelparallel import __version__ as SMP_VERSION  # type: ignore

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch
skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm # type: ignore
    import torch_xla.debug.metrics as met  # type: ignore
    import torch_xla.distributed.parallel_loader as pl # type: ignore

if is_apex_available():
    from apex import amp # type: ignore

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

# from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy

# import downstream_tasks

import numpy as np
from io import *
import psutil
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
from sklearn.decomposition import PCA, KernelPCA
# from .evaluate_down_stream import run_evaluation

logger = logging.get_logger(__name__)

global uniform_lls
uniform_lls = [] 
embs = []

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

CLUSTERERS = [
    "KMeans",
    "AffinityPropagation",
    "AgglomerativeClustering",
    "SpectralClustering",
    "HDBSCAN",
    "OPTICS",
    "LouvainCommunities",
]

METRICS = [
    "arccos",  # Manually implemented as unit length norm + euclidean distance
    "braycurtis",  # Like L1, but weights the result
    "canberra",  # Like L1, but weights dimensions by their magnitude
    "chebyshev",  # L-infinity
    "cityblock",  # L1
    "cosine",  # Supported by AgglomerativeClustering and OPTICS
    "euclidean",  # L2
    "infinity",
    "l1",
    "l2",
    "mahalanobis",  # Must provide either V or VI in ``metric_params``.
    "manhattan",  # L1
    "minkowski",  # Lp norm, Must provide a p value in ``p`` or ``metric_params``.
    "p",
    "seuclidean",  # Weighted L2. Needs an argument ``V`` with variances per dim.
]




class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp






class CLTrainer(Trainer):
    # def __init__(self, *args,V=None,precomputed_scores=None, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.V=V
    #     self.precomputed_scores = precomputed_scores
    
    def load_embeddings(self, fname,partitions=None, modalities=None):
        """
        Load cached embeddings.
        """
        # if partitions is None:
        #     partitions = config.partition
        # if modalities is None:
        #     modalities = getattr(config, "modality", "image")

        if not isinstance(partitions, str):
            embeddings = []
            y_true = []
            for partition in partitions:
                embeddings_i, y_true_i = self.load_embeddings(self, partition,modalities)
                embeddings.append(embeddings_i)
                y_true.append(y_true_i)
            # Concatenate the embeddings from each partition in the sample dimension
            embeddings = np.concatenate(embeddings, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            return embeddings, y_true

        partition = partitions

        if not isinstance(modalities, str):
            embeddings = []
            y_true = []
            for modality in modalities:
                embd_i, y_true_i = self.load_embeddings(self, partition, modality)
                embeddings.append(embd_i)
                y_true.append(y_true_i)
            # Concatenate the embeddings from each modality in the feature dimension
            embeddings = np.concatenate(embeddings, axis=1)
            for i in range(1, len(y_true)):
                assert np.all(y_true[i] == y_true[0]), "Mismatch in y_true labels"
            y_true = y_true[0]
            return embeddings, y_true

        modality = modalities

        # if config.model == "raw":
        #     print("Using raw image pixel data instead of model embedding.", flush=True)
        #     from torch import nn

        #     import zs_ssl_clustering.embed

        #     _config = copy.deepcopy(config)
        #     _config.modality = modality
        #     dataloader = zs_ssl_clustering.embed.make_dataloader(_config)
        #     return zs_ssl_clustering.embed.embed_dataset(dataloader, nn.Flatten(), "cpu")

        # fname = self.get_embeddings_path(config, partition, modality)
        print(f"Loading encoder embeddings from {fname}", flush=True)
        # Only need allow_pickle=True if we're using the saved config dict
        data = np.load(fname)
        embeddings = data["embeddings"]
        y_true = data["y_true"]

        # if config.prenorm == "none":
        #     pass

        # elif config.prenorm == "l2":
        #     # L2 normalize embeddings
        #     print("Using L2-normalized embeddings")
        #     embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        # elif config.prenorm == "l2_corrected":
        #     # L2 normalize embeddings, scaled by sqrt(d) where d is the dimensionality
        #     print("Using L2-normalized embeddings")
        #     embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        #     embeddings *= np.sqrt(embeddings.shape[1])

        # elif config.prenorm == "elementwise_zscore":
        #     # Standardize to zero mean, unit variance
        #     print("Using elementwise z-scored embeddings")
        #     embeddings -= np.mean(embeddings, axis=0)
        #     embeddings /= np.std(embeddings, axis=0)

        # elif config.prenorm == "average_zscore":
        #     # Fit clusterer on average z-scored embeddings
        #     print("Using average-z-scored embeddings")
        #     # Standardize to zero mean, AVERAGE of unit variance (a spherical scaling which
        #     # scales all distances equally, without altering importance of any dimensions)
        #     embeddings -= np.mean(embeddings, axis=0)
        #     embeddings /= np.mean(np.std(embeddings, axis=0))

        # else:
        #     raise ValueError(f"Unknown prenorm option: {config.prenorm}")

        return embeddings, y_true
    
    def get_pred_path(self,patrition=None):    
        
        """
        Generate path to y_pred file under output_dir/pred.
        """
        # 基础目录
        base_dir = os.path.join(self.args.output_dir, "pred")
        
        # 获取数据集和模型名称
        dataset_name = getattr(self.args, "dataset_name", "Bioscan-5M-zsc")
        model_name = getattr(self.args, "model_dna", "dnacse")  # 根据模态调整
        
        # 时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # 构造文件名
        fname = f"{patrition}__{dataset_name}__{model_name}__{timestamp}.npz"
        
        # 完整路径
        fname = os.path.join(base_dir, fname)
        # fname = self.sanitize_filename(fname)
        return fname
       
        
    def cluster(self,fname,patrition):
        start_all = time.time()
        ndim_reduced_man=50
        dim_reducer_man_nn = 30
        
        # Seed RNG state for reproducibility
        # seed=0
        # random.seed(seed)
        # np.random.seed(seed % 0xFFFF_FFFF)
        save_pred = True
            
            

        # if config.save_pred is None:
        #     config.save_pred = any("test" in p for p in config.partition)

    

        # Handle arccos distance metric by normalizing the vectors ourself and
        # passing euclidean to the clusterer, since it doesn't support arccos directly.
        _distance_metric = "euclidean"
        # _distance_metric = "euclidean" if _distance_metric == "arccos" else _distance_metric

        _distance_metric_man = "euclidean"
        # _distance_metric_man = (
        #     "euclidean" if _distance_metric_man == "arccos" else _distance_metric_man
        # )

     

        start_loading = time.time()
        # Load the embeddings of each modality and partition
        embeddings, y_true = self.load_embeddings(fname,partitions="test",modalities="dna")
        print(f"y_true shape: {y_true.shape}")
        print(f"Finished loading data in {time.time() - start_loading}", flush=True)

        n_clusters_gt = len(np.unique(y_true))
        print(f"Found {n_clusters_gt} ground truth clusters")
        encoding_dim = embeddings.shape[-1]

        og_embeddings = copy.deepcopy(embeddings)

        clusterer_args_used = set()
        results = {}

        start_reducing = time.time()
        
        # if ndim_reduced_man is None:
        #     raise ValueError(
        #          "Umap reduction was requested, but 'ndim_reduced_man' was not set."
        #     )

        from umap import UMAP

        _embeddings = embeddings
        

        # if config.dim_reducer_man_nn is None:
        #     # Default value is as per the guide in the UMAP documentation
        #     # https://umap-learn.readthedocs.io/en/latest/clustering.html#umap-enhanced-clustering
        #     config.dim_reducer_man_nn = 30

        start_reduce_man = time.time()
        reducer_man = UMAP(
            n_neighbors=dim_reducer_man_nn,
            n_components=ndim_reduced_man,
            min_dist=0.0,
            metric=_distance_metric_man,
            random_state=self.args.seed,
            n_jobs=-1,  # Only 1 worker used if RNG is manually seeded
            verbose=1>0,
        )
        reducerman_args_used = {"dim_reducer_man_nn", "dim_reducer_man_metric", "seed"}
        print(
            f"Fitting {reducer_man} on data shaped {_embeddings.shape}...", flush=True
        )
        embeddings = reducer_man.fit_transform(_embeddings)
        end_reduce_man = time.time()
        results["time_reduce_man"] = end_reduce_man - start_reduce_man
        print(
            f"UMAP fitting time: {results['time_reduce_man']:.2f}s"
        )

        
        end_reducing = time.time()

        reduced_dim = embeddings.shape[-1]

        clusterer_args = {
            "n_clusters",
            "distance_metric",
            "max_iter",
            "min_samples",
            "max_samples",
            "workers",
            "affinity_damping",
            "affinity_conv_iter",
            "spectral_affinity",
            "spectral_assigner",
            "spectral_n_components",
            "spectral_n_neighbors",
            "aggclust_linkage",
            "aggclust_dist_thresh",
            "hdbscan_method",
            "optics_method",
            "optics_xi",
        }

        
        # clusterer method:"AgglomerativeClustering"
        # Can work with specified number of clusters, as well as unknown (which requires a distance threshold)
        # We can also impose some structure metric through the "connectivity" argument
       
        n_clusters = n_clusters_gt
        clusterer = sklearn.cluster.AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=_distance_metric,
            linkage="ward",
            distance_threshold=None,
        )
        clusterer_args_used = clusterer_args_used.union(
            {
                "n_clusters",
                "distance_metric",
                "aggclust_linkage",
                "aggclust_dist_thresh",
            }
        )

      

        print("Standardizing data...", flush=True)
        zs2_embeddings = None
        azs2_embeddings = None
        nrm_embeddings = None
        zs2_nrm_embeddings = None
        nrm_azs2_embeddings = None

        _embeddings = embeddings

       

        # # Correct for impact of number of dimensions on distance measurements
        # # if not config.ndim_correction or config.distance_metric == ["arccos"]:
        # #     pass

        # # elif config.distance_metric in ["euclidean", "l2", "seuclidean"]:
        # # Correct for L2 distances scaling up like sqrt of number of dimensions
        #     _embeddings = _embeddings / np.sqrt(_embeddings.shape[-1])

        # # # Wipe the state of cluster arguments that were not relevant to the
        # # # chosen clusterer.(Optional)
        # # clusterer_args_unused = clusterer_args.difference(reducerman_args_used)
        # # clusterer_args_unused = clusterer_args_unused.difference(clusterer_args_used)
        # # for key in clusterer_args_unused:
        # #     setattr(config, key, None)
     
        print(
            f"Start fitting clusterer {clusterer} on data shaped {_embeddings.shape}...",
            flush=True,
        )
        start_cluster = time.time()
        clusterer.fit(_embeddings)
        end_cluster = time.time()
        print(f"Finished fitting clusterer in {end_cluster - start_cluster:.1f}s")

        print("Calculating performance metrics...")
        start_metrics = time.time() 
        y_pred = clusterer.labels_
        select_clustered = y_pred >= 0
        n_clusters_pred = len(np.unique(y_pred[select_clustered]))
        ratio_clustered = np.sum(select_clustered) / len(y_pred)
        ratio_unclustered = 1 - ratio_clustered
        _results = {
            "n_samples": len(embeddings),
            "encoding_dim": encoding_dim,
            "reduced_dim": reduced_dim,
            "time_reducing": end_reducing - start_reducing,
            "time_clustering": end_cluster - start_cluster,
            # "y_pred": y_pred,
            "num_cluster_true": n_clusters_gt,
            "num_cluster_pred": n_clusters_pred,
            "ratio_clustered": ratio_clustered,
            "ratio_unclustered": ratio_unclustered,
            "AMI": sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred),
            "ARI": sklearn.metrics.adjusted_rand_score(y_true, y_pred),
            "FMS": sklearn.metrics.fowlkes_mallows_score(y_true, y_pred),
            "completeness": sklearn.metrics.completeness_score(y_true, y_pred),
            "homogeneity": sklearn.metrics.homogeneity_score(y_true, y_pred),
            "CHS_true": sklearn.metrics.calinski_harabasz_score(embeddings, y_true),
            "CHS-fit_true": sklearn.metrics.calinski_harabasz_score(_embeddings, y_true),
            "CHS-og_true": sklearn.metrics.calinski_harabasz_score(og_embeddings, y_true),
            "DBS_true": sklearn.metrics.davies_bouldin_score(embeddings, y_true),
            "DBS-fit_true": sklearn.metrics.davies_bouldin_score(_embeddings, y_true),
            "DBS-og_true": sklearn.metrics.davies_bouldin_score(og_embeddings, y_true),
        }
        results.update(_results)

        if n_clusters_pred > 1 and len(np.unique(y_pred)) < len(embeddings):
            results["CHS_pred"] = sklearn.metrics.calinski_harabasz_score(
                embeddings, y_pred
            )
            results["CHS-fit_pred"] = sklearn.metrics.calinski_harabasz_score(
                _embeddings, y_pred
            )
            results["CHS-og_pred"] = sklearn.metrics.calinski_harabasz_score(
                og_embeddings, y_pred
            )
            results["DBS_pred"] = sklearn.metrics.davies_bouldin_score(embeddings, y_pred)
            results["DBS-fit_pred"] = sklearn.metrics.davies_bouldin_score(
                _embeddings, y_pred
            )
            results["DBS-og_pred"] = sklearn.metrics.davies_bouldin_score(
                og_embeddings, y_pred
            )

        # Repeat metrics, but considering only the samples that were clustered
        if ratio_clustered > 0:
            yct = y_true[select_clustered]
            ycp = y_pred[select_clustered]
            ec = embeddings[select_clustered]
            results["AMI_clus"] = sklearn.metrics.adjusted_mutual_info_score(yct, ycp)
            results["ARI_clus"] = sklearn.metrics.adjusted_rand_score(yct, ycp)
            results["FMS_clus"] = sklearn.metrics.fowlkes_mallows_score(yct, ycp)
            results["completeness_clus"] = sklearn.metrics.completeness_score(yct, ycp)
            results["homogeneity_clus"] = sklearn.metrics.homogeneity_score(yct, ycp)
            if n_clusters_pred > 1 and n_clusters_pred < len(ec):
                results["CHS_pred_clus"] = sklearn.metrics.calinski_harabasz_score(ec, ycp)
                results["CHS-fit_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                    _embeddings[select_clustered], ycp
                )
                results["CHS-og_pred_clus"] = sklearn.metrics.calinski_harabasz_score(
                    og_embeddings[select_clustered], ycp
                )
                results["DBS_pred_clus"] = sklearn.metrics.davies_bouldin_score(ec, ycp)
                results["DBS-fit_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                    _embeddings[select_clustered], ycp
                )
                results["DBS-og_pred_clus"] = sklearn.metrics.davies_bouldin_score(
                    og_embeddings[select_clustered], ycp
                )

        # # Compute silhouette scores with several distance metrics
        # for dm in ["euclidean", "l1", "chebyshev", "arccos"]:
        #     for space_name, embs in [
        #         ("reduced", embeddings),
        #         ("fit", _embeddings),
        #         ("nrm", nrm_embeddings),
        #         ("zs2", zs2_embeddings),
        #         ("azs2", azs2_embeddings),
        #         ("zs2-nrm", zs2_nrm_embeddings),
        #         ("nrm-azs2", nrm_azs2_embeddings),
        #         ("og", og_embeddings),
        #     ]:
        #         if embs is None:
        #             continue
        #         if space_name == "reduced":
        #             prefix = f"silhouette-{dm}"
        #         else:
        #             prefix = f"silhouette-{space_name}-{dm}"
        #         my_dstmtr = dm
        #         my_embs = embs
        #         # N.B. We don't bother with ndim correction here, since it has no impact
        #         # on the silhouette scores.
        #         if dm == "arccos":
        #             my_dstmtr = "euclidean"
        #             my_embs = my_embs / np.linalg.norm(my_embs, axis=1, keepdims=True)

        #         # Compute metrics on ground-truth clusters
        #         try:
        #             results[f"{prefix}_true"] = sklearn.metrics.silhouette_score(
        #                 my_embs, y_true, metric=my_dstmtr
        #             )
        #         except Exception as err:
        #             print(f"Error computing GT silhouette score with {dm}: {err}")

        #         # Compute metrics on ground-truth clusters, but considering only the
        #         # samples that were clustered
        #         if ratio_clustered > 0:
        #             try:
        #                 results[f"{prefix}_true_clus"] = sklearn.metrics.silhouette_score(
        #                     my_embs[select_clustered], yct, metric=my_dstmtr
        #                 )
        #             except Exception as err:
        #                 print(f"Error computing pred silhouette score with {dm}: {err}")

        #         # Compute metrics on predicted clusters
        #         if n_clusters_pred <= 1 or len(np.unique(y_pred)) == len(embeddings):
        #             continue
        #         try:
        #             results[f"{prefix}_pred"] = sklearn.metrics.silhouette_score(
        #                 my_embs, y_pred, metric=my_dstmtr
        #             )
        #         except Exception as err:
        #             print(
        #                 f"Error computing GT (clustered samples only) silhouette score with {dm}: {err}"
        #             )

        #         # Compute metrics on predicted clusters, but considering only the
        #         # samples that were clustered
        #         if ratio_clustered <= 0 or n_clusters_pred >= len(ec):
        #             continue
        #         try:
        #             results[f"{prefix}_pred_clus"] = sklearn.metrics.silhouette_score(
        #                 my_embs[select_clustered], ycp, metric=my_dstmtr
        #             )
        #         except Exception as err:
        #             print(
        #                 f"Error computing pred (clustered samples only) silhouette score with {dm}: {err}"
        #             )

        if hasattr(clusterer, "n_iter_"):
            results["iter"] = clusterer.n_iter_  # Number of iterations run.
            results["converged"] = clusterer.n_iter_ < 1_000

        end_metrics = time.time()
        print(
            f"Finished calculating performance metrics in {end_metrics - start_metrics:.1f}s"
        )
        # dim_reducer_man="UMAP"
        # drsL = ""
        # drsR = ""
        # if dim_reducer is not None and dim_reducer != "None":
        #     drsL = f"{config.dim_reducer}_{config.ndim_reduced})(" + drsL
        #     drsR = drsR + ")"
        # if dim_reducer_man is not None and dim_reducer_man != "None":
        #     drsL = f"{dim_reducer_man}_{ndim_reduced_man}(" + drsL
        #     drsR = drsR + ")"
        # print(
        #     f"\n{config.clusterer_name}({drsL}{config.model}({config.dataset_name}){drsR})"
        #     " evaluation results:"
        # )
        # for k, v in results.items():
        #     if "time" in k:
        #         print(f"  {k + ' ':.<36s} {v:10.4f} seconds")
        #     elif isinstance(k, int):
        #         print(f"  {k + ' ':.<36s} {v:>5d}")
        #     else:
        #         try:
        #             print(f"  {k + ' ':.<36s} {v:10.4f}")
        #         except TypeError:
        #             print(f"  {k + ' ':.<36s} {v}")

        # if config.log_wandb:
        #     print("Logging results to Weights & Biases...")
        #     start_wandb = time.time()
        #     wandb.log(results)
        #     end_wandb = time.time()
        #     print(
        #         f"Finished logging results to Weights & Biases in {end_wandb - start_wandb:.1f}s"
        #     )
        
        if save_pred:
            t1 = time.time()
            fname = self.get_pred_path(patrition=patrition)
            print(f"Saving y_pred to file {fname}")
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            np.savez_compressed(fname, y_pred=y_pred)
            print(f"Saved embeddings in {time.time() - t1:.2f}s")

        print(f"Finished everything in {time.time() - start_all:.1f}s")

        return results    
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
         Add support for streaming data

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__` unless streaming.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        # Handle datasets with streaming enabled
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            if eval_dataset.info.streaming:
                # Streaming datasets do not support `__len__`, nor do they need column removal
                return DataLoader(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    collate_fn=data_collator,
                )

            # Remove unused columns for non-streaming datasets
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        # Handle IterableDataset
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        # Handle regular datasets
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    @torch.no_grad()
    def concat_all_gather_ragged(self,tensor, **kwargs):
        r"""
        Gather a tensor over all processes and concatenate them into one.

        This version supports ragged tensors where the first dimension is not the
        same across all processes.

        Parameters
        ----------
        tensor : torch.Tensor
            The distributed tensor on the current process. The equivalent tensors
            on the other processes may differ in shape only in their first
            dimension.
        group : ProcessGroup, optional
            The process group to work on. If ``None``, the default process group
            will be used.
        async_op : bool, default=False
            Whether this op should be an async op.

        Returns
        -------
        gathered_tensor : torch.Tensor
            The contents of ``tensor`` from every distributed process, gathered
            together. None of the entries support a gradient.

        Warning
        -------
        As with :func:`torch.distributed.all_gather`, this has no gradient.
        """
        print("")
        if not torch.distributed.is_initialized():
           return tensor
        torch.distributed.barrier()
        # print(f"Rank {torch.distributed.get_rank()}: tensor shape = {tensor.shape}")
        tensor = tensor.contiguous()
        world_size = torch.distributed.get_world_size()
        # Gather the lengths of the tensors from all processes
        local_length = torch.tensor(tensor.shape[0], device=tensor.device)
        # print(f"Rank {torch.distributed.get_rank()}: local_length = {local_length}")
        torch.distributed.barrier()  # 确保同步
        all_length = [torch.zeros_like(local_length) for _ in range(world_size)]
        torch.distributed.all_gather(all_length, local_length, **kwargs)
        # print(f"Rank {torch.distributed.get_rank()}: all_length = {[x.item() for x in all_length]}")
        # We will have to pad them to be the size of the longest tensor
        max_length = max(x.item() for x in all_length)

        # Pad our tensor on the current process
        length_diff = max_length - local_length.item()
        if length_diff:
            pad_size = (length_diff, *tensor.shape[1:])
            padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding), dim=0).contiguous()
            
        print(f"Rank {torch.distributed.get_rank()}: padded tensor shape = {tensor.shape}")
        # Gather the padded tensors from all processes
        all_tensors_padded = [torch.zeros_like(tensor).contiguous() for _ in range(world_size)]
        torch.distributed.all_gather(all_tensors_padded, tensor.contiguous(), **kwargs)
        print(f"Rank {torch.distributed.get_rank()}: tensor shape after padding = {tensor.shape}")
        torch.distributed.barrier()
        # Remove padding
        all_tensors = []
        for tensor_i, length_i in zip(all_tensors_padded, all_length):
            all_tensors.append(tensor_i[:length_i])

        # Concatenate the tensors
        output = torch.cat(all_tensors, dim=0)
        return output
    
    
    def embed_dataset(self,dataloader, encoder, device, is_distributed=False, log_interval=20):
        r"""
        Embed a dataset using the given encoder.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The dataloader for the dataset to embed.
        encoder : torch.nn.Module
            The encoder.
        device : torch.device
            Device to run the model on.
        is_distributed : bool, default=False
            Whether the model is distributed across multiple GPUs.
        log_interval : int, default=20
            How often to print progress creating the embeddings.

        Returns
        -------
        embeddings : np.ndarray
            The embeddings for each sample in the dataset.
        y_true : np.ndarray
            The class index for each sample in the dataset.
        """
        encoder.eval()
        embeddings_list = []
        y_true_list = []
        for i_batch, batch in enumerate(dataloader):
            if i_batch % log_interval == 0:
                print(f"Processing batch {i_batch + 1:3d}/{len(dataloader):3d}", flush=True)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            y_true = batch["labels"].to(device)
            # print(f"Batch {i_batch}: y_true shape = {y_true.shape}")
            with torch.no_grad():
                outputs = encoder(**inputs, output_hidden_states=True,return_dict=True,sent_emb=True)
                embd = outputs.pooler_output #if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0].contiguous()
                # print(f"emb嵌入的形状：{embd.shape}")
            if is_distributed:
                torch.distributed.barrier()
                print(f"Rank {torch.distributed.get_rank()}: Before concat_all_gather_ragged")
                # Fetch results from other GPUs
                embd = self.concat_all_gather_ragged(embd)
            if i_batch == 0:
                print(embd.shape)
            embeddings_list.append(embd.cpu().numpy())
            
            y_true_list.append(y_true.cpu().numpy())

        # Concatenate the embeddings and targets from each batch
        embeddings = np.concatenate(embeddings_list)
        y_true = np.concatenate(y_true_list)
        # print(f"Before trim: embeddings shape = {embeddings.shape}, y_true shape = {y_true.shape}")
        # If the dataset size was not evenly divisible by the world size,
        # DistributedSampler will pad the end of the list of samples
        # with some repetitions. We need to trim these off.
        n_samples = len(dataloader.dataset)
        embeddings = embeddings[:n_samples]
        y_true = y_true[:n_samples]
        # print(f"After trim: embeddings shape = {embeddings.shape}, y_true shape = {y_true.shape}")
        return embeddings, y_true

    def sanitize_filename(text, allow_dotfiles=False):
        """
        Sanitise string so it can be a filename.

        Parameters
        ----------
        text : str
            A string.
        allow_dotfiles : bool, optional
            Whether to allow leading periods. Leading periods indicate hidden files
            on Linux. Default is `False`.

        Returns
        -------
        text : str
            The sanitised filename string.
        """
        # Remove non-ascii characters
        text = text.encode("ascii", "ignore").decode("ascii")

        # Folder names cannot end with a period in Windows. In Unix, a leading
        # period means the file or folder is normally hidden.
        # For this reason, we trim away leading and trailing periods as well as
        # spaces.
        if allow_dotfiles:
            text = text.strip().rstrip(".")
        else:
            text = text.strip(" .")

        # On Windows, a filename cannot contain any of the following characters:
        # \ / : * ? " < > |
        # Other operating systems are more permissive.
        # Replace / with a hyphen
        text = text.replace("/", "-")
        # Use a blacklist to remove any remaining forbidden characters
        text = re.sub(r'[\/:*?"<>|]+', "", text)
        return text

    def get_embeddings_path(self, partition="test", modality=None):
        """
        Generate path to embeddings file under output_dir/embedding.
        """
        # 基础目录
        # base_dir = os.path.join(self.args.output_dir, "embedding")
        
        # 获取模态、模型和数据集名称
        modality = modality or getattr(self.args, "modality", "dna")  # 默认 dna
        model = getattr(self.args, "model_dna", "dnacse") if modality == "dna" else getattr(self.args, "model", "unknown_model")
        dataset_name = getattr(self.args, "dataset_name", "bioscan-5m-zsc")
        
        # 时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        #and hasattr(self.args, "dna_maxlen") and self.args.dna_maxlen is not None
        # 子目录
        subdir = partition
        if modality == "dna" :
            subdir += f"__maxlen_128"
        
        # 构造文件名
        fname = f"{partition}_embeddings__{dataset_name}__{model}__{timestamp}.npz"
        
        # 完整路径
        full_path = os.path.join(self.args.output_dir, subdir, fname)
        return full_path

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_embedding: bool = False,
    ) -> Dict[str, float]:
        
        self.model.eval()
        device = self.args.device
        metrics={}
        rank = get_rank() if torch.distributed.is_initialized() else 0
        print(f"Rank {rank}: Starting evaluate()")
         # EMBED ===================================================================
        # Ensure encoder is on the correct device
        encoder = self.model
        # 仅主进程生成嵌入
        if self.args.local_rank <= 0:
            # Create embeddings
            t0 = time.time()
            print("Creating embeddings...")
            embeddings, y_true = self.embed_dataset(dataloader=self.eval_dataloader, encoder=encoder, device=device, is_distributed=False,log_interval=20)
            
            print(f"Created {len(embeddings)} embeddings in {time.time() - t0:.2f}s")
            
            # Save --------------------------------------------------------------------
            fname = self.get_embeddings_path(partition="test", modality="dna")
            # Save embeddings
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print(f"Saving embeddings to {fname}")
            t1 = time.time()
            tmp_a, tmp_b = os.path.split(fname)
            tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
        
            np.savez_compressed(tmp_fname, embeddings=embeddings, y_true=y_true)
            print(f"Rank 0: After savez, tmp_fname exists: {os.path.exists(tmp_fname)}")
            if not os.path.exists(tmp_fname):
                raise FileNotFoundError(f"Rank 0 failed to save {tmp_fname}")
            try:
                os.rename(tmp_fname, fname)
                os.sync()
                print(f"Rank 0: Renamed {tmp_fname} to {fname}")
            except OSError as e:
                print(f"Rank 0: Rename failed: {e}")
            print(f"Rank 0: Saved embeddings in {time.time() - t1:.2f}s")
        
        # 同步所有Rank
        if dist.is_initialized():
            dist.barrier()
        # rank = get_rank() if torch.distributed.is_initialized() else 0
        if self.args.local_rank <= 0:  # 仅主进程写入与评估
            results=self.cluster(fname=fname,patrition="test")
            metrics = {
                f"{metric_key_prefix}_{key}": value
                for key, value in results.items()
            }
            self.log(metrics)
            #  保存 metrics 到文件
            metrics_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    try:
                        existing_metrics = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e}, 重置为空列表")
                        existing_metrics = []
                existing_metrics.append(metrics)
            else:
                existing_metrics = [metrics]
            with open(metrics_file, "w") as f:
                json.dump(existing_metrics, f, indent=4)
       
            
         #最终同步
        if dist.is_initialized():
            dist.barrier()
            # 广播 metrics
            obj_list = [metrics] if rank == 0 else [None]
            dist.broadcast_object_list(obj_list, src=0)
            metrics = obj_list[0]
            print(f"Rank {rank}: Finished evaluate() and synced")
        print(f"Rank {rank}: evaluate() returning")
        # if dist.is_initialized():
        #     dist.barrier()
        #     print(f"Rank {dist.get_rank()}: Finished evaluate() and synced")
        
        # self.log(metrics)
        # if dist.is_initialized():
        #     dist.barrier()
        #     print(f"Rank {dist.get_rank()}: After barrier, before log")
        #     self.log(metrics)
        #     dist.barrier()  # 再次同步，确保 log 完成
        #     print(f"Rank {dist.get_rank()}: Finished evaluate() and synced")
        # else:
        #     self.log(metrics)
        
        print("evaluate()方法已经结束，返回训练过程")
        return metrics
    
    
    
        
    # def evaluate(
    #     self,
    #     eval_dataset: Optional[Dataset] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    #     eval_embedding: bool = False,
    # ) -> Dict[str, float]:
        
    #     self.model.eval()
    #     device = self.args.device
    #     metrics={}
    #      # EMBED ===================================================================
    #     # Ensure encoder is on the correct device
    #     encoder = self.model
    #     # Create embeddings
    #     t0 = time.time()
    #     print("Creating embeddings...")
    #     embeddings, y_true = self.embed_dataset(dataloader=self.eval_dataloader, encoder=encoder, device=device, is_distributed=self.args.local_rank != -1,log_interval=20)
        
    #     print(f"Created {len(embeddings)} embeddings in {time.time() - t0:.2f}s")
        
    #     # Save --------------------------------------------------------------------
    #     fname = self.get_embeddings_path(partition="test", modality="dna")
    #     # Save embeddings
    #     os.makedirs(os.path.dirname(fname), exist_ok=True)
    #     print(f"Saving embeddings to {fname}")
    #     t1 = time.time()
    #     tmp_a, tmp_b = os.path.split(fname)
    #     tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
    #     # 仅主进程保存文件
    #     if self.args.local_rank <= 0:
    #         np.savez_compressed(tmp_fname, embeddings=embeddings, y_true=y_true)
    #         print(f"Rank 0: After savez, tmp_fname exists: {os.path.exists(tmp_fname)}")
    #         if not os.path.exists(tmp_fname):
    #             raise FileNotFoundError(f"Rank 0 failed to save {tmp_fname}")
    #         try:
    #             os.rename(tmp_fname, fname)
    #             os.sync()
    #             print(f"Rank 0: Renamed {tmp_fname} to {fname}")
    #         except OSError as e:
    #             print(f"Rank 0: Rename failed: {e}")
    #         print(f"Rank 0: Saved embeddings in {time.time() - t1:.2f}s")
        
    #     # 主进程写入嵌入，需要同步所有进程
    #     if dist.is_initialized():
    #         dist.barrier()
    #     results=self.cluster(fname=fname,patrition="test")
    #     metrics = {
    #         f"{metric_key_prefix}_{key}": value
    #         for key, value in results.items()
    #     }
    #     rank = get_rank() if torch.distributed.is_initialized() else 0
    #     if self.args.local_rank <= 0:  # 仅主进程写入与评估
    #         #  保存 metrics 到文件
    #         metrics_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_metrics.json")
    #         if os.path.exists(metrics_file):
    #             with open(metrics_file, "r") as f:
    #                 try:
    #                     existing_metrics = json.load(f)
    #                 except json.JSONDecodeError as e:
    #                     print(f"JSON 解析错误: {e}, 重置为空列表")
    #                     existing_metrics = []
    #             existing_metrics.append(metrics)
    #         else:
    #             existing_metrics = [metrics]
    #         with open(metrics_file, "w") as f:
    #             json.dump(existing_metrics, f, indent=4)
    #     if rank == 0:
    #         self.log(metrics)
    #      #主进程写入Metrics，需要同步所有进程
    #     if dist.is_initialized():
    #         dist.barrier()
    #         print(f"Rank {rank}: Finished evaluate() and synced")
    #     print(f"Rank {rank}: evaluate() returning")
    #     # if dist.is_initialized():
    #     #     dist.barrier()
    #     #     print(f"Rank {dist.get_rank()}: Finished evaluate() and synced")
        
    #     # self.log(metrics)
    #     # if dist.is_initialized():
    #     #     dist.barrier()
    #     #     print(f"Rank {dist.get_rank()}: After barrier, before log")
    #     #     self.log(metrics)
    #     #     dist.barrier()  # 再次同步，确保 log 完成
    #     #     print(f"Rank {dist.get_rank()}: Finished evaluate() and synced")
    #     # else:
    #     #     self.log(metrics)
        
    #     print("evaluate()方法已经结束，返回训练过程")
    #     return metrics
        

        # V=V if V is not None else self.V
        # precomputed_scores=precomputed_scores if precomputed_scores is not None else self.precomputed_scores
        # # 使用传入的 eval_dataset 或默认的 self.eval_dataset
        # eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # if eval_dataset is None:
        #     raise ValueError("eval_dataset 未提供，请在初始化或调用时设置")
        
        # # 1. 保存原始模型状态
        # original_model = self.model
        
        # try:
        #     # 2. 临时替换为封装模型
        #     if self.args.average_rc:
        #        self.model = ModelVEPAverageRC(self.model)
        #     else:
        #        self.model = ModelVEP(self.model)
            
        #     # 3. 获取预测
        #     predictions = self.predict(test_dataset=eval_dataset).predictions
        #     # 4. 初始化结果字典 V
        #     # V = {}
        #     V["score"] = predictions  # 假设 predictions 是 1D 数组
            
        #     # 5. 加载预计算分数
        #     V["precomputed_score"] = precomputed_scores
        #     # pd.read_parquet(
        #     #     "TraitGym/mendelian_traits_matched_9/features/GPN_final_EuclideanDistance.parquet"
        #     # ).score.values
            
        #     # 6. 计算指标
        #     metrics = {}
            
        #     # 6.1 相关性
        #     corr_matrix = V[["score", "precomputed_score"]].corr()
        #     metrics[f"{metric_key_prefix}_pearson_corr"] = corr_matrix.loc["score", "precomputed_score"]
            
        #     # 6.2 全局 AUPRC
        #     global_auprc = average_precision_score(V["label"], V["score"])
        #     metrics[f"{metric_key_prefix}_global_auprc"] = global_auprc
            
        #     # 6.3 按 chrom 计算 AUPRC
        #     res_by_chrom = []
        #     for chrom in V["chrom"].unique():
        #         V_chrom = V[V["chrom"] == chrom]
        #         auprc = average_precision_score(V_chrom["label"], V_chrom["score"])
        #         res_by_chrom.append([chrom, len(V_chrom), auprc])
        #     res_by_chrom = pd.DataFrame(res_by_chrom, columns=["chrom", "n", "AUPRC"])
            
        #     # 保存每个 chrom 的 AUPRC
        #     for _, row in res_by_chrom.iterrows():
        #         metrics[f"{metric_key_prefix}_auprc_chrom_{row['chrom']}"] = row["AUPRC"]
            
        #     # 6.4 加权平均 AUPRC
        #     def stat(df):
        #         weight = df["n"] / df["n"].sum()
        #         return (df["AUPRC"] * weight).sum()
            
        #     weighted_auprc = stat(res_by_chrom)
        #     metrics[f"{metric_key_prefix}_weighted_auprc"] = weighted_auprc
            
        #     # 6.5 Bootstrap 标准误差
        #     def bootstrap_se(df, stat, n_bootstraps=1000):
        #         df = pl.DataFrame(df)
        #         bootstrap_stats = [
        #             stat(df.sample(len(df), with_replacement=True, seed=i))
        #             for i in range(n_bootstraps)
        #         ]
        #         return pl.Series(bootstrap_stats).std()
            
        #     bootstrap_std = bootstrap_se(res_by_chrom, stat)
        #     metrics[f"{metric_key_prefix}_weighted_auprc_bootstrap_se"] = bootstrap_std
            
            
        #     # 转换为 JSON 兼容格式
        #     for key, value in metrics.items():
        #         if isinstance(value, np.ndarray):
        #             metrics[key] = value.tolist()
        #         elif isinstance(value, np.floating):
        #             metrics[key] = float(value)
            
        #      # 保存直方图到输出目录
        #     if self.args.should_save:  # 确保只在主进程保存
        #         plt.figure(figsize=(2, 2))
        #         sns.histplot(
        #             data=V, x="score", bins=30, hue="label", stat="density",
        #             common_norm=False, common_bins=True
        #         )
        #         sns.despine()
        #         plot_path = os.path.join(self.args.output_dir, "eval_histogram.png")
        #         plt.savefig(plot_path, bbox_inches="tight")
        #         plt.close()
        #     # 保存 metrics 到文件
        #     metrics_file = os.path.join(self.args.output_dir, f"{metric_key_prefix}_metrics.json")
        #     rank = get_rank() if torch.distributed.is_initialized() else 0
        #     if rank == 0:  # 仅主进程写入
        #         if os.path.exists(metrics_file):
        #             with open(metrics_file, "r") as f:
        #                 try:
        #                     existing_metrics = json.load(f)
        #                 except json.JSONDecodeError as e:
        #                     print(f"JSON 解析错误: {e}, 重置为空列表")
        #                     existing_metrics = []
        #             existing_metrics.append(metrics)
        #         else:
        #             existing_metrics = [metrics]
        #         with open(metrics_file, "w") as f:
        #             json.dump(existing_metrics, f, indent=4)
            
            
        #     print(f"metrics已被记录:{metrics}")
        #     self.log(metrics)
            # 7. 返回评估结果
        return metrics
        # finally:
        #     # 8. 恢复原始模型
        #     # print("恢复训练状态")
        #     self.model = original_model
        #     self.model.train()
        #     # print("训练模型已恢复")
        
        
        
        
        
        # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    
        # eval_dataloader= self.get_eval_dataloader(eval_dataset)
        
        # loss = torch.tensor(0.0, device=self.args.device)
        # num_batches = torch.tensor(0, device=self.args.device)
        # total_loss = torch.tensor(0.0, device=self.args.device)
    
        # # 模型切换到评估模式
        # self.model.eval()
        # # 遍历 eval_dataloader
        # for step, inputs in enumerate(eval_dataloader):
        #     batch = inputs
        #     oov_to_id = batch.pop('oov_to_id')
        #     batch = self._prepare_inputs(batch)
        #     # print(f"Batch after preparation: {batch}")
        #     with torch.no_grad():
        #         outputs = self.model(
        #             **batch,
        #             oov_to_id=oov_to_id,
        #             output_hidden_states=True,
        #             return_dict=True,
        #             sent_emb=True
        #         )
        #         pooler_output = outputs.pooler_output
        #         # print(f"Pooler output shape: {pooler_output.shape}")
        #         batch_size = batch['input_ids'].size(0)
        #         num_sent = batch['input_ids'].size(1)
        #         # print(f"Batch size: {batch_size}, Num sent: {num_sent}")
        #         pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))
        #         z1, z2 = pooler_output[:,0], pooler_output[:,1]
        #         # Gather all embeddings if using distributed training
        #         if dist.is_initialized():
        #             # Dummy vectors for allgather
        #             z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        #             z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        #             # Allgather
        #             dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        #             dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        #             # Since allgather results do not have gradients, we replace the
        #             # current process's corresponding embeddings with original tensors
        #             z1_list[dist.get_rank()] = z1
        #             z2_list[dist.get_rank()] = z2
        #             # Get full batch embeddings: (bs x N, hidden)
        #             z1 = torch.cat(z1_list, 0)
        #             z2 = torch.cat(z2_list, 0)
        #         cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        #         pos = torch.exp(torch.diag(cos_sim))
        #         mu =self.mu
        #         sigma = self.sigma
        #         neg_score = cos_sim * 0.05
        #         weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp( - (neg_score - mu) ** 2 / (2 * math.pow(sigma, 2)))
        #         weight = weight / weight.mean(dim=-1, keepdim=True)
        #         neg = torch.sum(torch.exp(cos_sim) * weight.detach(), dim=1)
        #         loss = - torch.log(pos / neg).mean()
                
        #         total_loss += loss
        #         num_batches += 1
                
        # if dist.is_initialized():
        #     dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)  
                
                
                 
        # average_loss = total_loss.item() / num_batches.item() if num_batches.item() > 0 else 0.0
        
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     print(f"Validation average loss: {average_loss}")
        # metrics = {
        # f"{metric_key_prefix}_loss": average_loss,
        # f"{metric_key_prefix}_num_batches": num_batches.item(),
        # }
        # # print(metrics)
        # if self.is_world_process_zero():
        #    self.log(metrics)
        # self.model.eval()
        # metrics = { }  
        # metrics.update(self.cal_align_uniform(self.eval_dataloader))
        #     # 验证下游任务/root/shared-nvme/downstream_data
        #     # metrics=run_evaluation(test_model_dir=self.args.output_dir, model_list=["dnacse"], data_dir="/root/shared-nvme/downstream_data")
        # self.log(metrics)
        # # if world_size > 1:
        # #     torch.distributed.barrier()
        # print(f"Metrics received in _save_checkpoint: {metrics.keys()}")
        # return metrics
        
    def cal_align_uniform(self,dataloader):
        results = []
        labels = []
        embs = []
        print("进入评估")
        for batch in dataloader:
            with torch.no_grad():
                batch = [ele.to(self.model.device) for ele in batch]
                
                # print(batch[0].size(),batch[1].size(),batch[2].size())
                # print(batch[3].size(),batch[4].size(),batch[5].size())
                if len(batch) == 5:
                    outputs = self.model(input_ids=batch[0],attention_mask=batch[1],output_hidden_states=True, return_dict=True, sent_emb=True)
                    emba = outputs.pooler_output
                    outputs = self.model(input_ids=batch[2],attention_mask=batch[3],output_hidden_states=True, return_dict=True, sent_emb=True)
                    # print(f"sent_emboutput:{outputs}")
                    embb = outputs.pooler_output
                    label = batch[4].detach().cpu().numpy().tolist()
                else:
                    outputs = self.model(input_ids=batch[0],token_type_ids=batch[1],attention_mask=batch[2],output_hidden_states=True, return_dict=True, sent_emb=True)
                    emba = outputs.pooler_output
                    outputs = self.model(input_ids=batch[3],token_type_ids=batch[4],attention_mask=batch[5],output_hidden_states=True, return_dict=True, sent_emb=True)
                    embb = outputs.pooler_output
                    label = batch[6].detach().cpu().numpy().tolist()
                print("if 语句已经执行完毕")
                emba = F.normalize(emba,dim=-1)
                embb = F.normalize(embb,dim=-1)
                scores = torch.linalg.norm(emba-embb,dim=-1).pow(2)
                embs.append(emba)
                embs.append(embb)
                labels += label
            scores = scores.detach().cpu().numpy().tolist()
            results += scores
            
            labels += label
        print("for语句已经执行完毕")
        align = []
        uniform = []
        embs = torch.cat(embs,dim=0).detach().cpu()
        with torch.no_grad():
            uniform = F.pdist(embs).pow(2).neg().exp().mean().log().item()
            cos_scores = F.cosine_similarity(embs.unsqueeze(0),embs.unsqueeze(1),dim=-1)
            cos_scores = cos_scores * (1-torch.eye(cos_scores.size(0)))
            gamma = torch.arccos(torch.min(cos_scores)).item()
            E_costheta = torch.mean(cos_scores).item()
            V_costheta = torch.var(cos_scores).item()
            hist = torch.histc(cos_scores,bins=20,min=-1,max=1).cpu().numpy().tolist()
        for score,label in zip(results,labels):
            # if label > 4.0:
            align.append(score)
            # uniform.append(math.exp(-2 * score))
        align = sum(align) / len(align)
        metrics = {'align':align,'uniform':uniform,'E_costheta':E_costheta,'V_costheta':V_costheta,'hist':hist,'gamma':gamma}
        print(metrics)
        print("when you see this ,evaluation is over")
        return metrics
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs, dynamically passing oov_to_id to the model.

        Args:
            model (`nn.Module`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`): The inputs and targets of the model.

        Returns:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        # Prepare inputs (handles moving to device, etc.)
        inputs = self._prepare_inputs(inputs)
        

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        # Forward pass with loss computation
        with self.compute_loss_context_manager():
            # Standard behavior without oov_to_id
            loss = self.compute_loss(model, inputs)

        # Handle multi-GPU training (average the loss)
        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Handle gradient accumulation
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Backpropagation depending on mixed precision mode
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if os.path.exists(best_model_path) or os.path.exists(best_safe_model_path):
            if self.deepspeed:
                if self.model_wrapped is not None:
                    # this removes the pre-hooks from the previous engine
                    self.model_wrapped.destroy()
                    self.model_wrapped = None

                # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                    self,
                    num_training_steps=self.args.max_steps,
                    resume_from_checkpoint=self.state.best_model_checkpoint,
                )
                self.model = deepspeed_engine.module
                self.model_wrapped = deepspeed_engine
                self.deepspeed = deepspeed_engine
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                        # If the 'user_content.pt' file exists, load with the new smp api.
                        # Checkpoint must have been saved with the new smp api.
                        smp.resume_from_checkpoint(
                            path=self.state.best_model_checkpoint,
                            tag=WEIGHTS_NAME,
                            partial=False,
                            load_optimizer=False,
                        )
                    else:
                        # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                        # Checkpoint must have been saved with the old smp api.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        state_dict["_smp_is_partial"] = False
                        load_result = model.load_state_dict(state_dict, strict=True)
                else:
                    if isinstance(self.model, PreTrainedModel):
                        # model_args = getattr(self, "model_args", {})
                        # print(f"model_args type before conversion: {type(model_args)}")
                        # if not isinstance(model_args, dict):
                        #     model_args = vars(model_args)
                        # print(f"model_args type after conversion: {type(model_args)}")
                        self.model = self.model.from_pretrained(
                        self.state.best_model_checkpoint,
                        model_args=self.model_args,  # 通过 model_args 加载
                        )
                        
                        if not self.is_model_parallel:
                            self.model = self.model.to(self.args.device)
                    else:
                        # We load the model state dict on the CPU to avoid an OOM error.
                        if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                            state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                        else:
                            state_dict = torch.load(best_model_path, map_location="cpu")

                        # If the model is on the GPU, it still works!
                        # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                        # which takes *args instead of **kwargs
                        load_result = model.load_state_dict(state_dict, False)
                        if not is_sagemaker_mp_enabled():
                            self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # with self.tokenizer.batch_context(): 
                    total_batched_samples += 1
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    if (
                        (total_batched_samples % args.gradient_accumulation_steps != 0)
                        and args.parallel_mode == ParallelMode.DISTRIBUTED
                        and args._no_sync_in_gradient_accumulation
                        and hasattr(model, "no_sync")
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if self.deepspeed:
                        self.deepspeed.step()

                    if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Determine if this is the best model based on validation loss
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                # Save model checkpoint only if it's the best model
                self.save_model(output_dir, _internal_call=True)
                if self.deepspeed:
                    # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
                    # config `stage3_gather_16bit_weights_on_model_save` is True
                    self.deepspeed.save_checkpoint(output_dir)

                # Save optimizer and scheduler
                if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                    self.optimizer.consolidate_state_dict()

                if self.fsdp:
                    # FSDP has a different interface for saving optimizer states.
                    # Needs to be called on all ranks to gather all states.
                    # full_optim_state_dict will be deprecated after Pytorch 2.2!
                    full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                        reissue_pt_warnings(caught_warnings)
                elif is_sagemaker_mp_enabled():
                    opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
                    smp.barrier()
                    if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                        smp.save(
                            opt_state_dict,
                            os.path.join(output_dir, OPTIMIZER_NAME),
                            partial=True,
                            v3=smp.state.cfg.shard_optimizer_state,
                        )
                    if self.args.should_save:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                        reissue_pt_warnings(caught_warnings)
                        if self.do_grad_scaling:
                            torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
                elif self.args.should_save and not self.deepspeed:
                    # deepspeed.save_checkpoint above saves model/optim/sched
                    if self.fsdp:
                        torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
                    else:
                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
                    if self.do_grad_scaling:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
                
                # Save the Trainer state
                if self.args.should_save:
                    print(self.state)
                    self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

                # Save RNG state in non-distributed training
                rng_states = {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "cpu": torch.random.get_rng_state(),
                }
                if torch.cuda.is_available():
                    if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                        # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                        rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
                    else:
                        rng_states["cuda"] = torch.cuda.random.get_rng_state()

                if is_torch_tpu_available():
                    rng_states["xla"] = xm.get_rng_state()

                # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
                # not yet exist.
                os.makedirs(output_dir, exist_ok=True)

                if self.args.world_size <= 1:
                    torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
                else:
                    torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

                if self.args.push_to_hub:
                    self._push_from_checkpoint(output_dir)

                # Maybe delete some older checkpoints.
                if self.args.should_save:
                    self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
            
    
         
            
  

