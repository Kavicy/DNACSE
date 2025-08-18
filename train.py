import logging
import math
import os
import sys
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from datetime import timedelta
from packaging import version
import warnings
import torch
import collections
import random
import json
from datasets import load_dataset
from Bio.Seq import Seq
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import polars as pl
# from pyfaidx import Fasta
# import seaborn as sns
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import tempfile
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.utils import (
    ExplicitEnum,
    cached_property,
    get_full_repo_name,
    is_accelerate_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_neuroncore_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    requires_backends,
)
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property
from torch.utils.data import TensorDataset
import pandas as pd
from torch.utils.data import DataLoader
# from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from transformers.utils import is_torch_available, is_torch_tpu_available
from DNAcse.models import RobertaForCL, BertForCL
from DNAcse.trainers import CLTrainer
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import DistributedType
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp# type:ignore

    smp.init()
# window_size = 512    
def Construct_EvalDataset(datapath, tokenizer,max_seq_length):
    # 读取 CSV 文件，只加载需要的三列
    df = pd.read_csv(datapath, usecols=["dna_barcode",  "labels"])
    
    # 提取文本和标签
    text_a = df["dna_barcode"].tolist()
    # text_b = df["seq2"].tolist()
    labels = df["labels"].astype(float).tolist()
    print(f"Raw labels sample: {labels[:5]}")
    print(f"Set eval_tokenizer model_max_length to {max_seq_length}")
    # 使用 tokenizer 处理文本数据
    texta = tokenizer(text_a, padding="max_length", truncation=True, max_length=max_seq_length, return_tensors='pt')
    print(f"Tokenizer output keys: {list(texta.keys())}")
    # textb = tokenizer(text_b, padding="max_length", truncation=True, max_length=max_seq_length, return_tensors='pt')
    
    # 转换标签为 LongTensor
    labels = torch.LongTensor(labels)
    print(f"Tensor labels shape: {labels.shape}, sample: {labels[:5]}")
    # labels = torch.FloatTensor(labels)
    # 组合所有张量
    # tensors = list(texta.values()) + [labels] #+ list(textb.values())
    tensors = [texta["input_ids"], texta["attention_mask"], labels]
    dataset = TensorDataset(*tensors)
    sample = dataset[0]
    print(f"Dataset sample: input_ids shape={sample[0].shape}, attention_mask shape={sample[1].shape}, labels={sample[2]}")
    # return TensorDataset(*tensors)
    return dataset

# 定义一个简单的 collate_fn，将列表转换为字典
def eval_collate_fn(batch):
    """
    Collate function for DataLoader to process batches from TensorDataset.

    Parameters
    ----------
    batch : list
        List of tuples from TensorDataset, each containing (input_ids, attention_mask, labels).

    Returns
    -------
    dict
        Dictionary with batched input_ids, attention_mask, and labels.
    """
    # 提取标量标签
    labels = [x[2].item() for x in batch]
    
    return {
        "input_ids": torch.stack([x[0] for x in batch]),      # (batch_size, max_seq_length)
        "attention_mask": torch.stack([x[1] for x in batch]), # (batch_size, max_seq_length)
        "labels": torch.tensor(labels, dtype=torch.long)      # (batch_size,)
    }

# def eval_collate_fn(batch):
#     # batch 是 [(input_ids, attention_mask, labels), ...]
#     # input_ids 和 attention_mask 是 (max_seq_length,)，labels 是标量
#     return {
#         "input_ids": torch.stack([x[0] for x in batch]),          # (batch_size, max_seq_length)
#         "attention_mask": torch.stack([x[1] for x in batch]),     # (batch_size, max_seq_length)
#         "labels": torch.tensor([x[2] for x in batch])             # (batch_size,)
#     }
    


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lambdas:float = field(
        default=0.2,
        metadata={
            "help":"mix lambdas"
        }
    )
    

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0.0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    
    projector_dim: int = field(
        default=768,
        metadata={
            "help": "The dimension of projector"
        }
    )
    #Add some hyperparameters, such as dropout_rate,additive_noise_level=0.0, multiplicative_noise_level=0.0)
    global_std: float = field(
        default=0.15,
        metadata={
            "help": "The strength of additive noise"
        }
    )
    local_std: float = field(
        default=0.25,
        metadata={
            "help": "The strength of multiplicative noise"
        }
    )
    
    dropout_rate: float = field(
        default=0.1,
        metadata={
            "help": "The dropout rate of projector"
        }
    )
 
    mutation_rate: float = field(
        default=0.07,
        metadata={
            "help": "The mutation rate of the DNA sequence"
        }
    )
    lossweight:float = field(
        default=0.5,
        metadata={
            "help": "The weight of the loss function,for noisy emb and drop emb"
        }
    )
    pos_m:float = field(
        default=0.0,
        metadata={
            "help": "the angular margin (m) between the anchor (zi) and the positive pair (z_p)."
        }
    )
    num_sent:int =field(
        default=2,
        metadata={
            "help": "the number of sentences in each sample"
        }
    )
    apply_mixup: bool = field(
        default=True,
        metadata={
            "help":"whether to use mixup"
        }
    )
    mixup_alpha: float = field(
        default=1.0,
        metadata={
            "help":"the alpha parameter for mixup"
        }
    )
    drop_rate: float = field(
        default=0.1,
        metadata={
            "help":"the drop rate for the dropout layer"
        }
    )
    apply_strength : bool =field(
        default = True,
        metadata={
            "help":"whether to use noisy layer"
        }
    )
   
    
    
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    eval_file:Optional[str] = field(
        default=None,
        metadata={"help":"the validation data file (.txt or .csv)"}   
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    
    
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.eval_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    local_rank: int = field(
        default=-1, metadata={"help": "Local rank for distributed training"}
    )
    # Evaluation
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    greater_is_better:bool =field(
        default = True,
        metadata={"help":  "Whether the metric_for_best_model should be maximized or not."}   
    )
    # max_steps : int =field(
    #    default = 250000,
    #    metadata={"help":"max training steps."} 
    # )
    gradient_accumulation_steps : int =field(
       default =1,metadata={"help":"choose the gradient_accumulation_steps"} 
    )
    
    logging_dir: Optional[str] = field(
        default="./logs", 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    
    
    # resume_from_checkpoint :str = field(
    #     default ="/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased/checkpoint-10000",
    #     metadata={"help":"resume from checkpoint"}
    # )
    
   
    
    

    @cached_property
    def _setup_devices(self) -> "torch.device":
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if not is_sagemaker_mp_enabled() and not is_accelerate_available(check_partial_state=True):
            raise ImportError(
                "Using the `Trainer` with `PyTorch` requires `accelerate`: Run `pip install --upgrade accelerate`"
            )
        if self.no_cuda:
            self.distributed_state = PartialState(cpu=True, backend=self.ddp_backend)
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
            torch.cuda.set_device(device)
        elif self.deepspeed:
            # Need to do similar for Accelerator init
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = PartialState(timeout=timedelta(seconds=self.ddp_timeout))
            del os.environ["ACCELERATE_USE_DEEPSPEED"]
            self._n_gpu = 1
        else:
            self.distributed_state = PartialState(backend=self.ddp_backend)
            self._n_gpu = 1
        if not is_sagemaker_mp_enabled():
            device = self.distributed_state.device
            self.local_rank = self.distributed_state.local_process_index
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.distributed_state.distributed_type == DistributedType.NO
        ):
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        if is_torch_tpu_available():
            device = self.distributed_state.device
            self._n_gpu = 0
        elif is_sagemaker_dp_enabled():
            self._n_gpu = 1
        elif self.distributed_state.distributed_type == DistributedType.NO:
            if self.use_mps_device:
                if not torch.backends.mps.is_available():
                    if not torch.backends.mps.is_built():
                        raise AssertionError(
                            "MPS not available because the current PyTorch install was not "
                            "built with MPS enabled. Please install torch version >=1.12.0 on "
                            "your Apple silicon Mac running macOS 12.3 or later with a native "
                            "version (arm64) of Python"
                        )
                    else:
                        raise AssertionError(
                            "MPS not available because the current MacOS version is not 12.3+ "
                            "and/or you do not have an MPS-enabled device on this machine."
                        )
                else:
                    if not version.parse(version.parse(torch.__version__).base_version) > version.parse("1.12.0"):
                        warnings.warn(
                            "We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing)"
                            " on your MacOS machine. It has major fixes related to model correctness and performance"
                            " improvements for transformer based models. Please refer to"
                            " https://github.com/pytorch/pytorch/issues/82707 for more details."
                        )
                    device = torch.device("mps")
                    self._n_gpu = 1

            else:
                # if n_gpu is > 1 we'll use nn.DataParallel.
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
                # trigger an error that a device index is missing. Index 0 takes into account the
                # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
                # will use the first GPU in that env, i.e. GPU#1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
                # the default value.
                self._n_gpu = torch.cuda.device_count()
                if device.type == "cuda":
                    torch.cuda.set_device(device)
        return device



def main():
    
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    training_args.local_rank = local_rank
     
        
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    # if data_args.validation_file is not None:
    #     data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="/root/shared-nvme/data/tmp", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="/root/shared-nvme/data/tmp")
    
    
    #Choose the type of attention to use for the model

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,trust_remote_code=True,**config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True,**config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    #Choose the type of attention to use for the model
   
  
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        # tokenizer = BPEvarTokenizer.load_from(model_args.tokenizer_name,max_k=20,max_ratio=50, **tokenizer_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.model_max_length = data_args.max_seq_length
    print(f"Set tokenizer model_max_length to {tokenizer.model_max_length}")
    # device_instance = training_args._setup_devices()
    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args,
                trust_remote_code=True
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        
        sentences = examples[sent0_cname] + examples[sent1_cname]

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]
        
        
        # print(f"Tokenizer's model_max_length: {tokenizer.model_max_length}")
        # print(f"Max sequence length: {data_args.max_seq_length}")
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" ,#if data_args.pad_to_max_length else False
        )
        # max_observed_length = max([len(seq) for seq in sent_features['input_ids']])
        # print(f"After tokenization, max observed length: {max_observed_length}")

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
            
        return features
    
   
    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            remove_columns=column_names
        )
   
   

    #origin:List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]],new:List[Optional[Union[List[str], Tuple[str, ...]]]]
    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability
        
    
        
        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                
                special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
                bs = len(features)
                
                with open("sequence_lengths.log", "a") as f:
                    for idx, feature in enumerate(features):
                        if 'input_ids' in feature:
                            for i, seq in enumerate(feature['input_ids']):
                                f.write(f"Batch item {idx}, sequence {i}, length: {len(seq)}\n")
                
                
                if bs > 0:
                    num_sent = len(features[0]['input_ids'])
                else:
                    return
                flat_features = []
                for feature in features:
                    for i in range(num_sent):
                        flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
                
                
                
                batch = self.tokenizer.pad(
                    flat_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors="pt",
                )
                
                
                
                
                if model_args.do_mlm:
                    batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

                batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

                if "label" in batch:
                    batch["labels"] = batch["label"]
                    del batch["label"]
                if "label_ids" in batch:
                    batch["labels"] = batch["label_ids"]
                    del batch["label_ids"]
                # batch["oov_to_id"]=oov_to_id
                return batch
        
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels
    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer,max_length=data_args.max_seq_length)
    
    
    
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args
    
    # eval_dataset = Construct_EvalDataset(data_args.eval_file,tokenizer,max_seq_length=data_args.max_seq_length)
    # trainer.eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=training_args.per_device_train_batch_size)
    eval_dataset = Construct_EvalDataset(data_args.eval_file,tokenizer,data_args.max_seq_length)
    trainer.eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=training_args.per_device_eval_batch_size,collate_fn=eval_collate_fn)

    
    # # 加载 checkpoint-16000 到模型
    # trainer.state.best_model_checkpoint = "/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased/checkpoint-16000"  # 手动指定检查点路径
    # trainer._load_best_model()  # 修改为加载特定检查点（见下文）
    # trainer.model = trainer.model.from_pretrained("/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased/checkpoint-16000",model_args=model_args)

    # # 保存为完整模型
    # output_path = "/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased"
    # trainer.save_model(output_path)
    # print("保存完成")
    
    
    
    # print("data_collator:", data_collator(eval_dataset[:2]))  # 检查批处理结果
    # trainer.evaluate(eval_dataset=eval_dataset,V=V,precomputed_scores=precomputed_scores)
    # eval_dataset = Construct_EvaldDtaset(data_args.validation_file,tokenizer,max_seq_length=data_args.max_seq_length)
    # trainer.eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=training_args.per_device_train_batch_size)

    # # Training
    if training_args.do_train:
        # model_path = (
        #     model_args.model_name_or_path
        #     if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        #     else None
        # )
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        # train_result = trainer.train(resume_from_checkpoint="/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased/checkpoint-36999")
        # train_result = trainer.train(resume_from_checkpoint="/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased/checkpoint-16000")
        # train_result =  trainer.train(resume_from_checkpoint="/home/dnacsecode/DNAcse/result/my-unsup-dnacse-dnabert2-base-uncased_48/checkpoint-28000")
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
    print("训练已结束，不进行评估流程")

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


import optuna
from transformers import Trainer, TrainingArguments
import torch.distributed as dist
from optuna.integration import TorchDistributedTrial
# from optuna.integration import OptunaCallback
# from optuna.integration import TransformersPruningCallback
import torch.multiprocessing as mp
from transformers import TrainerCallback
import datetime 
import os
import shutil
print("RANK:", os.environ.get("RANK"))
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))

def setup_distributed():
    """检查分布式环境并返回 rank 和 world_size"""
    if not dist.is_initialized():
        # raise RuntimeError("Distributed environment not initialized. Use torchrun to launch the script.")
        logger.info("Distributed environment not initialized, initializing with NCCL backend")
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)  # 为每个进程分配 GPU
    logger.info(f"Rank {rank}/{world_size} initialized")
    return rank, world_size
def prepare_data(tokenizer, train_file, max_seq_length=128):
    """准备训练数据集"""
    logger.info(f"Rank {dist.get_rank()} starting data loading")
    data_files = {"train": train_file}
    extension = train_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files, cache_dir="/root/shared-nvme/data/tmp", delimiter="\t" if "tsv" in train_file else ",")

    column_names = datasets["train"].column_names
    sent0_cname = column_names[0]
    sent1_cname = column_names[0]  # Unsupervised case
    
    def prepare_features(examples):
        total = len(examples[sent0_cname])
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        sentences = examples[sent0_cname] + examples[sent1_cname]
        sent_features = tokenizer(
            sentences,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]
        return features

    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        remove_columns=column_names
    )
    # 同步 shuffle 和 select
    seed = 42
    dist.barrier()  # 确保所有进程完成 map
    train_dataset = train_dataset.shuffle(seed=seed).select(range(int(len(train_dataset) * 0.1)))
    logger.info(f"Rank {dist.get_rank()} finished data preparation")
    return train_dataset

def objective(trial, rank, world_size, train_dataset, tokenizer, eval_file):
       
        # 在主进程(rank 0)上生成超参数
        if rank == 0:
            params = {
                # "global_std": trial.suggest_float("global_std", 0.03, 0.08),
                # "global_std" : trial.suggest_float("global_std", 0.09, 0.095),
                "global_std":trial.suggest_categorical("global_std", [0.08]),
                # "global_std": 0.15,
                # "local_std": trial.suggest_float("local_std", 0.18, 0.21),
                "local_std": trial.suggest_categorical("local_std", [0.25]),
                # "local_std": 0.25,
                # "corr_coef": trial.suggest_float("corr_coef", 0.3, 0.9),
                # "corr_coef": 0.6,
                # "mutation_rate":trial.suggest_float("mutation_rate", 0.07, 0.08),
                "mutation_rate": trial.suggest_categorical("mutation_rate", [0.10]),
                # "mutation_rate": 0.55,
                "hard_negative_weight": 0.0,
                # "hard_negative_weight": trial.suggest_float("hard_negative_weight", 0.38, 0.39),
                # "hard_negative_weight": trial.suggest_categorical("hard_negative_weight", [0.40]),
                "temp": 0.05,
                "lambdas": 0.2,
                # "learning_rate": trial.suggest_categorical("learning_rate", [3e-5]),
                # "learning_rate":trial.suggest_loguniform("learning_rate", 3e-5, 5e-5),
                # "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-5, log=True),
                "learning_rate": 3e-5,
                # "batch_size": trial.suggest_categorical("batch_size", [64]),
                "batch_size":64,
                # "batch_size": trial.suggest_int("batch_size", 32, 128, step=16)
            }
            # params = {
            #             "global_std": 0.0750,
            #             "local_std": 0.08,
            #             "corr_coef": 0.8432,
            #             "temp": 0.05,
            #             "lambdas": 0.3927,
            #             "learning_rate": 2e-5,
            #             # "batch_size": 48
            #             "batch_size": trial.suggest_categorical("batch_size", [48])
            #         }
        else:
            params = None
        dist.barrier()  # 同步点
        params_list = [params] if rank == 0 else [None]
        dist.broadcast_object_list(params_list, src=0)
        params = params_list[0]
        

         # 使用同步后的参数
        global_std = params["global_std"]
        local_std = params["local_std"]
        # corr_coef = params["corr_coef"]
        temp = params["temp"]
        lambdas = params["lambdas"]
        learning_rate = params["learning_rate"]
        batch_size = params["batch_size"]
        mutation_rate = params["mutation_rate"]
        hard_negative_weight = params["hard_negative_weight"]
        # 打印确认每个进程使用相同的参数
        print(f"Rank {rank}: Using batch_size={batch_size}, lr={learning_rate}")
        
        # 模型参数设置代码..
        model_args = ModelArguments(
        model_name_or_path="/home/dnacsecode/DNAcse/bertmodel",  # 固定模型路径
        cache_dir=None,
        use_fast_tokenizer=True,
        model_revision="main",
        use_auth_token=False,
        # 以下为需要优化的参数
        global_std=global_std,
        local_std=local_std,
        # corr_coef=corr_coef,
        temp=temp,
        lambdas=lambdas,
        mutation_rate=mutation_rate,
        # 以下为固定参数
        num_sent=2,
        apply_noise=True,
        do_mlm=False,
        pooler_type="cls",#trial.suggest_categorical("pooler_type", ["cls", "cls_before_pooler"]),
        projector_dim=768,
        dropout_rate=0.1,
        mlm_weight=0.1,
        hard_negative_weight=hard_negative_weight,
        mlp_only_train=True
        )
        data_args = DataTrainingArguments(
            eval_file=eval_file,
            max_seq_length=128
        )
        #1%pretrain->5%
        # eval_interval = max(20, 5000// batch_size)
        eval_interval = 100
        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=f"./results/trial_{trial.number}",
            num_train_epochs=0.5,
            max_steps=1000,
            evaluation_strategy="steps",
            eval_steps=eval_interval,
            save_steps=eval_interval,
            save_strategy="steps",
            save_total_limit=1,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=128,
            metric_for_best_model="AMI",      # 指定早停指标
            greater_is_better=True,
            load_best_model_at_end=True,
            gradient_accumulation_steps=1,
            fp16=True,
            logging_dir=f"./logs/trial_{trial.number}",
            logging_steps=eval_interval,
            local_rank=rank,
            lr_scheduler_type="linear",
            warmup_ratio=0.05,
        )
        

        # download model & vocab.
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name,trust_remote_code=True,**config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True,**config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")    
     
        model = BertForCL.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    model_args=model_args,
                    trust_remote_code=True
                )
        if model_args.do_mlm:
            pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
            model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        
        @dataclass
        class OurDataCollatorWithPadding:

            tokenizer: PreTrainedTokenizerBase
            padding: Union[bool, str, PaddingStrategy] = True
            max_length: Optional[int] = None
            pad_to_multiple_of: Optional[int] = None
            mlm: bool = True
            mlm_probability: float = data_args.mlm_probability
            
        
            
            def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                    
                    special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
                    bs = len(features)
                    
                    with open("sequence_lengths.log", "a") as f:
                        for idx, feature in enumerate(features):
                            if 'input_ids' in feature:
                                for i, seq in enumerate(feature['input_ids']):
                                    f.write(f"Batch item {idx}, sequence {i}, length: {len(seq)}\n")
                    
                    
                    if bs > 0:
                        num_sent = len(features[0]['input_ids'])
                    else:
                        return
                    flat_features = []
                    for feature in features:
                        for i in range(num_sent):
                            flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
                    
                    
                    
                    batch = self.tokenizer.pad(
                        flat_features,
                        padding=self.padding,
                        max_length=self.max_length,
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        return_tensors="pt",
                    )
                    
                    
                    
                    
                    if model_args.do_mlm:
                        batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

                    batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

                    if "label" in batch:
                        batch["labels"] = batch["label"]
                        del batch["label"]
                    if "label_ids" in batch:
                        batch["labels"] = batch["label_ids"]
                        del batch["label_ids"]
                    # batch["oov_to_id"]=oov_to_id
                    return batch
            
            
            def mask_tokens(
                self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
                """
                inputs = inputs.clone()
                labels = inputs.clone()
                # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                if special_tokens_mask is None:
                    special_tokens_mask = [
                        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                    ]
                    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
                else:
                    special_tokens_mask = special_tokens_mask.bool()

                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                inputs[indices_random] = random_words[indices_random]

                # The rest of the time (10% of the time) we keep the masked input tokens unchanged
                return inputs, labels
        data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer,max_length=data_args.max_seq_length)
        
        class OptunaCallback(TrainerCallback):
            def __init__(self, trial, rank):
                self.trial = trial
                self.rank = rank
            
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                logger.info(f"Rank {self.rank} metrics: {metrics}")
                if self.rank == 0:
                    if "eval_AMI" not in metrics:
                        logger.error("eval_AMI not found in metrics")
                        metric_value = 0
                    else:
                        metric_value = metrics.get("eval_AMI")
                    logger.info(f"Rank 0 reporting metric_value: {metric_value} at step {state.global_step}")
                    self.trial.report(metric_value, state.global_step)
                    should_prune = self.trial.should_prune()
                else:
                    should_prune = False
                
                # 同步剪枝决定
                should_prune = torch.tensor(should_prune, dtype=torch.bool, device="cuda")
                dist.broadcast(should_prune, src=0)
        
                if should_prune.item():
                    if self.rank == 0:
                        logger.info("Trial pruned")
                    raise optuna.exceptions.TrialPruned()
        trainer = CLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[OptunaCallback(trial, rank)]
            )
        trainer.model_args = model_args
        # 检查 DataLoader
        train_dataloader = trainer.get_train_dataloader()
        for i, batch in enumerate(train_dataloader):
            # rank = dist.get_rank() if dist.is_initialized() else 0
            shapes = {k: v.shape for k, v in batch.items()}
            print(f"Rank {rank}: Batch {i}, shapes = {shapes}")
            # print(f"Rank {rank}: Batch {i}, shapes = {[k: v.shape for k, v in batch.items()]}")
            if i > 3:
                break
        
        eval_dataset = Construct_EvalDataset(data_args.eval_file,tokenizer,data_args.max_seq_length)
        trainer.eval_dataloader = DataLoader(eval_dataset,shuffle=False,batch_size=training_args.per_device_eval_batch_size,collate_fn=eval_collate_fn)

        # 开始训练并返回验证集准确率
        trainer.train()
        
        #获取最佳指标和 epoch
        #只在Rank0上完成
        # 仅在 Rank 0 上返回结果
        logger.info(f"Rank {rank} starting final evaluate()")
        result = trainer.evaluate()
        logger.info(f"Rank {rank} evaluate() completed: {result}")

        if rank == 0:
            return result.get("eval_AMI", 0)
        return None

def cleanup_previous_trial(trial_number, rank):
    """清理上一个 trial 的中间文件，仅保留 eval_metrics.json 文件"""
    if rank == 0 and trial_number > 0:
        prev_trial_dir = f"./results/trial_{trial_number - 1}"
        if os.path.exists(prev_trial_dir):
            try:
                # 遍历目录中的所有文件和子目录
                for item in os.listdir(prev_trial_dir):
                    item_path = os.path.join(prev_trial_dir, item)
                    # 跳过 eval_metrics.json 文件
                    if item != "eval_metrics.json":
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        logger.info(f"Cleaned up {item_path}")
            except Exception as e:
                logger.error(f"Failed to clean up files in {prev_trial_dir}: {e}")
        else:
            logger.info(f"No directory found for {prev_trial_dir}")

def hyperparameter_optimization():
    logging.basicConfig(level=logging.INFO)
    logger.info(f"RANK: {os.environ.get('RANK')}")
    logger.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    logger.info(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    logger.info(f"Before setup_distributed, dist.is_initialized(): {dist.is_initialized()}")
    rank, world_size = setup_distributed()
    logger.info(f"Rank {rank}/{world_size} started")
    logger.info(f"After setup_distributed, dist.is_initialized(): {dist.is_initialized()}")
    
    train_file ="/root/shared-nvme/bioscan-5M/pretrain/pretrain.csv"
    eval_file="/root/shared-nvme/bioscan-5M/zsc.csv"
    tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": "main",
            "use_auth_token": None,
        }
    model_name_or_path="/home/dnacsecode/DNAcse/bertmodel"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    tokenizer.model_max_length=128
    train_dataset = prepare_data(tokenizer, train_file)
    # 创建或加载 Optuna study（仅 Rank 0）
    study_name = "hyperparam_search_v24"
    storage = "sqlite:///optuna_study_v24.db"
    if rank == 0:
        study = optuna.create_study(study_name=study_name,storage=storage,direction="maximize", pruner=optuna.pruners.MedianPruner(),load_if_exists=True)
        logger.info(f"Rank 0 loaded or created study with {len(study.trials)} completed trials")
    else:
        study = None
    # 定义回调函数
    def trial_callback(study, trial):
        if rank == 0:
            trial_data = {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": str(trial.state)  # 此时 state 已可用
            }
            # 试验开始时（RUNNING），只保存超参数
        # if trial.state == optuna.trial.TrialState.RUNNING:
        #     with open(f"./trial_{trial.number}_params.json", "w") as f:
        #         json.dump({"trial_number": trial.number, "params": trial.params}, f)
        #     logger.info(f"Trial {trial.number} started: params={trial.params}")
        
        # # 试验结束时（COMPLETED 或 PRUNED），保存超参数和结果
        # elif trial.state in [optuna.trial.TrialState.COMPLETED, optuna.trial.TrialState.PRUNED]:
            # with open(f"./trial_{trial.number}_result.json", "w") as f:
            #     json.dump(trial_data, f)
            # logger.info(f"Trial {trial.number} completed: params={trial.params}, value={trial.value}, state={trial.state}"
            with open(f"./{trial.number}_result.json", "w") as f:
                json.dump(trial_data, f)
            logger.info(f"new_params_{trial.number} completed: params={trial.params}, value={trial.value}")
    # 定义目标函数包装器
    def objective_wrapper(trial):
        # 在开始当前 trial 前清理上一个 trial
        cleanup_previous_trial(trial.number, rank)
        dist.barrier()  # 确保所有进程在清理后同步
        value = objective(trial, rank, world_size, train_dataset, tokenizer, eval_file)
       
        return value
        
    # 运行优化
    if rank == 0:
        remaining_trials =  30- len(study.trials)
        if remaining_trials > 0:
            logger.info(f"Rank 0 starting optimization with {remaining_trials} trials remaining")
            study.optimize(objective_wrapper, n_trials=remaining_trials,callbacks=[trial_callback])
        else:
            logger.info("all 10 trials already completed")
        print(f"Best AMI: {study.best_value}")
        print(f"Best params: {study.best_params}")
        with open("./best_params.json", "w") as f:
            json.dump({"best_params": study.best_params, "best_value": study.best_value}, f)
    else:
        for _ in range(30):  # 非 Rank 0 进程同步执行
            # objective_wrapper(optuna.trial.FixedTrial({}))
            try:
                objective_wrapper(optuna.trial.FixedTrial({}))
            except optuna.exceptions.TrialPruned:
                pass

    dist.barrier()  # 确保所有进程完成
    







def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # hyperparameter_optimization()
     # 如果是主进程，可以进一步使用best_params
    # if dist.get_rank() == 0 and best_params is not None:
    #     print("超参数优化完成，最佳参数已保存。")
    main()
