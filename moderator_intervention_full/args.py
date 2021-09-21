# based on transformers/examples/pytorch/text-classification/run_glue.py

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from transformers.file_utils import ExplicitEnum


class ModerationType(ExplicitEnum):
    ANY_MODERATION = "any_moderation"
    QUALITY_MODERATION = "quality_moderation"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    moderation_type: ModerationType = field(
        metadata={
            "help": "The type of moderation to model."
        }
    )
    data_dir: Optional[str] = field(
        default=str(Path.home() / 'data')
    )
    in_domain: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use only in-domain data (has effect only on quality moderation)."
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    comment_parents_num: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of comment parents to include."
        },
    )
    use_post_id: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use post id as context for the comment."
        },
    )
    split_rand_state: Optional[int] = field(
        default=42,
        metadata={
            "help": "The random state to split the data according to"
        },
    )

    def __post_init__(self):
        self.moderation_type = ModerationType(self.moderation_type)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    labels_num: Optional[int] = field(
        default=2, metadata={"help": "number of labels in the model output"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )


@dataclass
class KfoldTrainingArguments(TrainingArguments):
    optimize_threshold: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Optimize threshold."
        },
    )
    folds_num: Optional[int] = field(
        default=None,
        metadata={"help": "The number of folds."},
    )
    output_dir_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Output path prefix"}
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name in wandb"}
    )


def parse_arguments() -> tuple[ModelArguments, DataTrainingArguments, KfoldTrainingArguments, bool]:
    if debug := 'DEBUG' in os.environ:
        print('running in debug mode')
        sys.argv += ['--data_dir', str(Path.home() / 'data')]
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, KfoldTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # overwrite with smaller model for debugging if not loading locally
    if debug and 'results' not in model_args.model_name_or_path:
        model_args.model_name_or_path = 'squeezebert/squeezebert-uncased'
    if debug and model_args.tokenizer_name and 'results' not in model_args.tokenizer_name:
        model_args.tokenizer_name = 'squeezebert/squeezebert-uncased'

    for x in (model_args, data_args, training_args, debug):
        pprint(x)
    return model_args, data_args, training_args, debug
