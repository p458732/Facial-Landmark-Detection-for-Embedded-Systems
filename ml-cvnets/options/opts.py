#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Optional

from common import SUPPORTED_MODALITIES
from cvnets import modeling_arguments
from data.collate_fns import arguments_collate_fn
from data.datasets import arguments_dataset
from data.sampler import add_sampler_arguments
from data.text_tokenizer import arguments_tokenizer
from data.transforms import arguments_augmentation
from data.video_reader import arguments_video_reader
from loss_fn import add_loss_fn_arguments
from metrics import METRICS_REGISTRY, arguments_stats
from optim import arguments_optimizer
from optim.scheduler import arguments_scheduler
from options.utils import load_config_file
from utils import logger


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # convert values into dict
        override_dict = {}
        for val in values:
            if val.find("=") < 0:
                logger.error(
                    "For override arguments, a key-value pair of the form key=value is expected. Got: {}".format(
                        val
                    )
                )
            val_list = val.split("=")
            if len(val_list) != 2:
                logger.error(
                    "For override arguments, a key-value pair of the form key=value is expected with only one value per key. Got: {}".format(
                        val
                    )
                )
            override_dict[val_list[0]] = val_list[1]

        # determine the type of each value from parser actions and set accordingly
        options = parser._actions
        for option in options:
            option_dest = option.dest
            if option_dest in override_dict:
                val = override_dict[option_dest]
                if type(option.default) == bool and option.nargs == 0:
                    # Boolean argument
                    # value could be false, False, true, True
                    override_dict[option_dest] = (
                        True if val.lower().find("true") > -1 else False
                    )
                elif option.nargs is None:
                    # when nargs is not defined, it is usually a string, int, and float.
                    override_dict[option_dest] = option.type(val)
                elif option.nargs in ["+", "*"]:
                    # for list, we expect value to be comma separated
                    val_list = val.split(",")
                    override_dict[option_dest] = [option.type(v) for v in val_list]
                else:
                    logger.error(
                        "Following option is not yet supported for overriding. Please specify in config file. Got: {}".format(
                            option
                        )
                    )
        setattr(namespace, "override_args", override_dict)


def arguments_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="Common arguments", description="Common arguments"
    )

    group.add_argument(
        "--taskname",
        type=str,
        default="",
        help="Name of the task (can have arbitrary values)",
    )
    group.add_argument("--common.seed", type=int, default=0, help="Random seed")
    group.add_argument(
        "--common.config-file", type=str, default=None, help="Configuration file"
    )
    group.add_argument(
        "--common.results-loc",
        type=str,
        default="results",
        help="Directory where results will be stored",
    )
    group.add_argument(
        "--common.run-label",
        type=str,
        default="run_1",
        help="Label id for the current run",
    )
    group.add_argument(
        "--common.eval-stage-name",
        type=str,
        default="evaluation",
        help="Name to be used while logging in evaluation stage.",
    )

    group.add_argument(
        "--common.resume", type=str, default=None, help="Resume location"
    )
    group.add_argument(
        "--common.finetune",
        type=str,
        default=None,
        help="Checkpoint location to be used for finetuning",
    )
    group.add_argument(
        "--common.finetune-ema",
        type=str,
        default=None,
        help="EMA Checkpoint location to be used for finetuning",
    )

    group.add_argument(
        "--common.mixed-precision", action="store_true", help="Mixed precision training"
    )
    group.add_argument(
        "--common.mixed-precision-dtype",
        type=str,
        default="float16",
        help="Mixed precision training data type",
    )
    group.add_argument(
        "--common.accum-freq",
        type=int,
        default=1,
        help="Accumulate gradients for this number of iterations",
    )
    group.add_argument(
        "--common.accum-after-epoch",
        type=int,
        default=0,
        help="Start accumulation after this many epochs",
    )
    group.add_argument(
        "--common.log-freq",
        type=int,
        default=100,
        help="Display after these many iterations",
    )
    group.add_argument(
        "--common.auto-resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    group.add_argument(
        "--common.grad-clip", type=float, default=None, help="Gradient clipping value"
    )
    group.add_argument(
        "--common.k-best-checkpoints",
        type=int,
        default=5,
        help="Keep k-best checkpoints",
    )
    group.add_argument(
        "--common.save-all-checkpoints",
        action="store_true",
        default=False,
        help="If True, will save checkpoints from all epochs",
    )

    group.add_argument(
        "--common.inference-modality",
        type=str,
        default="image",
        choices=SUPPORTED_MODALITIES,
        help="Inference modality. Image or videos",
    )

    group.add_argument(
        "--common.channels-last",
        action="store_true",
        default=False,
        help="Use channel last format during training. "
        "Note 1: that some models may not support it, so we recommend to use it with caution"
        "Note 2: Channel last format does not work with 1-, 2-, and 3- tensors. "
        "Therefore, we support it via custom collate functions",
    )

    group.add_argument(
        "--common.tensorboard-logging",
        action="store_true",
        help="Enable tensorboard logging",
    )
    group.add_argument(
        "--common.override-kwargs",
        nargs="*",
        action=ParseKwargs,
        help="Override arguments. Example. To override the value of --sampler.vbs.crop-size-width, "
        "we can pass override argument as "
        "--common.override-kwargs sampler.vbs.crop_size_width=512 \n "
        "Note that keys in override arguments do not contain -- or -",
    )

    group.add_argument(
        "--common.enable-coreml-compatible-module",
        action="store_true",
        help="Use coreml compatible modules (if applicable) during inference",
    )

    group.add_argument(
        "--common.debug-mode",
        action="store_true",
        help="You can use this flag for debugging purposes.",
    )

    # intermediate checkpoint related args
    group.add_argument(
        "--common.save-interval-freq",
        type=int,
        default=0,
        help="Save checkpoints every N updates. Defaults to 0",
    )

    try:
        from internal.utils.opts import arguments_internal

        parser = arguments_internal(parser=parser)
    except ModuleNotFoundError:
        logger.debug("Cannot load internal arguments, skipping.")

    return parser


def arguments_ddp(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="DDP arguments")
    group.add_argument(
        "--ddp.rank",
        type=int,
        default=0,
        help="Node rank for distributed training. Defaults to 0.",
    )
    group.add_argument(
        "--ddp.world-size",
        type=int,
        default=-1,
        help="World size for DDP. Defaults to -1, meaning use all GPUs.",
    )
    group.add_argument(
        "--ddp.dist-url", type=str, default=None, help="DDP URL. Defaults to None."
    )
    group.add_argument(
        "--ddp.dist-port",
        type=int,
        default=30786,
        help="DDP Port. Only used when --ddp.dist-url is not specified. Defaults to 30768.",
    )
    group.add_argument(
        "--ddp.device-id", type=int, default=None, help="Device ID. Defaults to None."
    )
    group.add_argument(
        "--ddp.backend", type=str, default="nccl", help="DDP backend. Default is nccl"
    )
    group.add_argument(
        "--ddp.find-unused-params",
        action="store_true",
        default=False,
        help="Find unused params in model. useful for debugging with DDP. Defaults to False.",
    )

    group.add_argument(
        "--ddp.use-deprecated-data-parallel",
        action="store_true",
        default=False,
        help="Use Data parallel for training. This flag is not recommended for training and should be used only for debugging. \
            The support for this flag will be deprecating in future.",
    )

    return parser


def parser_to_opts(parser: argparse.ArgumentParser, args: Optional[List[str]] = None):
    # parse args
    opts = parser.parse_args(args)
    opts = load_config_file(opts)
    return opts


def get_training_arguments(
    parse_args: Optional[bool] = True, args: Optional[List[str]] = None
):
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # dataset related arguments
    parser = arguments_dataset(parser=parser)

    # cvnet arguments, including models
    parser = modeling_arguments(parser=parser)

    # sampler related arguments
    parser = add_sampler_arguments(parser=parser)

    # collate fn  related arguments
    parser = arguments_collate_fn(parser=parser)

    # transform related arguments
    parser = arguments_augmentation(parser=parser)

    # Video reader related arguments
    # Should appear after arguments_augmentations(parser=parser) because "--frame-augmentation.*" depends on "--image-augmentation.*"
    parser = arguments_video_reader(parser=parser)

    # loss function arguments
    parser = add_loss_fn_arguments(parser=parser)

    # optimizer arguments
    parser = arguments_optimizer(parser=parser)
    parser = arguments_scheduler(parser=parser)

    # DDP arguments
    parser = arguments_ddp(parser=parser)

    # stats arguments
    parser = arguments_stats(parser=parser)

    # common
    parser = arguments_common(parser=parser)

    # text tokenizer arguments
    parser = arguments_tokenizer(parser=parser)

    # metric arguments
    parser = METRICS_REGISTRY.all_arguments(parser=parser)

    if parse_args:
        return parser_to_opts(parser, args)
    else:
        return parser


def get_eval_arguments(parse_args=True, args: Optional[List[str]] = None):
    return get_training_arguments(parse_args=parse_args, args=args)


def get_conversion_arguments(args: Optional[List[str]] = None):
    parser = get_training_arguments(parse_args=False)

    # Arguments related to coreml conversion
    group = parser.add_argument_group("Conversion arguments")
    group.add_argument(
        "--conversion.coreml-extn",
        type=str,
        default="mlmodel",
        help="Extension for converted model. Default is mlmodel",
    )
    group.add_argument(
        "--conversion.input-image-path",
        type=str,
        default=None,
        help="Path of the image to be used for conversion",
    )

    # Arguments related to server.
    group.add_argument(
        "--conversion.bucket-name", type=str, help="Model job's bucket name"
    )
    group.add_argument("--conversion.task-id", type=str, help="Model job's id")
    group.add_argument(
        "--conversion.viewers",
        type=str,
        nargs="+",
        default=None,
        help="Users who can view your models on server",
    )

    # parse args
    return parser_to_opts(parser, args=args)


def get_benchmarking_arguments(args: Optional[List[str]] = None):
    parser = get_training_arguments(parse_args=False)

    #
    group = parser.add_argument_group("Benchmarking arguments")
    group.add_argument(
        "--benchmark.batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    group.add_argument(
        "--benchmark.warmup-iter", type=int, default=10, help="Warm-up iterations"
    )
    group.add_argument(
        "--benchmark.n-iter",
        type=int,
        default=100,
        help="Number of iterations for benchmarking",
    )
    group.add_argument(
        "--benchmark.use-jit-model",
        action="store_true",
        help="Convert the model to JIT and then benchmark it",
    )

    # parse args
    return parser_to_opts(parser, args=args)


def get_loss_landscape_args(args: Optional[List[str]] = None):
    parser = get_training_arguments(parse_args=False)

    group = parser.add_argument_group("Loss landscape related arguments")
    group.add_argument(
        "--loss-landscape.n-points",
        type=int,
        default=11,
        help="No. of grid points. Default is 11, so we have 11x11 grid",
    )
    group.add_argument(
        "--loss-landscape.min-x",
        type=float,
        default=-1.0,
        help="Min. value along x-axis",
    )
    group.add_argument(
        "--loss-landscape.max-x",
        type=float,
        default=1.0,
        help="Max. value along x-axis",
    )
    group.add_argument(
        "--loss-landscape.min-y",
        type=float,
        default=-1.0,
        help="Min. value along y-axis",
    )
    group.add_argument(
        "--loss-landscape.max-y",
        type=float,
        default=1.0,
        help="Max. value along y-axis",
    )

    # parse args
    return parser_to_opts(parser, args=args)
