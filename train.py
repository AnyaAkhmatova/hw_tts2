import argparse
import collections
import warnings

import numpy as np
import torch

import hw_tts.loss as module_loss
import hw_tts.model as module_arch
import hw_tts.mel_spectrogram as module_mel_spectrogram
from hw_tts.trainer import Trainer
from hw_tts.utils import prepare_device
from hw_tts.datasets import get_dataloader
from hw_tts.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloader = get_dataloader(**config["data"])

    mel_spectrogram = config.init_obj(config["mel_spectrogram"], module_mel_spectrogram)

    # build model architecture, then print to console
    generator = config.init_obj(config["generator_arch"], module_arch)
    discriminator_mpd = config.init_obj(config["mpd_arch"], module_arch)
    discriminator_msd = config.init_obj(config["msd_arch"], module_arch)
    logger.info(generator)
    logger.info(discriminator_mpd)
    logger.info(discriminator_msd)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    discriminator_mpd = discriminator_mpd.to(device)
    discriminator_msd = discriminator_msd.to(device)

    mel_spectrogram = mel_spectrogram.to(device)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss, mel_spectrogram=mel_spectrogram).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = [*filter(lambda p: p.requires_grad, generator.parameters()),
                        *filter(lambda p: p.requires_grad, discriminator_mpd.parameters()),
                        *filter(lambda p: p.requires_grad, discriminator_msd.parameters())]
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = None
    if "lr_scheduler" in config.config:
        if config.config["lr_scheduler"]["type"] == "OneCycleLR":
            lr_scheduler = config.init_obj(config["lr_scheduler"],
                                        torch.optim.lr_scheduler, 
                                        optimizer, steps_per_epoch=len(dataloader))
        else:
            lr_scheduler = config.init_obj(config["lr_scheduler"],
                            torch.optim.lr_scheduler, 
                            optimizer)

    trainer = Trainer(
        generator,
        discriminator_mpd,
        discriminator_msd,
        mel_spectrogram,
        loss_module,
        optimizer,
        lr_scheduler,
        config=config,
        device=device,
        dataloader=dataloader
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
