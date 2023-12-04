import argparse
import json
import os
from pathlib import Path
import glob
from glob import glob
import io

import torch
import torchaudio
from torchvision.transforms import ToTensor

import hw_tts.model as module_model
import hw_tts.mel_spectrogram as module_mel_spectrogram
from hw_tts.utils.parse_config import ConfigParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

import wandb

import warnings 
warnings.filterwarnings("ignore")


def plot_spectrogram_to_buf(spectrogram_tensor, name=None):
    plt.figure(figsize=(20, 5))
    plt.imshow(spectrogram_tensor)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["generator_arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["generator_state_dict"]
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    wav_files_path = config["inference"]["path"]
    wav_files_names = []

    for name in glob(wav_files_path+'/*.wav'):
        wav_files_names.append(name)

    mel_spectrogram = config.init_obj(config["mel_spectrogram"], module_mel_spectrogram).to(device)

    output_dir = config["inference"]["output_path"]
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    wandb_logging = config["inference"]["wandb_logging"]
    if wandb_logging:
        wandb.init(
                project=config['trainer'].get('wandb_project'),
                config=config
            )

    df = pd.DataFrame(columns=["generated audio", "real audio", "mel_spectogram"])
    idx = 0

    with torch.no_grad():
        for name in wav_files_names:
            wav, sr = torchaudio.load(name)
            wav.squeeze_()
            wav = wav.unsqueeze(0).to(device)
            spec = mel_spectrogram(wav)
            gen_output = model(spec)
            wav = wav[0].cpu().detach()
            spec = spec[0].cpu().detach()
            gen_output = gen_output[0].cpu().detach()

            torchaudio.save(output_dir + '/' + name.split('/')[-1].split('.')[0] + '_inf.wav', 
                            src=gen_output, sample_rate=sr)
            if wandb_logging:
                spec = spec.numpy()
                image_spec = PIL.Image.open(plot_spectrogram_to_buf(spec))

                df.loc[idx] = [wandb.Audio(gen_output.reshape(-1, 1), sample_rate=sr),
                               wandb.Audio(wav.reshape(-1, 1), sample_rate=sr), 
                               wandb.Image(ToTensor()(image_spec))]
                idx += 1
            
    if wandb_logging:
        wandb.log({"inference_results": wandb.Table(dataframe=df)})
        print('Table is added to wandb')


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

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    assert config.config.get("inference", {}) is not None

    main(config)
