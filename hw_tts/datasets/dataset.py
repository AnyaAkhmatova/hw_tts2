import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

import pathlib

import pandas as pd
import random


class LJSpeech(Dataset):
    def __init__(self, path2csv, dataset_root, max_time_samples=8192, train=True):
        super().__init__()
        self.path2csv = path2csv
        self.dataset_root = pathlib.Path(dataset_root)
        self.max_time_samples = max_time_samples
        self.train = train

        df = pd.read_csv(
            path2csv,
            names=['id', 'gt', 'gt_letters_only'],
            sep="|"
        )
        df = df.dropna()

        self.names = list(df['id'])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset_root / f'wavs/{self.names[idx]}.wav')
        wav.squeeze_()

        if self.train:
            if wav.size(-1) - self.max_time_samples > 0:
                start_sample = random.randint(0, wav.size(-1) - self.max_time_samples)
                wav = wav[start_sample:start_sample + self.max_time_samples]

        return wav


def collate_fn(wav_list):
    max_wav_len = max([wav.shape[-1] for wav in wav_list])
    wav_tensor = torch.zeros(len(wav_list), max_wav_len)
    wav_lens = torch.zeros(len(wav_list))

    for i in range(len(wav_list)):
        wav_tensor[i, :wav_list[i].shape[-1]] = wav_list[i]
        wav_lens[i] = wav_list[i].shape[-1]

    return {"targets": wav_tensor, "targets_len": wav_lens}


def get_dataloader(path2csv, dataset_root, max_time_samples, batch_size, num_workers, **kwargs):
    dataset = LJSpeech(path2csv, dataset_root, max_time_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers, 
                            collate_fn=collate_fn, pin_memory=True)
    return dataloader
