import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import numpy as np
import PIL
from torchvision.transforms import ToTensor

from .base_trainer import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.logger.utils import plot_spectrogram_to_buf


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            discriminator_mpd,
            discriminator_msd,
            mel_spectrogram,
            criterion,
            optimizer,
            lr_scheduler,
            config,
            device,
            dataloader,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, discriminator_mpd, discriminator_msd, criterion, optimizer, lr_scheduler, config, device)
        self.config = config
        self.train_dataloader = dataloader
        self.skip_oom = skip_oom
        self.mel_spectrogram = mel_spectrogram

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.log_step = self.len_epoch // 5

        self.train_metrics = MetricTracker(
            "total_loss", "gen_loss", "mel_loss", "disc_loss", "fmap_loss", "grad norm"
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["targets"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.discriminator_mpd.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.discriminator_msd.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.discriminator_mpd.train()
        self.discriminator_msd.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.len_epoch)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=len(self.train_dataloader))
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} total_loss: {:.6f}, gen_loss: {:.6f}, mel_loss: {:.6f}"
                    " disc_loss: {:.6f}, fmap_loss: {:.6f}".format(
                        epoch, 
                        self._progress(batch_idx), 
                        batch["total_loss"].item(), 
                        batch["gen_loss"].item(), 
                        batch["mel_loss"].item(), 
                        batch["disc_loss"].item(),
                        batch["fmap_loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_sample(**batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        batch["targets_spec"] = self.mel_spectrogram(batch["targets"])
        batch["gen_outputs"] = self.generator(batch["targets_spec"]).squeeze()

        max_len = max(batch["targets"].shape[-1], batch["gen_outputs"].shape[-1])
        batch["targets"] = nn.functional.pad(batch["targets"], (0, max_len-batch["targets"].shape[-1]))
        batch["gen_outputs"] = nn.functional.pad(batch["gen_outputs"], (0, max_len-batch["gen_outputs"].shape[-1]))

        mpd_outputs, mpd_fmaps, mpd_target_outputs, mpd_target_fmaps = self.discriminator_mpd(batch["gen_outputs"], batch["targets"])
        msd_outputs, msd_fmaps, msd_target_outputs, msd_target_fmaps = self.discriminator_msd(batch["gen_outputs"], batch["targets"])
        batch["mpd_outputs"] = mpd_outputs
        batch["mpd_target_outputs"] = mpd_target_outputs
        batch["mpd_fmaps"] = mpd_fmaps
        batch["mpd_target_fmaps"] = mpd_target_fmaps
        batch["msd_outputs"] = msd_outputs
        batch["msd_target_outputs"] = msd_target_outputs
        batch["msd_fmaps"] = msd_fmaps
        batch["msd_target_fmaps"] = msd_target_fmaps

        total_loss, gen_loss, mel_loss, disc_loss, fmap_loss = self.criterion(**batch)
        batch["total_loss"] = total_loss
        batch["gen_loss"] = gen_loss
        batch["mel_loss"] = mel_loss
        batch["disc_loss"] = disc_loss
        batch["fmap_loss"] = fmap_loss

        if is_train:
            batch["total_loss"].backward()
            # self._clip_grad_norm()
            if torch.isnan(torch.tensor(self.get_grad_norm())).item() == 1:
                self.optimizer.zero_grad()
                self.logger.warning("Nan grad. Skipping batch.")
            else:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        metrics.update("total_loss", total_loss.item())
        metrics.update("gen_loss", gen_loss.item())
        metrics.update("mel_loss", mel_loss.item())
        metrics.update("disc_loss", disc_loss.item())
        metrics.update("fmap_loss", fmap_loss.item())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_sample(self, targets, gen_outputs, **kwargs):
        ind = np.random.choice(targets.shape[0])
        self.writer.add_audio("real_audio", targets[ind], sample_rate=22050)
        self.writer.add_audio("generated_audio", gen_outputs[ind], sample_rate=22050)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = [*self.generator.parameters(), *self.discriminator_mpd.parameters(), *self.discriminator_msd.parameters()]
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

