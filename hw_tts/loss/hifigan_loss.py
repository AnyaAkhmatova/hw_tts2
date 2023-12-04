import torch
import torch.nn as nn


def adv_generator_loss(discr_outputs):
    loss = 0
    for out in discr_outputs:
        cur_loss = ((1.0 - out)**2).mean()
        loss += cur_loss
    return loss


def adv_discriminator_loss(discr_outputs, discr_target_outputs):
    loss = 0
    for outputs, target_outputs in zip(discr_outputs, discr_target_outputs):
        cur_loss = (outputs**2).mean()
        cur_target_loss = ((1.0 - target_outputs)**2).mean()
        loss = loss + cur_loss + cur_target_loss
    return loss


def feature_maps_loss(discr_fmaps, discr_target_fmaps):
    loss = 0
    for fmaps, target_fmaps in zip(discr_fmaps, discr_target_fmaps):
        for fmap, target_fmap in zip(fmaps, target_fmaps):
            loss += torch.abs(fmap - target_fmap).mean()
    return loss


def mel_specs_loss(mel_spectrogram, gen_outputs, targets):
    gen_mel_specs = mel_spectrogram(gen_outputs)
    target_mel_specs = mel_spectrogram(targets)
    loss = torch.abs(gen_mel_specs - target_mel_specs).mean()
    return loss


class HiFiGANLoss(nn.Module):
    def __init__(self, fm_coef, mel_coef, mel_spectrogram):
        super().__init__()
        self.fm_coef = fm_coef
        self.mel_coef = mel_coef
        self.mel_spectrogram = mel_spectrogram

    def forward(self, 
                gen_outputs, 
                targets, 
                mpd_outputs, 
                mpd_target_outputs,
                mpd_fmaps, 
                mpd_target_fmaps,
                msd_outputs, 
                msd_target_outputs,
                msd_fmaps, 
                msd_target_fmaps,
                **kwargs):
        
        gen_loss = adv_generator_loss(mpd_outputs) + adv_generator_loss(msd_outputs)

        mel_loss = mel_specs_loss(self.mel_spectrogram, gen_outputs, targets) * self.mel_coef

        disc_loss = adv_discriminator_loss(mpd_outputs, mpd_target_outputs) + \
                    adv_discriminator_loss(msd_outputs, msd_target_outputs)

        fmap_loss = (feature_maps_loss(mpd_fmaps, mpd_target_fmaps) + \
                     feature_maps_loss(msd_fmaps, msd_target_fmaps)) * self.fm_coef
        
        total_loss = gen_loss + mel_loss + disc_loss + fmap_loss
    
        return total_loss, gen_loss, mel_loss, disc_loss, fmap_loss
    
    