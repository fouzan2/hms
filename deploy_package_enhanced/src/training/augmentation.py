"""
Advanced data augmentation for EEG classification.
Includes time-domain and frequency-domain augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import random
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


class TimeDomainAugmentation:
    """Time-domain augmentation for raw EEG signals."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_config = config['training']['augmentation']['time_domain']
        
        # Build augmentation pipeline
        self.augmentations = self._build_augmentations()
        
    def _build_augmentations(self) -> List[Callable]:
        """Build list of augmentation functions."""
        augmentations = []
        
        if self.augmentation_config.get('time_shift', 0) > 0:
            augmentations.append(
                TimeShift(max_shift=self.augmentation_config['time_shift'])
            )
            
        if 'amplitude_scale' in self.augmentation_config:
            augmentations.append(
                AmplitudeScale(scale_range=self.augmentation_config['amplitude_scale'])
            )
            
        if self.augmentation_config.get('gaussian_noise', 0) > 0:
            augmentations.append(
                GaussianNoise(noise_level=self.augmentation_config['gaussian_noise'])
            )
            
        if self.augmentation_config.get('channel_dropout', 0) > 0:
            augmentations.append(
                ChannelDropout(drop_prob=self.augmentation_config['channel_dropout'])
            )
            
        if self.augmentation_config.get('time_warp', False):
            augmentations.append(TimeWarp())
            
        if self.augmentation_config.get('band_stop', False):
            augmentations.append(RandomBandStop())
            
        return augmentations
        
    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply augmentations to input tensor."""
        if not training:
            return x
            
        for aug in self.augmentations:
            if random.random() < aug.p:
                x = aug(x)
                
        return x


class TimeShift:
    """Random time shift augmentation."""
    
    def __init__(self, max_shift: float = 0.1, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random time shift.
        
        Args:
            x: Input tensor (batch, channels, time)
        """
        batch_size, n_channels, seq_len = x.shape
        max_shift_samples = int(seq_len * self.max_shift)
        
        if max_shift_samples == 0:
            return x
            
        # Random shift for each sample in batch
        shifts = torch.randint(-max_shift_samples, max_shift_samples + 1, 
                              (batch_size,))
        
        # Apply shifts
        x_shifted = torch.zeros_like(x)
        for i, shift in enumerate(shifts):
            if shift > 0:
                x_shifted[i, :, shift:] = x[i, :, :-shift]
            elif shift < 0:
                x_shifted[i, :, :shift] = x[i, :, -shift:]
            else:
                x_shifted[i] = x[i]
                
        return x_shifted


class AmplitudeScale:
    """Random amplitude scaling augmentation."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), 
                 p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling."""
        batch_size = x.shape[0]
        
        # Random scale for each sample
        scales = torch.empty(batch_size, 1, 1).uniform_(*self.scale_range)
        scales = scales.to(x.device)
        
        return x * scales


class GaussianNoise:
    """Add Gaussian noise augmentation."""
    
    def __init__(self, noise_level: float = 0.01, p: float = 0.5):
        self.noise_level = noise_level
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to signal."""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


class ChannelDropout:
    """Randomly drop EEG channels."""
    
    def __init__(self, drop_prob: float = 0.1, p: float = 0.5):
        self.drop_prob = drop_prob
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Drop random channels."""
        batch_size, n_channels, _ = x.shape
        
        # Create channel mask
        mask = torch.bernoulli(
            torch.ones(batch_size, n_channels, 1) * (1 - self.drop_prob)
        ).to(x.device)
        
        return x * mask


class TimeWarp:
    """Time warping augmentation using dynamic time warping."""
    
    def __init__(self, n_speed_changes: int = 3, max_speed_ratio: float = 1.2,
                 p: float = 0.5):
        self.n_speed_changes = n_speed_changes
        self.max_speed_ratio = max_speed_ratio
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        batch_size, n_channels, seq_len = x.shape
        
        # Generate random speed change points
        change_points = torch.sort(
            torch.randint(0, seq_len, (self.n_speed_changes + 2,))
        )[0]
        change_points[0] = 0
        change_points[-1] = seq_len
        
        # Generate random speed ratios
        speed_ratios = torch.empty(self.n_speed_changes + 1).uniform_(
            1 / self.max_speed_ratio, self.max_speed_ratio
        )
        
        # Apply time warping
        warped_x = torch.zeros_like(x)
        for i in range(batch_size):
            for ch in range(n_channels):
                warped_signal = self._warp_signal(
                    x[i, ch].cpu().numpy(),
                    change_points.numpy(),
                    speed_ratios.numpy()
                )
                warped_x[i, ch] = torch.from_numpy(warped_signal).to(x.device)
                
        return warped_x
        
    def _warp_signal(self, signal: np.ndarray, change_points: np.ndarray,
                     speed_ratios: np.ndarray) -> np.ndarray:
        """Warp a single signal."""
        warped_segments = []
        
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            segment = signal[start:end]
            
            # Resample segment
            new_length = int(len(segment) * speed_ratios[i])
            if new_length > 0:
                resampled = np.interp(
                    np.linspace(0, len(segment) - 1, new_length),
                    np.arange(len(segment)),
                    segment
                )
                warped_segments.append(resampled)
                
        # Concatenate and resize to original length
        warped = np.concatenate(warped_segments)
        return np.interp(
            np.linspace(0, len(warped) - 1, len(signal)),
            np.arange(len(warped)),
            warped
        )


class RandomBandStop:
    """Randomly remove frequency bands."""
    
    def __init__(self, n_bands: int = 1, band_width: float = 5.0,
                 freq_range: Tuple[float, float] = (1, 50), 
                 sampling_rate: float = 200, p: float = 0.3):
        self.n_bands = n_bands
        self.band_width = band_width
        self.freq_range = freq_range
        self.sampling_rate = sampling_rate
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random band-stop filter."""
        batch_size = x.shape[0]
        
        # Generate random frequency bands to remove
        filtered_x = x.clone()
        
        for i in range(batch_size):
            for _ in range(self.n_bands):
                # Random center frequency
                center_freq = np.random.uniform(*self.freq_range)
                
                # Create band-stop filter
                low_freq = max(0.5, center_freq - self.band_width / 2)
                high_freq = min(self.sampling_rate / 2 - 0.5, 
                               center_freq + self.band_width / 2)
                
                # Apply filter
                sos = signal.butter(4, [low_freq, high_freq], 
                                   btype='bandstop', 
                                   fs=self.sampling_rate, 
                                   output='sos')
                
                for ch in range(x.shape[1]):
                    filtered_x[i, ch] = torch.from_numpy(
                        signal.sosfiltfilt(sos, x[i, ch].cpu().numpy())
                    ).to(x.device)
                    
        return filtered_x


class FrequencyDomainAugmentation:
    """Frequency-domain augmentation for spectrograms."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_config = config['training']['augmentation']['frequency_domain']
        
        # Build augmentation pipeline
        self.augmentations = self._build_augmentations()
        
    def _build_augmentations(self) -> List[Callable]:
        """Build list of augmentation functions."""
        augmentations = []
        
        if self.augmentation_config.get('freq_mask', 0) > 0:
            augmentations.append(
                FrequencyMasking(freq_mask_param=self.augmentation_config['freq_mask'])
            )
            
        if self.augmentation_config.get('time_mask', 0) > 0:
            augmentations.append(
                TimeMasking(time_mask_param=self.augmentation_config['time_mask'])
            )
            
        if self.augmentation_config.get('spec_augment', False):
            augmentations.append(SpecAugment())
            
        if self.augmentation_config.get('mixup', False):
            augmentations.append(SpectrogramMixup())
            
        return augmentations
        
    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply augmentations to spectrogram."""
        if not training:
            return x
            
        for aug in self.augmentations:
            if random.random() < aug.p:
                x = aug(x)
                
        return x


class FrequencyMasking:
    """Frequency masking for spectrograms."""
    
    def __init__(self, freq_mask_param: float = 0.1, p: float = 0.5):
        self.freq_mask_param = freq_mask_param
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking.
        
        Args:
            x: Spectrogram tensor (batch, channels, freq, time)
        """
        batch_size, _, n_freq, _ = x.shape
        mask_size = int(n_freq * self.freq_mask_param)
        
        if mask_size == 0:
            return x
            
        # Apply masking to each sample
        masked_x = x.clone()
        for i in range(batch_size):
            # Random frequency band to mask
            f0 = torch.randint(0, n_freq - mask_size + 1, (1,)).item()
            masked_x[i, :, f0:f0 + mask_size, :] = 0
            
        return masked_x


class TimeMasking:
    """Time masking for spectrograms."""
    
    def __init__(self, time_mask_param: float = 0.1, p: float = 0.5):
        self.time_mask_param = time_mask_param
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking.
        
        Args:
            x: Spectrogram tensor (batch, channels, freq, time)
        """
        batch_size, _, _, n_time = x.shape
        mask_size = int(n_time * self.time_mask_param)
        
        if mask_size == 0:
            return x
            
        # Apply masking to each sample
        masked_x = x.clone()
        for i in range(batch_size):
            # Random time segment to mask
            t0 = torch.randint(0, n_time - mask_size + 1, (1,)).item()
            masked_x[i, :, :, t0:t0 + mask_size] = 0
            
        return masked_x


class SpecAugment:
    """SpecAugment: multiple frequency and time masks."""
    
    def __init__(self, n_freq_masks: int = 2, n_time_masks: int = 2,
                 freq_mask_param: float = 0.1, time_mask_param: float = 0.1,
                 p: float = 0.8):
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_masking = FrequencyMasking(freq_mask_param, p=1.0)
        self.time_masking = TimeMasking(time_mask_param, p=1.0)
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment."""
        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            x = self.freq_masking(x)
            
        # Apply time masks
        for _ in range(self.n_time_masks):
            x = self.time_masking(x)
            
        return x


class SpectrogramMixup:
    """Mixup augmentation for spectrograms."""
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
        
    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    ]:
        """
        Apply mixup to spectrograms.
        
        Args:
            x: Spectrogram batch
            y: Labels (optional)
            
        Returns:
            Mixed spectrograms and optionally mixed labels
        """
        batch_size = x.shape[0]
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix spectrograms
        mixed_x = lam * x + (1 - lam) * x[index]
        
        if y is not None:
            return mixed_x, y, y[index], lam
        else:
            return mixed_x


class CutMix:
    """CutMix augmentation for spectrograms."""
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        self.alpha = alpha
        self.p = p
        
    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
    ]:
        """Apply CutMix augmentation."""
        batch_size, _, h, w = x.shape
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        if y is not None:
            return mixed_x, y, y[index], lam
        else:
            return mixed_x


class AugmentationPipeline:
    """Complete augmentation pipeline for EEG data."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize augmentations
        self.time_augment = TimeDomainAugmentation(config)
        self.freq_augment = FrequencyDomainAugmentation(config)
        
        # Mixup/CutMix
        aug_config = config['training']['augmentation']
        self.mixup_alpha = aug_config['time_domain'].get('mixup_alpha', 1.0)
        self.use_cutmix = aug_config.get('use_cutmix', False)
        
        if self.use_cutmix:
            self.mix_augment = CutMix(alpha=self.mixup_alpha)
        else:
            self.mix_augment = SpectrogramMixup(alpha=self.mixup_alpha)
            
    def augment_batch(self, batch: Dict[str, torch.Tensor], 
                     training: bool = True) -> Dict[str, torch.Tensor]:
        """Apply augmentations to a batch."""
        if not training:
            return batch
            
        augmented_batch = batch.copy()
        
        # Time-domain augmentation for raw EEG
        if 'eeg' in batch:
            augmented_batch['eeg'] = self.time_augment(batch['eeg'], training)
            
        # Frequency-domain augmentation for spectrograms
        if 'spectrogram' in batch:
            augmented_batch['spectrogram'] = self.freq_augment(
                batch['spectrogram'], training
            )
            
        # Mixup/CutMix augmentation
        if 'label' in batch and random.random() < 0.5:
            if 'spectrogram' in augmented_batch:
                mixed_spec, y_a, y_b, lam = self.mix_augment(
                    augmented_batch['spectrogram'], batch['label']
                )
                augmented_batch['spectrogram'] = mixed_spec
                augmented_batch['label_a'] = y_a
                augmented_batch['label_b'] = y_b
                augmented_batch['mix_lambda'] = lam
                
        return augmented_batch


class TestTimeAugmentation:
    """Test-time augmentation for improved predictions."""
    
    def __init__(self, n_augmentations: int = 5, augmentation_config: Dict = None):
        self.n_augmentations = n_augmentations
        
        # Light augmentations for test time
        self.augmentations = [
            TimeShift(max_shift=0.05, p=1.0),
            AmplitudeScale(scale_range=(0.95, 1.05), p=1.0),
            GaussianNoise(noise_level=0.005, p=1.0)
        ]
        
    def __call__(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Apply test-time augmentation.
        
        Args:
            x: Input tensor
            model: Model for prediction
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        # Original prediction
        with torch.no_grad():
            pred = model(x)
            if isinstance(pred, dict):
                pred = pred['logits']
            predictions.append(F.softmax(pred, dim=1))
            
        # Augmented predictions
        for _ in range(self.n_augmentations - 1):
            # Apply random augmentation
            aug = random.choice(self.augmentations)
            x_aug = aug(x)
            
            with torch.no_grad():
                pred = model(x_aug)
                if isinstance(pred, dict):
                    pred = pred['logits']
                predictions.append(F.softmax(pred, dim=1))
                
        # Average predictions
        return torch.stack(predictions).mean(dim=0) 