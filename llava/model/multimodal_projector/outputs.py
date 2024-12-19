from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AudioModuleOutput:
    audio_features: torch.FloatTensor
    padding_mask: torch.BoolTensor


@dataclass
class CifProjectorOutput(AudioModuleOutput):
    ctc_loss: Optional[torch.FloatTensor] = None
    quantity_loss: Optional[torch.FloatTensor] = None


@dataclass
class CtcProjectorOutput(AudioModuleOutput):
    ctc_loss: Optional[torch.FloatTensor] = None
