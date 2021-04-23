import math
import numbers
from typing import Optional

import numpy as np
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("samp_fbank")
class SampFbankTransform(AudioFeatureTransform):

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return SampFbankTransform(
            _config.get("max_samp_fbank", 5),
            _config.get("num_samp_fbank", -1),
            _config.get("sampleFbank_prob", 1.0)
        )

    def __init__(self, max_samp_fbank, num_samp_fbank, sampleFbank_prob):
        assert max_samp_fbank >= 1
        self.max_samp_fbank = max_samp_fbank
        self.num_samp_fbank = num_samp_fbank
        self.sampleFbank_prob = sampleFbank_prob

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"max_samp_fbank={self.max_samp_fbank}",
                    f"num_samp_fbank={self.num_samp_fbank}",
                    f"sampleFbank_prob={self.sampleFbank_prob}",
                ]
            )
            + ")"
        )

    def __call__(self, spectrogram):
        '''
            To maintain consistsnt API with feature_transform
        '''

        return spectrogram
