import math
import numbers
from typing import Optional

import numpy as np
from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    register_audio_feature_transform,
)


@register_audio_feature_transform("alignAugment")
class alignAugmentTransform(AudioFeatureTransform):

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return alignAugmentTransform(
            _config.get("alignAugment_prob", 0.3),
            _config.get("apply_alignAugment", True),
            _config.get("alignMask", False),
            _config.get("skip_source", False),
            _config.get("percentMaskedTokens", 0.2),
            _config.get("thresholdMaskedTokens", 5),
            _config.get("time_mask_limited", False),
            _config.get("time_mask_N", 2),
            _config.get("time_mask_T", 40),
            _config.get("time_mask_p", 1.0),
            _config.get("freq_mask_N", 2),
            _config.get("freq_mask_F", 30),
            _config.get("random_mask_value", None),
            _config.get("align_mask_value", 0.),
            _config.get("path_roberta", None),
            _config.get("skip_roberta", False),
        )

    def __init__(self, 
                 alignAugment_prob, 
                 apply_alignAugment, 
                 alignMask, 
                 skip_source,
                 percentMaskedTokens,
                 thresholdMaskedTokens,
                 time_mask_limited,
                 time_mask_N,
                 time_mask_T,
                 time_mask_p, 
                 freq_mask_N,
                 freq_mask_F,
                 random_mask_value,
                 align_mask_value,
                 path_roberta,
                 skip_roberta,
        ):

        self.alignAugment_prob = alignAugment_prob
        self.apply_alignAugment = apply_alignAugment
        self.alignMask = alignMask
        self.skip_source = skip_source
        self.percentMaskedTokens = percentMaskedTokens
        self.thresholdMaskedTokens = thresholdMaskedTokens

        self.time_mask_limited = time_mask_limited
        self.time_mask_N = time_mask_N
        self.time_mask_T = time_mask_T
        self.time_mask_p = time_mask_p
        self.freq_mask_N = freq_mask_N
        self.freq_mask_F = freq_mask_F
        self.align_mask_value = align_mask_value
        self.random_mask_value = random_mask_value
        self.path_roberta = path_roberta
        self.skip_roberta = skip_roberta
        

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"alignAugment_prob={self.alignAugment_prob}",
                    f"apply_alignAugment={self.apply_alignAugment}",
                    f"alignMask={self.alignMask}",
                    f"skip_source={self.skip_source}",
                    f"percentMaskedTokens={self.percentMaskedTokens}",
                    f"thresholdMaskedTokens={self.thresholdMaskedTokens}",
                    f"time_mask_limited={self.time_mask_limited}",
                    f"time_mask_N={self.time_mask_N}",
                    f"time_mask_T={self.time_mask_T}",
                    f"time_mask_p={self.time_mask_p}",
                    f"freq_mask_N={self.freq_mask_N}",
                    f"freq_mask_F={self.freq_mask_F}",
                    f"align_mask_value={self.align_mask_value}",
                    f"random_mask_value={self.random_mask_value}",
                    f"path_roberta={self.path_roberta}",
                    f"skip_roberta={self.skip_roberta}",
                ]
            )
            + ")"
        )

    def __call__(self, spectrogram):
        '''
        To maintain consistsnt API with feature_transform
        '''
        return spectrogram
