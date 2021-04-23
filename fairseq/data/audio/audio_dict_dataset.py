# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os.path as op
import re
import math
import random
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import get_fbank, get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator
)
from fairseq.data.audio.speech_to_text_dataset import (
    get_features_or_waveform,
    _collate_frames
)


logger = logging.getLogger(__name__)


class AudioDictDataset(SpeechToTextDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        audio_dict,
        align_time_min,
        align_time_max,
        total_time,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        # For aligned augmentation
        self.align_time_min = align_time_min
        self.align_time_max = align_time_max
        self.audio_dict = audio_dict
        self.audio_dict_size = len(self.audio_dict)
        self.total_time = total_time

        # Used in the +AuioDict part of ADA-LM/ADA-RT
        self.max_samp_fbank = self.data_cfg.max_samp_fbank 
        if self.max_samp_fbank is not None:
            assert isinstance(self.max_samp_fbank, int) and \
                self.max_samp_fbank >= 1
        self.num_samp_fbank = self.data_cfg.num_samp_fbank

        # Used in aligned masking (target side only  w/o audio dict)
        self.max_mask_fbank = self.data_cfg.max_mask_fbank
        self.num_mask_fbank = self.data_cfg.num_mask_fbank
    
        # % of data in a mini-batch to be applied with sampleFbank
        # prob: should be -1 when sample_fbank is not used
        self.sampleFbank_prob = self.data_cfg.sampleFbank_prob

        self.apply_alignAugment = self.data_cfg.apply_alignAugment

        self.roberta = None
        self.skip_roberta = self.data_cfg.skip_roberta
        logger.info('Skip roberta: {}'.format(self.skip_roberta))
        if self.apply_alignAugment:
            if not self.skip_roberta:
                from fairseq.models.roberta import RobertaModel
                self.roberta = RobertaModel.from_pretrained(
                    self.data_cfg.path_roberta, checkpoint_file='model.pt'
                )

                if self.data_cfg.roberta_fp16:
                    self.roberta.half()
                    
                logger.info('Inference of roberta with dtype: {}'.format(
                    (next(self.roberta.parameters())).dtype)
                )
                self.roberta.cuda() 
                self.roberta.eval()
            else:
                self.audio_dict_keys = list(self.audio_dict.keys())
                

        self.alignAugment_prob = self.data_cfg.alignAugment_prob
        self.alignMask = self.data_cfg.alignMask
        self.skip_source = self.data_cfg.skip_source
        self.percentMaskedTokens = self.data_cfg.percentMaskedTokens
        self.thresholdMaskedTokens = self.data_cfg.thresholdMaskedTokens
        if self.alignAugment_prob > 0 and self.alignAugment_prob <= 1:
            assert self.thresholdMaskedTokens >= 1

        self.random_time_mask_N = self.data_cfg.random_time_mask_N
        self.random_time_mask_T = self.data_cfg.random_time_mask_T
        self.random_time_mask_p = self.data_cfg.random_time_mask_p
        self.random_time_mask_limited = self.data_cfg.random_time_mask_limited
        if self.random_time_mask_N is not None \
            and self.random_time_mask_T is not None:
            self.time_mask_max = self.random_time_mask_N * \
                                    self.random_time_mask_T
    
        self.random_freq_mask_N = self.data_cfg.random_freq_mask_N
        self.random_freq_mask_F = self.data_cfg.random_freq_mask_F
        self.random_mask_value = self.data_cfg.random_mask_value #specaugment after ADA
        self.align_mask_value = self.data_cfg.align_mask_value 

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        logger.info(self.__repr__())

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples}, '
            f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
            f"roberta={self.roberta}, "
            f"skip_roberta={self.skip_roberta}, "
            f"alignAugment_prob={self.alignAugment_prob}, "
            f"self.alignMask={self.alignMask}, "
            f"self.skip_source={self.skip_source}, "
            f"self.percentMaskedTokens={self.percentMaskedTokens}, "
            f"self.thresholdMaskedTokens={self.thresholdMaskedTokens}, "
            f"self.random_time_mask_N={self.random_time_mask_N}, "
            f"self.random_time_mask_T={self.random_time_mask_T}, "
            f"self.random_time_mask_p={self.random_time_mask_p}, "
            f"self.random_time_mask_limited={self.random_time_mask_limited}, "
            f"self.random_freq_mask_N={self.random_freq_mask_N}, "
            f"self.random_freq_mask_F={self.random_freq_mask_F}, "
            f"self.random_mask_value={self.random_mask_value}, "
            f"self.align_mask_value={self.align_mask_value}, "
            f"self.sampleFbank_prob={self.sampleFbank_prob}, "
            f"self.max_samp_fbank={self.max_samp_fbank}, "
            f"self.num_samp_fbank={self.num_samp_fbank}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms}, "
        )

    def _augment_target(self, orig_sentence):
        '''
        To augment the target side based on Roberta model or
            random replacements from the keys of audio dictionary

        Arguments:
         orig_sentence (str): an input transcription

        Return:
         1. container (List[Tuple(position, word_from_roberta)])
         2. updated (str):
             The transcription with words prediced by roberta,
             or sampled from the keys of audio dictionary
        '''

        container, collect_sent = [], []
        updated = orig_sentence.split()

        positions = random.sample(
            range(len(updated)), 
            min(
                max(1, int(len(updated)*self.percentMaskedTokens)),
                self.thresholdMaskedTokens
            )
        )

        positions.sort()

        if not self.skip_roberta:
            with torch.no_grad():
                for pos in positions:
                    sent_list = orig_sentence.split()
                    sent_list[pos] = '<mask>'

                    collect_sent.append(' '.join(sent_list))

                _info = self.roberta.batch_fill_mask(collect_sent, topk=2)

                for pos, info in zip(positions, _info):
                    try:
                        item = info[1][-1].strip()
                    except:
                        item = info[0][-1].strip()
                    
                    if item in string.punctuation:
                        continue

                    item = item.upper()
                    updated[pos] = item
                    container.append((pos, item))
        else:
            # ADA-RT
            idx_tokens = random.sample(
                range(self.audio_dict_size), 
                len(positions)
            )
            for pos, tok in zip(positions, idx_tokens):
                updated[pos] = self.audio_dict_keys[tok]
                container.append((pos, self.audio_dict_keys[tok]))


        return container, ' '.join(updated), positions
        

    def _sample_fbank(self, 
                      spectrogram, 
                      transcription, 
                      time_min, 
                      time_max, 
                      scaling
    ):
        '''
        This is the data augmentation part by sampling from AudioDict.
        Since passing the audio_dict to funct inside can be slow.
        We do it here
        '''
        align_time_min = time_min.split('-')
        align_time_max = time_max.split('-')

        # Sample words for sampling fbanks
        transp_list = transcription.split()
        len_transp_list = len(transp_list)
        
        if int(self.num_samp_fbank) >= 1:
            _number_swapped = int(self.num_samp_fbank)
        elif float(self.num_samp_fbank) >= 0. and float(self.num_samp_fbank) < 1.:
            _number_swapped = math.floor(len_transp_list*self.num_samp_fbank)
        else:
            _number_swapped = len_transp_list

        number_swapped = min(max(1, _number_swapped), int(self.max_samp_fbank))
        positions = np.sort(
            np.random.choice(range(0, len_transp_list),
                             size=number_swapped,
                             replace=False)
        )
        positions.sort()

        collect_fbank_min_pos, collect_fbank_max_pos = [], []
        collect_sampled_fbanks = []

        for pos in positions:
            if transp_list[pos] not in self.audio_dict.keys():
                continue
            
            if len(self.audio_dict[transp_list[pos]]) <= 3:
                # Not enough varants for this word
                continue

            sampled_idx = np.random.choice(
                range(len(self.audio_dict[transp_list[pos]])),
                replace=False, size=1
            )

            word_sampled_fbank = self.audio_dict[
                transp_list[pos]][sampled_idx[0]
            ]
            sampled_fbank = np.concatenate(
                [v for k, v in word_sampled_fbank.items() if k != '_id']
            )
            
            fbank_min_pos = int(float(align_time_min[pos]) * scaling)
            fbank_max_pos = int(float(align_time_max[pos]) * scaling)

            collect_fbank_min_pos.append(fbank_min_pos)
            collect_fbank_max_pos.append(fbank_max_pos)
            collect_sampled_fbanks.append(sampled_fbank)

        if len(collect_fbank_max_pos) == 0:
            assert len(collect_fbank_min_pos) == 0
            # Words for positions sampled do not exist in AD
            return spectrogram

        # Update the fbank
        collect_fbank_max_pos.insert(0, 0)
        collect_fbank_min_pos.append(spectrogram.shape[0])
        collect_pos = [(max_pos, min_pos) for max_pos, min_pos in
                        zip(collect_fbank_max_pos, collect_fbank_min_pos)]
        collect_sampled_fbanks.append(np.array([])) # to maintain the same length

        fbank_updated = []
        for idx, ((max_idx, min_idx), fb) in enumerate(
            zip(collect_pos, collect_sampled_fbanks)
        ):
            remained_fbank = spectrogram[max_idx:(min_idx), :]
            fbank_updated.append(remained_fbank)

            if fb.shape[0] == 0:
                # because of the "maintain the same length"
                continue
            else:
                fbank_updated.append(fb)
        fbank_updated = np.concatenate(fbank_updated)

        return fbank_updated

    def _ADAMask(self, spectrogram, frames_masked):
        '''
        SpecAugment for ADA with extension to control the amount of
            random time maskings given the number of frames masked in
            aligned time maskings

        Note:
        #mask_value: in previous version: 0 here but mean in SpecAugment
        '''
        
        distorted = spectrogram.copy()
        num_frames = spectrogram.shape[0]
        num_freqs = spectrogram.shape[1]

        if self.random_mask_value is None:
            mask_value = spectrogram.mean()
        else:
            mask_value = self.random_mask_value

        for _i in range(self.random_freq_mask_N):
            f = np.random.randint(0, self.random_freq_mask_F)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0: f0 + f] = mask_value

        if self.random_time_mask_limited:
            # Restrict the amount of random time masking given
            # the amount of aligned time masking
            remained = self.time_mask_max - frames_masked
            if remained > 0:
                max_time_mask_t = (remained // self.random_time_mask_N)
            else:
                max_time_mask_t = -1
        else:
            # Normal specaugment
            max_time_mask_t = min(
                self.random_time_mask_T, 
                math.floor(num_frames * self.random_time_mask_p)
            )

        if max_time_mask_t < 1:
            return distorted

        for _i in range(self.random_time_mask_N):
            t = np.random.randint(0, max_time_mask_t)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0 : t0 + t, :] = mask_value

        return distorted


    def _alignAugment(self, source, index, scaling, align_mask=False, skip_source=False):
        '''
        Not sure if it is better to pass copies of align_time_min/max and tgt_texts instead

        Arguments:
         source: fbanks in numpy format
         index: index of data instance
         scaling: conversion factor between raw audio time and fbank time steps
         align_mask: Replace the corresponding fbanks with variable
            align_mask_value
         skip_source: No aligned masking or 
            audio dictionary is applied on source side.
            It is used in target-only augmentation

        Returns:
         1. spectrograms (np array)
         2. augmented transcriptions (str)
         3. number of frames masked in ADA (int)
         4. number of tokens replaced in transcriptions (int)
         5. number of hits on audio dictionary (int)
        '''

        aug_info, aug_tp, positions = self._augment_target(self.tgt_texts[index])
        align_time_min = self.align_time_min[index].split('-')
        align_time_max = self.align_time_max[index].split('-')
        frames_masked = 0
        hit_audioDict = 0

        assert len(aug_tp.split())==len(align_time_min)==len(align_time_max)

        if skip_source:
            ## Only target side augmentation
            return source, aug_tp, frames_masked, len(aug_info), 0

        # Generate fbanks for augmented words
        collect_fbank_min_pos, collect_fbank_max_pos = [], []
        collect_sampled_fbanks = []

        if self.align_mask_value is None:
            align_mask_value = source.mean()
        else:
            align_mask_value = self.align_mask_value

        for pos, word in aug_info:
            fbank_min_pos = int(float(align_time_min[pos]) * scaling)
            fbank_max_pos = int(float(align_time_max[pos]) * scaling)

            if align_mask or word not in self.audio_dict.keys():
                # Return masked spectrogram
                frames_masked += (fbank_max_pos - fbank_min_pos + 1)
                assert frames_masked >= 0
                source[fbank_min_pos:(fbank_max_pos+1),:] = align_mask_value
            else:
                # sample fbanks from AD
                hit_audioDict += 1

                sampled_idx = np.random.choice(
                    range(len(self.audio_dict[word])),
                    replace=False, size=1
                )

                word_sampled_fbank = self.audio_dict[word][sampled_idx[0]]
                sampled_fbank = np.concatenate(
                    [v for k, v in word_sampled_fbank.items() if k != '_id']
                )

                collect_fbank_min_pos.append(fbank_min_pos)
                collect_fbank_max_pos.append(fbank_max_pos)
                collect_sampled_fbanks.append(sampled_fbank)

        if not collect_fbank_min_pos and not collect_fbank_max_pos:
            # No augmented words exist in AD or no augmented target words
            assert (hit_audioDict == 0) and (frames_masked == 0) and \
                (hit_audioDict == 0)
            return source, aug_tp, frames_masked, len(aug_info), hit_audioDict

        # Update the fbank
        assert len(collect_fbank_min_pos)==len(collect_fbank_max_pos)\
                ==len(collect_sampled_fbanks)

        collect_fbank_max_pos.insert(0, 0)
        collect_fbank_min_pos.append(source.shape[0])
        collect_pos = [(max_pos, min_pos) for max_pos, min_pos in
                        zip(collect_fbank_max_pos, collect_fbank_min_pos)]
        collect_sampled_fbanks.append(np.array([])) # to maintain the same length

        fbank_updated = []
        for idx, ((max_idx, min_idx), fb) in enumerate(
            zip(collect_pos, collect_sampled_fbanks)
        ):
            remained_fbank = source[max_idx:(min_idx), :]
            fbank_updated.append(remained_fbank)

            if fb.shape[0] == 0:
                # because of the "maintain the same length"
                continue
            else:
                fbank_updated.append(fb)
        fbank_updated = np.concatenate(fbank_updated)

        return fbank_updated, aug_tp, frames_masked, len(aug_info), hit_audioDict


    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
        )

        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input

            scaling = source.shape[0] / float(self.total_time[index])
            transp_list = self.tgt_texts[index].split()
            tgt_texts, align_time_min, align_time_max = None, None, None
    
            if \
                self.is_train_split and \
                self.apply_alignAugment and \
                torch.rand([1]).item() <= float(self.alignAugment_prob) \
            :
                source, tgt_texts, frames_masked, tokens_masked, hit = \
                    self._alignAugment(
                        source, index, scaling, 
                        align_mask=self.alignMask,
                        skip_source=self.skip_source
                    )

                source = self._ADAMask(source, frames_masked)
            else:
                if tgt_texts is None:
                    assert align_time_min is None
                    assert align_time_max is None

                    tgt_texts = self.tgt_texts[index]
                    align_time_min = self.align_time_min[index]
                    align_time_max = self.align_time_max[index]
                  
                if \
                    self.is_train_split and \
                    self.audio_dict is not None and \
                    torch.rand([1]).item() <= self.sampleFbank_prob \
                :
                    ## Allow the original fbanks to be used under certain prob
                    source = self._sample_fbank(
                        source,
                        tgt_texts,
                        align_time_min,
                        align_time_max,
                        scaling
                    )

                # Call the standard SpecAugment
                source = self.feature_transforms(source)
                tokens_masked = hit = 0
            
        source = torch.from_numpy(source).float()

        target = None
        if self.tgt_texts is not None:
            #tokenized = self.tokenize_text(self.tgt_texts[index])
            tokenized = self.tokenize_text(tgt_texts)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        return index, source, target, tokens_masked, hit

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}

        indices = torch.tensor([i for i, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _ in samples], self.data_cfg.use_audio_input
        )

        tokens_masked = torch.tensor([i for _, _, _, i, _ in samples])
        hit = torch.tensor([i for _, _, _, _, i in samples])

        ntokens_masked = torch.sum(tokens_masked)
        nhit = torch.sum(hit)

        n_frames = torch.tensor([s.size(0) for _, s, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _ in samples)
        
        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "ntokens_masked": ntokens_masked,
            "nhit": nhit
        }

        return out


class AudioDictDatasetCreator(SpeechToTextDatasetCreator):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""
    # columns for alignment info.
    KEY_TIME_MIN, KEY_TIME_MAX = "align_time_min", "align_time_max"
    KEY_TOTAL_TIME = "total_time"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        audio_dict,
    ) -> AudioDictDataset:

        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        align_time_min, align_time_max, total_time = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])

            align_time_min.extend([ss[cls.KEY_TIME_MIN] for ss in s])
            align_time_max.extend([ss[cls.KEY_TIME_MAX] for ss in s])
            total_time.extend([ss[cls.KEY_TOTAL_TIME] for ss in s])

        return AudioDictDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            audio_dict,
            align_time_min,
            align_time_max,
            total_time,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )


    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        audio_dict
    ) -> AudioDictDataset:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                audio_dict
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
