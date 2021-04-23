#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import os.path as op
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils import (
    create_zip,
    create_zip_list,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

#SPLITS = [
#    "train-clean-100",
#    "train-clean-360",
#    "train-other-500",
#    "dev-clean",
#    "dev-other",
#    "test-clean",
#    "test-other",
#]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker", "align_time_min", "align_time_max", "total_time"]
#MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


def process(args):
    ''' 
    This version assume that the fbank features are extracted before e.g. for purpose of data augmentation

    Arguments:
     args.feature_root: a list containing the paths of extracted fbanks for zipping
     args.info_dict: a dict with *split* as key and file path as *value*
    '''

    os.makedirs(args.output_root, exist_ok=True)

    if args.feature_root is None:
        # Do not create new zip files
        assert args.path_fbankzip_root is not None, \
            'Please provide zipped filter banks'
        print('Load zipfile')
        zip_manifest = get_zip_manifest(args.path_fbankzip_root, 'fbank80.zip')
    else:
        zip_filename = "fbank80.zip"
        zip_path = op.join(args.output_root, zip_filename)
        print("ZIPing features...")
        create_zip_list(args.feature_root, zip_path) # Allow fbanks to be saved over different dirs. but are gathered for one zip file
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(args.output_root, zip_filename)

    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []

    # Take the info file for each split and generate the .tsv files
    # info file has 3 columns: 
    #  1) n_frames, 2) utterance id, and 3) transcription
    for split, info_path in args.info_dict.items():
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        with open(info_path, "r") as fin:
            for idx, l in enumerate(fin):
                line = l.strip().split()

                # Note: the utt_id here has no extra zeros for padding
                n_frames, uid, tp, tmin, tmax, total_time = line[0], line[1], \
                                                    line[2:-3], line[-3], \
                                                    line[-2], line[-1]
                _uid = uid.split("-")
                
                if args.augment:
                    spk_id, chapter_id, utt_id, aug_id = _uid[0], _uid[1], \
                                                            _uid[2], _uid[3]
                    sample_id = f"{spk_id}-{chapter_id}-{utt_id}-{aug_id}"
                else:
                    spk_id, chapter_id, utt_id = _uid[0], _uid[1], _uid[2]
                    sample_id = f"{spk_id}-{chapter_id}-{utt_id}"

                manifest["id"].append(sample_id)
                manifest["audio"].append(zip_manifest[sample_id])
                manifest["n_frames"].append(n_frames)
                manifest["tgt_text"].append(" ".join(tp))
                manifest["speaker"].append(spk_id)

                if split.startswith('train'):
                    manifest["align_time_min"].append(tmin)
                    manifest["align_time_max"].append(tmax)
                    manifest["total_time"].append(total_time)
                
            save_df_to_tsv(
                pd.DataFrame.from_dict(manifest), op.join(args.output_root, "{}.tsv".format(split))
            )

            if split.startswith('train'):
                print(f'Add {split} to train_text')
                train_text.extend(manifest["tgt_text"])
                print("length of train_text: {}".format(len(train_text)))

    # Generate vocab
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}"
    with NamedTemporaryFile(mode="w") as f_tmp:
        for t in train_text:
            f_tmp.write(t + "\n")
        gen_vocab(
            f_tmp.name,
            op.join(args.output_root, spm_filename_prefix),
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        args.output_root, spm_filename_prefix + ".model", specaugment_policy="ld"
    )
    # Clean up
    #shutil.rmtree(feature_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--feature-root", "-f", nargs='+', type=str)
    parser.add_argument("--info_dict", "-i", required=True, type=json.loads)
    parser.add_argument("--path-fbankzip-root", default=None, type=str)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
