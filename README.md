Implementation of our paper: On-the-Fly Aligned Data Augmentation for Sequence-to-Sequence ASR

#### Requirements
- Python 3.6
- fairseqv1.0

#### Scripts added/modified for ADA
- fairseq/criterions/s2t_xent_ctc_loss.py
- fairseq/criterions/label_smoothed_cross_entropy.py
- fairseq/models/speech_to_text/__init__.py
- fairseq/models/speech_to_text/s2t_transformer.py
- fairseq/models/speech_to_text/s2t_ctc_transformer.py
- fairseq/models/roberta/hub_interface.py
- fairseq/models/transformer.py
- fairseq/data/audio/speech_to_text_dataset.py
- fairseq/data/audio/audio_dict_dataset.py
- fairseq/data/audio/feature_transforms/alignAugment.py
- fairseq/data/audio/feature_transforms/samp_fbank.py
- fairseq/data/audio/feature_transforms/specaugment.py 
- fairseq/tasks/speech_to_text.py
- fairseq/sequence_generator.py
