#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerModel
from torch import Tensor


logger = logging.getLogger(__name__)


@register_model("s2t_ctc_transformer")
class S2T_CTC_TransformerModel(S2TTransformerModel):
    """
    Add extra output layer after TransformerEncoder for 
    multi-tasking training with CTC
    """


    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerDecoderScriptable(
                args, task.target_dictionary, embed_tokens
                )


class TransformerDecoderScriptable(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, 
            no_encoder_attn=no_encoder_attn
        )
        self.ctc_output_layer = nn.Linear(
            args.encoder_embed_dim, len(dictionary)
        )

        
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

        ctc_output = self.ctc_output_layer(encoder_out.encoder_out)

        #return x, None
        return x, {'ctc_output': ctc_output}


@register_model_architecture(model_name="s2t_ctc_transformer", arch_name="s2t_ctc_transformer")
def base_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)


@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_transformer_s")
def s2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_transformer_espnetSmall")
def s2t_transformer_espnetSmall(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 256)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_transformer_espnetSmallConv512")
def s2t_transformer_espnetSmallConv512(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base")
def s2t_ctc_base(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_ker5")
def s2t_ctc_base_ker5(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512")
def s2t_ctc_base_enc512(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512_head8")
def s2t_ctc_base_enc512_head8(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "transf_may")
def transf_may(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512_ffn2048")
def s2t_ctc_base_enc512_ffn2048(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512_ffn2048_drop15p")
def s2t_ctc_base_enc512_ffn2048(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512_ffn2048_head8")
def s2t_ctc_base_enc512_ffn2048_head8(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_base_enc512_ffn2048_drop15p_head8")
def s2t_ctc_base_enc512_ffn2048_head8(args):
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "3,3")
    args.conv_channels = getattr(args, "conv_channels", 512)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)

@register_model_architecture("s2t_ctc_transformer", "s2t_ctc_transformer_m")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)

