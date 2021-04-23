# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy \
    import LabelSmoothedCrossEntropyCriterion


@register_criterion("xent_ctc_loss")
class Xent_CTC_Criterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        ctc_weight=0.3,
        zero_infinity=False
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, 
            ignore_prefix_size=ignore_prefix_size, 
            report_accuracy=report_accuracy
        )

        # For CTC 
        self.blank_idx = task.target_dictionary.bos()
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()

        self.ctc_weight = ctc_weight
        self.zero_infinity = zero_infinity
        print('criterion sentence-avg: ', self.sentence_avg)
        print('ctc_weight: ', self.ctc_weight)
        print('zero_infinity: ', self.zero_infinity)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--ctc-weight', default=0.3, type=float,
                            help='0.3: => 0.7 for attn loss')
        parser.add_argument('--zero-infinity', default=False, type=bool)
                            
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        ctc_loss = self.compute_ctc_loss(model, net_output, sample, reduce=reduce)

        # The gradient is first computed over 'sum' 
        # then average in reduce_metrics():
        # fairseq/tasks/fairseq_task.py: compute gradients
        # fairseq/trainer.py: reduce_metrics and grad update
        # Another solution: set reduction='mean' in ctc.
         # Multiply it by sample_size so that its sample_size
        # is cancelled by the /sample_size in reduce_metrics
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        loss = (1-self.ctc_weight)*loss + self.ctc_weight*ctc_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ctc_loss": ctc_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if 'nhit' in sample and 'ntokens_masked' in sample:
            logging_output['nhit'] = sample['nhit'].data
            logging_output['ntokens_masked'] = sample['ntokens_masked'].data
        else:
            logging_output['nhit'] = 0.
            logging_output['ntokens_masked'] = 0.

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, sample, reduce=True):
        '''
        lptobs: T x B x C
        targets: B x T'
        '''
        ctc_output = net_output[1]['ctc_output'] #T X B X C
        lprobs = F.log_softmax(ctc_output, dim=-1)

        if "src_lengths" in sample["net_input"]:
            # 4: the down-sampling factor
            input_lengths = sample["net_input"]["src_lengths"]
            input_lengths = torch.floor_divide(input_lengths, 4)
        else:
            ## Assume same length for each source input
            input_lengths = torch.full(
                (ctc_output.size(1),), ctc_output.size(0), dtype=torch.long
            )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        target_lengths = pad_mask.sum(-1)
        targets_flat = sample["target"].masked_select(pad_mask)

        #with torch.backends.cudnn.flags(enabled=False):
        loss = F.ctc_loss(
            lprobs,
            targets=targets_flat,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=self.blank_idx,
            reduction="sum",
            zero_infinity=self.zero_infinity,
        )
            #target_lengths=target_lengths,

        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)

        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        if 'nhit' in logging_outputs[0] and \
            'ntokens_masked' in logging_outputs[0]:
            nhit = sum(log['nhit'] for log in logging_outputs)
            ntokens_masked = sum(
                log['ntokens_masked'] for log in logging_outputs
            )
            assert nhit <= ntokens_masked
            if ntokens_masked > 0:
                hit_rate = nhit / ntokens_masked
            else:
                hit_rate = -1

            #TODO: check how to fill the 3 arguments below
            metrics.log_scalar("nhit", nhit, round=3, weight=0)
            metrics.log_scalar("ntokens_masked", ntokens_masked, round=3, weight=0)
            metrics.log_scalar("hit_rate", hit_rate, round=3, weight=0)

        # May have to adjust below for CTC loss as well 
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
