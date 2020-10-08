f"""
Shell script tho reproduce results for BERTScores in data from WMT18/19 Metrics Shared task.
"""
import argparse
import hashlib
import logging
import os
import sys
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from tqdm import tqdm

from fairseq import utils
from fairseq import checkpoint_utils
from fairseq.data import LanguagePairDataset

#!/usr/bin/env python3
logger = logging.getLogger('prism')
logger.setLevel(logging.INFO)

MODELS = {
    '8412b2044da4b9b2c0a8ce87b305d0d1': {
        'name': 'm39v1',
        'path': 'todo',
        'date': '2020-04-30',
        'description': 'model released with arXiv paper April 2020',
        'langs': ['ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eo', 'fi', 'fr', 'he',
                  'hr', 'hu', 'id', 'it', 'ja', 'kk', 'lt', 'lv', 'mk', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
                  'sk', 'sl', 'sq', 'sr', 'sv', 'tr', 'uk', 'vi', 'zh'],
    }
}


def hash_model(model_dir):
    md5 = hashlib.md5()
    block_size = 2 ** 20
    for fname in ('checkpoint.pt', 'spm.model', 'dict.src.txt', 'dict.tgt.txt'):
        with open(os.path.join(model_dir, fname), "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                md5.update(data)
    md5.digest()
    return md5.hexdigest()

"""
Copy of https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_scorer.py
with softmax temperature control added 
"""
class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, temperature=1.0):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        self.temperature = temperature
        assert self.softmax_batch > 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model.forward(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample['target'] = tgt

                # divide the logits by temperature prior to softmax
                # for example, see https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py:
                #   decoder_out[0][:, -1:, :].div_(temperature)
                bd[0].div_(self.temperature)

                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            probs = probs.view(sample['target'].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                alignment = utils.extract_hard_alignment(avg_attn_i, sample['net_input']['src_tokens'][i],
                                                         sample['target'][i], self.pad, self.eos)
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
            }])
        return hypos

class Prism:
    def __init__(self, model_dir, lang, temperature):
        '''
        model_dir should contain:
         1) checkpoint.pt: the fairseq model
         2) spm.model: the sentencepiece model
         3) dict.src.txt: the fairseq source dictionary
         4) dict.tgt.txt: the fairseq target dictionary (likely a copy of the source)

        lang: ISO 639-1 Code (e.g. "en"). Must be a language compatable with the model.
        '''
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_dir + '/spm.model')

        self.lang = lang
        self.temperature = temperature

        # this prints things and I can't figure out how to disable it
        sys.stdout = open(os.devnull, 'w')
        self.models, self.args, self.task = checkpoint_utils.load_model_ensemble_and_task(
            [model_dir + '/checkpoint.pt', ],
            arg_overrides=dict(data=model_dir + '/'),
        )
        sys.stdout = sys.__stdout__

        self.use_cuda = torch.cuda.is_available()

        self.generator = SequenceScorer(self.task.target_dictionary, temperature=temperature)

        for model in self.models:
            if self.use_cuda:
                model.cuda()
            model.make_generation_fast_(
                beamable_mm_beam_size=None,
                need_attn=False,
            )
            # if model.args.fp16:
            #    model.half()

        # hash model
        self.model_hash = hash_model(model_dir)

        if self.model_hash in MODELS:
            model_langs = MODELS[self.model_hash]['langs']
            if lang not in model_langs:
                model_name = MODELS[self.model_hash]['name']
                logger.warning(f'Language "{lang}" is unsupported for model "{model_name}"')
                logger.warning(f'Supported languages for {model_name}: {", ".join(model_langs)}')
        else:
            logger.warning('unrecognized model, so cannot check language')

    def identifier(self):
        if self.model_hash in MODELS:
            model_name = MODELS[self.model_hash]['name']
        else:
            logger.warning('unrecognized model, using hash to identify')
            model_name = self.model_hash

        return dict(version='0.1', model=model_name, seg_scores='avg_log_prob', 
            sys_scores='avg_log_prob', log_base=2, temperature=self.temperature)

    def _binarize(self, sentence: str) -> torch.LongTensor:
        return self.task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long()

    def _encode(self, sent, prepend=True):
        sent = ' '.join(self.sp.EncodeAsPieces(sent))
        if prepend:
            sent = f'<{self.lang}> ' + sent
        return self._binarize(sent)

    def _build_batches(self,
                       source_tokens: List[List[int]],
                       target_tokens: List[List[int]],
                       skip_invalid_size_inputs: bool) -> Iterator[Dict[str, Any]]:
        source_lengths = torch.LongTensor([t.numel() for t in source_tokens])
        target_lengths = torch.LongTensor([t.numel() for t in target_tokens])

        batch_iterator = self.task.get_batch_iterator(
            dataset=LanguagePairDataset(source_tokens, source_lengths, self.task.source_dictionary,
                                        tgt=target_tokens, tgt_sizes=target_lengths,
                                        tgt_dict=self.task.target_dictionary),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=(2000, 2000),  # ???
            ignore_invalid_inputs=skip_invalid_size_inputs,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def _score_forward(self, tok_sents_in, tok_sents_out):
        assert len(tok_sents_in) == len(tok_sents_out)
        tok_level_scores = [None, ] * len(tok_sents_in)  # for debug
        results = [None, ] * len(tok_sents_in)
        for batch in self._build_batches(tok_sents_in, tok_sents_out, skip_invalid_size_inputs=False):
            if self.use_cuda:  # must be a better way
                batch['id'] = batch['id'].cuda()
                batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'].cuda()
                batch['net_input']['src_lengths'] = batch['net_input']['src_lengths'].cuda()
                batch['net_input']['prev_output_tokens'] = batch['net_input']['prev_output_tokens'].cuda()
                batch['target'] = batch['target'].cuda()

            translations = self.task.inference_step(self.generator, self.models, batch)

            ids = batch['id'].cpu().numpy()

            tok_scores = [x[0]['positional_scores'].cpu().numpy() for x in translations]

            # [1:] to skip language tag log prob
            sent_scores = [np.mean(x[1:]) for x in tok_scores]

            for _id, sent_score, _tok_score in zip(ids, sent_scores, tok_scores):
                results[_id] = sent_score
                tok_level_scores[_id] = _tok_score

        if logger.level == logging.DEBUG:
            for ii, (sent_in, scores_out, sent_out) in enumerate(zip(tok_sents_in, tok_level_scores, tok_sents_out)):
                sent_in_str = ' '.join([self.task.source_dictionary[x] for x in sent_in])
                logger.debug(f'Input[{ii}] = ' + sent_in_str)
                sent_out_tok = [self.task.source_dictionary[x] for x in sent_out]
                logger.debug(f'Output[{ii}] = ' + \
                             f' '.join([f'{a}[{b:.02f}]' for a, b in zip(sent_out_tok, scores_out)]))

        if None in results:
            raise Exception('Missing one or more sentence scores')

        return np.array(results)

    def score(self, cand, ref=None, src=None, segment_scores=False):

        if not (ref is None) ^ (src is None):
            raise Exception('Must provide exactly one of "ref" or "src"')

        tokenized_cand = [self._encode(sentence, prepend=False) for sentence in cand]
        tokenized_cand_prep = [self._encode(sentence, prepend=True) for sentence in cand]

        if src is not None:
            # Prism-src: score candidate given on source
            if len(cand) != len(src):
                raise Exception(f'Length of cand ({len(cand)}) does not match length of src ({len(src)})')
            tokenized_src = [self._encode(sentence, prepend=False) for sentence in src]
            scores = self._score_forward(tokenized_src, tokenized_cand_prep)

        else:
            # Prism-ref: average candidate given reference and reference given candidate
            if len(cand) != len(ref):
                raise Exception(f'Length of cand ({len(cand)}) does not match length of ref ({len(ref)})')
            tokenized_ref = [self._encode(sentence, prepend=False) for sentence in ref]
            tokenized_ref_prep = [self._encode(sentence, prepend=True) for sentence in ref]
            forward_scores = self._score_forward(tok_sents_in=tokenized_ref, tok_sents_out=tokenized_cand_prep)
            reverse_scores = self._score_forward(tok_sents_in=tokenized_cand, tok_sents_out=tokenized_ref_prep)
            scores = 0.5 * forward_scores + 0.5 * reverse_scores

        if not segment_scores:
            scores = np.mean(scores)

        return scores


def compute_kendall(
    hyp1_scores: list, hyp2_scores: list, dataframe: pd.DataFrame
) -> (int, list):
    """ Computes the official WMT19 shared task Kendall correlation score. """
    assert len(hyp1_scores) == len(hyp2_scores) == len(data)
    conc, disc = 0, 0
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Kendall eval..."):
        if hyp1_scores[i] > hyp2_scores[i]:
            conc += 1
        else:
            disc += 1

    return (conc - disc) / (conc + disc)


def run_prism(mt: list, ref: list, language=False, temperature=1.0) -> list:
    prism = Prism(model_dir="m39v1", lang=language, temperature=temperature)
    scores = prism.score(cand=mt, ref=ref, segment_scores=True)
    return list(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates BERTScores against relative preferences."
    )
    parser.add_argument(
        "--test_path",
        default="wmt-metrics/wmt19/de-en/relative-ranks.csv",
        help="Path to the test dataframe with relative preferences.",
        type=str,
    )
    parser.add_argument(
        "--language", default="en", help="Target language of the testset.", type=str,
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=1.0, 
        help='Softmax temperature: values >1.0 produce more uniform samples and values <1.0 produce sharper samples')
    parser.add_argument(
        "--run_wmt18",
        default=False,
        help="Runs entire WMT18 evaluation.",
        action="store_true",
    )
    parser.add_argument(
        "--run_wmt19",
        default=False,
        help="Runs entire WMT19 evaluation.",
        action="store_true",
    )

    args = parser.parse_args()
    if args.run_wmt18:
        lps = [
            "en-cs",
            "en-de",
            "en-et",
            "en-fi",
            "en-ru",
            "en-tr",
            "en-zh",
            "cs-en",
            "de-en",
            "et-en",
            "fi-en",
            "ru-en",
            "tr-en",
            "zh-en",
        ]
        kendall_scores = {}
        for lp in lps:
            data = pd.read_csv(f"wmt-metrics/wmt18/{lp}/relative-ranks.csv")
            hyp1_scores = run_prism([str(s) for s in data.hyp1], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
            hyp2_scores = run_prism([str(s) for s in data.hyp2], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
            #hyp1_scores = run_prism([str(s) for s in data.hyp1], list(data.ref), list(data.src), language=lp.split('-')[1])
            #hyp2_scores = run_prism([str(s) for s in data.hyp2], list(data.ref), list(data.src), language=lp.split('-')[1])
            kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
            print("Results for {}: {}".format(lp, kendall))
            kendall_scores[lp] = kendall
        print(kendall_scores)

    elif args.run_wmt19:
        lps = [
            "en-cs",
            "en-de",
            "en-fi",
            "en-gu",
            "en-kk",
            "en-lt",
            "en-ru",
            "en-zh",
            "de-en",
            "fi-en",
            "gu-en",
            "kk-en",
            "lt-en",
            "ru-en",
            "zh-en",
            "de-cs",
            "de-fr",
            "fr-de",
        ]
        kendall_scores = {}
        for lp in lps:
            data = pd.read_csv(f"wmt-metrics/wmt19/{lp}/relative-ranks.csv")
            hyp1_scores = run_prism([str(s) for s in data.hyp1], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
            hyp2_scores = run_prism([str(s) for s in data.hyp2], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
            kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
            print("Results for {}: {}".format(lp, kendall))
            kendall_scores[lp] = kendall
        print(kendall_scores)

    else:
        data = pd.read_csv(args.test_path)
        kendall_scores = {}
        hyp1_scores = run_prism([str(s) for s in data.hyp1], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
        hyp2_scores = run_prism([str(s) for s in data.hyp2], list(data.ref), language=lp.split('-')[1], temperature=args.temperature)
        kendall = compute_kendall(hyp1_scores, hyp2_scores, data)
        print("Results for {}: {}".format(args.test_path, kendall))
        kendall_scores[lp] = kendall
        print(kendall_scores)
