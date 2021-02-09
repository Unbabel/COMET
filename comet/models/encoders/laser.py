# -*- coding: utf-8 -*-
r"""
LASER Encoder Model
===================================
    Pretrained LASER Encoder model from Facebook.
    https://github.com/facebookresearch/LASER

    Check the original papers:
        - https://arxiv.org/abs/1704.04154
        - https://arxiv.org/abs/1812.10464
    
    and the original implementation: https://github.com/facebookresearch/LASER
"""
import os
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from comet.models.encoders.encoder_base import Encoder
from comet.models.utils import convert_padding_direction, sort_sequences
from comet.tokenizers import FastBPEEncoder
from torchnlp.download import download_file_maybe_extract
from torchnlp.utils import lengths_to_mask

# LASER model trained for 93 different languages.
L93_LASER_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt"
)
L93_MODEL_NAME = "bilstm.93langs.2018-12-26.pt"


# LASER model trained with europarl parallel data.
EPARL_LASER_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/laser/models/bilstm.eparl21.2018-11-19.pt"
)
EPARL_MODEL_NAME = "bilstm.eparl21.2018-11-19.pt"

if "HOME" in os.environ:
    saving_directory = os.environ["HOME"] + "/.cache/torch/unbabel_comet/"
else:
    raise Exception("HOME environment variable is not defined.")


class LASEREncoder(Encoder):
    """
    Bidirectional LASER Encoder

    :param num_embeddings: Size of the vocabulary (73640 BPE tokens).
    :param padding_idx: Index of the padding token in the vocabulary.
    :param embed_dim: Size of the embeddings.
    :param hidden_size: Number of features of the LSTM hidden layer.
    :param num_layers: Number of LSTM stacked layers.
    :param bidirectinal: Flag to initialize a Bidirectional LSTM.
    :param left_pad: If set to True the inputs can be left padded.
        (internaly they will be converted to right padded inputs)
    :param padding_value: Value of the padding token.
    """

    def __init__(
        self,
        num_embeddings: int = 73640,
        padding_idx: int = 1,
        embed_dim: int = 320,
        hidden_size: int = 512,
        num_layers: int = 5,
        bidirectional: bool = True,
        left_pad: bool = True,
        padding_value: float = 0.0,
    ) -> None:
        super().__init__(None)
        self._output_units = hidden_size * 2 if bidirectional else hidden_size
        self._n_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim

        self.padding_idx = padding_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, embed_dim, padding_idx=self.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.left_pad = left_pad
        self.padding_value = padding_value

    @property
    def max_positions(self):
        return 4096  # More than that is not recommended

    @property
    def num_layers(self):
        return 1  # In LASER we can only use the last layer

    def freeze_embeddings(self):
        """ Freezes the embedding layer of the network to save some memory while training. """
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, hparams: Namespace):
        """Function that loads a pretrained LASER encoder and the respective tokenizer.

        :param hparams: Namespace.

        :returns: LASER Encoder model
        """
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)

        download_file_maybe_extract(
            L93_LASER_MODEL_URL,
            directory=saving_directory,
            check_files=[L93_MODEL_NAME],
        )
        state_dict = torch.load(saving_directory + L93_MODEL_NAME)
        encoder = LASEREncoder(**state_dict["params"])
        encoder.load_state_dict(state_dict["model"])
        encoder.tokenizer = FastBPEEncoder(state_dict["dictionary"])
        return encoder

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of sequences.

        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the lenght of each sequence [seq_len].

        :return: Dictionary with `sentemb` (tensor with dims [batch_size x output_units]), `wordemb`
            (tensor with dims [batch_size x seq_len x output_units]), `mask` (input mask),
            `all_layers` (List with word_embeddings from all layers, `extra` (tuple with the LSTM outputs,
            hidden states and cell states).
        """
        self.lstm.flatten_parameters()  # Is it required? should this be in the __init__?
        tokens, lengths, unsorted_idx = sort_sequences(tokens, lengths)

        if self.left_pad:
            # convert left-padding to right-padding
            tokens = convert_padding_direction(
                tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = tokens.size()

        # embed tokens
        x = self.embed_tokens(tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self._n_layers, bsz, self.hidden_size
        else:
            state_size = self._n_layers, bsz, self.hidden_size

        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        assert list(x.size()) == [seqlen, bsz, self.output_units]
        word_embeddings = x

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self._n_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        model_out = self.reorder_output(
            encoder_out={
                "sentemb": sentemb,
                "extra": (word_embeddings, final_hiddens, final_cells),
            },
            new_order=unsorted_idx,
        )
        model_out["mask"] = lengths_to_mask(lengths, device=tokens.device)
        model_out["wordemb"] = model_out["extra"][0].transpose(0, 1)
        model_out["all_layers"] = [model_out["wordemb"]]
        return model_out

    def reorder_output(
        self, encoder_out: Dict[str, torch.Tensor], new_order: torch.LongTensor
    ) -> Dict[str, torch.Tensor]:
        """
        Function that reorders the LASER encoder outputs at the batch level.

        :param encoder_out: the output of the forward function.
        :param new_order: the new order inside the batch.
        """
        # reorder encoder_out
        encoder_out["extra"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["extra"]
        )
        # reorder sentence embeddings
        encoder_out["sentemb"] = encoder_out["sentemb"].index_select(0, new_order)
        return encoder_out
