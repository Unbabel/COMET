# -*- coding: utf-8 -*-
from .bert import BERTEncoder
from .xlmr import XLMREncoder
from .encoder_base import Encoder

str2encoder = {"BERT": BERTEncoder, "XLMR": XLMREncoder}

__all__ = ["BERTEncoder", "XLMREncoder"]
