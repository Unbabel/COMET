# -*- coding: utf-8 -*-
from .laser import LASEREncoder
from .bert import BERTEncoder
from .xlmr import XLMREncoder
from .encoder_base import Encoder

str2encoder = {"LASER": LASEREncoder, "BERT": BERTEncoder, "XLMR": XLMREncoder}

__all__ = ["LASEREncoder", "BERTEncoder", "XLMREncoder"]
