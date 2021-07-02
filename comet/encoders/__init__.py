# -*- coding: utf-8 -*-
from .bert import BERTEncoder
from .xlmr import XLMREncoder

str2encoder = {"BERT": BERTEncoder, "XLM-RoBERTa": XLMREncoder}
