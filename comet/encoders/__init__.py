# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import transformers
from packaging import version

from .bert import BERTEncoder
from .minilm import MiniLMEncoder
from .xlmr import XLMREncoder

str2encoder = {"BERT": BERTEncoder, "XLM-RoBERTa": XLMREncoder, "MiniLM": MiniLMEncoder}

if version.parse(transformers.__version__) >= version.parse("4.17.0"):
    from .xlmr_xl import XLMRXLEncoder

    str2encoder["XLM-RoBERTa-XL"] = XLMRXLEncoder
