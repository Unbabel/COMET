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
from .bert import BERTEncoder
from .minilm import MiniLMEncoder
from .xlmr import XLMREncoder
from .rembert import RemBERTEncoder
from .xlmr_xl import XLMRXLEncoder

str2encoder = {
    "BERT": BERTEncoder,
    "XLM-RoBERTa": XLMREncoder,
    "MiniLM": MiniLMEncoder,
    "XLM-RoBERTa-XL": XLMRXLEncoder,
    "RemBERT": RemBERTEncoder,
}
