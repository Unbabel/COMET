# flake8: noqa
# -*- coding: utf-8 -*-
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

import logging

from .models import load_from_checkpoint, download_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


__version__ = "2.2.0"
__copyright__ = "2020 Unbabel. All rights reserved."
