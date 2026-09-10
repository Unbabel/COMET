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

# Library code must not call logging.basicConfig(): it configures the root
# logger for the whole process the moment `comet` is imported, which is why
# e.g. `logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)` after
# `import comet` doesn't fully suppress PL's info/debug output (see #259).
# The comet-* CLI entry points call basicConfig themselves; library users
# (e.g. `evaluate.load("comet")`) now get to decide their own logging setup.
logger = logging.getLogger(__name__)


__version__ = "2.2.7"
__copyright__ = "2020 Unbabel. All rights reserved."
