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
import itertools
from typing import List, Tuple, Any

import torch
from torch.utils.data import Sampler
from collections import OrderedDict


class ModelOutput(OrderedDict):
    """ This was copied from previous versions of HuggingFace Transformers. Latest
    version made some breaking changes into ModelOutputs which impacted Prediction
    and Target classes defined bellow.

    Base class for all model outputs as dataclass. Has a `__getitem__` that allows
    indexing by integer or slice (like a tuple) or strings (like a dictionary) 
    that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`to_tuple`] method to 
    convert it to a tuple before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class Prediction(ModelOutput):
    """Renamed ModelOutput"""

    pass


class Target(ModelOutput):
    """Renamed ModelOutput into Targets to keep same behaviour"""

    pass


class LabelSet:
    """Taken from: https://github.com/LightTag/sequence-labeling-with-transformers/"""

    def __init__(self, labels: List[str] = ["minor", "major", "critical"]):
        self.labels_to_id = {}
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "I")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l


class OrderedSampler(Sampler[int]):
    """Sampler that returns the indices in a deterministic order."""

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def flatten_metadata(metadata):
    """Metadata from the model output can be in various forms and this function
    will gather all metadata and flatten everything.
    """
    metadata = Prediction(**{k: [dic[k] for dic in metadata] for k in metadata[0]})
    for k, v in metadata.items():
        if torch.is_tensor(v[0]):
            # If we have tensors we can use cat to flatten them.
            metadata[k] = torch.cat(v, dim=0).tolist()
        else:
            # for other predictions such as word tags we have to flatten the list.
            metadata[k] = [item for sublist in v for item in sublist]
    return metadata


def restore_list_order(sorted_list, sort_ids):
    """Restores the original ids of a given list."""
    unsorted_list = [None for _ in range(len(sorted_list))]
    for i, s in zip(sort_ids, sorted_list):
        unsorted_list[i] = s
    return unsorted_list


def split_sequence_into_sublists(sequence, n):
    elements_per_chunk, remainder = divmod(len(sequence), n)
    return [
        sequence[
            i * elements_per_chunk
            + min(i, remainder) : (i + 1) * elements_per_chunk
            + min(i + 1, remainder)
        ]
        for i in range(n)
    ]
