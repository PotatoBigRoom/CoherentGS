# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Base Configs"""

from __future__ import annotations

from typing import Any, Tuple, Type

from dataclasses import dataclass


# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)
