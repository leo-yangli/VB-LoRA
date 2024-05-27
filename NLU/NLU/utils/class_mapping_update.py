# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

# this script remaps classes to class strings so that it's quick to load such maps and not require
# loading all possible modeling files
#
# it can be extended to auto-generate other dicts that are needed at runtime


import os
import sys
from os.path import abspath, dirname, join


git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)

src = "src/transformers/models/auto/modeling_auto.py"
dst = "src/transformers/utils/modeling_auto_mapping.py"

if os.path.exists(dst) and os.path.getmtime(src) < os.path.getmtime(dst):
    # speed things up by only running this script if the src is newer than dst
    sys.exit(0)

# only load if needed
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING  # noqa


entries = "\n".join(
    [
        f'        ("{k.__name__}", "{v.__name__}"),'
        for k, v in MODEL_FOR_QUESTION_ANSWERING_MAPPING.items()
    ]
)
content = [
    "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
    "# 1. modify: models/auto/modeling_auto.py",
    "# 2. run: python utils/class_mapping_update.py",
    "from collections import OrderedDict",
    "",
    "",
    "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(",
    "    [",
    entries,
    "    ]",
    ")",
    "",
]
print(f"updating {dst}")
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(content))
