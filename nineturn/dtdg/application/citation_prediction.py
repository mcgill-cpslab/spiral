# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Use GCN-LSTM DTDG learning framework to  preform citation prediction."""

from nineturn.core.config import get_logger
from nineturn.dtdg.dataloader import ogb_dataset


def citation_prediction(dataset_name: str):
    """Perform citation prediction on the input dataset."""
    logger = get_logger()
    citation_graph = ogb_dataset('ogbl-citation2')
    total_observations = len(citation_graph)
    logger.info(
        f"""
                Total of observations in the dataset {dataset_name} is {total_observations}.
                """
    )
