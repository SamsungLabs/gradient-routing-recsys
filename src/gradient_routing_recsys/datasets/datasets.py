import copy
from typing import List, Tuple

import torch
from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType


class SubModuleHierarchicalRecommenderDataset(Dataset):
    @property
    def field2type(self):
        return self._field2type

    @property
    def field2source(self):
        return self._field2source

    def num(self, field):
        if field == "cat_id":
            return self.num_cat_id
        elif field == "user_prefs":
            return self.num_user_prefs
        else:
            return super().num(field)

    @classmethod
    def cast(
        cls,
        dataset: Dataset,
        cat_feat_field: str,
        cols_to_remove: List[str],
        cols_to_add: List[Tuple[str, FeatureType, FeatureSource]],
    ) -> Dataset:
        casted_dataset = copy.copy(dataset)
        casted_dataset.__class__ = SubModuleHierarchicalRecommenderDataset
        casted_dataset._field2type = copy.copy(dataset.field2type)
        casted_dataset._field2source = copy.copy(dataset.field2source)

        casted_dataset.num_cat_id = dataset.num(cat_feat_field)
        casted_dataset.num_user_prefs = int(
            torch.max(dataset.item_feat[cat_feat_field])
        )

        for col in cols_to_remove:
            casted_dataset._field2type.pop(col)
            casted_dataset._field2source.pop(col)

        for col in cols_to_add:
            casted_dataset._field2type[col[0]] = col[1]
            casted_dataset._field2source[col[0]] = col[2]

        return casted_dataset
