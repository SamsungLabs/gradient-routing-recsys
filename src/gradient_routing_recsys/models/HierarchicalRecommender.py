import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.config import Config
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.model.abstract_recommender import ContextRecommender
from torch import Tensor

from gradient_routing_recsys.models.utils import sparsemax


class HierarchicalRecommender(ContextRecommender):
    def __init__(
        self,
        recbole_config: Config,
        dataset: Dataset,
        user_pref_net: nn.Module,
        recommender_net: nn.Module,
        loss: nn.MSELoss | nn.BCEWithLogitsLoss,
        cat_feat_field: str,
        input_user_fields_rec_net_flag: bool,
        user_pref_activation: str = "softmax",
        gumbel_softmax_temperature: float = 1.0,
    ):
        super(HierarchicalRecommender, self).__init__(recbole_config, dataset)

        self.user_pref_net = user_pref_net
        self.recommender_net = recommender_net
        self.loss = loss
        self.cat_feat_field = cat_feat_field
        self.input_user_fields_rec_net_flag = input_user_fields_rec_net_flag

        self.cat_feat_num = int(torch.max(dataset.item_feat[cat_feat_field]))
        self.user_pref_activation = user_pref_activation
        self.gumbel_softmax_temperature = gumbel_softmax_temperature

    def forward(self, interaction: Interaction) -> Tensor:
        batch_size = len(interaction)

        user_pref_net_interaction = self._prepare_user_pref_net_interaction(
            interaction, batch_size
        )

        user_pref_net_output = self._process_user_pref_net(
            user_pref_net_interaction, batch_size
        )

        user_pref_net_softmax = self._apply_activation_function(user_pref_net_output)

        user_pref_net_masked = self._mask_gradient_propagation(
            interaction, user_pref_net_softmax
        )

        recommender_net_interaction = self._prepare_recommender_net_interaction(
            interaction, user_pref_net_masked, batch_size
        )

        output = self.recommender_net(recommender_net_interaction)

        return output.squeeze(-1)

    def calculate_loss(self, interaction: Interaction) -> float:
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction: Interaction) -> Tensor:
        return self.forward(interaction)

    def _prepare_user_pref_net_interaction(
        self, interaction: Interaction, batch_size: int
    ) -> Interaction:
        user_pref_net_interaction = Interaction(
            {key: interaction.interaction[key] for key in self.user_field_names}
        ).repeat_interleave(self.cat_feat_num)

        # create the integer vector and tile it across the batch
        cat_id = (
            torch.arange(
                1,
                self.cat_feat_num + 1,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
        )  # (B·K,)

        user_pref_net_interaction["cat_id"] = cat_id

        return user_pref_net_interaction

    def _process_user_pref_net(
        self, user_pref_net_interaction: Interaction, batch_size: int
    ) -> Tensor:
        user_pref_net_output = self.user_pref_net(user_pref_net_interaction)

        reshaped_user_pref_net_output = torch.reshape(
            user_pref_net_output, (batch_size, -1)
        )

        return reshaped_user_pref_net_output

    def _apply_activation_function(self, output: Tensor) -> Tensor:
        """
        Apply the configured activation function to the user preference network output.

        Parameters
        ----------
        output : torch.Tensor
            The raw output tensor from the user preference network, shape (batch_size, num_classes)

        Returns
        -------
        torch.Tensor
            The activated tensor, shape (batch_size, num_classes)
        """
        if self.user_pref_activation == "softmax":
            return F.softmax(output, dim=1)
        elif self.user_pref_activation == "sigmoid":
            return F.sigmoid(output)
        elif self.user_pref_activation == "gumbel_softmax":
            return F.gumbel_softmax(
                output, tau=self.gumbel_softmax_temperature, hard=False, dim=1
            )
        elif self.user_pref_activation == "sparsemax":
            return sparsemax(output, dim=1)
        elif self.user_pref_activation == "relu":
            return F.relu(output)
        elif self.user_pref_activation == "silu":
            return F.silu(output)
        elif self.user_pref_activation == "tanh":
            return F.tanh(output)
        else:
            raise ValueError(
                f"Unknown activation function: {self.user_pref_activation}. "
                f"Supported options: softmax, sigmoid, gumbel_softmax, sparsemax, relu, silu, tanh"
            )

    def _mask_gradient_propagation(
        self, interaction: Interaction, user_pref_net_softmax: Tensor
    ) -> Tensor:
        multi_hot_vector = self._create_multi_hot_vector(
            interaction[self.cat_feat_field]
        )

        user_pref_net_masked = (
            multi_hot_vector * user_pref_net_softmax
            + (1 - multi_hot_vector) * user_pref_net_softmax.detach()
        )

        return user_pref_net_masked

    def _create_multi_hot_vector(self, idx_tensor: Tensor) -> Tensor:
        """
        Convert a (B, K) integer tensor of 1‑based indices (padded with 0)
        into a (B, C) multi‑hot matrix. On example:
        [[1, 2, 0],
         [4, 0, 0],
         [2, 3, 4]]
        should be converted into:
        [[1, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 1, 1, 1]]
        provided that there are 4 classes (C=4).

        Parameters
        ----------
        idx_tensor : torch.Tensor
            2‑D integer tensor (int64/int32). 0 is treated as padding and ignored.

        Returns
        -------
        torch.Tensor
            Multi‑hot tensor of shape (B, C) containing only 0/1.
        """

        # -------------------------------------------------
        # 1. Determine output width C
        # -------------------------------------------------
        C = self.cat_feat_num

        # -------------------------------------------------
        # 2. Shift to zero‑based indices, keep padding as –1
        # -------------------------------------------------
        #   (subtract 1 **only** for the real indices)
        #   Padding stays –1 → it will never match any column number.
        zero_based = idx_tensor - 1  # shape (B, K)
        zero_based[idx_tensor == 0] = -1  # padding → –1

        # -------------------------------------------------
        # 3. Build a boolean mask of shape (B, C)
        # -------------------------------------------------
        #   Compare each zero‑based index with every possible column.
        #   Broadcasting: (B, K, 1) vs (1, 1, C) → (B, K, C)
        col_range = torch.arange(C, device=idx_tensor.device)  # (C,)
        #   `unsqueeze(-1)` makes it (B, K, 1); `col_range` becomes (1, 1, C)
        match = zero_based.unsqueeze(-1) == col_range  # (B, K, C) bool

        #   Any True in the K‑dimension means that column should be hot.
        multi_hot = match.any(dim=1).to(idx_tensor.dtype)  # (B, C)

        return multi_hot

    def _prepare_recommender_net_interaction(
        self, interaction: Interaction, user_pref_net_masked: Tensor, batch_size: int
    ) -> Interaction:
        recommender_net_interaction = Interaction(
            {
                key: interaction.interaction[key]
                for key in self.user_field_names + self.item_field_names
            }
            if self.input_user_fields_rec_net_flag
            else {key: interaction.interaction[key] for key in self.item_field_names}
        )

        pos = (
            torch.arange(self.cat_feat_num, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .float()
        )
        final_user_pref_net_output = torch.stack([user_pref_net_masked, pos], dim=-1)

        recommender_net_interaction["user_prefs"] = final_user_pref_net_output

        return recommender_net_interaction
