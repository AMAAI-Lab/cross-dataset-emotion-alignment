import torch
from torchmetrics import Metric
from torchmetrics.functional import f1_score, precision, recall
from torchmetrics.utilities import dim_zero_cat


class SongLevelMetric(Metric):
    def __init__(self, num_labels, top_k):
        super().__init__()
        self.add_state("similarities", default=[], dist_reduce_fx="cat")
        self.add_state("track_ids", default=[], dist_reduce_fx="cat")
        self.add_state("true_labels", default=[], dist_reduce_fx="cat")
        self.num_labels = num_labels
        self.top_k = torch.tensor(top_k)

    def update(
        self, similarities: torch.Tensor, track_ids: torch.Tensor, true_labels: torch.Tensor
    ):
        self.similarities.append(similarities)
        self.track_ids.append(track_ids)
        self.true_labels.append(true_labels)

    def compute(self):
        similarities = dim_zero_cat(self.similarities)
        track_ids = dim_zero_cat(self.track_ids)
        true_labels = dim_zero_cat(self.true_labels)

        unique_track_ids = torch.unique(track_ids)
        predicted_labels = torch.zeros(
            (len(unique_track_ids), self.num_labels), device=self.device
        )
        true_labels_per_song = torch.zeros(
            (len(unique_track_ids), self.num_labels), device=self.device
        )

        for i, track_id in enumerate(unique_track_ids):
            mask = track_ids == track_id
            track_similarities = similarities[mask]
            track_true_labels = true_labels[mask][
                0
            ]  # Assuming all segments of a track have the same labels

            # Majority voting
            _, top_k_indices = torch.topk(track_similarities, k=self.top_k, dim=1)
            votes = torch.zeros(self.num_labels, device=self.device)
            votes.scatter_add_(
                0,
                top_k_indices.view(-1),
                torch.ones_like(top_k_indices.view(-1), dtype=torch.float),
            )

            _, top_k_indices = torch.topk(votes, k=self.top_k)
            predicted_labels[i, top_k_indices] = 1.0

            true_labels_per_song[i] = track_true_labels

        prec = precision(
            predicted_labels,
            true_labels_per_song,
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
        )
        rec = recall(
            predicted_labels,
            true_labels_per_song,
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
        )
        f1 = f1_score(
            predicted_labels,
            true_labels_per_song,
            task="multilabel",
            num_labels=self.num_labels,
            average="macro",
        )

        return {"precision": prec, "recall": rec, "f1_score": f1}
