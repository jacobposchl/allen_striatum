"""Symmetric NuCLR/DCL loss with within-sample negatives and matched unit IDs."""
import torch


def _gather_rows(matrix, rows):
    return matrix.gather(1, rows.unsqueeze(-1).expand(*rows.shape, matrix.shape[-1]))


def symmetric_nuclr_loss(
    first, second, anchor_rows_a, anchor_rows_b, anchor_mask,
    unit_mask_a, unit_mask_b, temperature=0.15, decoupled=True, full_denominator=True,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Accumulate logits in float32 even under mixed precision.
    first, second = first.float(), second.float()
    similarity = torch.einsum("bnd,bmd->bnm", first, second) / temperature
    logits_a = _gather_rows(similarity, anchor_rows_a).masked_fill(~unit_mask_b[:, None], -torch.inf)
    logits_b = _gather_rows(similarity.transpose(1, 2), anchor_rows_b).masked_fill(~unit_mask_a[:, None], -torch.inf)
    positive_a = logits_a.gather(2, anchor_rows_b.unsqueeze(-1)).squeeze(-1)
    positive_b = logits_b.gather(2, anchor_rows_a.unsqueeze(-1)).squeeze(-1)
    denom_a, denom_b = logits_a, logits_b
    if decoupled:
        denom_a = denom_a.scatter(2, anchor_rows_b.unsqueeze(-1), -torch.inf)
        denom_b = denom_b.scatter(2, anchor_rows_a.unsqueeze(-1), -torch.inf)
    if full_denominator:
        same_a = _gather_rows(first @ first.transpose(1, 2) / temperature, anchor_rows_a)
        same_b = _gather_rows(second @ second.transpose(1, 2) / temperature, anchor_rows_b)
        same_a = same_a.masked_fill(~unit_mask_a[:, None], -torch.inf).scatter(2, anchor_rows_a.unsqueeze(-1), -torch.inf)
        same_b = same_b.masked_fill(~unit_mask_b[:, None], -torch.inf).scatter(2, anchor_rows_b.unsqueeze(-1), -torch.inf)
        denom_a, denom_b = torch.cat((denom_a, same_a), -1), torch.cat((denom_b, same_b), -1)
    valid = anchor_mask & positive_a.isfinite() & positive_b.isfinite()
    valid &= denom_a.isfinite().any(-1) & denom_b.isfinite().any(-1)
    # logsumexp(all -inf) has NaN gradients even if its result is masked later.
    denom_a = denom_a.masked_fill(~valid[..., None], 0)
    denom_b = denom_b.masked_fill(~valid[..., None], 0)
    positive_a, positive_b = positive_a.masked_fill(~valid, 0), positive_b.masked_fill(~valid, 0)
    loss_a = -positive_a + denom_a.logsumexp(-1)
    loss_b = -positive_b + denom_b.logsumexp(-1)
    weight = valid.sum()
    normalizer = weight.clamp_min(1)
    loss = (0.5 * (loss_a + loss_b)).masked_fill(~valid, 0).sum() / normalizer
    accuracy = 0.5 * (
        ((logits_a.argmax(-1) == anchor_rows_b) & valid).sum()
        + ((logits_b.argmax(-1) == anchor_rows_a) & valid).sum()
    ) / normalizer
    chance = 0.5 * (
        (valid / unit_mask_b.sum(1, keepdim=True).clamp_min(1)).sum()
        + (valid / unit_mask_a.sum(1, keepdim=True).clamp_min(1)).sum()
    ) / normalizer
    return loss, accuracy, chance, weight


class NuCLRLoss:
    def __init__(self, temperature=0.15, decoupled=True, full_denominator=True):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature, self.decoupled, self.full_denominator = temperature, decoupled, full_denominator

    def __call__(self, output_a, output_b, batch):
        loss, accuracy, chance, weight = symmetric_nuclr_loss(
            output_a.projection, output_b.projection,
            batch["anchor_rows_a"], batch["anchor_rows_b"], batch["anchor_mask"],
            output_a.unit_mask, output_b.unit_mask,
            self.temperature, self.decoupled, self.full_denominator,
        )
        return loss, {
            "loss": loss.detach().item(), "retrieval_accuracy": accuracy.item(),
            "retrieval_chance": chance.item(), "selection_score": (accuracy - chance).item(),
            "_weight": weight.item(),
        }
