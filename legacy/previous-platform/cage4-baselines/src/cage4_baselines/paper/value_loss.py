"""Symmetric critic loss for the external MACPO core.

The pinned upstream Huber helper uses e > delta for its linear branch and
silently drops e < -delta. Finite-episode MC targets regularly reach that tail.
Keep the author clipping, masks and reduction while fixing this loss branch.
"""
import torch


def symmetric_value_loss(trainer,values,previous,returns,active):
    target=trainer.value_normalizer(returns) if trainer._use_popart else returns
    def pointwise(prediction):
        error=target-prediction
        if not trainer._use_huber_loss:return error.square()/2
        magnitude=error.abs();delta=trainer.huber_delta
        return torch.where(magnitude<=delta,error.square()/2,delta*(magnitude-delta/2))
    loss=pointwise(values)
    if trainer._use_clipped_value_loss:
        clipped=previous+(values-previous).clamp(-trainer.clip_param,trainer.clip_param)
        loss=torch.maximum(loss,pointwise(clipped))
    return (loss*active).sum()/active.sum() if trainer._use_value_active_masks else loss.mean()
