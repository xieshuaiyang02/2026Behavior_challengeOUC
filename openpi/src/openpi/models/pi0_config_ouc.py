"""Stage-Aware pi0.5 configuration on the official JAX/Flax backend."""

import dataclasses
import math

import flax.nnx as nnx

from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class Pi0ConfigOUC(Pi0Config):
    pi05: bool = True
    num_stage_classes: int = dataclasses.field(kw_only=True)
    num_stage_queries: int = 3
    done_stage_id: int = 0
    stage_loss_weight: float = 0.1
    next_stage_loss_weight: float = 0.1
    progress_loss_weight: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input or self.num_stage_queries != 3:
            raise ValueError("OUC requires pi0.5, discrete state input and exactly three stage queries")
        if self.num_stage_classes < 2 or self.done_stage_id != 0:
            raise ValueError("Stage vocabulary must include <DONE>=0 and at least one real stage")
        for value in (self.stage_loss_weight, self.next_stage_loss_weight, self.progress_loss_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Auxiliary loss weights must be finite and nonnegative")

    def create(self, rng):
        from openpi.models.pi0_ouc import Pi0OUC

        return Pi0OUC(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, *args, **kwargs):
        raise NotImplementedError("This OUC variant uses official JAX params checkpoints")
