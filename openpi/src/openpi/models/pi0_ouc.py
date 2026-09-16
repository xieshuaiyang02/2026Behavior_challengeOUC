"""OUC queries on the official pi0.5 JAX model; original parameter paths are preserved."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask


def mask_stage_logits(current, following, allowed=None):
    allowed = jnp.ones_like(current, dtype=jnp.bool_) if allowed is None else jnp.broadcast_to(allowed, current.shape)
    current_allowed = allowed.at[:, 0].set(False)
    next_allowed = allowed.at[:, 0].set(True)
    return jnp.where(current_allowed, current, -1e9), jnp.where(next_allowed, following, -1e9)


def compute_stage_losses(output, targets, config):
    """Static-shape masked losses, averaged over the global JAX batch's valid rows."""
    current_ids = jnp.asarray(targets["current_stage_id"])
    next_ids = jnp.asarray(targets["next_stage_id"])
    num_classes = output["current_stage_logits"].shape[-1]
    # Never use joint stage_valid to discard a valid current/progress target.
    current_valid = jnp.asarray(targets["current_stage_valid"], dtype=jnp.bool_) & (
        (current_ids > 0) & (current_ids < num_classes)
    )
    next_valid = jnp.asarray(targets["next_stage_valid"], dtype=jnp.bool_) & (
        (next_ids >= 0) & (next_ids < num_classes)
    )
    current, following = mask_stage_logits(
        output["current_stage_logits"], output["next_stage_logits"], targets.get("allowed_stage_mask")
    )
    # Sanitize before CE/gather and Huber, including NaN placeholders in invalid rows.
    current_safe = jnp.where(current_valid[:, None], current, 0)
    next_safe = jnp.where(next_valid[:, None], following, 0)
    current_target = jnp.where(current_valid, current_ids, 0)
    next_target = jnp.where(next_valid, next_ids, 0)
    progress_target = jnp.where(current_valid, targets["progress"], 0)
    progress_pred = jnp.where(current_valid, output["progress_pred"], 0)

    def reduce(value, valid):
        # These are global arrays under the official jax.jit/NamedSharding path.
        # sum() reduces across data/FSDP shards, rather than averaging device means.
        return jnp.where(valid, value, 0).sum() / jnp.maximum(valid.sum(), 1)

    current_loss = reduce(optax.softmax_cross_entropy_with_integer_labels(current_safe, current_target), current_valid)
    next_loss = reduce(optax.softmax_cross_entropy_with_integer_labels(next_safe, next_target), next_valid)
    progress_loss = reduce(optax.huber_loss(progress_pred, progress_target, delta=1.0), current_valid)
    flow = output["flow_loss"].mean()
    return {
        "loss": flow
        + config.stage_loss_weight * current_loss
        + config.next_stage_loss_weight * next_loss
        + config.progress_loss_weight * progress_loss,
        "flow_loss": flow,
        "current_stage_loss": current_loss,
        "next_stage_loss": next_loss,
        "progress_loss": progress_loss,
        "stage_valid_fraction": (current_valid & next_valid).astype(jnp.float32).mean(),
        "current_stage_valid_fraction": current_valid.astype(jnp.float32).mean(),
        "next_stage_valid_fraction": next_valid.astype(jnp.float32).mean(),
        "current_stage_valid_count": current_valid.sum(),
        "next_stage_valid_count": next_valid.sum(),
        "current_stage_accuracy": reduce((current.argmax(-1) == current_target).astype(jnp.float32), current_valid),
        "next_stage_accuracy": reduce((following.argmax(-1) == next_target).astype(jnp.float32), next_valid),
        "progress_mae": reduce(jnp.abs(progress_pred - progress_target), current_valid),
    }


class StageHead(nnx.Module):
    def __init__(self, width, output_dim, *, rngs):
        self.norm = nnx.LayerNorm(width, rngs=rngs)
        self.hidden = nnx.Linear(width, width // 2, rngs=rngs)
        self.output = nnx.Linear(width // 2, output_dim, rngs=rngs)

    def __call__(self, x):
        return self.output(nnx.gelu(self.hidden(self.norm(x.astype(jnp.float32)))))


class Pi0OUC(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        width = gemma.get_config(config.paligemma_variant).width
        self.stage_queries = nnx.Param(jax.random.normal(rngs.params(), (3, width)) * 0.02)
        self.current_stage_head = StageHead(width, config.num_stage_classes, rngs=rngs)
        self.next_stage_head = StageHead(width, config.num_stage_classes, rngs=rngs)
        self.progress_head = StageHead(width, 1, rngs=rngs)
        self.stage_config = config

    def embed_prefix(self, observation):
        tokens, mask, ar = super().embed_prefix(observation)
        queries = jnp.broadcast_to(
            self.stage_queries.value.astype(tokens.dtype), (tokens.shape[0], 3, tokens.shape[-1])
        )
        return (
            jnp.concatenate([tokens, queries], axis=1),
            jnp.concatenate([mask, jnp.ones((tokens.shape[0], 3), dtype=jnp.bool_)], axis=1),
            jnp.concatenate([ar, jnp.array([True, False, False])]),
        )

    def predict_stage_info(self, prefix_out):
        return {
            "current_stage_logits": self.current_stage_head(prefix_out[:, -3]),
            "next_stage_logits": self.next_stage_head(prefix_out[:, -2]),
            "progress_pred": nnx.sigmoid(self.progress_head(prefix_out[:, -1]))[:, 0],
        }

    def compute_outputs(self, rng, observation, actions, *, train=False):
        # Same RNG split, preprocessing, FM noise, timestep and AdaRMS as official Pi0.compute_loss.
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar])
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1) - 1,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return {"flow_loss": jnp.mean(jnp.square(velocity - u_t), axis=-1), **self.predict_stage_info(prefix_out)}

    def compute_loss(self, rng, observation, actions, *, train=False):
        return self.compute_outputs(rng, observation, actions, train=train)["flow_loss"]

    # Pi0.sample_actions is inherited unchanged: its prefix prefill calls the
    # overridden embed_prefix once, caching context+stage KV for every layer.
