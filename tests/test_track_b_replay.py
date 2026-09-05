from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from token_reconstruction.experiment_runtime import seed_everything
from token_reconstruction.inverse import ResidualAffineInverse
from token_reconstruction.standalone_decoder import decoder_from_method


_PATH = Path("experiments/TRR-0003/track_b/replay_selected.py").resolve()
_SPEC = spec_from_file_location("trr0003_track_b_replay", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _constructor_trace(*, consume_tiny: bool) -> dict[str, str]:
    seed_everything(1737)
    trace: dict[str, str] = {}
    for method in _MODULE.METHOD_ORDER:
        trace[f"{method}.before_main"] = _MODULE._rng_digest()
        if method == "angular_inverse_control":
            _ = ResidualAffineInverse(_MODULE.HIDDEN_SIZE)
        else:
            _ = decoder_from_method(
                method,
                hidden_size=_MODULE.HIDDEN_SIZE,
                vocab_size=_MODULE.VOCAB_SIZE,
                logit_scale=16.0,
                bottleneck_size=256,
            )
        trace[f"{method}.after_main_initialization"] = _MODULE._rng_digest()
        if consume_tiny:
            _MODULE._discard_tiny_initialization(method)
        trace[f"{method}.after_tiny_initialization"] = _MODULE._rng_digest()
    return trace


def test_replay_constructor_rng_trace_matches_original_order() -> None:
    assert _constructor_trace(consume_tiny=True) == _MODULE._reference_initialization_trace()


def test_tiny_constructor_draw_changes_later_mlp_initialization() -> None:
    with_tiny = _constructor_trace(consume_tiny=True)
    without_tiny = _constructor_trace(consume_tiny=False)

    assert with_tiny["residual_mlp256_token_ce.before_main"] != without_tiny[
        "residual_mlp256_token_ce.before_main"
    ]
