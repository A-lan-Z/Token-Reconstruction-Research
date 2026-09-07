"""Strict CPU-only header construction check for the frozen TRR-P09 B1 state."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.trr0010_p09_fixed_loader import load_p09_fixed_state

STATE = Path("/tmp/trr-p09-runtime/fixed-control-b1-r3/states/checkpoint_step_013000.safetensors")

def run() -> dict[str, object]:
    model = load_p09_fixed_state(
        STATE,
        expected_state_sha256="5757506003f0a4b82cb1cd6de2a82f11345f2dbcbce13c8cf18c4f515de3ec3a",
        expected_selected_step=13000,
        expected_bank_manifest_sha256="aefaa5f47aa1042dffa7210767ed4a0be1fd848c373f8cd6de1ad4c739035a31",
        expected_fit_manifest_sha256="aefaa5f47aa1042dffa7210767ed4a0be1fd848c373f8cd6de1ad4c739035a31",
        expected_schedule_semantic_sha256="8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5",
        expected_base_state_sha256="5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14",
        expected_embedding_sha256="ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1",
        expected_runner_state_sha256="209155048df936359870a1419803800d7bad4ab72a46e09bf3287648079964da",
    )
    return {
        "status": "PASS_CPU_STRICT_P09_HEADER",
        "state": str(STATE),
        "selected_step": 13000,
        "method_id": "continued_fixed_readout",
        "tensor_count": len(model.state_dict()),
        "forward_run": False,
        "gpu_used": False,
        "truth_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
