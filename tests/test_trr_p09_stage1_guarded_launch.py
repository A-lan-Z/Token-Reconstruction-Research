"""CPU-only checks for the P09 guarded launch helper."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.trr_p09 import stage1_guarded_launch as launch


def _launch_args(tmp_path: Path) -> list[str]:
    plan = tmp_path / "plan.json"
    inputs = tmp_path / "inputs.json"
    model = tmp_path / "model"
    plan.write_text("{}", encoding="utf-8")
    inputs.write_text("{}", encoding="utf-8")
    model.mkdir()
    return [
        "--mode", "qualify",
        "--input-manifest", str(inputs),
        "--stage1-plan", str(plan),
        "--output-root", str(tmp_path / "qualification"),
        "--watchdog-output-root", str(tmp_path / "watchdog"),
        "--model-snapshot", str(model),
        "--authorization-receipt", str(tmp_path / "authorization.json"),
        "--watchdog-receipt", str(tmp_path / "watchdog-receipt.json"),
        "--repository-root", str(REPOSITORY_ROOT),
    ]


def test_dry_run_uses_real_wrapper_and_distinguishes_inner_guards(tmp_path: Path) -> None:
    args = launch._parser().parse_args(_launch_args(tmp_path))
    command, env, metadata = launch.build_launch(args)
    assert command[1] == str(launch.WRAPPER_PATH)
    assert command[command.index("--timeout-seconds") + 1] == "600"
    assert metadata["wrapper"]["sha256"] == launch.WRAPPER_SHA256
    assert metadata["wrapper_enforcement"]["gpu_reserved_or_free"] is False
    assert metadata["wrapper_enforcement"]["retained_output_bytes"] is False
    assert metadata["child_inner_enforcement"]["gpu_reserved_bytes_max"] == 8 * launch.GIB
    assert metadata["child_inner_enforcement"]["retained_capture_bytes_max"] == 20 * launch.GIB
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "scripts/trr_p09/stage1_public_capture.py" in command


def test_estimate_scales_only_measured_qualification_and_explicit_overhead(tmp_path: Path) -> None:
    receipt = tmp_path / "qualification.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "QUALIFICATION_PASS",
                "condition": "public_base",
                "truth_opened": False,
                "elapsed_seconds": 2.0,
                "batches": [
                    {
                        "repeat_torch_equal": True,
                        "future_padding_active_torch_equal": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    estimate = launch.estimate_production_wall(receipt, fixed_overhead_seconds=10.0)
    assert estimate["qualification_receipt_sha256"] == launch._sha256(receipt)
    assert estimate["qualification_forward_calls"] == 3
    assert estimate["production_forward_calls"] == 1350
    assert estimate["forward_scaled_seconds"] == pytest.approx(900.0)
    assert estimate["estimated_wall_seconds"] == pytest.approx(910.0)
    assert estimate["within_production_wall_cap"] is True


def test_estimate_rejects_nonfinite_overhead(tmp_path: Path) -> None:
    receipt = tmp_path / "qualification.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "QUALIFICATION_PASS",
                "condition": "public_base",
                "truth_opened": False,
                "elapsed_seconds": 1.0,
                "batches": [{"repeat_torch_equal": True, "future_padding_active_torch_equal": True}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(launch.LaunchError, match="finite"):
        launch.estimate_production_wall(receipt, fixed_overhead_seconds=float("nan"))
