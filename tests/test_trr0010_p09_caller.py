from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from trr0010_p09_caller import (
    A2SourceBinding,
    FrozenArtifact,
    P09CallerError,
    ProductionBindings,
    ResourceQualification,
    make_serialization_only_checkpoint_callback,
    prepare_directional_runtime,
    restore_selected_and_export,
    validate_optimizer_configuration,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _protocol(adapter_path: Path) -> SimpleNamespace:
    def shared_decoder_rows(*args, **kwargs):
        del args, kwargs
        raise AssertionError("synthetic caller tests do not train")

    return SimpleNamespace(
        __file__=str(adapter_path),
        BankContract=object,
        TrainingContract=object,
        ReadoutHook=object,
        shared_decoder_rows=shared_decoder_rows,
        contract_digest=lambda _contract: "a" * 64,
    )


def _bindings(tmp_path: Path) -> tuple[ProductionBindings, dict[str, Path], SimpleNamespace]:
    def write(rel: str, payload: bytes) -> Path:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def artifact(label: str, status: str) -> FrozenArtifact:
        path = write(f"artifacts/{label}.json", f"{label}-bound".encode())
        payload = path.read_bytes()
        return FrozenArtifact(label, str(path), len(payload), _sha_bytes(payload), status)

    source_paths: dict[str, Path] = {}
    sources = []
    for index, path in enumerate(
        (
            "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
            "scripts/trr_p09/fixed_control_runner.py",
            "scripts/trr_p09/prepare_streamed_bank.py",
        )
    ):
        source = write(path, f"a2-source-{index}".encode())
        source_paths[path] = source
        sources.append(A2SourceBinding(path, "b" * 40, _sha_bytes(source.read_bytes())))

    resource = ResourceQualification(
        status="PASS",
        gpu_peak_reserved_bytes=100,
        gpu_reserved_limit_bytes=200,
        host_peak_rss_bytes=100,
        host_rss_limit_bytes=200,
        host_available_bytes=300,
        host_available_floor_bytes=200,
        wall_seconds=1.0,
        wall_limit_seconds=2.0,
    )
    bindings = ProductionBindings(
        contract=artifact("contract", "FROZEN"),
        bank_manifest=artifact("bank", "VERIFIED"),
        schedule=artifact("schedule", "FROZEN"),
        resource_qualification=resource,
        a2_sources=tuple(sources),
    )
    return bindings, source_paths, _protocol(source_paths["src/token_reconstruction/trr_p09_fixed_control_adapter.py"])


def _runtime(tmp_path: Path):
    torch.manual_seed(101001)
    decoder = build_residual_mlp512(
        hidden_size=8,
        vocabulary_size=17,
        context_width=4,
        bottleneck_size=3,
        seed=4005,
    )
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    bindings, source_paths, protocol = _bindings(tmp_path)
    runtime = prepare_directional_runtime(
        protocol=protocol,
        base_decoder=decoder,
        support_ids=torch.tensor([1, 4, 8, 13]),
        support_counts=torch.tensor([1, 4, 25, 100]),
        public_embedding=embedding,
        bindings=bindings,
        source_paths=source_paths,
        base_learning_rate=2.0e-4,
    )
    return runtime, embedding


def test_runtime_binds_shared_decoder_device_and_explicit_foreach_false(tmp_path: Path) -> None:
    runtime, _embedding = _runtime(tmp_path)
    assert runtime.hook.base is runtime.decoder
    assert runtime.optimizer.defaults["foreach"] is False
    assert runtime.optimizer.defaults["weight_decay"] == 0.0
    assert [group["name"] for group in runtime.optimizer.param_groups] == [
        "decoder",
        "directional_delta",
    ]
    assert {parameter.device for parameter in runtime.hook.parameters()} == {torch.device("cpu")}
    assert runtime.binding_receipt["verification_sha256"]


def test_prepare_verifies_changed_bound_artifact(tmp_path: Path) -> None:
    bindings, source_paths, protocol = _bindings(tmp_path)
    Path(bindings.contract.path).write_bytes(b"changed")
    decoder = build_residual_mlp512(hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005)
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    with pytest.raises(P09CallerError, match="bytes or SHA-256"):
        prepare_directional_runtime(
            protocol=protocol,
            base_decoder=decoder,
            support_ids=[1, 4],
            support_counts=[1, 4],
            public_embedding=embedding,
            bindings=bindings,
            source_paths=source_paths,
            base_learning_rate=2.0e-4,
        )


def test_runtime_refuses_unaccepted_final_bindings(tmp_path: Path) -> None:
    bindings, _source_paths, _protocol_obj = _bindings(tmp_path)
    bad = ProductionBindings(
        contract=FrozenArtifact(
            "contract",
            bindings.contract.path,
            bindings.contract.bytes,
            bindings.contract.sha256,
            "PENDING",
        ),
        bank_manifest=bindings.bank_manifest,
        schedule=bindings.schedule,
        resource_qualification=bindings.resource_qualification,
        a2_sources=bindings.a2_sources,
    )
    with pytest.raises(P09CallerError, match="contract status"):
        bad.validate()


def test_serialization_callback_and_selected_export_are_coherent(tmp_path: Path) -> None:
    runtime, embedding = _runtime(tmp_path)
    with torch.no_grad():
        runtime.hook.delta_rows.normal_(mean=0.0, std=0.05)
    # The shared runner decays both groups before a nonzero-step callback;
    # construction-time rates must not be re-imposed at checkpoint time.
    runtime.optimizer.param_groups[0]["lr"] = 1.0e-4
    runtime.optimizer.param_groups[1]["lr"] = 5.0e-5
    before = {name: value.detach().clone() for name, value in runtime.hook.state_dict().items()}
    callback = make_serialization_only_checkpoint_callback(
        output_root=tmp_path / "checkpoints",
        runtime=runtime,
        base_state={"sha256": "d" * 64},
        fit_manifest={"sha256": "e" * 64},
    )
    receipt = callback({"step": 7, "state_sha256": "f" * 64}, runtime.decoder, runtime.hook)
    checkpoint = Path(receipt["checkpoint"]["path"])
    assert checkpoint.exists()
    assert receipt["serialization_only"] is True
    assert receipt["resume_requires_separate_optimizer_artifact"] is True
    assert all(torch.equal(before[name], value) for name, value in runtime.hook.state_dict().items())

    with torch.no_grad():
        runtime.hook.delta_rows.zero_()
    export_receipt = restore_selected_and_export(
        checkpoint_path=checkpoint,
        expected_checkpoint=receipt["checkpoint"],
        export_path=tmp_path / "selected_effective.safetensors",
        runtime=runtime,
        public_embedding=embedding,
        selected_step=7,
    )
    exported = load_file(export_receipt["export"]["path"])["embeddings"]
    activation = torch.randn(2, 4, 8)
    valid = torch.ones(2, 4, dtype=torch.bool)
    primary = runtime.hook(activation, valid, embedding)
    deployed = runtime.hook.merged_forward(activation, valid, exported)
    assert torch.equal(primary, deployed)
    assert torch.equal(primary.argmax(dim=-1), deployed.argmax(dim=-1))


def test_serialization_callback_allows_terminal_zero_schedule(tmp_path: Path) -> None:
    runtime, _embedding = _runtime(tmp_path)
    for group in runtime.optimizer.param_groups:
        group["lr"] = 0.0
    callback = make_serialization_only_checkpoint_callback(
        output_root=tmp_path / "checkpoints",
        runtime=runtime,
        base_state={"sha256": "d" * 64},
        fit_manifest={"sha256": "e" * 64},
    )
    receipt = callback({"step": 3000}, runtime.decoder, runtime.hook)
    assert Path(receipt["checkpoint"]["path"]).is_file()


def test_restore_rejects_wrong_selected_metadata(tmp_path: Path) -> None:
    runtime, embedding = _runtime(tmp_path)
    callback = make_serialization_only_checkpoint_callback(
        output_root=tmp_path / "checkpoints",
        runtime=runtime,
        base_state={"sha256": "d" * 64},
        fit_manifest={"sha256": "e" * 64},
    )
    receipt = callback({"step": 7}, runtime.decoder, runtime.hook)
    with pytest.raises(P09CallerError, match="selected checkpoint metadata"):
        restore_selected_and_export(
            checkpoint_path=Path(receipt["checkpoint"]["path"]),
            expected_checkpoint=receipt["checkpoint"],
            export_path=tmp_path / "selected_effective.safetensors",
            runtime=runtime,
            public_embedding=embedding,
            selected_step=8,
        )


def test_runtime_rejects_foreach_true_optimizer_policy(tmp_path: Path) -> None:
    runtime, _embedding = _runtime(tmp_path)
    optimizer = torch.optim.AdamW(
        runtime.optimizer.param_groups,
        weight_decay=0.0,
        foreach=True,
    )
    with pytest.raises(P09CallerError, match="foreach"):
        validate_optimizer_configuration(
            optimizer,
            runtime.decoder,
            runtime.hook,
            expected_base_learning_rate=2.0e-4,
            expected_weight_decay=0.0,
        )
