"""
Unit & Integration Tests for Swizzle Robotics VLA & World-Model SDK (`swizzle.robotics`, `swizzle.world`)
"""

from swizzle import (
    HardwareDetector,
    PatchCacheFilter,
    KinematicSpecActPredictor,
    TemporalActionEnsembler,
    VLAPolicy,
    WorldPlanner,
    extract_patch_intensities_from_frame,
    quantize_action_to_token,
    dequantize_token_to_action,
)


def test_hardware_detector_student_tiers():
    p2gb = HardwareDetector.detect(override_vram_gb=2.0)
    assert p2gb.recommended_vla == "swizzle/octo-small-fp16"
    assert p2gb.needs_colab_tunnel_for_7b is True

    p4gb = HardwareDetector.detect(override_vram_gb=4.0)
    assert p4gb.recommended_vla == "swizzle/smolvla-450m-int8"
    assert p4gb.recommended_world_model == "swizzle/dino-wm-latent"

    p8gb = HardwareDetector.detect(override_vram_gb=8.0)
    assert p8gb.recommended_vla == "swizzle/pi0-fast-int4"
    assert p8gb.recommended_world_model == "swizzle/vjepa2-ac-int4"


def test_patch_cache_spatio_temporal_reuse():
    filt = PatchCacheFilter(grid_size=14, delta_threshold=0.025)
    frame1 = [0.5] * 196
    d1 = filt.evaluate(frame1)
    assert d1.reused_patches == 0
    assert d1.recomputed_patches == 196

    # Frame 2: only 22 patches (robot gripper) change; 174 background patches stay invariant
    frame2 = list(frame1)
    for i in range(22):
        frame2[i] = 0.90
    d2 = filt.evaluate(frame2)
    assert d2.reused_patches == 174
    assert d2.patch_hit_ratio > 0.88
    assert d2.prefill_speedup > 3.6


def test_kinematic_spec_act_7dof():
    for v in [-1.0, -0.35, 0.0, 0.45, 1.0]:
        tok = quantize_action_to_token(v)
        assert 31744 <= tok <= 31999
        assert abs(dequantize_token_to_action(tok) - v) < 0.01

    pred = KinematicSpecActPredictor(bin_tolerance=2)
    steps = [
        [0.10, -0.05, 0.20, 0.0, 0.0, 0.0, 1.0],
        [0.11, -0.05, 0.21, 0.0, 0.0, 0.0, 1.0],
        [0.12, -0.05, 0.22, 0.0, 0.0, 0.0, 1.0],
        [0.13, -0.05, 0.23, 0.0, 0.0, 0.0, 1.0],
    ]
    last = None
    for s in steps:
        last = pred.draft_and_verify(s)
    assert last is not None
    assert last.accepted_dof_tokens >= 6
    assert last.decoding_speedup >= 2.5


def test_vla_policy_end_to_end():
    policy = VLAPolicy(
        model="swizzle/smolvla-450m-int8",
        patch_cache=True,
        speculative_kinematics=True,
        temporal_ensemble_hz=50,
    )
    frame1 = [0.45] * 196
    out1 = policy.act(frame1, instruction="pick up the red block")
    assert len(out1.action) == 7
    assert len(out1.substeps_50hz) == 4

    frame2 = list(frame1)
    for i in range(20):
        frame2[i] = 0.82
    out2 = policy.act(frame2, instruction="pick up the red block")
    assert out2.patch_cache.patch_hit_ratio > 0.85
    assert out2.effective_hz > 25.0


def test_world_planner_cem_convergence():
    planner = WorldPlanner(
        model="swizzle/dino-wm-latent",
        num_trajectories=128,
        horizon=5,
        cem_iterations=3,
    )
    curr = [0.10, -0.20, 0.05, 0.0, 0.0, 0.0, 1.0]
    goal = [0.55, 0.25, 0.40, 0.10, -0.10, 0.0, 1.0]
    res = planner.plan(curr, goal)
    assert len(res.optimal_action_sequence) == 5
    assert res.final_predicted_distance < res.initial_latent_distance
    assert res.planning_speedup > 7.5
    assert res.vram_used_gb <= 1.5


def test_student_cli_subcommands(capsys):
    import json
    from swizzle.cli import main

    # 1. `swizzle doctor --vram-gb 4.0 --json`
    rc_doc = main(["doctor", "--vram-gb", "4.0", "--json"])
    assert rc_doc == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["recommended_vla"] == "swizzle/smolvla-450m-int8"
    assert doc_out["recommended_world_model"] == "swizzle/dino-wm-latent"

    # 2. `swizzle bench --steps 12 --json`
    rc_bench = main(["bench", "--steps", "12", "--json"])
    assert rc_bench == 0
    bench_out = json.loads(capsys.readouterr().out)
    assert bench_out["steps"] == 12
    assert bench_out["avg_patch_cache_hit_ratio"] > 0.85
    assert bench_out["world_model_speedup"] > 7.5

    # 3. `swizzle act --instruction "pick up the red cube" --steps 4 --json`
    rc_act = main(["act", "--instruction", "pick up the red cube", "--steps", "4", "--json"])
    assert rc_act == 0
    act_out = json.loads(capsys.readouterr().out)
    assert len(act_out["steps"]) == 4
    assert len(act_out["steps"][0]["action_7dof"]) == 7
    assert len(act_out["steps"][0]["tokens_256bin"]) == 7


def test_webcam_frame_ingestion():
    import numpy as np

    # Simulate 480x640x3 OpenCV camera frame
    webcam_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Simulate a moving object across 100x100 pixels
    webcam_frame[150:250, 200:300] = 250

    patches = extract_patch_intensities_from_frame(webcam_frame, grid_size=14)
    assert len(patches) == 196
    assert any(p > 0.8 for p in patches)
    assert any(p < 0.6 for p in patches)

    policy = VLAPolicy(grid_size=14)
    # Feed raw frame directly to policy.act()
    out = policy.act(webcam_frame, instruction="track moving object")
    assert len(out.action) == 7
    assert out.patch_cache.total_patches == 196
    assert out.effective_hz > 20.0



