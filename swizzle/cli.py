"""
Swizzle One-Command Student & Researcher CLI (`swizzle`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides terminal diagnostics, local 4-engine robotics/world-model benchmarks,
and interactive 7-DoF VLA + CEM World-Model rollout execution:
  - `swizzle doctor`
  - `swizzle bench --steps 60`
  - `swizzle act --instruction "pick up the red block and place it in the green tray"`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Sequence

from . import __version__
from .robotics import HardwareDetector, VLAPolicy, quantize_action_to_token
from .world import WorldPlanner


def cmd_doctor(args: argparse.Namespace) -> int:
    profile = HardwareDetector.detect(override_vram_gb=args.vram_gb)
    if args.json:
        print(json.dumps(asdict(profile), indent=2))
        return 0

    print("=" * 72)
    print(f"  SWIZZLE STUDENT HARDWARE & EDGE VLA DOCTOR (v{__version__})")
    print("=" * 72)
    print(f"  Detected Device       : {profile.device_name} ({profile.device_type.upper()})")
    print(f"  Available VRAM / Mem  : {profile.vram_gb:.2f} GB")
    print(f"  Recommended VLA       : {profile.recommended_vla} ({profile.recommended_quantization})")
    print(f"  Recommended World Mdl : {profile.recommended_world_model}")
    print(
        f"  7B Colab Tunnel Mode  : "
        f"{'RECOMMENDED (for OpenVLA-7B / Pi0-FAST)' if profile.needs_colab_tunnel_for_7b else 'OPTIONAL (runs locally)'}"
    )
    print("-" * 72)
    print("  Quickstart Commands:")
    print(f"    1. Benchmark 4-Engine Pipeline : swizzle bench --model {profile.recommended_vla}")
    print(f"    2. Run 7-DoF Closed-Loop Agent : swizzle act --instruction \"pick up the red block\"")
    print("    3. Interactive Web Playground  : https://swizzle-ten.vercel.app/#student-playground")
    print("=" * 72)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    steps = max(2, int(args.steps))
    policy = VLAPolicy(
        model=args.model,
        grid_size=14,
        patch_cache=True,
        speculative_kinematics=True,
        temporal_ensemble_hz=50,
    )
    planner = WorldPlanner(
        model=args.world_model,
        num_trajectories=128,
        horizon=5,
        cem_iterations=3,
    )

    t0 = time.perf_counter()
    base_frame = [0.42] * 196
    hit_ratios: List[float] = []
    accept_rates: List[float] = []
    step_latencies: List[float] = []
    effective_hz_list: List[float] = []

    for s in range(steps):
        frame = list(base_frame)
        # Simulate moving robot gripper across 18 of 196 ViT patches (~90.8% static background)
        offset = (s * 3) % (196 - 18)
        for i in range(offset, offset + 18):
            frame[i] = 0.88
        out = policy.act(frame, instruction="benchmark closed-loop tabletop trajectory")
        if s > 0:
            hit_ratios.append(out.patch_cache.patch_hit_ratio)
        accept_rates.append(out.spec_act.acceptance_rate)
        step_latencies.append(out.end_to_end_ms)
        effective_hz_list.append(out.effective_hz)

    curr_lat = [0.10, -0.20, 0.05, 0.0, 0.0, 0.0, 1.0]
    goal_lat = [0.55, 0.25, 0.40, 0.10, -0.10, 0.0, 1.0]
    wm_plan = planner.plan(curr_lat, goal_lat)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    avg_hit = sum(hit_ratios) / max(1, len(hit_ratios))
    avg_acc = sum(accept_rates) / max(1, len(accept_rates))
    avg_lat = sum(step_latencies) / max(1, len(step_latencies))
    avg_hz = sum(effective_hz_list) / max(1, len(effective_hz_list))

    summary = {
        "vla_model": args.model,
        "world_model": args.world_model,
        "steps": steps,
        "avg_patch_cache_hit_ratio": round(avg_hit, 4),
        "avg_spec_act_acceptance_rate": round(avg_acc, 4),
        "avg_vla_step_latency_ms": round(avg_lat, 2),
        "avg_effective_control_hz": round(avg_hz, 1),
        "world_model_cem_trajectories": wm_plan.num_trajectories,
        "world_model_batched_ms": wm_plan.swizzle_batched_ms,
        "world_model_speedup": wm_plan.planning_speedup,
        "world_model_initial_dist": wm_plan.initial_latent_distance,
        "world_model_final_dist": wm_plan.final_predicted_distance,
        "local_execution_wall_ms": round(wall_ms, 2),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("=" * 72)
    print(f"  SWIZZLE 4-ENGINE ROBOTICS & WORLD-MODEL BENCHMARK ({steps} STEPS)")
    print("=" * 72)
    print(f"  1. Swizzle-PatchCache (14x14 ViT) : {avg_hit * 100:.1f}% static patch reuse")
    print(f"  2. Swizzle-SpecAct (7-DoF Taylor) : {avg_acc * 100:.1f}% token acceptance (1 pass)")
    print(f"  3. Swizzle-Teleop (50Hz Spline)   : {avg_lat:.2f} ms/step ({avg_hz:.1f} Hz effective)")
    print(
        f"  4. Swizzle-MPC ({wm_plan.num_trajectories}-Traj CEM)     : "
        f"{wm_plan.swizzle_batched_ms:.1f} ms ({wm_plan.planning_speedup:.2f}x vs unbatched)"
    )
    print(
        f"     Latent Goal Distance           : "
        f"{wm_plan.initial_latent_distance:.4f} -> {wm_plan.final_predicted_distance:.4f}"
    )
    print("-" * 72)
    print(f"  Completed {steps} closed-loop steps + 128-rollout CEM plan in {wall_ms:.1f} ms.")
    print("=" * 72)
    return 0


def cmd_act(args: argparse.Namespace) -> int:
    steps = max(1, int(args.steps))
    policy = VLAPolicy(model=args.model, grid_size=14)
    planner = WorldPlanner(model=args.world_model, num_trajectories=128, horizon=max(3, steps))

    curr_lat = [0.05, -0.15, 0.10, 0.0, 0.0, 0.0, 0.0]
    goal_lat = [0.48, 0.22, 0.35, 0.05, -0.05, 0.0, 1.0]
    plan_res = planner.plan(curr_lat, goal_lat)

    base_frame = [0.45] * 196
    records = []
    for s in range(steps):
        frame = list(base_frame)
        offset = (s * 4) % (196 - 16)
        for i in range(offset, offset + 16):
            frame[i] = 0.85
        hint = plan_res.optimal_action_sequence[min(s, len(plan_res.optimal_action_sequence) - 1)]
        out = policy.act(frame, instruction=args.instruction, target_action_hint=hint)
        tokens = [quantize_action_to_token(v) for v in out.action]
        records.append(
            {
                "step": s + 1,
                "action_7dof": out.action,
                "tokens_256bin": tokens,
                "patch_hit_ratio": out.patch_cache.patch_hit_ratio,
                "latency_ms": out.end_to_end_ms,
                "effective_hz": out.effective_hz,
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "instruction": args.instruction,
                    "vla_model": args.model,
                    "world_model": args.world_model,
                    "cem_initial_dist": plan_res.initial_latent_distance,
                    "cem_final_dist": plan_res.final_predicted_distance,
                    "steps": records,
                },
                indent=2,
            )
        )
        return 0

    print("=" * 76)
    print(f"  SWIZZLE CLOSED-LOOP 7-DoF ACTION RUNNER · {args.model}")
    print(f"  Task Instruction : \"{args.instruction}\"")
    print(
        f"  World-Model Plan : {args.world_model} "
        f"(dist {plan_res.initial_latent_distance:.3f} -> {plan_res.final_predicted_distance:.3f})"
    )
    print("-" * 76)
    for rec in records:
        act_str = ", ".join(f"{v:+.3f}" for v in rec["action_7dof"])
        print(
            f"  Step {rec['step']:02d} | 7-DoF: [{act_str}] | "
            f"Cache: {rec['patch_hit_ratio'] * 100:4.1f}% | "
            f"{rec['latency_ms']:5.1f} ms ({rec['effective_hz']:4.1f} Hz)"
        )
    print("=" * 76)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swizzle",
        description="Swizzle Student & Researcher CLI for LLM, Robotics VLA, and World-Model Inference.",
    )
    parser.add_argument("--version", action="version", version=f"swizzle {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_doc = sub.add_parser("doctor", help="Inspect laptop GPU/CPU tier and get recommended VLA & World Model.")
    p_doc.add_argument("--vram-gb", type=float, default=None, help="Simulate a specific VRAM capacity in GB.")
    p_doc.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_bench = sub.add_parser("bench", help="Run a local benchmark of PatchCache, SpecAct, Teleop, and CEM MPC.")
    p_bench.add_argument("--steps", type=int, default=40, help="Number of closed-loop VLA steps to benchmark.")
    p_bench.add_argument("--model", type=str, default="swizzle/smolvla-450m-int8", help="VLA policy model ID.")
    p_bench.add_argument("--world-model", type=str, default="swizzle/dino-wm-latent", help="World model ID.")
    p_bench.add_argument("--json", action="store_true", help="Emit JSON output.")

    p_act = sub.add_parser("act", help="Execute a multi-step 7-DoF VLA + World-Model CEM rollout in the terminal.")
    p_act.add_argument(
        "--instruction",
        type=str,
        default="pick up the red block and place it in the green target zone",
        help="Natural language manipulation instruction.",
    )
    p_act.add_argument("--steps", type=int, default=5, help="Number of closed-loop steps to run.")
    p_act.add_argument("--model", type=str, default="swizzle/smolvla-450m-int8", help="VLA policy model ID.")
    p_act.add_argument("--world-model", type=str, default="swizzle/dino-wm-latent", help="World model ID.")
    p_act.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "bench":
        return cmd_bench(args)
    if args.command == "act":
        return cmd_act(args)
    # Default to `doctor` when invoked with no subcommand so `swizzle` immediately helps students
    args.vram_gb = None
    args.json = False
    return cmd_doctor(args)


if __name__ == "__main__":
    sys.exit(main())

