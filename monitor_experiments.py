#!/usr/bin/env python3
import os, sys, json, time, select, argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # Fallback handled below

# ----------------------------- Config / CLI ---------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("logs_dir", nargs="?", default="logs", help="Logs directory (default: ./logs)")
    env_thr = os.getenv("EASYTUNA_ALIVE_THRESHOLD_SECONDS")
    p.add_argument(
        "--alive-threshold",
        type=float,
        default=float(env_thr) if env_thr else 5.0,
        help="Seconds considered 'alive' since last_alive (default: 5, or env EASYTUNA_ALIVE_THRESHOLD_SECONDS).",
    )
    p.add_argument(
        "--tz",
        default=os.getenv("EASYTUNA_TZ", None),
        help="Timezone to interpret naive timestamps (e.g., Europe/Vienna). "
             "Also used for dashboard 'now'. Can be set via EASYTUNA_TZ.",
    )
    return p.parse_args()

# ------------------------------- Utilities ---------------------------------- #

def load_json_safe(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    s = int(seconds)
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

def resolve_zone(tz_hint: Optional[str]) -> Optional[datetime.tzinfo]:
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(tz_hint) if tz_hint else datetime.now().astimezone().tzinfo
    except Exception:
        return datetime.now().astimezone().tzinfo

def parse_iso(ts: Optional[str], tz_hint: Optional[datetime.tzinfo]) -> Optional[datetime]:
    """Accept ISO strings; 'Z' => UTC. Naive => interpret in tz_hint."""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None and tz_hint is not None:
            return dt.replace(tzinfo=tz_hint)
        return dt if dt.tzinfo else None
    except Exception:
        return None

def is_alive(last_alive_str: Optional[str], threshold_seconds: float, now: datetime) -> Tuple[bool, Optional[float]]:
    """True iff |now - last_alive| <= threshold_seconds. Returns (alive, drift_seconds)."""
    if not last_alive_str:
        return (False, None)
    # tz_hint already embedded in 'now'
    last_alive = parse_iso(last_alive_str, now.tzinfo)
    if not last_alive:
        return (False, None)
    try:
        drift = (now - last_alive).total_seconds()
    except Exception:
        return (False, None)
    return (abs(drift) <= float(threshold_seconds), drift)

def get_study_base_path(study_path: Path) -> Path:
    return study_path.parent if study_path.name == ".easytuna" else study_path

# ----------------------------- Liveness / Discover --------------------------- #

def discover_experiments(logs_path: Path) -> List[Tuple[str, List[Tuple[str, Path]]]]:
    """Return [(experiment_name, [(study_id, study_data_path), ...]), ...]."""
    experiments: List[Tuple[str, List[Tuple[str, Path]]]] = []
    if not logs_path.exists():
        return experiments

    items = [p for p in logs_path.iterdir() if p.is_dir()]
    top_studies: List[Tuple[str, Path]] = []
    nested: List[Tuple[str, List[Tuple[str, Path]]]] = []

    def study_path(p: Path) -> Optional[Path]:
        et = p / ".easytuna"
        if (et / "status.json").exists() or (et / "results.json").exists():
            return et
        if (p / "status.json").exists() or (p / "results.json").exists():
            return p
        return None

    for item in items:
        sp = study_path(item)
        if sp:
            top_studies.append((item.name, sp))
        else:
            subs: List[Tuple[str, Path]] = []
            for sub in item.iterdir():
                if sub.is_dir():
                    sps = study_path(sub)
                    if sps:
                        subs.append((sub.name, sps))
            if subs:
                nested.append((item.name, subs))

    if top_studies and not nested:
        ex_name = logs_path.name if logs_path.name != "logs" else "Experiment"
        experiments.append((ex_name, top_studies))
    elif nested:
        experiments.extend(nested)
        if top_studies:
            ex_name = f"{logs_path.name}_studies" if logs_path.name != "logs" else "Individual_Studies"
            experiments.append((ex_name, top_studies))
    else:
        for item in items:
            sp = study_path(item)
            if sp:
                experiments.append((item.name, [(item.name, sp)]))
    return experiments

def study_status(study_path: Path, threshold_seconds: float, now: datetime) -> Tuple[bool, Optional[float], Optional[str], dict, dict]:
    """Return (alive, drift, last_alive_str, status_json or {}, results_json or {})."""
    status = load_json_safe(study_path / "status.json") or {}
    # If status carries its own tz, rebuild 'now' in that zone to compare consistently
    tz_key = status.get("timezone") or status.get("tz")
    now_z = now.astimezone(resolve_zone(tz_key)) if tz_key else now
    alive, drift = is_alive(status.get("last_alive"), threshold_seconds, now_z)
    return alive, drift, status.get("last_alive"), status, (load_json_safe(study_path / "results.json") or {})

# ----------------------------- Dashboard UI ---------------------------------- #

def display_dashboard(logs_dir: str, threshold_seconds: float, tz_name: Optional[str]) -> Dict[str, dict]:
    logs_path = Path(logs_dir)
    tzinfo = resolve_zone(tz_name)
    now = datetime.now(tzinfo) if tzinfo else datetime.now().astimezone()

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"🔄 EasyTuna Experiment Monitor - {now.isoformat()}")
    print(f"🔍 Monitoring experiments in: {logs_path.absolute()}")
    print("=" * 80)

    if not logs_path.exists():
        print(f"❌ Logs directory '{logs_dir}' not found")
        return {}

    experiments = discover_experiments(logs_path)
    if not experiments:
        print("📭 No experiments found")
        return {}

    terminable: Dict[str, dict] = {}
    e_i = s_i = t_i = 1

    for ex_name, studies in sorted(experiments):
        print(f"\n🧪 EXPERIMENT [E{e_i}]: {ex_name}")
        print("-" * 80)

        # Experiment alive if any study is alive
        ex_alive = any(study_status(sp, threshold_seconds, now)[0] for _, sp in studies)
        if ex_alive:
            exp_path_guess = logs_path / ex_name
            exp_term_path = exp_path_guess if exp_path_guess.exists() else logs_path
            terminable[f"E{e_i}"] = {"type": "experiment", "path": exp_term_path, "id": ex_name, "display_name": f"Experiment: {ex_name}"}

        for sid, spath in sorted(studies):
            alive, drift, last_alive_str, status, results = study_status(spath, threshold_seconds, now)
            print(f"  📊 Study [S{s_i}]: {sid}")

            if not status:
                print(f"    Status: ⚪ no status")
                print(f"    Host: N/A | PID: N/A")
            else:
                hostname = status.get("hostname", "N/A")
                pid = status.get("pid", "N/A")
                status_text = status.get("status", "unknown")

                if last_alive_str:
                    drift_str = f"{drift:.1f}s" if isinstance(drift, (int, float)) else "N/A"
                    print(f"    Last alive: {last_alive_str} (Δ={drift_str})")

                if alive:
                    print(f"    Status: 🟢 {status_text} | Host: {hostname} | PID: {pid}")
                    term_path = get_study_base_path(spath)
                    terminable[f"S{s_i}"] = {"type": "study", "path": term_path, "id": f"{ex_name}/{sid}", "display_name": f"Study: {sid}"}
                    # Trials: list non-terminated subfolders under trial_runs
                    trial_runs = term_path / "trial_runs"
                    if trial_runs.exists():
                        for tpath in sorted(p for p in trial_runs.iterdir() if p.is_dir()):
                            if any(tpath.glob("*_terminate*")):
                                continue
                            terminable[f"T{t_i}"] = {
                                "type": "trial", "path": tpath, "id": f"{ex_name}/{sid}/{tpath.name}", "display_name": f"Trial: {tpath.name}"
                            }
                            t_i += 1
                else:
                    shown = "inactive" if status_text in (None, "", "inactive", "stopped", "failed") else f"inactive (was: {status_text})"
                    print(f"    Status: 🔴 {shown}")
                    print(f"    Host: {hostname} | PID: {pid}")

                if results.get("duration_seconds"):
                    print(f"    Duration: {format_duration(results.get('duration_seconds'))}")

                if results:
                    n_complete = results.get("n_complete_trials", 0)
                    n_total = results.get("n_trials", "N/A")
                    n_pruned = results.get("n_pruned_trials", 0)
                    n_failed = results.get("n_failed_trials", 0)
                    print(f"    Trials: {n_complete}/{n_total} complete, {n_pruned} pruned, {n_failed} failed")

                    if results.get("best_value") is not None:
                        metric = results.get("metric_name", "metric")
                        best_val = results.get("best_value")
                        best_trial_num = results.get("best_trial", {}).get("number", "N/A")
                        try:
                            print(f"    Best {metric}: {float(best_val):.4f} (Trial #{best_trial_num})")
                        except Exception:
                            print(f"    Best {metric}: {best_val} (Trial #{best_trial_num})")

            print()
            s_i += 1
        e_i += 1

    return terminable

def input_with_timeout(prompt: str, timeout: int) -> Optional[str]:
    print(prompt, end='', flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip() if ready else None

# ---------------------------------- Main ------------------------------------- #

def main():
    args = parse_args()
    logs_dir, thr, tz_name = args.logs_dir, float(args.alive_threshold), args.tz

    print("🚀 Starting EasyTuna Experiment Monitor")
    print(f"📁 Monitoring directory: {Path(logs_dir).absolute()}")
    print(f"⏱️ Alive threshold: {thr:g}s")
    print("🔧 Termination levels: E<num>=Experiment, S<num>=Study, T<num>=Trial")
    print("=" * 100)

    while True:
        try:
            items = display_dashboard(logs_dir, thr, tz_name)

            if items:
                ex = [k for k in items if k.startswith("E")]
                st = [k for k in items if k.startswith("S")]
                tr = [k for k in items if k.startswith("T")]

                parts = []
                if ex: parts.append(f"{len(ex)} experiment{'s' if len(ex) != 1 else ''}")
                if st: parts.append(f"{len(st)} stud{'ies' if len(st) != 1 else 'y'}")
                if tr: parts.append(f"{len(tr)} trial{'s' if len(tr) != 1 else ''}")
                if parts:
                    print(f"\n🎯 Available for termination: {', '.join(parts)}")

            prompt = (
                "\n" + "=" * 80 +
                "\nEnter an ID like E1 / S7 / T12 to terminate, press Enter to refresh (10s), or type 'q' to quit: "
            )
            user_input = input_with_timeout(prompt, 10)
            if user_input is None or user_input == "":
                continue
            if user_input.lower() == "q":
                break

            key = user_input.upper().strip()
            if key in items:
                item = items[key]; t = item["type"]
                if t == "experiment":
                    msg = f"❓ TERMINATE ENTIRE EXPERIMENT '{item['id']}'? This will stop ALL studies and trials! (y/N): "
                elif t == "study":
                    msg = f"❓ TERMINATE STUDY '{item['id']}'? This will stop all trials in this study! (y/N): "
                else:
                    msg = f"❓ TERMINATE TRIAL '{item['id']}'? (y/N): "
                if input(msg).lower() == "y":
                    path = Path(item["path"])
                    term_file = path / ( "_terminate_experiment.txt" if t=="experiment" else "_terminate_study.txt" if t=="study" else "_terminate_trial.txt" )
                    try:
                        term_file.touch()
                        print(f"✅ Termination signal sent to {item['display_name']}")
                        print(f"📝 Signal file created: {term_file}")
                        print("🔄 Refreshing in 3s..."); time.sleep(3)
                    except Exception as e:
                        print(f"❌ Failed to send termination signal: {e}")
                        print("🔄 Refreshing in 2s..."); time.sleep(2)
                else:
                    print("❌ Termination cancelled. Refreshing in 2s..."); time.sleep(2)
            else:
                avail = list(items.keys())
                if avail:
                    print(f"❌ Invalid ID '{key}'. Available: {', '.join(avail)}")
                else:
                    print(f"❌ No terminable experiments/studies/trials found.")
                print("🔄 Refreshing in 2s..."); time.sleep(2)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🔄 Continuing in 2s..."); time.sleep(2)

    print("\n👋 EasyTuna Monitor stopped")

if __name__ == "__main__":
    main()
