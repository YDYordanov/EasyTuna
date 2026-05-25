import os, sys, logging, math, shutil, inspect, ast, json, socket, importlib.util
from datetime import datetime
import optuna


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def _lcm_many(vals):
    out = 1
    for v in vals:
        out = _lcm(out, v)
    return out

def _first_multiple_at_least(lo, d):
    return ((lo + d - 1) // d) * d

def _last_multiple_at_most(hi, d):
    return (hi // d) * d

def _snap_to_multiple(raw, d, lo, hi, mode='nearest'):
    """Return nearest valid multiple of d within [lo, hi]; None if no feasible multiple."""
    first = _first_multiple_at_least(lo, d)
    last = _last_multiple_at_most(hi, d)
    if first > last:
        return None  # no feasible value

    if mode == 'floor':
        v = (raw // d) * d
        if v < first:
            v = first
        return min(v, last)

    if mode == 'ceil':
        v = ((raw + d - 1) // d) * d
        if v > last:
            v = last
        return max(v, first)

    # nearest (default)
    floor_v = (raw // d) * d
    ceil_v = floor_v if raw % d == 0 else floor_v + d
    candidates = []
    if first <= floor_v <= last:
        candidates.append(floor_v)
    if first <= ceil_v <= last:
        candidates.append(ceil_v)
    if not candidates:
        # raw is outside feasible grid; choose boundary closest to raw
        return first if abs(raw - first) <= abs(raw - last) else last
    # Choose the nearest candidate; ties use the smaller value for determinism.
    return min(candidates, key=lambda x: (abs(x - raw), x))

# --------------------------------------------------------------------------- #
#  Helpers for divisibility snapping from discrete candidate lists
# --------------------------------------------------------------------------- #
def _snap_from_val_list(raw, candidates, mode='nearest'):
    """
    Given a raw sampled value (already one of the *unconstrained* val_list choices),
    and a filtered list of *candidates* that satisfy divisibility constraints,
    choose a snapped value according to rounding mode:

        nearest : min |raw - v|; tie -> smaller
        floor   : max v <= raw; fallback to min(candidates) if all > raw
        ceil    : min v >= raw; fallback to max(candidates) if all < raw

    Assumes `candidates` is non-empty and sorted ascending.
    """
    if not candidates:
        return None
    candidates = sorted(candidates)
    if mode == 'floor':
        le = [v for v in candidates if v <= raw]
        return max(le) if le else candidates[0]
    if mode == 'ceil':
        ge = [v for v in candidates if v >= raw]
        return min(ge) if ge else candidates[-1]
    # nearest (default)
    return min(candidates, key=lambda v: (abs(v - raw), v))

# --------------------------------------------------------------------------- #
#  File logger: helps log optuna studies to file
# --------------------------------------------------------------------------- #
def setup_file_logger(exper_id, log_dir: str = 'logs', filename: str = "file.log"):
    """
    Set up file logger for EasyTuna experiments.
    
    Args:
        exper_id: Can be a full path (for new hierarchy) or just experiment ID (for old style)
        log_dir: Base log directory (ignored if exper_id is a full path)
        filename: Log file name
    """
    # Handle both old and new calling conventions
    if os.path.isabs(exper_id) or ('/' in exper_id and not exper_id.startswith('logs')):
        # New style: exper_id is already a full path
        log_dir_path = exper_id
    else:
        # Old style: exper_id is just an ID, combine with log_dir
        log_dir_path = os.path.join(log_dir, exper_id)
    
    os.makedirs(log_dir_path, exist_ok=True)
    log_path = os.path.join(log_dir_path, filename)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any old handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File handler (auto‐flushes on each record)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)

    # Console handler (optional; remove if you don't want console echo)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Tell Optuna to *not* use its own stderr handler,
    # and to *propagate* logs up to the root logger instead.
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    return logger



def _get_package_name(file_path, project_root):
    """
    Determines the package name of a Python file relative to a project root.
    This is used to provide context for resolving relative imports.
    """
    file_path = os.path.abspath(file_path)
    project_root = os.path.abspath(project_root)
    if not file_path.startswith(project_root):
        return None
    
    relative_path = os.path.relpath(os.path.dirname(file_path), project_root)
    if relative_path == '.':
        return ""  # Indicates top-level in project root
    return relative_path.replace(os.path.sep, '.')


def _snapshot_scripts(study_path: str, train_model_script: str):
    """
    Recursively find and snapshot all project-local python scripts, starting
    from a given entry point. It correctly resolves both absolute and relative
    imports to ensure the entire dependency tree is captured.
    
    Args:
        study_path: Absolute path to the study directory.
        train_model_script: Path to the main training script.
    """
    scripts_dir = os.path.join(study_path, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    project_root = os.getcwd()
    copied_files = set()
    
    files_to_process = [os.path.abspath(train_model_script)]

    while files_to_process:
        current_file = files_to_process.pop()

        if current_file in copied_files:
            continue

        if not current_file.startswith(project_root):
            continue

        try:
            relative_path = os.path.relpath(current_file, project_root)
            dest_path = os.path.join(scripts_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(current_file, dest_path)
            copied_files.add(current_file)
        except (IOError, shutil.SameFileError) as e:
            logging.warning(f"Could not copy {current_file}: {e}")
            continue

        try:
            with open(current_file, "r", encoding="utf-8") as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=current_file)
        except (SyntaxError, IOError, UnicodeDecodeError) as e:
            logging.warning(f"Could not parse {current_file} for imports: {e}")
            continue
        
        current_pkg = _get_package_name(current_file, project_root)

        for node in ast.walk(tree):
            found_origins = []

            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        spec = importlib.util.find_spec(alias.name)
                        if spec and spec.origin:
                            found_origins.append(spec.origin)
                    except (ModuleNotFoundError, ValueError):
                        pass

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 and current_pkg is None:
                    continue

                names_to_find = []
                if node.module:
                    # from .foo import bar -> name is '.foo'
                    names_to_find.append('.' * node.level + node.module)
                else:
                    # from . import foo, bar -> names are '.foo', '.bar'
                    for alias in node.names:
                        names_to_find.append('.' * node.level + alias.name)

                for name in names_to_find:
                    try:
                        pkg_context = current_pkg if name.startswith('.') else None
                        spec = importlib.util.find_spec(name, pkg_context)
                        if spec and spec.origin:
                            found_origins.append(spec.origin)
                    except (ModuleNotFoundError, ValueError):
                        pass

            for origin in found_origins:
                if origin.startswith(project_root) and origin.endswith('.py'):
                    if origin not in copied_files:
                        files_to_process.append(os.path.abspath(origin))

def handle_exception(exc_type, exc_value, exc_traceback, logger):
    # Log the full traceback
    logger.error("Uncaught exception", 
                 exc_info=(exc_type, exc_value, exc_traceback))
    # Then fall back to the default handler (prints to console)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# --------------------------------------------------------------------------- #
#  JSON Status and Results Tracking
# --------------------------------------------------------------------------- #
def update_status_json(study_log_dir: str, status_data: dict = None):
    """
    Update or create status.json with current experiment status.
    
    Args:
        study_log_dir: Path to study log directory
        status_data: Additional status data to merge (optional)
    """
    easytuna_dir = os.path.join(study_log_dir, ".easytuna")
    status_path = os.path.join(easytuna_dir, "status.json")
    
    # Default status data
    current_status = {
        "last_alive": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "status": "running"
    }
    
    # Try to load existing status and preserve non-default fields
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                existing_status = json.load(f)
                # Preserve fields that aren't being updated
                for key, value in existing_status.items():
                    if key not in current_status:
                        current_status[key] = value
        except (json.JSONDecodeError, IOError):
            pass  # Use defaults if file is corrupted
    
    # Merge any additional status data
    if status_data:
        current_status.update(status_data)
    
    # Write updated status (live in .easytuna)
    try:
        os.makedirs(easytuna_dir, exist_ok=True)
        with open(status_path, 'w') as f:
            json.dump(current_status, f, indent=2)
        # Copy to study dir and set read-only (force overwrite)
        study_status_path = os.path.join(study_log_dir, "status.json")
        try:
            if os.path.exists(study_status_path):
                os.chmod(study_status_path, 0o666)  # Allow overwrite of read-only mirror.
                os.remove(study_status_path)
        except Exception:
            pass
        shutil.copy2(status_path, study_status_path)
        try:
            os.chmod(study_status_path, 0o444)
        except Exception:
            pass
    except IOError as e:
        # Don't fail the experiment if status update fails
        pass


def update_results_json(study_log_dir: str, study, metric_name: str, optim_direction: str, 
                       experiment_finished: bool = False, additional_data: dict = None,
                       experiment_config: dict = None):
    """
    Update or create results.json with current experiment results.
    
    Args:
        study_log_dir: Path to study log directory
        study: Optuna study object
        metric_name: Name of the optimization metric
        optim_direction: Optimization direction ('maximize' or 'minimize')
        experiment_finished: Whether the experiment has completed
        additional_data: Additional result data to include
        experiment_config: Experiment configuration (sampler, constraints, etc.)
    """
    easytuna_dir = os.path.join(study_log_dir, ".easytuna")
    results_path = os.path.join(easytuna_dir, "results.json")
    
    # Build results data
    results = {
        "last_updated": datetime.now().astimezone().isoformat(),
        "experiment_finished": experiment_finished,
        "n_trials": len(study.trials),
        "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "metric_name": metric_name,
        "optimization_direction": optim_direction,
    }
    
    # Add experiment configuration if provided
    if experiment_config:
        results["experiment_config"] = experiment_config
    
    # Add best trial information if available
    try:
        best_trial = study.best_trial
        results["best_trial"] = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "datetime_start": best_trial.datetime_start.isoformat() if best_trial.datetime_start else None,
            "datetime_complete": best_trial.datetime_complete.isoformat() if best_trial.datetime_complete else None,
        }
        results["best_value"] = best_trial.value
        # Effective parameters after snapping:
        # Filter user_attrs to drop "_sampled"
        results["best_params"] = {k: v for k, v in best_trial.user_attrs.items() if not k.endswith("_sampled")}
    except ValueError:
        # No feasible trials yet
        results["best_trial"] = None
        results["best_value"] = None
        results["best_params"] = None
        results["best_raw_params"] = None
    
    # Add any additional data
    if additional_data:
        results.update(additional_data)
    
    # Write results (live in .easytuna)
    try:
        os.makedirs(easytuna_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        # Copy to study dir and set read-only (force overwrite)
        study_results_path = os.path.join(study_log_dir, "results.json")
        try:
            if os.path.exists(study_results_path):
                os.chmod(study_results_path, 0o666)  # Allow overwrite of read-only mirror.
                os.remove(study_results_path)
        except Exception:
            pass
        shutil.copy2(results_path, study_results_path)
        try:
            os.chmod(study_results_path, 0o444)
        except Exception:
            pass
    except IOError as e:
        # Don't fail the experiment if results update fails
        pass


class DotDict(dict):
    """
    A dictionary that supports attribute-style access:
        d = DotDict({'x': 1, 'y': {'z': 2}})
        print(d.x)        # 1
        print(d.y.z)      # 2
        d.a = 3
        print(d['a'])     # 3
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to DotDict
        for key, value in list(self.items()):
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        # Wrap any assigned dicts automatically
        if isinstance(value, dict):
            value = DotDict(value)
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
