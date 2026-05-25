# main_logic.py
"""
Experiment driver integrating Optuna hyperparameter (hyp) search with:
  • c-TPE constraint on total model parameter count
  • generalized post-sampling hyp divisibility constraints
  • optional *expandable grid* snapping for discrete val_list params
  • handling val_list as ordinals for well-behaved Optuna hyp optimisation

**Directory Structure**
Storage structure: <log_dir>/<exper_id>/<study_id>/
- exper_id: Experiment container (can hold multiple studies)
- study_id: Individual hyperparameter study
- If exper_id is None, it defaults to study_id (backward compatibility)

Each study directory contains:
- study.out: Main experiment log
- status.json: Read-only snapshot of status (updated after every change)
- results.json: Read-only snapshot of results (updated after every change)
- scripts/: Snapshotted training scripts
- trial_runs/: Individual trial outputs
- .easytuna/: Internal EasyTuna files (live status.json, results.json, Optuna journal DB)

**Expandable grid (Option 2)**
If an integer hyperparameter is defined with a `val_list` and also declares
`divisible_by`, we behave as follows:

    1. Sample a raw value from the list (Optuna sees coarse anchors; good for TPE).
    2. Filter the list to values divisible by the divisor LCM.
       • If non-empty: snap among that filtered list (nearest/floor/ceil).
       • If empty *and* `allow_snap_outside_list=True`:
            Snap to the nearest valid multiple of the divisor LCM within the
            numeric envelope [min(val_list), max(val_list)] — even if the final
            value is not in the original list.
       • Else: prune the trial.

Raw sampled values remain in `trial.params`. Effective, snapped values are
passed to training and recorded in `trial.user_attrs`.
"""
import os, sys, time, warnings
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from ._utils import (
    DotDict, _lcm_many, _snap_to_multiple, _snap_from_val_list, setup_file_logger, handle_exception,
    _snapshot_scripts, update_status_json, update_results_json)
from ._isolrun import run_isolated
from ._termination_utils import check_termination_signal, acknowledge_termination
from ._multi_seed_utils import run_multi_seed_training


# Toggle to reduce debug noise (overridden by run_experiment arg).
VERBOSE_DEFAULT = False

# --------------------------------------------------------------------------- #
#  Divisibility post-processing
# --------------------------------------------------------------------------- #
def apply_divisibility_constraints(trial, logger, sampled_hyps: dict, hyp_config: dict, *, verbose: bool):
    """
    Mutate `sampled_hyps` in place: for each hyperparameter that declares
    `divisible_by`, snap `sampled_hyps[name]` to a valid multiple of the LCM
    of its referenced divisors.

    Supported schema additions in hyp_config[name]:
        divisible_by            : str | list[str] (names of other int hypers)
        rounding                : 'nearest' | 'floor' | 'ceil' (default 'nearest')
        allow_snap_outside_list : bool (val_list only; see Option 2 docs)

    Behavior:
        • If param has a val_list:
            - Filter list to values divisible by divisor LCM.
            - If any remain: snap among them.
            - Else if allow_snap_outside_list: treat [min(list), max(list)] as interval and snap there.
            - Else prune.
        • If param has an interval: snap within that interval.

    Records (trial.user_attrs):
        <name>_raw : value Optuna actually sampled (pre-snap)
        <name>     : snapped (effective) value used for training
    """
    for name, cfg in hyp_config.items():
        if not isinstance(cfg, DotDict):
            continue

        div = cfg.get('divisible_by')
        if not div:
            continue  # nothing to do

        # Normalize to list
        divisors = [div] if isinstance(div, str) else list(div)

        # Collect divisor values (must be ints)
        divisor_vals = []
        for dv in divisors:
            if dv not in sampled_hyps:
                raise RuntimeError(f"divisible_by reference '{dv}' not sampled before coercion.")
            dv_val = sampled_hyps[dv]
            if not isinstance(dv_val, int):
                raise RuntimeError(f"divisible_by='{dv}' requires int but got {type(dv_val)}.")
            divisor_vals.append(dv_val)

        # Combine via LCM so value is divisible by *all* divisors
        d = _lcm_many(divisor_vals)
        if d <= 0:
            raise RuntimeError(f"Invalid divisor computed for {name}: {d}")

        raw = sampled_hyps[name]
        mode = cfg.get('rounding', 'nearest')

        if mode not in ('nearest', 'floor', 'ceil'):
            raise ValueError(f"Invalid rounding='{mode}' for {name}.")

        # --- Case A: val_list ------------------------------------------------ #
        if 'val_list' in cfg:
            valid_vals = [v for v in cfg.val_list if (v % d) == 0]

            if valid_vals:
                snapped = _snap_from_val_list(raw, valid_vals, mode)
                if verbose and snapped != raw:
                    logger.info(f"[DBG] snapping {name} (in-list): raw={raw} -> {snapped} (d={d}, mode={mode})")

            else:
                # Fallback if allowed: use numeric envelope of the list
                if cfg.get('allow_snap_outside_list', False):
                    lo, hi = min(cfg.val_list), max(cfg.val_list)
                    snapped = _snap_to_multiple(raw, d, lo, hi, mode)
                    if snapped is None:
                        raise optuna.TrialPruned(
                            f"No feasible {name} in [{lo}, {hi}] divisible by {divisors} (LCM={d}) even after expansion."
                        )
                    if verbose:
                        logger.info(f"[DBG] {name}: no divisible value in val_list; snapped outside list → {snapped} "
                              f"(raw={raw}, d={d}, mode={mode})")
                else:
                    raise optuna.TrialPruned(
                        f"No feasible {name} in val_list divisible by {divisors} (LCM={d})."
                    )

        # --- Case B: interval ------------------------------------------------ #
        else:
            assert 'interval' in cfg, f"divisible_by requires 'interval' or 'val_list' for {name}"
            lo, hi = cfg.interval
            snapped = _snap_to_multiple(raw, d, lo, hi, mode)
            if snapped is None:
                raise optuna.TrialPruned(
                    f"No feasible {name} in [{lo}, {hi}] divisible by {divisors} (LCM={d})."
                )
            if verbose and snapped != raw:
                logger.info(f"[DBG] snapping {name}: raw={raw} -> {snapped} (d={d}, mode={mode})")

        # Record attrs for analysis
        trial.set_user_attr(f"{name}_raw", raw)
        trial.set_user_attr(name, snapped)

        # Update the value that will be passed to training
        sampled_hyps[name] = snapped

# --------------------------------------------------------------------------- #
#  Hyperparameter sampling
# --------------------------------------------------------------------------- #
def _process_one_config(trial, hyp_name: str, cfg: DotDict):
    """
    Process one hyperparameter configuration for name `hyp_name`.
    Returns a sampled value in native dtype (int or float).

    Supports:
      • int/float w/ interval or val_list (mutually exclusive).
      • log scale for floats only.
      • For val_list, Optuna samples an index (`<hyp>_idx`) so TPE gets an ordinal space.
      • Special handling for 'seed' type with multiple seeds configuration.
    """
    assert 'type' in cfg.keys(), "please specify hyperparameter data type"
    
    # Handle special seed type
    if cfg.type == 'seed':
        assert 'seeds' in cfg.keys(), "seed type requires 'seeds' list"
        assert 'parallel' in cfg.keys(), "seed type requires 'parallel' boolean"
        # For seed type, we return the entire configuration for multi-seed handling
        return cfg
    
    assert cfg.type in [int, float], "hyperparameter type must be int or float"
    assert (
        ('val_list' in cfg.keys()) ^ ('interval' in cfg.keys())
    ), "exactly one of 'val_list' and 'interval' should be specified"

    # Validate init is in range / list
    if 'init' in cfg:
        if 'interval' in cfg:
            lo, hi = cfg.interval
            assert lo <= cfg.init <= hi, (
                f"init {hyp_name}={cfg.init} not in [{lo}, {hi}]"
            )
        if 'val_list' in cfg:
            assert cfg.init in cfg.val_list, (
                f"init {hyp_name}={cfg.init} not in allowed values {cfg.val_list}"
            )

    # Type checks
    if cfg.type in [int, float]:
        data_type = cfg.type
        error_message = f"please specify numbers of the given hyperparameter data type {data_type}"
        if 'init' in cfg.keys():
            assert type(cfg.init) == data_type, error_message
        if 'interval' in cfg.keys():
            assert type(cfg.interval[0]) == type(cfg.interval[1]) == data_type, error_message
        if 'val_list' in cfg.keys():
            assert all(isinstance(x, data_type) for x in cfg.val_list), error_message

    if cfg.type == int and cfg.get('log', False):
        raise AssertionError(
            "log scale for integer hyp-s not implemented; use 'val_list' or a float->int cast."
        )

    # Discrete list sampling
    if 'val_list' in cfg.keys():
        values = cfg.val_list
        assert all(x < y for x, y in zip(values, values[1:])), "'val_list' should be strictly increasing"
        idx = trial.suggest_int(name=f"{hyp_name}_idx", low=0, high=len(values)-1)
        return values[idx]

    # Interval sampling
    suggest_dict = {
        'name': hyp_name,
        'low': cfg.interval[0],
        'high': cfg.interval[1],
    }
    if 'step' in cfg.keys():
        suggest_dict['step'] = cfg['step']
    if 'log' in cfg.keys():
        suggest_dict['log'] = cfg['log']

    if cfg.type == int:
        return trial.suggest_int(**suggest_dict)
    else:
        return trial.suggest_float(**suggest_dict)

# --------------------------------------------------------------------------- #
#  Objective (internal)
# --------------------------------------------------------------------------- #
def _objective(trial, metric_name: str, exper_id: str, study_id: str, log_dir: str, study_log_dir: str, hyp_config: DotDict, 
               constraints_config: dict, train_model_script: str, 
               trial_timeout: int, logger, verbose: bool, termination_flag = None,
               tot_num_params_range: list = None):
    # Check for termination signal before starting trial  
    experiment_dir = os.path.join(log_dir, exper_id)
    if check_termination_signal(study_log_dir, "", experiment_dir):
        logger.info(f"🛑 Termination signal detected, pruning trial {trial.number}")
        acknowledge_termination(study_log_dir, "", experiment_dir, logger)
        raise optuna.TrialPruned("Experiment terminated by user request")
    
    # Store constraints configuration for the constraint function
    if constraints_config is not None:
        trial.set_user_attr("constraints_config", constraints_config)
    
    # Backward compatibility: still set tot_num_params_range if provided
    if tot_num_params_range is not None:
        assert tot_num_params_range[0] < tot_num_params_range[1], \
            "Please leave space between [min, max] number of parameters."
        trial.set_user_attr("tot_num_params_range", tot_num_params_range)

    # 1) Sample all hyperparameters (raw)
    final_hyps = {}
    seed_config = None
    seed_arg_name = None
    for hyp, cfg in hyp_config.items():
        val = _process_one_config(trial, hyp, cfg) if isinstance(cfg, DotDict) else cfg
        
        # Handle special seed configuration
        if isinstance(cfg, DotDict) and cfg.get('type') == 'seed':
            seed_config = val
            seed_arg_name = hyp  # Extract the seed argument name from the hyperparameter name
            continue  # Don't include in final_hyps
            
        final_hyps[hyp] = val
        # Keep the raw sampled value in attrs (for debugging / analysis)
        trial.set_user_attr(f"{hyp}_sampled", val)

    # 2) Apply divisibility snapping (in-place mutation)
    apply_divisibility_constraints(
        trial, sampled_hyps=final_hyps, hyp_config=hyp_config, logger=logger, verbose=verbose)
    # Mirror the *effective* values of all hyperparameters for convenience.
    for k, v in final_hyps.items():
        trial.set_user_attr(k, v)

    # 3) Create trial directory under <experiment>/<study_id>/trial_runs/
    trial_runs_dir = os.path.join(log_dir, exper_id, study_id, "trial_runs")
    per_trial_dir = os.path.join(trial_runs_dir, f"trial{trial.number:03d}")
    per_trial_dir = os.path.abspath(per_trial_dir)
    os.makedirs(per_trial_dir, exist_ok=True)

    # 4) Handle multi-seed or single-seed training
    snapshotted_script_path = os.path.join(study_log_dir, "scripts", os.path.basename(train_model_script))
    original_script_dir = os.path.dirname(os.path.abspath(train_model_script))
    
    # The root for imports should be the 'scripts' directory itself
    import_root = os.path.join(study_log_dir, "scripts")

    # Determine what constraint metrics to fetch
    constraint_metrics = list(constraints_config.keys()) if constraints_config else ["n_params"]
    
    if seed_config is not None:
        # Multi-seed setup
        metric_val, constraint_values = run_multi_seed_training(
            seed_config, seed_arg_name, final_hyps, snapshotted_script_path,
            metric_name, constraint_metrics, trial.number, per_trial_dir, trial_timeout,
            verbose, logger, trial, termination_flag,
            original_script_dir=original_script_dir, import_root=import_root
        )
    else:
        # Single-seed training (traditional behavior)
        args_list = []
        for name, value in final_hyps.items():
            if isinstance(value, bool):
                if value:
                    args_list.append(f"--{name}")
            else:
                args_list.extend([f"--{name}", str(value)])

        if verbose:
            logger.info(f"[DBG] Trial {trial.number} effective args_list: {args_list}")

        # Prepare fetch list for run_isolated
        fetch_list = [metric_name] + constraint_metrics

        # Use run_isolated for robust execution and result parsing
        try:
            res = run_isolated(
                script_path=snapshotted_script_path,
                args_list=args_list,
                fetch=fetch_list,
                timeout=trial_timeout,
                log_dir=per_trial_dir,
                env={"LOG_DIR": per_trial_dir},
                termination_flag=termination_flag,
                logger=logger,
                cwd=original_script_dir,
                import_root=import_root
            )
            
            if verbose:
                logger.info(f"[DBG] Training stdout:\n{res['stdout']}")
                if res['stderr']:
                    logger.warning(f"[DBG] Training stderr:\n{res['stderr']}")
            
            metric_val = res['outputs'][metric_name]
            
            # Extract constraint values
            constraint_values = {}
            for constraint_name in constraint_metrics:
                constraint_values[constraint_name] = res['outputs'][constraint_name]
            
        except Exception as e:
            logger.error(f"Training script failed: {e}")
            raise optuna.TrialPruned(f"Training script failed: {e}")

    # Guard against train_model returning None/NaN
    for constraint_name, value in constraint_values.items():
        if value is None or value <= 0:
            raise RuntimeError(f"{train_model_script} returned invalid {constraint_name}={value}")
    
    if metric_val is None or not (metric_val == metric_val):  # NaN check
        metric_val = float('-inf')

    # 5) Save constraint values for the constraint function
    for constraint_name, value in constraint_values.items():
        trial.set_user_attr(f"constraint_{constraint_name}", value)
    
    # Backward compatibility: also set num_params if n_params is one of the constraints
    if 'n_params' in constraint_values:
        trial.set_user_attr("num_params", constraint_values['n_params'])

    # 6) Log constraint values and objective value
    constraint_log = ", ".join([f"{name}: {value:,}" for name, value in constraint_values.items()])
    logger.info(f"[I] Trial {trial.number} → {constraint_log}  Objective value: {metric_val:.4f}")

    return metric_val

# --------------------------------------------------------------------------- #
#  Constraint function used by c-TPE sampler (internal)
# --------------------------------------------------------------------------- #
def _constraints_func(trial):
    """
    Return a list of constraint violation magnitudes for Optuna's c-TPE:

        value <= 0  ⇒ constraint satisfied

    Supports multiple custom constraints with user-defined names.
    For pruned trials (e.g., divisibility failure before training),
    returns positive violations and does not assert.

    Backward compatibility: Still supports the legacy tot_num_params_range.
    """
    constraints_config = trial.user_attrs.get("constraints_config", None)
    
    # Backward compatibility: Check for legacy tot_num_params_range
    if constraints_config is None:
        n = trial.user_attrs.get("num_params", None)
        tot_num_params_range = trial.user_attrs.get("tot_num_params_range", None)
        
        if n is None or tot_num_params_range is None:
            # Trial did not reach training; mark infeasible.
            return [1.0, 1.0]

        return [
            tot_num_params_range[0] - n,  # ≤ 0 ⇒ n ≥ min
            n - tot_num_params_range[1],  # ≤ 0 ⇒ n ≤ max
        ]
    
    # New multi-constraint system
    violations = []
    for constraint_name, constraint_config in constraints_config.items():
        metric_value = trial.user_attrs.get(f"constraint_{constraint_name}", None)
        
        if metric_value is None:
            # Trial did not reach training or constraint metric missing; mark infeasible
            violations.append(1.0)
            if 'min_value' in constraint_config and 'max_value' in constraint_config:
                violations.append(1.0)  # Add second violation for max constraint
        else:
            # Check min constraint if specified
            if 'min_value' in constraint_config:
                violations.append(constraint_config['min_value'] - metric_value)  # ≤ 0 ⇒ metric ≥ min
            
            # Check max constraint if specified
            if 'max_value' in constraint_config:
                violations.append(metric_value - constraint_config['max_value'])  # ≤ 0 ⇒ metric ≤ max
    
    return violations if violations else [0.0]  # Return at least one constraint

# --------------------------------------------------------------------------- #
#  Normalize user hyp_config into DotDicts with metadata
# --------------------------------------------------------------------------- #
def _normalize_hyp_config(hyp_config: DotDict):
    """
    Wrap any plain dict hyperparameter configs in DotDict, attach .hyp_name.
    We add .hyp_name for convenience.
    """
    for hyp in list(hyp_config.keys()):
        if isinstance(hyp_config[hyp], dict):
            hyp_config[hyp]['hyp_name'] = hyp
            hyp_config[hyp] = DotDict(hyp_config[hyp])
    return hyp_config


# --------------------------------------------------------------------------- #
#  Build initial trial param dict from cfg.init values
# --------------------------------------------------------------------------- #
def _build_init_params(hyp_config: DotDict):
    """
    Extract initial parameter suggestions for enqueue_trial.

      • For val_list params: Optuna expects the *index*.
      • For interval params: use the numeric init value.
      • Constants are ignored (no dict wrapper).
      • Seed configurations are skipped as they don't need Optuna sampling.
    """
    init_params = {}
    for hyp, cfg in hyp_config.items():
        if isinstance(cfg, DotDict) and 'init' in cfg:
            # Skip seed configurations as they don't need Optuna sampling
            if cfg.get('type') == 'seed':
                continue
            if 'val_list' in cfg:
                idx = cfg.val_list.index(cfg.init)
                init_params[f"{hyp}_idx"] = idx
            else:
                init_params[hyp] = cfg.init
    return init_params

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def run_experiment(
    hyp_config,
    train_model_script: str,
    sampler_name: str = "cTPE",   # "cTPE" or "cBO"
    metric_name: str = "accuracy",
    optim_direction: str = "maximize",
    study_id: str = 'my_study123',
    exper_id: str = None,  # Optional - if None, uses study_id
    resume_if_exists: bool = False,
    n_trials: int = 50,
    n_parallel_trials: int = 1,
    timeout: int = None,
    trial_timeout: int = None,
    seed: int = 42,
    n_startup_trials: int = 10,
    storage: optuna.storages.BaseStorage | str | None = None,
    log_dir: int = 'logs',
    verbose: bool = VERBOSE_DEFAULT,
    enable_termination_check: bool = True,
    termination_check_interval: int = 1,  # Check every N trials
    constraints: dict = None,
    # Backward compatibility for legacy parameter count constraint
    tot_num_params_range: list = None,
) -> optuna.Study:
    """
    Public entry point for running an Optuna experiment.

    Parameters
    ----------
    hyp_config : DotDict
        User-defined hyperparameter space. See schema in docstring top of file.
    train_model_script : str
        Path to the training script executed via run_isolated.
    constraints : dict, optional
        Multiple custom constraints with user-defined names. Dictionary format:
        {
            'constraint_name': {
                'min_value': float (optional),  # Minimum allowed value
                'max_value': float (optional),  # Maximum allowed value
            },
            'another_constraint': {
                'min_value': float,
                'max_value': float,
            },
            # ... more constraints
        }
        
        Your training script should set these as global variables matching the constraint names.
        Example: For constraint 'n_params', set n_params = 1234567 in your training script.
        
        At least one of min_value or max_value must be specified for each constraint.
        
    tot_num_params_range : list [min_params, max_params], optional
        DEPRECATED: Legacy constraint for total model parameters. Use 'constraints' parameter instead.
        Maintained for backward compatibility. If both constraints and tot_num_params_range are 
        provided, constraints takes precedence.
    study_id : str
        ID for this specific hyperparameter study.
    exper_id : str, optional
        ID for the experiment containing multiple studies. If None, uses study_id.
        Storage structure: <log_dir>/<exper_id>/<study_id>/
    n_trials : int
        Number of NEW trials to run in this optimize() call. This is NOT the total
        number of trials for the study - each call to optimize() runs this many additional trials.
        When resuming a study, this specifies how many MORE trials to add beyond existing ones.
    n_parallel_trials : int
        Number of trials to run in parallel.
    timeout : int (seconds)
        Global optimization wallclock timeout.
    seed : int
        Random seed for the TPE sampler.
    n_startup_trials : int
        Number of random sampling trials before using TPE/BO algorithm. 
        Default is 10, which works well for <=5 hyperparameters. Increase for larger hyp spaces.
        When resuming: if the study already has >= n_startup_trials, TPE/BO is used immediately.
        If resuming with fewer trials than n_startup_trials, random sampling continues until
        the total reaches n_startup_trials, then switches to TPE/BO.
    storage : optuna.storages.BaseStorage
        Ignore this. For advanced users.
    log_dir : str
        Directory to save the logs and the optuna configuration for resumption, etc.
        Recommended to be on network storage (NFS, SMB, etc.) for multi-node access.
    verbose : bool
        Print debugging info (snap decisions, args_list, etc.).
    enable_termination_check : bool
        Enable experiment termination mechanism.
    termination_check_interval : int
        Check for termination signal every N trials.

    Returns
    -------
    optuna.Study
        Fully populated study; inspect `study.best_trial`, `study.trials`, etc.
    """
    if n_trials <= n_startup_trials:
        warnings.warn(
            f"n_trials ({n_trials}) must be greater than n_startup_trials ({n_startup_trials}). "
            "If you are resuming a study, please ignore this warning."
        )
    
    # Handle exper_id logic: if None, use study_id (backward compatibility)
    if exper_id is None:
        exper_id = study_id
    
    # Set up directory structure: <log_dir>/<exper_id>/<study_id>/
    study_log_dir = os.path.join(log_dir, exper_id, study_id)
    
    # Forward error messages to logs
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: handle_exception(
        exc_type, exc_value, exc_traceback, logger
    )

    # Accept plain dict or DotDict.
    if not isinstance(hyp_config, DotDict):
        hyp_config = DotDict(hyp_config)

    # Normalize user config (wrap per-hparam dicts, attach names).
    hyp_config = _normalize_hyp_config(hyp_config)

    # Validate and process constraints
    if constraints is not None and tot_num_params_range is not None:
        raise ValueError("Cannot specify both 'constraints' and 'tot_num_params_range'. Use 'constraints' for new projects.")
    
    if constraints is None and tot_num_params_range is None:
        raise ValueError("Must specify either 'constraints' or 'tot_num_params_range' parameter.")
    
    # Process constraints configuration
    constraints_config = None
    if constraints is not None:
        # Validate constraints format
        if not isinstance(constraints, dict):
            raise TypeError("'constraints' must be a dictionary")
        
        constraints_config = {}
        for constraint_name, constraint_spec in constraints.items():
            if not isinstance(constraint_spec, dict):
                raise TypeError(f"Constraint '{constraint_name}' must be a dictionary with 'min_value' and/or 'max_value'")
            
            if 'min_value' not in constraint_spec and 'max_value' not in constraint_spec:
                raise ValueError(f"Constraint '{constraint_name}' must specify at least one of 'min_value' or 'max_value'")
            
            # Validate constraint values
            if 'min_value' in constraint_spec and 'max_value' in constraint_spec:
                if constraint_spec['min_value'] >= constraint_spec['max_value']:
                    raise ValueError(f"Constraint '{constraint_name}': min_value ({constraint_spec['min_value']}) must be less than max_value ({constraint_spec['max_value']})")
            
            constraints_config[constraint_name] = constraint_spec.copy()
        
    elif tot_num_params_range is not None:
        # Legacy mode: convert tot_num_params_range to constraints format
        if len(tot_num_params_range) != 2:
            raise ValueError("tot_num_params_range must be a list of [min_params, max_params]")
        if tot_num_params_range[0] >= tot_num_params_range[1]:
            raise ValueError("tot_num_params_range: min_params must be less than max_params")
        
        constraints_config = {
            'n_params': {
                'min_value': tot_num_params_range[0],
                'max_value': tot_num_params_range[1]
            }
        }

    # Log important info at start of each study.
    logger = setup_file_logger(exper_id=study_log_dir, log_dir="", filename="study.out")
    
    # Initialize status tracking
    update_status_json(study_log_dir, {
        "status": "initializing",
        "experiment_id": exper_id,
        "study_id": study_id,
        "train_script": train_model_script,
        "metric_name": metric_name,
        "optimization_direction": optim_direction,
        "n_trials_target": n_trials,
        "n_parallel_trials": n_parallel_trials
    })
    
    logger.info(f"Experiment ID          : {exper_id}")
    logger.info(f"Study ID               : {study_id}")
    logger.info(f"Training script path   : {train_model_script}")

    # --- Termination watcher thread (poll every 2s) ---
    import threading
    import atexit
    termination_flag = {'stop': False}
    study_ref = {'study': None}  # Will hold reference to study for stopping

    def termination_watcher():
        from ._termination_utils import check_termination_signal, acknowledge_termination
        experiment_dir = os.path.join(log_dir, exper_id)
        while not termination_flag['stop']:
            # Update status to show we're alive (only if study exists)
            try:
                if study_ref.get('study') is not None:
                    update_status_json(study_log_dir, {"status": "running"})
            except Exception:
                pass  # Don't fail if status update fails
            
            if check_termination_signal(study_log_dir, "", experiment_dir):
                logger.info("🛑 Termination signal detected by watcher. Acknowledging and setting termination flag.")
                update_status_json(study_log_dir, {"status": "terminating"})
                acknowledge_termination(study_log_dir, "", experiment_dir, logger)
                # Confirm deletion (network storage safety)
                import time as _time
                for _ in range(10):
                    _time.sleep(0.2)
                    if not check_termination_signal(study_log_dir, "", experiment_dir):
                        break
                else:
                    logger.warning("⚠️ Termination file could not be deleted after multiple attempts.")
                termination_flag['stop'] = True

                # Try to stop the study gracefully
                if study_ref['study'] is not None:
                    try:
                        study_ref['study'].stop()
                        logger.info("🛑 Study stop() called successfully.")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to call study.stop(): {e}")

                break
            time.sleep(2)

    if enable_termination_check:
        watcher_thread = threading.Thread(target=termination_watcher, daemon=True)
        watcher_thread.start()
        atexit.register(lambda: termination_flag.update({'stop': True}))

    # Git info
    try:
        import subprocess
        root = os.path.dirname(os.path.abspath(train_model_script))
        sha    = subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],  text=True).strip()
        branch = subprocess.check_output(['git','-C',root,'rev-parse','--abbrev-ref','HEAD'], text=True).strip()
        status = subprocess.check_output(['git','-C',root,'status','--porcelain'], text=True)
        logger.info(f"Git branch             : {branch}")
        logger.info(f"Git commit             : {sha}")
        if status:
            logger.info("Git has uncommitted changes:\n" + status.strip())
    except Exception as e:
        logger.warning(f"Could not gather git info: {e}")
    # Other info
    try:
        import torch
        logger.info(f"PyTorch version        : {torch.__version__}")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            logger.info(f"CUDA devices           : {devices}")
    except ImportError:
        pass
    logger.info(f"Target metric          : {metric_name}")
    logger.info(f"Optimization direction : {optim_direction}")
    logger.info(f"Sampler                : {sampler_name}")
    logger.info(f"Number of trials       : {n_trials}")
    logger.info(f"Startup trials         : {n_startup_trials}")
    logger.info(f"Parallel jobs          : {n_parallel_trials}")
    logger.info(f"Global timeout (s)     : {timeout}")
    logger.info(f"Per‑trial timeout (s)  : {trial_timeout}")
    logger.info("Hyperparameter config  :")
    for name, cfg in hyp_config.items():
        if isinstance(cfg, DotDict):
            if cfg.get('type') == 'seed':
                logger.info(f"  • {name}: seeds={cfg.seeds}, parallel={cfg.parallel}")
            else:
                logger.info(f"  • {name}: " +
                            ("interval="+str(cfg.interval) if 'interval' in cfg else
                            "values="+str(cfg.val_list)) +
                            (", log" if cfg.get('log') else "") +
                            (f", divisible_by={cfg.divisible_by}" if cfg.get('divisible_by') else ""))
    
    # Log constraints information
    if tot_num_params_range is not None:
        # Legacy mode
        logger.info(f"Param‑count constraint : [{tot_num_params_range[0]}, {tot_num_params_range[1]}]")
    else:
        # New multi-constraint mode
        logger.info("Constraints:")
        for constraint_name, constraint_spec in constraints_config.items():
            min_val = constraint_spec.get('min_value', 'None')
            max_val = constraint_spec.get('max_value', 'None')
            logger.info(f"  • {constraint_name}: [{min_val}, {max_val}]")
    
    # Prepare experiment configuration for results.json
    experiment_config = {
        "sampler_name": sampler_name,
        "n_startup_trials": n_startup_trials,
        "n_parallel_trials": n_parallel_trials,
        "timeout": timeout,
        "trial_timeout": trial_timeout,
    }
    
    # Add constraints information
    if tot_num_params_range is not None:
        experiment_config["constraints"] = {
            "type": "legacy",
            "tot_num_params_range": tot_num_params_range
        }
    else:
        experiment_config["constraints"] = {
            "type": "multiple",
            "constraints": constraints_config
        }
    
    t0 = time.time()

    # Build sampler + study (with JournalStorage by default)
    if sampler_name.lower() == "ctpe":
        sampler = TPESampler(
            multivariate=True,
            seed=seed,
            n_startup_trials=n_startup_trials,
            constraints_func=_constraints_func,
        )
    elif sampler_name.lower() == "cbo":
        try: from optuna.integration import BoTorchSampler
        except ImportError:
            try: from optuna.integration.botorch import BoTorchSampler
            except ImportError as e:
                raise ImportError(
                    "BoTorchSampler not found. "
                    " Install via: pip install optuna-integration[botorch]"
                    "and restart your Python process."
                ) from e
        sampler = BoTorchSampler(
            # Uses the same constraints function as c-TPE.
            constraints_func=_constraints_func,
            n_startup_trials=n_startup_trials,
            seed=seed,
            # you can pass other BoTorchSampler args here, e.g.:
            # gpytorch_model_args={"covar_module": ...},
            # acq_func="EI", etc.
        )
    else:
        raise ValueError(f"Unknown sampler_name={sampler_name!r}")
    # Default to a local log file if no storage is supplied.
    if storage is None:
        easytuna_dir = os.path.join(study_log_dir, '.easytuna')
        log_path = os.path.join(easytuna_dir, 'study.log')
        os.makedirs(easytuna_dir, exist_ok=True)
        storage = JournalStorage(JournalFileBackend(log_path))
    study = optuna.create_study(
        study_name=study_id,
        load_if_exists=resume_if_exists,
        direction=optim_direction,
        sampler=sampler,
        storage=storage,
    )
    
    # Set study reference for termination watcher
    study_ref['study'] = study

    # Handle snapshotting logic based on resumption and study state
    if resume_if_exists:
        # Check if this is a truly existing study (has trials) or just a "resume" flag on a new study
        if len(study.trials) > 0:
            # This is a real resumption of an existing study - ensure snapshot exists
            scripts_dir = os.path.join(study_log_dir, "scripts")
            snapshotted_script = os.path.join(scripts_dir, os.path.basename(train_model_script))
            
            if not os.path.exists(snapshotted_script):
                raise FileNotFoundError(
                    f"Cannot resume study: Required script snapshot not found.\n"
                    f"Expected location: {snapshotted_script}\n"
                    f"Script snapshots are created automatically when a study is first run. "
                    f"If you're resuming an old study created before snapshotting was implemented, "
                    f"you may need to recreate the study or manually copy your scripts to: {scripts_dir}"
                )
            
            if verbose:
                logger.info(f"Resuming existing study with {len(study.trials)} trials using snapshotted scripts from: {scripts_dir}")
        else:
            # resume_if_exists=True was set, but this study has no previous trials.
            if verbose:
                logger.info("Study marked for resumption but has no previous trials. Creating fresh snapshot.")
            _snapshot_scripts(study_log_dir, train_model_script)
    else:
        # New study - create snapshot
        _snapshot_scripts(study_log_dir, train_model_script)

    # Mark stale RUNNING trials as failed and re-enqueue their params.
    if resume_if_exists:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                # append the "set_trial_state_values" op to the journal
                storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL, values=None)
                logger.warning(f"Journal: marked trial {t.number} as FAIL")
                # now re‑enqueue the same params so they actually run
                study.enqueue_trial(t.params)
    
    # Enqueue an init trial for faster warm start (optional but recommended)
    if not resume_if_exists or len(study.trials) == 0:
        init_params = _build_init_params(hyp_config)
        if verbose:
            logger.info(f'Init parameters: {init_params}')
        study.enqueue_trial(init_params)

    # Launch optimisation
    def _objective_with_termination_flag(trial, *args, **kwargs):
        if termination_flag['stop']:
            logger.info("🛑 Termination flag set. Pruning trial and stopping study.")
            raise optuna.TrialPruned("Experiment terminated by user request (flag)")
        return _objective(trial, *args, **kwargs, termination_flag=termination_flag)

    # Callback to update results after each trial
    def results_callback(study, trial):
        update_results_json(study_log_dir, study, metric_name, optim_direction, 
                           experiment_config=experiment_config)

    try:
        update_status_json(study_log_dir, {"status": "optimizing"})
        study.optimize(
            lambda trial: _objective_with_termination_flag(
                trial,
                metric_name=metric_name,
                exper_id=exper_id,
                study_id=study_id,
                log_dir=log_dir,
                study_log_dir=study_log_dir,
                hyp_config=hyp_config,
                constraints_config=constraints_config,
                train_model_script=train_model_script,
                trial_timeout=trial_timeout,
                logger=logger,
                verbose=verbose,
                tot_num_params_range=tot_num_params_range,
            ),
            n_trials=n_trials,
            n_jobs=n_parallel_trials,
            timeout=timeout,
            callbacks=[results_callback]
        )
    finally:
        termination_flag['stop'] = True

    t1 = time.time()
    dur = t1 - t0
    logger.info(f"Experiment finished at : {datetime.now().astimezone().isoformat()}")
    logger.info(f"Total duration (s)     : {dur:.1f}")

    # Update final status and results
    update_status_json(study_log_dir, {
        "status": "completed",
        "duration_seconds": dur,
        "finished_at": datetime.now().astimezone().isoformat()
    })
    update_results_json(study_log_dir, study, metric_name, optim_direction, 
                       experiment_finished=True, 
                       additional_data={"duration_seconds": dur},
                       experiment_config=experiment_config)

    # Report best
    try:
        best = study.best_trial
        logger.info(f"🎯 Best feasible trial #{best.number}")
        logger.info(f"    {metric_name} = {best.value:.4f}")
        logger.info(f"    params(raw)={best.params}")
        logger.info(f"    # Parameters: {best.user_attrs.get('num_params', 'NA')}")
        # Show effective snapped values
        logger.info("    Effective (user_attrs):")
        for k, v in best.user_attrs.items():
            if k.endswith("_raw") or k.endswith("_sampled"):
                continue
            logger.info(f"      {k}: {v}")
    except ValueError:
        logger.error("⚠️ No feasible trials found! Try relaxing your parameter window or search ranges.")
        return

    return study
