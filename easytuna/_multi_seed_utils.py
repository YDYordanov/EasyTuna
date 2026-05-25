# multi_seed_utils.py
"""
Multi-seed training utilities for EasyTuna.
"""
import os
import sys
import concurrent.futures
from ._isolrun import run_isolated


def run_single_seed(seed, seed_arg_name, final_hyps, train_model_script, metric_name, constraint_metrics,
                   trial_number, per_trial_dir, trial_timeout, verbose, logger, termination_flag=None, original_script_dir=None, import_root=None):
    """
    Run training with a single seed and return (metric_val, constraint_values).
    
    Parameters
    ----------
    seed : int
        The random seed to use for this run
    seed_arg_name : str
        The argument name to use for the seed (e.g., 'seed', 'random_seed', etc.)
    final_hyps : dict
        The hyperparameters (excluding multi-seed config)
    train_model_script : str
        Path to training script
    metric_name : str
        Name of metric to extract
    constraint_metrics : list
        List of constraint metric names to extract
    trial_number : int
        Trial number for logging
    per_trial_dir : str
        Base directory for this trial
    trial_timeout : int
        Timeout for training script
    verbose : bool
        Verbose logging flag
    logger : Logger
        Logger instance
    original_script_dir : str, optional
        The original directory of the training script to use as cwd.
    import_root : str, optional
        The root directory for imports (the snapshot directory).
        
    Returns
    -------
    tuple
        (metric_val, constraint_values_dict) from training
    """
    # Create args list for this seed run
    args_list = []
    for name, value in final_hyps.items():
        if isinstance(value, bool):
            # only emit the flag if True; omit if False (for 'store_true' args)
            if value:
                args_list.append(f"--{name}")
        else:
            args_list.extend([f"--{name}", str(value)])
    
    # Add seed argument using the provided argument name
    args_list.extend([f"--{seed_arg_name}", str(seed)])

    if verbose:
        logger.info(f"[DBG] Trial {trial_number} seed {seed} args_list: {args_list}")
    
    # Create seed-specific subdirectory under per_trial_dir (which is now under trial_runs)
    seed_dir = os.path.join(per_trial_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    # Log file for this seed
    seed_log_file = os.path.join(seed_dir, "terminal.out")
    with open(seed_log_file, 'w') as f:
        f.write("[ARGS] " + " ".join(args_list) + "\n\n")
    
    # Prepare fetch list for run_isolated
    fetch_list = [metric_name] + constraint_metrics
    
    # Run isolated training
    res = run_isolated(
        script_path=train_model_script,
        args_list=args_list,
        fetch=fetch_list,
        timeout=trial_timeout,
        log_dir=seed_dir,
        env={"LOG_DIR": seed_dir},
        termination_flag=termination_flag,
        logger=logger,
        cwd=original_script_dir,
        import_root=import_root
    )
    
    if verbose:
        print(res['stdout'], end='')
        print(res['stderr'], file=sys.stderr, end='')
    
    metric_val = res['outputs'][metric_name]
    
    # Extract constraint values
    constraint_values = {}
    for constraint_name in constraint_metrics:
        constraint_values[constraint_name] = res['outputs'][constraint_name]
    
    # Guard against train_model returning None/NaN for constraints
    for constraint_name, value in constraint_values.items():
        if value is None or value <= 0:
            raise RuntimeError(f"{train_model_script} returned invalid {constraint_name}={value} for seed {seed}")
    
    if metric_val is None or not (metric_val == metric_val):  # NaN check
        metric_val = float('-inf')
    
    return metric_val, constraint_values


def run_multi_seed_training(seed_config, seed_arg_name, final_hyps, train_model_script, 
                           metric_name, constraint_metrics, trial_number, per_trial_dir, trial_timeout, 
                           verbose, logger, trial, termination_flag=None, original_script_dir=None, import_root=None):
    """
    Handle multi-seed training execution with support for multiple constraint metrics.
    
    Parameters
    ----------
    constraint_metrics : list
        List of constraint metric names to extract from training script
    
    Returns
    -------
    tuple
        (averaged_metric_val, dict_of_constraint_values)
    """
    seeds = seed_config['seeds']
    parallel = seed_config['parallel']
    
    if verbose:
        logger.info(f"[I] Trial {trial_number} running with {len(seeds)} seeds: {seeds}")
        logger.info(f"[I] Parallel execution: {parallel}")
    
    # Store seed configuration in trial attributes
    trial.set_user_attr("seeds", seeds)
    trial.set_user_attr("parallel_seeds", parallel)
    
    if parallel:
        # Run all seeds in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
            futures = [
                executor.submit(
                    run_single_seed, seed, seed_arg_name, final_hyps, train_model_script,
                    metric_name, constraint_metrics, trial_number, per_trial_dir, trial_timeout,
                    verbose, logger, termination_flag, original_script_dir, import_root
                )
                for seed in seeds
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        # Run seeds sequentially
        results = []
        for seed in seeds:
            result = run_single_seed(
                seed, seed_arg_name, final_hyps, train_model_script, metric_name, constraint_metrics,
                trial_number, per_trial_dir, trial_timeout, verbose, logger, termination_flag,
                original_script_dir, import_root
            )
            results.append(result)
    
    # Extract metrics and constraint values
    metric_vals = [result[0] for result in results]
    constraint_results = [result[1] for result in results]
    
    # Aggregate constraint values (ensure all seeds have same constraint values)
    aggregated_constraints = {}
    for constraint_name in constraint_metrics:
        constraint_vals = [constraint_dict[constraint_name] for constraint_dict in constraint_results]
        
        # Parameter counts should be identical; other constraints are averaged.
        if constraint_name == 'n_params':
            # Parameter count should be identical across seeds
            assert all(val == constraint_vals[0] for val in constraint_vals), \
                f"Parameter counts differ across seeds: {constraint_vals}"
            aggregated_constraints[constraint_name] = constraint_vals[0]
        else:
            # For other constraints, take the average
            aggregated_constraints[constraint_name] = sum(constraint_vals) / len(constraint_vals)
    
    # Average the metric values
    metric_val = sum(metric_vals) / len(metric_vals)
    
    # Store individual seed results for analysis
    for i, (seed, mv) in enumerate(zip(seeds, metric_vals)):
        trial.set_user_attr(f"seed_{seed}_{metric_name}", mv)
    
    if verbose:
        logger.info(f"[I] Trial {trial_number} seed results: {dict(zip(seeds, metric_vals))}")
        logger.info(f"[I] Trial {trial_number} averaged {metric_name}: {metric_val:.4f}")
    
    return metric_val, aggregated_constraints
