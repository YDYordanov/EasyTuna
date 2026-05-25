# demo_experiment.py
"""
Demo experiment for EasyTuna hyperparameter optimization.

This example shows how to:
- Define hyperparameter spaces with different types (int, float, seed)
- Use divisibility constraints between parameters
- Set up multi-seed training with averaging
- Configure multiple custom constraints (n_params, memory_usage, inference_time, etc.)
- Configure experiment tracking (all logs and results are in the experiment directory tree)
- Organize experiments with hierarchical structure

Your training script should:
- Accept hyperparameters as CLI arguments (via argparse)
- Set target metrics as global variables (e.g., accuracy = 0.95)
- Set constraint metrics as global variables (e.g., n_params = 12345, memory_usage = 1500, inference_time = 45)
- Use os.environ.get('LOG_DIR') for trial-specific logging/saving
"""

from easytuna import run_experiment


def build_demo_space():
    """
    Build the demo hyperparameter config.

    Hyp config settings:
      type: hyperparameter type - int, float, or 'seed'
      init: initial value
      interval: interval of possible values
      val_list: list of possible values (mutually exclusive with 'interval')
      log: use logarithmic scale for sampling or not
      divisible_by: name or list of names of other *int* hypers
      rounding: 'nearest' | 'floor' | 'ceil'
      allow_snap_outside_list: when using val_list + divisible_by, permit snapping
        to multiples *outside* list but within numeric envelope of list (Option 2).
      
      For 'seed' type:
        seeds: list of random seeds to run
        parallel: whether to run seeds in parallel (True) or sequentially (False)
    """
    hyp_config = {

        # --- Optimisation hyperparameters ---
        'lr': {
            'type': float, 'init': 1e-3, 'interval': [1e-5, 1e-2], 'log': True
        },
        'weight_decay': {
            'type': float, 'init': 1e-2, 'interval': [1e-5, 1e-1], 'log': True
        },
        'dropout_rate': {
            'type': float, 'init': 0.1, 'interval': [0.05, 0.5]
        },
        'batch_size': {
            'type': int, 'init': 32, 'val_list': [16, 32, 64, 128]
        },
        
        # --- Architecture hyperparameters ---
        # It's recommended to use val_list for *integer* hyp-s whenever possible, to restrict the hyp search space
        # Just make sure that the val_list is ordered and spaced reasonably (e.g. avoid [1, 10, 11, 15]) 
        # because it will be encoded as ordinals (1, 2, 3,..., n) in the hyp search space in Optuna
        'num_heads': {
            'type': int, 'init': 8, 'val_list': [4, 6, 8, 12, 16]
        },
        'interm_size_ratio': {
            'type': int, 'init': 2, 'interval': [1, 6]  # step defaults to 1
        },
        'num_layers': {
            'type': int, 'interval': [3, 12], 'step': 1
        },

        # --- Hyperparameters divisible by other hyperparameters ---
        'hidden_size': {
            'type': int,
            'init': 128,
            'val_list': [64, 96, 128, 192, 256, 384, 512],
            'divisible_by': 'num_heads',
            'rounding': 'nearest',
            'allow_snap_outside_list': True,  # Reduces pruning when no listed value is divisible.
        },
        
        # --- Multi-seed setup ---
        # Remove this section for single-seed training.
        'seed': {
            'type': 'seed',
            'seeds': [42, 123, 456],  # list of seeds to run
            'parallel': True,  # run all seeds in parallel or sequentially
        },
        
        # --- Constants (passed through verbatim; no dict wrapper) ---
        'epochs': 3,
    }

    return hyp_config

def main():
    hyp_config = build_demo_space()

    # Training script path (relative to project root)
    # The training script should accept hyperparameters as CLI arguments and expose
    # target metric and constraint values as globals. EasyTuna provides a per-trial
    # output directory via os.environ.get('LOG_DIR').
    train_model_script = "demo_model_train.py"

    # Define multiple constraints with custom names
    # Constraint names must match globals set by the training script.
    constraints = {
        'n_params': {
            'min_value': 2e6,    # At least 2M parameters
            'max_value': 8e6,    # At most 8M parameters (allows smaller-medium models)
        },
        'memory_usage': {
            'max_value': 100,    # Max 100MB memory usage
            # Note: no min_value specified, so only upper bound is enforced
        },
        'inference_time': {
            'min_value': 1.0,    # At least 1ms (avoid overly simple models)
            'max_value': 5.0,    # At most 5ms inference time
        },
        'model_flops': {
            'max_value': 200e6,  # At most 200M FLOPs
        }
    }

    out = run_experiment(
        # --- Experiment Organization ---
        exper_id='my_experiment_2025',  # Experiment container (can hold multiple studies)
        study_id='hyperopt_study_v1',   # Individual hyperparameter study 
        # NOTE: Directory structure: logs/<exper_id>/<study_id>/
        # If exper_id is None, it defaults to study_id (backward compatibility)
        
        # --- Optimization Settings ---
        resume_if_exists=True,  # resume the experiment if it exists by study_id 
        sampler_name='cTPE',  # cTPE(default) or cBO hyperparameter sampler in Optuna
        metric_name='accuracy',  # the metric to optimize: this should match the variable name in your run script
        optim_direction='maximize',  # maximize or minimize
        hyp_config=hyp_config,
        constraints=constraints,  # Multiple custom constraints (replaces tot_num_params_range)
        train_model_script=train_model_script,  # the path to your .py training script
        
        # --- Trial Configuration ---
        n_trials=5,  # Run 5 additional trials, not 5 total. Use >=20 total for real experiments.
        timeout=3600,  # or stop after <timeout> seconds, whichever comes first
        trial_timeout=600,  # stop each trial if reached <trial_timeout> seconds
        n_parallel_trials=1,  # run this many trials in parallel
        n_startup_trials=10,  # random exploration before TPE/BO (10 works well for <=5 hyperparams; increase for larger hyp spaces)
        seed=42,
        
        # --- Logging & Tracking ---
        log_dir='logs',  # where to store all logs
        verbose=True,
        # Termination is handled by placing a file with _exit, _end, _terminate, or _kill in the experiment, study, or trial directory.
    )

if __name__ == "__main__":
    main()
    
    print("\n=== Summary ===")
    print("✅ Multiple custom constraints with user-defined names")
    print("✅ Flexible min/max bounds for each constraint")  
    print("✅ Multi-seed training with constraint aggregation")
    print("✅ Seamless integration with existing hyperparameter optimization")
    
    print("\nTo use constraints in your projects:")
    print("1. Update your training script to set constraint metrics as global variables")
    print("   Example: n_params = 1234567; memory_usage = 1500; inference_time = 45")
    print("2. Define constraints dict with min_value/max_value for each constraint")
    print("3. Pass constraints to run_experiment() instead of tot_num_params_range")
