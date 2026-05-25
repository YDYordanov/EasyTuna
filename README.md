# EasyTuna

Easy Optuna integration for realistic hyperparameter optimization in research settings. Combines powerful Optuna optimization with practical features for fair model comparisons, multi-seed evaluation, and experiment tracking.

## Features

- **🔧 Easy Integration**: Set hyperparameters via CLI arguments; extract relevant metrics from global variables
- **🛡️ Reproducibility via Code Snapshotting**: Guarantees reproducibility by running trials from a complete snapshot of all project Python files, saved to a `scripts/` subfolder within the study log. This isolates the experiment from code changes while preserving relative path access to data.
- **🔄 Resuming Studies**: Easily resume an existing study by setting `resume_if_exists=True`. EasyTuna will continue the optimization from where it left off, using the original snapshotted code to maintain consistency.
- **⚖️ Multiple Constraints**: Define multiple constraint metrics with custom names (e.g., `n_params`, `memory_usage`, `inference_time`) for comprehensive model optimization
- **🎯 Smart Discrete Handling**: Optimal handling of discrete parameter sets via ordinal encoding for better TPE performance  
- **🔗 Divisibility Constraints**: Enforce divisibility relationships between parameters (e.g., `hidden_size` divisible by `num_heads`)
- **🌱 Multi-seed Training**: Run trials with multiple random seeds and average results for robust evaluation
- **📊 Experiments Monitoring Dashboard**: Real-time monitoring of running experiments via the `monitor_experiments.py` script. You can view progress and terminate running trials interactively from the terminal.
- **🛑 Graceful Termination**: Stop long-running experiments from any node via the monitoring dashboard.
- **📁 Hierarchical Organization**: Organize experiments and studies with a clean directory structure.

## Installation

**From this repository:**
```bash
pip install .
```

**Optional extras:**
```bash
pip install ".[botorch]"  # cBO / BoTorch sampler support
pip install ".[demo]"     # dependencies for demo_model_train.py
pip install ".[all]"      # both optional groups
```

**From Git:**
```bash
pip install git+https://github.com/<user>/<repo>.git
```

**Recommended Setup:**
- Place your `log_dir` (experiments directory) on network storage (NFS, SMB, etc.) for multi-node access.
- This enables monitoring and terminating experiments from any compute node.

## How It Works: Code Snapshotting and Resuming

EasyTuna is designed for reproducible research. Here’s how it ensures your experiments are reliable:

1.  **Initial Run**: When you first execute `run_experiment`, EasyTuna performs a one-time snapshot of your code.
    - It starts with your main training script (e.g., `my_train_script.py`).
    - It recursively scans all `import` statements to find every local Python script your project uses.
    - It copies this entire collection of files into a `scripts/` directory inside your study's log path (e.g., `logs/my_experiment/my_study/scripts/`).

2.  **Trial Execution**: Every trial in your study is executed using the code from this `scripts/` snapshot, not the original files. This isolates the experiment from any changes you might make to the source code while the study is running.

3.  **Resuming a Study**: If you stop and later want to continue a study, simply call `run_experiment` again with the same `study_id` and set `resume_if_exists=True`.
    - EasyTuna will detect the existing study and its `scripts/` snapshot.
    - It will **not** create a new snapshot. Instead, it will resume the study, continuing to use the code that was snapshotted on the very first run.
    - This guarantees that all trials—both old and new—are run with the exact same codebase.

```python
# To resume a study, use the same study_id and set resume_if_exists=True
run_experiment(
    # ... same configuration as before ...
    study_id="hyperopt_study_v1",
    resume_if_exists=True,
    n_trials=20,  # This will add 20 MORE trials beyond any existing ones
)
```

### Important Notes for Resuming:

- **`n_trials`**: Always specifies how many new trials to run, not the total. If resuming a study with 30 existing trials and `n_trials=20`, you'll get 50 total trials.
- **`n_startup_trials`**: Keep the same value when resuming. If the study already has ≥ `n_startup_trials` completed, TPE/BO starts immediately. If fewer, random sampling continues until the total reaches `n_startup_trials`.

## Quick Start

### 1. Set up your training script

Your training script should accept hyperparameters as CLI arguments and set target metrics as global variables:

```python
import argparse

# Parse hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=128)
args = parser.parse_args()

# Your training code here...
model = create_model(hidden_size=args.hidden_size)
val_accuracy = train_model(model, lr=args.lr, batch_size=args.batch_size)

# Set target metrics as global variables (REQUIRED - EasyTuna extracts these)
accuracy = val_accuracy
n_params = count_parameters(model)
```

### 2. Define your experiment

```python
from easytuna import run_experiment

# Define hyperparameter space
hyp_config = {
    'lr': {
        'type': float, 
        'interval': [1e-5, 1e-2], 
        'log': True,
        'init': 1e-3
    },
    'batch_size': {
        'type': int, 
        'val_list': [16, 32, 64, 128],
        'init': 32
    },
    'hidden_size': {
        'type': int,
        'val_list': [64, 128, 256, 512],
        'divisible_by': 'num_heads',  # ensure divisibility
        'init': 128
    },
    'num_heads': {
        'type': int,
        'val_list': [4, 8, 12],
        'init': 8
    },
    # Multi-seed configuration
    'seed': {
        'type': 'seed',
        'seeds': [42, 123, 456],
        'parallel': True  # run seeds in parallel
    }
}

# Run optimization
study = run_experiment(
    # Experiment organization
    exper_id='transformer_optimization',    # experiment container
    study_id='base_architecture_v1',       # individual study
    
    # Training setup  
    train_model_script='train.py',
    hyp_config=hyp_config,
    metric_name='accuracy',
    optim_direction='maximize',
    
    # Constraints (choose one approach)
    # Option 1: Multiple custom constraints
    constraints={
        'n_params': {'min_value': 1e6, 'max_value': 10e6},    # parameter count
        'memory_usage': {'max_value': 4096},                   # max 4GB memory
        'inference_time': {'max_value': 50},                   # max 50ms inference
    },
    
    # Option 2: Legacy single constraint (still supported)
    # tot_num_params_range=[1e6, 10e6],
    
    # Optimization settings
    sampler_name='cTPE',     # or 'cBO' for Bayesian
    n_trials=50,            # run 50 additional trials, not 50 total
    n_startup_trials=10,
    timeout=3600,         # 1 hour limit
    
    # Tracking & organization
    log_dir='experiments',
    verbose=True,
)
```

## Directory Structure

EasyTuna creates organized experiment directories:

```
experiments/
└── transformer_optimization/           # exper_id
    └── base_architecture_v1/          # study_id  
        ├── study.out                  # main log file
        ├── status.json                # real-time experiment status
        ├── results.json               # current best results & metrics
        ├── scripts/                   # snapshotted scripts
        ├── .optuna/                   # optuna database (hidden)
        └── trial_runs/                # individual trial outputs
            ├── trial000/
            │   └── terminal.out
            └── trial001/
                └── terminal.out
```

**Status & Results Tracking:**
- `status.json`: Updated every 2 seconds with experiment status, hostname, PID, and last_alive timestamp
- `results.json`: Updated after each trial with current best results, trial counts, and experiment progress

**Network Storage Recommended:**
The logs directory should be placed on network storage (NFS, SMB, etc.) accessible by all compute nodes. This enables:
- Monitoring progress from any location
- Graceful termination from any node
- Centralized result collection

## Experiment Tracking & Termination

EasyTuna does not require any external dashboard. All experiment logs, metrics, and results are stored in the experiment directory tree. This structure allows using tools like Jupyter Lab or VS Code to browse the directory tree, open the results, view or terminate running experiments and write your own reports.

**Real-time Status Tracking:**
- `status.json` provides live experiment status including:
  - Last alive timestamp (updated every 2 seconds)
  - Hostname and process ID
  - Current experiment phase (initializing, optimizing, completed, etc.)
  - Target trial count and parallelization settings

**Results Monitoring:**
- `results.json` contains current optimization results:
  - Best trial found so far with parameters and metric value
  - Trial completion statistics (complete, pruned, failed)
  - Experiment progress and duration
  - Updated after each completed trial

**Monitoring Dashboard:**
Use the included monitoring script to track all running experiments from your terminal:
```bash
python monitor_experiments.py [path/to/your/experiments_dir]
```
This provides a real-time dashboard displaying the status, progress, and best results for all studies. The dashboard view is continuously updated, giving you a live overview of all your experiments.

**Graceful Termination:**
- To stop an experiment, create a file with any of these substrings in its name: `_exit`, `_end`, `_terminate`, or `_kill`.
- Place the file in the experiment, study, or trial directory. The experiment will terminate all processes at the corresponding level as gracefully as possible.
- Example:
  ```bash
  touch experiments/transformer_optimization/base_architecture_v1/stop_terminate
  ```

## Advanced Features

### Metric Extraction

EasyTuna extracts target metrics from your training script's global variables:

```python
# Set target metrics as global variables (REQUIRED)
accuracy = 0.95
n_params = 1024768
```

**Key Points:**
- Set your target metric names (e.g., `accuracy`, `f1_score`) as global variables
- Always include `n_params` as a global variable for parameter constraints  
- Variable names must match the `metric_name` parameter in `run_experiment()`

### Multiple Constraints

Define multiple constraint metrics with custom names for comprehensive optimization:

```python
# In your training script, set constraint metrics as global variables
accuracy = None        # Optimization target
n_params = None        # Model parameter count  
memory_usage = None    # Peak memory usage in MB
inference_time = None  # Average inference time in ms
model_flops = None     # Model computational complexity

# In run_experiment(), specify constraints
constraints = {
    'n_params': {'min_value': 1e6, 'max_value': 10e6},     # 1M-10M parameters
    'memory_usage': {'max_value': 4096},                    # Max 4GB memory
    'inference_time': {'max_value': 50},                    # Max 50ms inference
    'model_flops': {'max_value': 1e9},                      # Max 1B FLOPs
}

study = run_experiment(
    constraints=constraints,
    # ... other parameters
)
```

**Constraint Format:**
- Each constraint needs `min_value`, `max_value`, or both
- Constraint names must match global variables in your training script
- Backward compatible: `tot_num_params_range=[min, max]` still works

### Divisibility Constraints

Ensure parameters maintain mathematical relationships:

```python
'hidden_size': {
    'type': int,
    'val_list': [64, 128, 256, 512],
    'divisible_by': ['num_heads', 'key_size'],  # multiple constraints
    'rounding': 'nearest',                       # or 'floor', 'ceil'
    'allow_snap_outside_list': True             # expand search if needed
}
```

### Multi-seed Training

Run multiple seeds per trial for robust evaluation:

```python
'seed': {
    'type': 'seed',
    'seeds': [42, 123, 456, 789, 999],    # 5 random seeds
    'parallel': False                      # sequential execution
}
```

### Experiment Termination

Stop experiments gracefully from any node:

- **Command Line**: Create a file with `_exit`, `_end`, `_terminate`, or `_kill` in its name in the experiment, study, or trial directory.
- **Programmatic**: Create the termination signal file in the appropriate directory.

## Tips for Best Results

1. **Use `val_list` for integers** when possible - gives TPE better ordinal structure
2. **Set reasonable `tot_num_params_range`** - helps constrain search space effectively  
3. **Scale `n_startup_trials` appropriately** - 10 works well for <=5 hyperparams; increase for larger hyp spaces
4. **When resuming studies** - keep `n_startup_trials` consistent with the original value
5. **Remember `n_trials` is additive** - when resuming, it adds MORE trials, not a total cap
6. **Use multi-seed for final evaluation** - more robust than single-seed results
7. **Monitor experiments via your own tools** – all logs and results are in the experiment directory tree
8. **Use `cBO` sampler for expensive evaluations** - more sample-efficient than TPE

## Backup & Storage Management

### Backing Up Your Experiment Database

To backup your experiments directory from a remote server to your local machine:

```bash
# Basic sync (copy everything)
rsync -avz --progress user@server:/path/to/experiments/ ./local_experiments/

# Exclude large model files to save space and time
rsync -avz --progress --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
      user@server:/path/to/experiments/ ./local_experiments/

# Include additional exclusions for other large files
rsync -avz --progress \
      --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
      --exclude='*.bin' --exclude='*.safetensors' --exclude='*.h5' \
      --exclude='*.pkl' --exclude='*.pickle' \
      user@server:/path/to/experiments/ ./local_experiments/
```

### Automated Backup with Cron

Set up automatic backups that persist across restarts:

```bash
# Edit crontab
crontab -e

# Add a line for daily backup at 2 AM
0 2 * * * /usr/bin/rsync -avz --progress --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' user@server:/path/to/experiments/ /home/user/experiment_backups/ >> /home/user/backup.log 2>&1

# For hourly backups during work hours (9 AM - 6 PM, weekdays)
0 9-18 * * 1-5 /usr/bin/rsync -avz --progress --exclude='*.pth' --exclude='*.pt' user@server:/path/to/experiments/ /home/user/experiment_backups/ >> /home/user/backup.log 2>&1
```

**Backup Best Practices:**
- Use `--exclude` to skip large model checkpoints unless specifically needed
- Monitor backup logs regularly: `tail -f ~/backup.log`
- Consider incremental backups for large experiment directories
- Store backups on a different physical device/location for redundancy

## License

MIT License
