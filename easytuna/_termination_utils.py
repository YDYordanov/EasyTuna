# termination_utils.py
"""
Experiment termination utilities for EasyTuna.
"""
import os
import glob


def _termination_files(directory: str, patterns: list[str]):
    if not directory:
        return []
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return []

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(directory, f"*{pattern}*")))
    return matches


def check_termination_signal(study_dir: str, trial_dir: str, exper_dir: str) -> bool:
    """
    Check for termination files at multiple levels: trial, study, and experiment.
    A termination file at a higher level (e.g., experiment) terminates all sub-processes.
    """
    # Define termination file patterns
    termination_patterns = ["_exit", "_end", "_terminate", "_kill"]
    
    return any(
        _termination_files(directory, termination_patterns)
        for directory in (trial_dir, study_dir, exper_dir)
    )


def acknowledge_termination(study_dir: str, trial_dir: str, exper_dir: str, logger):
    """
    Acknowledge termination by removing the signal file(s) from all levels.
    """
    termination_patterns = ["_exit", "_end", "_terminate", "_kill"]
    
    for level, directory in (
        ("trial", trial_dir),
        ("study", study_dir),
        ("experiment", exper_dir),
    ):
        for f in _termination_files(directory, termination_patterns):
            try:
                os.remove(f)
                logger.info(f"✅ Removed {level}-level termination file: {f}")
            except OSError as e:
                logger.warning(f"⚠️ Failed to remove termination file {f}: {e}")
