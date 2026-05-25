"""
isolrun_child.py

Isolated running of a child process (for running a model training instance)
"""
import sys, os, runpy, pickle, pathlib
try:
    import cloudpickle as cpickle
except ImportError:
    cpickle = pickle

def main():
    # argv: [self, target_script, input_pickle, output_pickle]
    _, target_path, in_p, out_p = sys.argv

    with open(in_p, "rb") as f:
        payload = cpickle.load(f)

    fetch  = payload["fetch"]
    args   = payload["args"]
    run_id = payload["run_id"]

    # Simulate the target script's CLI
    sys.argv = [os.path.basename(target_path), *args]

    # Build isolated global namespace
    g = {
        "__name__": "__main__",
        "__file__": target_path,
        "__package__": None,
        "__cached__": None,
        "__builtins__": __builtins__,
    }

    # Execute target script
    g_after = runpy.run_path(target_path, init_globals=g, run_name="__main__")

    # Collect requested outputs
    outputs = {}
    for k in fetch:
        if k in g_after:
            outputs[k] = g_after[k]

    with open(out_p, "wb") as f:
        cpickle.dump({"outputs": outputs, "meta": {"run_id": run_id, "fetched": fetch}}, f)

if __name__ == "__main__":
    main()
