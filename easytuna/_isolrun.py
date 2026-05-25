import os, sys, uuid, tempfile, subprocess, pathlib, pickle, threading, time
from queue import Queue, Empty

_CHILD = pathlib.Path(__file__).with_name("_isolrun_child.py")


def _stream_subprocess_output(proc, log_path, flush_interval=2.0, termination_flag=None, logger=None):
    q = Queue()

    def enqueue_output(stream, label):
        for line in iter(stream.readline, ''):
            q.put((label, line))
        stream.close()

    threading.Thread(target=enqueue_output, args=(proc.stdout, 'STDOUT'), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(proc.stderr, 'STDERR'), daemon=True).start()

    buffer = []
    last_flush = time.time()
    all_stdout = []
    all_stderr = []

    with open(log_path, 'a', buffering=1) as f:
        while proc.poll() is None or not q.empty():
            # Check for termination flag
            if termination_flag and termination_flag.get('stop'):
                if logger:
                    logger.info("🛑 Termination flag detected in isolrun. Killing subprocess.")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()
                break
                
            try:
                label, line = q.get(timeout=0.1)
                prefix = "[OUT] " if label == "STDOUT" else "[ERR] "
                f.write(prefix + line)
                buffer.append(prefix + line)

                if label == "STDOUT":
                    all_stdout.append(line)
                else:
                    all_stderr.append(line)

            except Empty:
                pass

            if time.time() - last_flush >= flush_interval and buffer:
                f.flush()
                buffer.clear()
                last_flush = time.time()

        if buffer:
            f.flush()

    return ''.join(all_stdout), ''.join(all_stderr)


def run_isolated(
    script_path,
    args_list=None,       # simulated CLI args for target script
    log_dir=None,         # where to write terminal.out
    timeout=None,
    termination_flag=None,  # flag to check for early termination
    logger=None,
    fetch=None,           # list of global names to retrieve  
    python_exe=sys.executable,
    env=None,
    cwd=None,
    import_root=None,
):
    fetch = list(fetch) if fetch else []
    args_list = list(args_list) if args_list else []
    ser = pickle

    run_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=f"isolrun_{run_id}_") as td:
        td = pathlib.Path(td)
        in_p = td / "input.pkl"
        out_p = td / "outputs.pkl"

        with in_p.open("wb") as f:
            ser.dump({"fetch": fetch, "args": args_list, "run_id": run_id}, f)

        cmd = [python_exe, str(_CHILD), str(script_path), str(in_p), str(out_p)]
        # Inherit the parent environment, then apply explicit overrides.
        child_env = os.environ.copy()
        if env:
            child_env.update(env)

        # Set up PYTHONPATH: The import_root (snapshot dir) must come first to ensure
        # snapshotted code is used. The original cwd is included as a fallback and
        # to ensure other project modules can be found if needed.
        project_root = cwd or os.getcwd()
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [import_root, project_root, existing_pythonpath]))

        proc = subprocess.Popen(
            cmd,
            cwd=project_root,                  # run inside your project directory
            env=child_env,                     # with PYTHONPATH pointing here
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Set up logging path if log_dir is provided
        log_file_path = None
        if log_dir:
            log_file_path = os.path.join(log_dir, "terminal.out")

        if log_file_path:
            stdout, stderr = _stream_subprocess_output(proc, log_file_path, termination_flag=termination_flag, logger=logger)
        else:
            # Handle termination checking even without logging
            if termination_flag:
                start_time = time.time()
                while proc.poll() is None:
                    if termination_flag.get('stop'):
                        if logger:
                            logger.info("🛑 Termination flag detected in isolrun. Killing subprocess.")
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except:
                            proc.kill()
                        break
                    
                    # Check timeout
                    if timeout and (time.time() - start_time) > timeout:
                        proc.kill()
                        stdout, stderr = proc.communicate()
                        raise RuntimeError(f"Process timed out.\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}")
                    
                    time.sleep(0.1)
                
                stdout, stderr = proc.communicate()
            else:
                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    raise RuntimeError(f"Process timed out.\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}")

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"Isolated run failed (rc={rc})\n"
                f"--- STDOUT ---\n{stdout}\n"
                f"--- STDERR ---\n{stderr}"
            )

        if not out_p.exists():
            raise RuntimeError("Child produced no output pickle.")

        with out_p.open("rb") as f:
            payload = ser.load(f)

    return {
        "outputs": payload.get("outputs", {}),
        "run_id": payload.get("meta", {}).get("run_id", run_id),
        "stdout": stdout,
        "stderr": stderr,
        "meta": payload.get("meta", {}),
    }
