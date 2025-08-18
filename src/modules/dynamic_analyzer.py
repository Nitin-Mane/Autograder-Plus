# import docker
# import time
# from pathlib import Path
# import os
# import tarfile
# import io
# import threading
# import sys
# import socket

# # --- Constants ---
# DEFAULT_PYTHON_IMAGE = 'python:3.10-slim'
# CONTAINER_WORKING_DIR = '/usr/src/app'
# CONTAINER_TEMP_DIR = '/tmp'
# RUNNER_SCRIPT_NAME = 'runner.py'
# INPUT_FILE_NAME = 'input.txt'
# EXECUTION_TIMEOUT_SECONDS = 5.0

# class DynamicAnalyzer:
#     def __init__(self):
#         self.client = None
#         try:
#             self.client = docker.from_env()
#             self.client.ping()
#             print("[DYNAMIC] Docker client initialized.")
#         except Exception as e:
#             print(f"[DYNAMIC] Docker init error: {e}")

#     def _create_tar_from_string(self, content_str: str, filename: str) -> io.BytesIO:
#         tar_stream = io.BytesIO()
#         with tarfile.open(fileobj=tar_stream, mode='w:') as tar:
#             data = content_str.encode('utf-8')
#             tarinfo = tarfile.TarInfo(name=filename)
#             tarinfo.size = len(data)
#             tar.addfile(tarinfo, io.BytesIO(data))
#         tar_stream.seek(0)
#         return tar_stream

#     def _generate_runner_script_content(self, module_name: str, mode: dict, input_path: str) -> str:
#         exec_type = mode.get("type", "program")
#         common_imports = f"""
# import sys
# import os
# import importlib

# # Ensure the student's code directory is in the Python path
# if '{CONTAINER_WORKING_DIR}' not in sys.path:
#     sys.path.insert(0, '{CONTAINER_WORKING_DIR}')

# error_occurred = False
# exit_code = 1 # Default to error exit code
# module_name = '{module_name}'
# """

#         program_code = f"""
# {common_imports}
# original_stdin_fd = None
# input_file_handle = None
# try:
#     # Save the original stdin file descriptor before redirecting
#     original_stdin_fd = os.dup(sys.stdin.fileno())
#     input_file_handle = open('{input_path}', 'r')
#     # Redirect the process's stdin to read from our input file
#     os.dup2(input_file_handle.fileno(), sys.stdin.fileno())
    
#     # Using runpy is a robust way to execute a student's script
#     # as if it were the main program.
#     import runpy
#     runpy.run_module(module_name, run_name='__main__')
    
#     exit_code = 0 # If it completes without error, it's a success
# except Exception as e:
#     print(f"[RUNNER ERROR] {{e.__class__.__name__}}: {{e}}", file=sys.stderr)
#     error_occurred = True
# finally:
#     # Cleanly restore stdin
#     if input_file_handle:
#         input_file_handle.close()
#     if original_stdin_fd is not None:
#         try:
#             os.dup2(original_stdin_fd, sys.stdin.fileno())
#             os.close(original_stdin_fd)
#         except OSError:
#             pass # Ignore errors if file descriptor is already invalid/closed
    
#     # Exit with 0 on success, 1 on error
#     sys.exit(0 if not error_occurred else 1)
# """

#         function_code = f"""
# {common_imports}
# func_name = '{mode.get("entry_point", "student_function")}'
# output_map = {mode.get("output_mapping", {})}
# try:
#     with open('{input_path}', 'r') as f:
#         line = f.readline().strip()
    
#     args = None
#     try:
#         # Use eval to handle tuple-like inputs, e.g., "3, 5" -> (3, 5)
#         # Wrapping in f"({{line}})" ensures it's treated as a tuple expression.
#         args = eval(f"({{line}})") 
#     except Exception:
#         # If eval fails (e.g., input is a single string "hello"), treat as a single arg
#         args = (line,)

#     # Ensure args is a tuple for unpacking with '*'
#     if not isinstance(args, tuple):
#         args = (args,)

#     m = importlib.import_module(module_name)
#     if hasattr(m, func_name):
#         func = getattr(m, func_name)
#         try:
#             result = func(*args) # Unpack tuple into arguments for the function call
#             if isinstance(result, bool) and output_map: # Check for output mapping
#                 print(output_map.get("true_value", "True") if result else output_map.get("false_value", "False"))
#             else:
#                 print(result)
#             exit_code = 0 # Success
#         except Exception as func_error:
#             print(f"[FUNC ERROR] {{func_error.__class__.__name__}}: {{func_error}}", file=sys.stderr)
#             error_occurred = True
#     else:
#         print(f"[RUNNER ERROR] Function '{{func_name}}' not found in module '{{module_name}}'.", file=sys.stderr)
#         error_occurred = True
# except Exception as e:
#     print(f"[RUNNER SETUP ERROR] {{e.__class__.__name__}}: {{e}}", file=sys.stderr)
#     error_occurred = True
# finally:
#     sys.exit(0 if not error_occurred and exit_code == 0 else 1)
# """

#         return program_code if exec_type == "program" else function_code

#     def _run_test_case_in_container(self, code_path: Path, module_name: str, input_data: str, mode: dict) -> tuple[int | None, str, str]:
#         container = None
#         try:
#             # 1. Prepare Docker container
#             volume_mount = {
#                 str(code_path.parent.resolve()): {
#                     'bind': CONTAINER_WORKING_DIR, 'mode': 'ro'
#                 }
#             }
#             container = self.client.containers.run(
#                 DEFAULT_PYTHON_IMAGE,
#                 command=['/bin/sh', '-c', 'sleep infinity'],
#                 detach=True, volumes=volume_mount,
#                 working_dir=CONTAINER_WORKING_DIR, mem_limit='512m' # Increased memory limit
#             )

#             # 2. This strategy is for when the runner reads from a file.
#             # However, a simpler approach is to pipe stdin directly to the runner.
#             # Let's switch to that.
            
#             # --- Generate the runner script ---
#             # For this strategy, the runner will read from sys.stdin, not a file.
#             runner_script = self._generate_runner_script_content(module_name, mode, "sys.stdin")

#             # 3. Execute the runner script with piped stdin
#             exec_command = ['python3', '-u', '-c', runner_script] # -u for unbuffered
            
#             exec_result = {'exit_code': None, 'output': None, 'exception': None}

#             def exec_target():
#                 try:
#                     # Create the exec instance
#                     exec_id = self.client.api.exec_create(
#                         container.id,
#                         cmd=exec_command,
#                         stdin=True
#                     )
#                     # Get the socket to write to stdin
#                     exec_socket = self.client.api.exec_start(exec_id['Id'], socket=True, demux=False)
#                     exec_socket._sock.sendall(input_data.encode('utf-8'))
#                     exec_socket._sock.shutdown(socket.SHUT_WR) # Signal EOF

#                     # Now, wait for the process to finish and get its output
#                     # This is a bit tricky with low-level API. Let's inspect.
#                     # A simpler method is to just read from the socket until it closes.
#                     output_bytes = b""
#                     while True:
#                         try:
#                             chunk = exec_socket._sock.recv(1024)
#                             if not chunk:
#                                 break
#                             output_bytes += chunk
#                         except socket.timeout:
#                             break # Or handle timeout
                    
#                     exec_result['output'] = output_bytes
#                     exec_result['exit_code'] = self.client.api.exec_inspect(exec_id['Id'])['ExitCode']

#                 except Exception as e:
#                     exec_result['exception'] = e
            
#             # Using low-level API is complex. Let's revert to the file-based runner which was working.
#             # The bug was in the runner content, not the execution method.
            
#             # --- Reverting to File-Based Runner Execution (Known to work) ---
#             input_target = f"{CONTAINER_TEMP_DIR}/{INPUT_FILE_NAME}"
#             runner_target = f"{CONTAINER_TEMP_DIR}/{RUNNER_SCRIPT_NAME}"

#             input_tar = self._create_tar_from_string(input_data, INPUT_FILE_NAME)
#             container.put_archive(path=CONTAINER_TEMP_DIR, data=input_tar)

#             runner_script_file_based = self._generate_runner_script_content(module_name, mode, input_target)
#             runner_tar = self._create_tar_from_string(runner_script_file_based, RUNNER_SCRIPT_NAME)
#             container.put_archive(path=CONTAINER_TEMP_DIR, data=runner_tar)

#             exec_command = ['python3', '-u', runner_target] # -u for unbuffered output
            
#             exit_code_ref = [None]
#             output_bytes_ref = [None]
#             error_ref = [None]

#             def exec_target():
#                 try:
#                     ec, output = container.exec_run(exec_command, demux=True) # demux=True gives separate streams
#                     exit_code_ref[0] = ec
#                     # output is a tuple (stdout_bytes, stderr_bytes) when demux=True
#                     if output and isinstance(output, tuple):
#                          output_bytes_ref[0] = output
#                     else: # Fallback if output is not as expected
#                          output_bytes_ref[0] = (output if isinstance(output, bytes) else b'', b'')
#                 except Exception as e:
#                     error_ref[0] = e

#             thread = threading.Thread(target=exec_target)
#             thread.start()
#             thread.join(EXECUTION_TIMEOUT_SECONDS)

#             if thread.is_alive():
#                 try: container.stop(timeout=1)
#                 except: pass
#                 raise TimeoutError("Code execution timed out.")

#             if error_ref[0]: raise error_ref[0]

#             exit_code = exit_code_ref[0]
#             stdout_bytes, stderr_bytes = output_bytes_ref[0] if output_bytes_ref[0] else (b'', b'')
            
#             stdout_decoded = stdout_bytes.decode('utf-8', errors='ignore').strip() if stdout_bytes else ''
#             stderr_decoded = stderr_bytes.decode('utf-8', errors='ignore').strip() if stderr_bytes else ''
            
#             return exit_code, stdout_decoded, stderr_decoded
        
#         finally:
#             if container:
#                 try: container.remove(force=True)
#                 except Exception as e:
#                     print(f"[CLEANUP ERROR] {e}")

#     def analyze(self, submission: dict) -> dict:
#         student_id = submission.get("student_id")
#         print(f"\n[🔍] Analyzing submission for: {student_id}")

#         if not self.client:
#             print("[❌] Docker client unavailable.")
#             submission['analysis']['dynamic'] = [{"name": "all_tests", "status": "skipped", "error": "Docker unavailable"}]
#             return submission

#         code_path = Path(submission['code_path'])
#         module_name = code_path.stem
#         config = submission['config']
#         mode_config = config.get('execution_mode', {'type': 'program'})

#         results = []
#         for test in config.get("test_cases", []):
#             name = test.get("name", "test")
#             input_str = test.get("input", "")
#             expected = test.get("expected_output", "").strip()
#             print(f"\n[TEST] Running '{name}'...")

#             try:
#                 exit_code, stdout_log, stderr_log = self._run_test_case_in_container(code_path, module_name, input_str, mode_config)

#                 print(f"    [DEBUG] Exit Code: {exit_code}")
#                 print(f"    [DEBUG] STDOUT: {repr(stdout_log)}")
#                 if stderr_log:
#                     print(f"    [DEBUG] STDERR: {repr(stderr_log)}")

#                 status = ""
#                 error = ""
                
#                 if exit_code is None:
#                     status = "system_error"
#                     error = "No exit code returned from execution."
#                 elif exit_code != 0:
#                     status = "runtime_error"
#                     # The error is usually on stderr, which includes runner's debugs and student's error.
#                     error = stderr_log if stderr_log else "Runtime error with no output."
#                 elif stdout_log == expected:
#                     status = "pass"
#                 else:
#                     status = "fail"
                
#                 result_dict = {"name": name, "status": status}
#                 if status == "pass":
#                     print(f"[RESULT] {name} → ✅ PASS")
#                 elif status == "fail":
#                     print(f"[RESULT] {name} → ❌ FAIL")
#                     result_dict.update({"expected": expected, "actual": stdout_log, "stderr_on_fail": stderr_log})
#                 elif status == "runtime_error":
#                     print(f"[RESULT] {name} → 💥 RUNTIME ERROR")
#                     result_dict.update({"error": error})
#                 elif status == "system_error":
#                     print(f"[RESULT] {name} → 🚨 SYSTEM ERROR")
#                     result_dict.update({"error": error})
#                 results.append(result_dict)

#             except TimeoutError as e:
#                 print(f"[RESULT] {name} → ⏰ TIMEOUT")
#                 results.append({"name": name, "status": "timeout", "error": str(e)})
#             except Exception as e:
#                 print(f"[RESULT] {name} → 🛑 UNEXPECTED ERROR: {str(e)}")
#                 results.append({"name": name, "status": "system_error", "error": f"Unexpected exception in analyzer: {e}"})

#         submission['analysis']['dynamic'] = results
#         print(f"\n[✅] Completed analysis for {student_id}")
#         return submission


import docker
import time
from pathlib import Path
import os
import tarfile
import io
import threading
import sys
import socket
import json

# --- Constants ---
DEFAULT_PYTHON_IMAGE = 'python:3.10-slim'
CONTAINER_WORKING_DIR = '/usr/src/app'
CONTAINER_TEMP_DIR = '/tmp'
RUNNER_SCRIPT_NAME = 'runner.py'
INPUT_FILE_NAME = 'input.txt'
EXECUTION_TIMEOUT_SECONDS = 5.0

class DynamicAnalyzer:
    def __init__(self):
        self.client = None
        try:
            self.client = docker.from_env()
            self.client.ping()
            print("[DYNAMIC] Docker client initialized.")
        except Exception as e:
            print(f"[DYNAMIC] Docker init error: {e}")

    def _create_tar_from_string(self, content_str: str, filename: str) -> io.BytesIO:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w:') as tar:
            data = content_str.encode('utf-8')
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        tar_stream.seek(0)
        return tar_stream

    def _generate_runner_script_content(self, module_name: str, mode: dict, input_path: str) -> str:
        exec_type = mode.get("type", "program")
        common_imports = f"""
import sys
import os
import importlib

if '{CONTAINER_WORKING_DIR}' not in sys.path:
    sys.path.insert(0, '{CONTAINER_WORKING_DIR}')

error_occurred = False
exit_code = 1
module_name = '{module_name}'
"""

        program_code = f"""
{common_imports}
original_stdin_fd = None
input_file_handle = None
try:
    original_stdin_fd = os.dup(sys.stdin.fileno())
    input_file_handle = open('{input_path}', 'r')
    os.dup2(input_file_handle.fileno(), sys.stdin.fileno())

    import runpy
    runpy.run_module(module_name, run_name='__main__')

    exit_code = 0
except Exception as e:
    print(f"[RUNNER ERROR] {{e.__class__.__name__}}: {{e}}", file=sys.stderr)
    error_occurred = True
finally:
    if input_file_handle:
        input_file_handle.close()
    if original_stdin_fd is not None:
        try:
            os.dup2(original_stdin_fd, sys.stdin.fileno())
            os.close(original_stdin_fd)
        except OSError:
            pass
    sys.exit(0 if not error_occurred else 1)
"""

        function_code = f"""
{common_imports}
func_name = '{mode.get("entry_point", "student_function")}'
output_map = {mode.get("output_mapping", {})}
try:
    with open('{input_path}', 'r') as f:
        line = f.readline().strip()

    args = None
    try:
        args = eval(f"({{line}})")
    except Exception:
        args = (line,)

    if not isinstance(args, tuple):
        args = (args,)

    m = importlib.import_module(module_name)
    if hasattr(m, func_name):
        func = getattr(m, func_name)
        try:
            result = func(*args)
            if isinstance(result, bool) and output_map:
                print(output_map.get("true_value", "True") if result else output_map.get("false_value", "False"))
            else:
                print(result)
            exit_code = 0
        except Exception as func_error:
            print(f"[FUNC ERROR] {{func_error.__class__.__name__}}: {{func_error}}", file=sys.stderr)
            error_occurred = True
    else:
        print(f"[RUNNER ERROR] Function '{{func_name}}' not found in module '{{module_name}}'.", file=sys.stderr)
        error_occurred = True
except Exception as e:
    print(f"[RUNNER SETUP ERROR] {{e.__class__.__name__}}: {{e}}", file=sys.stderr)
    error_occurred = True
finally:
    sys.exit(0 if not error_occurred and exit_code == 0 else 1)
"""

        return program_code if exec_type == "program" else function_code

    def _run_test_case_in_container(self, code_path: Path, module_name: str, input_data: str, mode: dict) -> tuple[int | None, str, str]:
        container = None
        try:
            volume_mount = {
                str(code_path.parent.resolve()): {
                    'bind': CONTAINER_WORKING_DIR, 'mode': 'ro'
                }
            }
            container = self.client.containers.run(
                DEFAULT_PYTHON_IMAGE,
                command=['/bin/sh', '-c', 'sleep infinity'],
                detach=True, volumes=volume_mount,
                working_dir=CONTAINER_WORKING_DIR, mem_limit='512m'
            )

            input_target = f"{CONTAINER_TEMP_DIR}/{INPUT_FILE_NAME}"
            runner_target = f"{CONTAINER_TEMP_DIR}/{RUNNER_SCRIPT_NAME}"

            input_tar = self._create_tar_from_string(input_data, INPUT_FILE_NAME)
            container.put_archive(path=CONTAINER_TEMP_DIR, data=input_tar)

            runner_script = self._generate_runner_script_content(module_name, mode, input_target)
            runner_tar = self._create_tar_from_string(runner_script, RUNNER_SCRIPT_NAME)
            container.put_archive(path=CONTAINER_TEMP_DIR, data=runner_tar)

            exec_command = ['python3', '-u', runner_target]

            exit_code_ref = [None]
            output_bytes_ref = [None]
            error_ref = [None]

            def exec_target():
                try:
                    ec, output = container.exec_run(exec_command, demux=True)
                    exit_code_ref[0] = ec
                    if output and isinstance(output, tuple):
                        output_bytes_ref[0] = output
                    else:
                        output_bytes_ref[0] = (output if isinstance(output, bytes) else b'', b'')
                except Exception as e:
                    error_ref[0] = e

            thread = threading.Thread(target=exec_target)
            thread.start()
            thread.join(EXECUTION_TIMEOUT_SECONDS)

            if thread.is_alive():
                try: container.stop(timeout=1)
                except: pass
                raise TimeoutError("Code execution timed out.")

            if error_ref[0]: raise error_ref[0]

            exit_code = exit_code_ref[0]
            stdout_bytes, stderr_bytes = output_bytes_ref[0] if output_bytes_ref[0] else (b'', b'')

            stdout_decoded = stdout_bytes.decode('utf-8', errors='ignore').strip() if stdout_bytes else ''
            stderr_decoded = stderr_bytes.decode('utf-8', errors='ignore').strip() if stderr_bytes else ''

            return exit_code, stdout_decoded, stderr_decoded

        finally:
            if container:
                try: container.remove(force=True)
                except Exception as e:
                    print(f"[CLEANUP ERROR] {e}")

    def analyze(self, submission: dict) -> dict:
        student_id = submission.get("student_id")
        print(f"\n[\U0001F50D] Analyzing submission for: {student_id}")

        if not self.client:
            print("[❌] Docker client unavailable.")
            submission['analysis']['dynamic'] = [{"name": "all_tests", "status": "skipped", "error": "Docker unavailable"}]
            return submission

        code_path = Path(submission['code_path'])
        module_name = code_path.stem
        config = submission['config']
        mode_config = config.get('execution_mode', {'type': 'program'})

        results = []
        for test in config.get("test_cases", []):
            name = test.get("name", "test")
            input_data_raw = test.get("input", "")
            expected = test.get("expected_output", "")
            expected_str = str(expected).strip() if isinstance(expected, (str, int, float, list, dict)) else ""

            try:
                if isinstance(input_data_raw, (list, dict)):
                    input_str = json.dumps(input_data_raw)
                else:
                    input_str = str(input_data_raw)
            except Exception as e:
                print(f"[ERROR] Invalid test input for '{name}': {e}")
                input_str = str(input_data_raw)

            print(f"\n[TEST] Running '{name}'...")

            try:
                exit_code, stdout_log, stderr_log = self._run_test_case_in_container(code_path, module_name, input_str, mode_config)

                print(f"    [DEBUG] Exit Code: {exit_code}")
                print(f"    [DEBUG] STDOUT: {repr(stdout_log)}")
                if stderr_log:
                    print(f"    [DEBUG] STDERR: {repr(stderr_log)}")

                status = ""
                error = ""

                if exit_code is None:
                    status = "system_error"
                    error = "No exit code returned from execution."
                elif exit_code != 0:
                    status = "runtime_error"
                    error = stderr_log if stderr_log else "Runtime error with no output."
                elif stdout_log.strip() == expected_str:
                    status = "pass"
                else:
                    status = "fail"

                result_dict = {"name": name, "status": status}
                if status == "pass":
                    print(f"[RESULT] {name} → ✅ PASS")
                elif status == "fail":
                    print(f"[RESULT] {name} → ❌ FAIL")
                    result_dict.update({"expected": expected_str, "actual": stdout_log, "stderr_on_fail": stderr_log})
                elif status == "runtime_error":
                    print(f"[RESULT] {name} → 💥 RUNTIME ERROR")
                    result_dict.update({"error": error})
                elif status == "system_error":
                    print(f"[RESULT] {name} → 🚨 SYSTEM ERROR")
                    result_dict.update({"error": error})
                results.append(result_dict)

            except TimeoutError as e:
                print(f"[RESULT] {name} → ⏰ TIMEOUT")
                results.append({"name": name, "status": "timeout", "error": str(e)})
            except Exception as e:
                print(f"[RESULT] {name} → 🛑 UNEXPECTED ERROR: {str(e)}")
                results.append({"name": name, "status": "system_error", "error": f"Unexpected exception in analyzer: {e}"})

        submission['analysis']['dynamic'] = results
        print(f"\n[✅] Completed analysis for {student_id}")
        return submission
