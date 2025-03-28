import sys
import subprocess
import numpy as np

EPSILON = 1e-7
CONFIG_FILES = [f"./data/func{i}.cfg" for i in range(1, 5)]
PATH = "build/integrate_parallel"

def run_integrate(config_file, num_runs, num_threads, path_to_exe = PATH):
    results = []
    times = []

    func_number = int(config_file[-5])

    for _ in range(num_runs):
        process = subprocess.run([path_to_exe, str(func_number), config_file, str(num_threads)]
                                 if path_to_exe == PATH else
                                 [path_to_exe, str(func_number), config_file],
                                 capture_output=True, text=True)
        output = process.stdout.strip().split("\n")

        try:
            integral = float(output[0])
            abs_error = float(output[1])
            rel_error = float(output[2])
            exec_time = float(output[3])

            results.append(integral)
            times.append(exec_time)
        except (ValueError, IndexError):
            print(f"Error for output {config_file}: {output}")
            return None

    return results, abs_error, rel_error, times

def analyze_results(results, abs_error, rel_error, times, return_res = False):
    if not all(abs(results[0] - r) < EPSILON for r in results):
        print("The results do not coincide within the accuracy limits")
        return 16

    min_time = min(times)
    avg_time = np.mean(times)
    std_dev = np.std(times, ddof=1) if len(times) > 1 else 0

    if return_res:
        return [min_time, avg_time, std_dev]

    print(results[0])
    print(abs_error)
    print(rel_error)
    print(f"{min_time:.7f}")
    print(f"{avg_time:.7f}")
    print(f"{std_dev:.7f}\n")

def main():
    if len(sys.argv) != 3:
        print("Error, must be: python script.py <num_runs> <num_threads>")
        return 1

    try:
        num_runs = int(sys.argv[1])
        num_threads = int(sys.argv[2])
        if num_runs < 1 or num_threads < 1:
            raise ValueError
    except ValueError:
        print("Error, both arguments must be positive integers")
        return 2

    for config_file in CONFIG_FILES:
        result = run_integrate(config_file, num_runs, num_threads)
        if result:
            analyze_results(*result)

if __name__ == "__main__":
    main()
