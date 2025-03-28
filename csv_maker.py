import numpy as np
import matplotlib.pyplot as plt

from runner import run_integrate, analyze_results, CONFIG_FILES

NUM_RUNS = 10
THREADS = [1, 2, 3, 4, 8, 16, 18, 100, 254]

TABLES =  False
TABLE_NAME = './data/table.csv'

CONFIG_FILES = CONFIG_FILES[:-1]
avg_times = [[] for _ in CONFIG_FILES]
avg_old_times = []

# print(avg_old_times)
to_write = ',min_time,avg_time,std_dev\n'

for num_threads in THREADS:

    print(f'THREADS: {num_threads}')

    for idx, config_file in enumerate(CONFIG_FILES):

        result = run_integrate(config_file, NUM_RUNS, num_threads)
        # print(result)

        if result:
            min_time, avg_time, std_dev = analyze_results(*result, return_res=True)
            avg_times[idx].append(avg_time)
            # print(avg_time)

            to_write += f'{config_file[7:-4]}_th{num_threads},{min_time},{avg_time},{std_dev}' + ("\n" if (num_threads != THREADS[-1] or idx != len(CONFIG_FILES)-1) else '')
        else:
            raise "Mistake in config files"

with open(TABLE_NAME, 'w', encoding='UTF-8') as file:
    file.write(to_write)

