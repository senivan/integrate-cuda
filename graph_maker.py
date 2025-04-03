import csv
import numpy as np
import matplotlib.pyplot as plt

dct_cuda = {}
dct_res = {}

with open('table.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for idx, row in enumerate(spamreader):
        if idx == 0:
            continue

        # print(row)
        dct_cuda[row[0][:5]] = [float(row[1]), float(row[2]), float(row[3])]

# print(dct_cuda)
dct_res['cuda'] = dct_cuda

with open('other_methods.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    len_ = sum(1 for row in spamreader)

with open('other_methods.csv') as csvfile:
    spamreader = csv.reader(csvfile)

    dct_ = {}
    method_name = None

    for idx, row in enumerate(spamreader):
        if idx == 0:
            continue

        if (method_name and method_name != row[0]):
            dct_res[method_name] = dct_.copy()
            dct_.clear()
            # print('here')

        # print(row)

        method_name = row[0]
        dct_[row[1][:5]] = [float(row[2]), float(row[3]), float(row[4])]

        if (idx == len_-1):
            dct_res[method_name] = dct_.copy()

del dct_res['serial']

for key, val in dct_res.items():
    print(key)
    print(val)

# dct_func = {}
#
# for method_name, val in dct_res.items():
#
#     for func_name, times in val.items():
#
#         print(func_name, method_name, times)
#
#         if func_name not in dct_func:
#             dct_func[func_name] = {}
#         dct_func[func_name][method_name] = times
#
# for key, val in dct_func.items():
#     print(key)
#     print(val)

x_axis = np.arange(3)

# print(avarage_times, minimum_times, deviations)

for idx, [method_name, val] in enumerate(dct_res.items()):

    min_times = [time[0] for time in val.values()]
    times = [time[1] for time in val.values()]
    sds = [time[2] for time in val.values()]

    # plt.bar(x_axis-0.375 + idx * 0.15, times, 0.15, yerr=sds, capsize = 4)
    # plt.bar(x_axis-0.375 + idx * 0.15, min_times, 0.15, label=method_name)
    plt.bar(x_axis-0.3 + idx * 0.2, times, 0.2, yerr=sds, capsize = 4)
    plt.bar(x_axis-0.3 + idx * 0.2, min_times, 0.2, label=method_name)

    # break

plt.xticks(x_axis, ["func1", "func2", "func3"])
plt.ylabel("Time, in ms")
plt.legend()

plt.show()