import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
# 定义ping延迟和带宽数组
# 单位 秒 s

ping_latency = np.array([
                        [0	,0.034866,0.017796	,0.076512,	0.030985],
                        [0.03257,	0,	0.027257	,0.079517,	0.018823],
                        [0.108798	,0.065485	,0,	0.064679	,0.043849],
                        [0.206233	,0.028255	,0.034938	,0,	0.02909],
                        [0.021318	,0.034842	,0.019245,	0.042098,	0]])

# MB ps


bandwidths = np.array([
                        [float("inf")	,1.666069031,	1.81388855	,2.56729126	,4.906654358],
                        [0.312805176	,float("inf")	,1.200675964	,0.575065613,	0.590324402],
                        [0.994682312,	2.725601196	,float("inf")	,0.332832336,	0.495910645],
                        [0.772476196,	1.02519989	,1.168251038,	float("inf")	,1.210212708],
                        [2.743721008,	1.764297485	,1.543045044,	3.170013428,	float("inf")]])
# 定义通信数据大小
data_size_kb = 20  # 20 KB = 160 千比特

# 定义加载时间和推理时间的函数
def load_time(M):
    return 5.57 * M ** 0.44

def inference_time(M):
    return 0.15 * M ** 0.73

# 定义总模型大小和设备数量
M_total = 1 # 设备3上的完整模型大小
n = 2  # n个设备
# 计算设备之间的通信时间
def communication_time(i, j, data_size_kb=20):
    data_size_kilobits = data_size_kb/ 1024 # 转换为MB
    return (ping_latency[i, j] + (data_size_kilobits / bandwidths[i, j]))
# Simulated annealing function


def simulated_annealing(M_total, n, iterations=5000, initial_temp=100, cooling_rate=0.99):
    # Initialize model allocation randomly
    Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

    current_temp = initial_temp
    best_Mi = Mi
    best_time, stages = calc_total_time(best_Mi)

    # Annealing process
    for iteration in range(iterations):
        # Generate new allocation, sorted for fairness
        new_Mi = np.sort(np.random.dirichlet(np.ones(n), size=1)[0] * M_total)

        # Calculate total time
        new_time, new_stages = calc_total_time(new_Mi)

        # Accept the new allocation if it's better or by chance
        if new_time < best_time or random.uniform(0, 1) < np.exp((best_time - new_time) / current_temp):
            best_Mi = new_Mi
            best_time = new_time
            stages = new_stages

        # Cool down
        current_temp *= cooling_rate

    return best_Mi, best_time, stages


# Calculating the total time for all devices
def calc_total_time(Mi):
    total_times = []
    stages = []

    # First device timing (starts at t=0)
    load_start = 0
    load_end = load_time(Mi[0])
    infer_start = load_end
    infer_end = infer_start + inference_time(Mi[0])
    comm_start = infer_end
    comm_end = comm_start + communication_time(0, 1)  # First device communicates with the next

    total_times.append(comm_end)

    # Record stage timing
    stages.append({
        'load_start': load_start, 'load_end': load_end,
        'infer_start': infer_start, 'infer_end': infer_end,
        'comm_start': comm_start, 'comm_end': comm_end
    })

    # Compute timing for all other devices
    for i in range(1, n):
        load_start = 0
        load_end = load_time(Mi[i])
        infer_start = max(stages[i - 1]['comm_end'], load_end) # 上一轮通信结束或者本轮load结束之后才可以开启推理

        infer_end = infer_start + inference_time(Mi[i])
        comm_start = infer_end
        comm_end = comm_start + communication_time(i, (i + 1) % n)  # Circular communication between devices

        total_times.append(comm_end)

        # Record stage timing
        stages.append({
            'load_start': load_start, 'load_end': load_end,
            'infer_start': infer_start, 'infer_end': infer_end,
            'comm_start': comm_start, 'comm_end': comm_end
        })

    # Return the maximum total time, representing the total recovery time
    return max(total_times), stages


def print_stage_info(stages):
    # 创建数据表格
    table = []
    headers = ['设备', '加载开始 (s)', '加载结束 (s)', '推理开始 (s)', '推理结束 (s)', '通信开始 (s)', '通信结束 (s)']

    # 填充数据
    for i, stage in enumerate(stages):
        row = [
            i,
            stage['load_start'],
            stage['load_end'],
            stage['infer_start'],
            stage['infer_end'],
            stage['comm_start'],
            stage['comm_end']
        ]
        table.append(row)

    # 使用 tabulate 格式化输出表格
    print(tabulate(table, headers=headers, floatfmt=".2f", tablefmt="grid"))


def plot_pipeline(stages,last_time,recovery_time_single_device):
    fig, ax = plt.subplots(figsize=(10, n+1))

    for i, stage in enumerate(stages):
        # 加载阶段
        ax.broken_barh([(stage['load_start'], stage['load_end'] - stage['load_start'])], (i - 0.4, 0.8),
                       facecolors='tab:blue', label="load" if i == 0 else "")
        # 推理阶段
        ax.broken_barh([(stage['infer_start'], stage['infer_end'] - stage['infer_start'])], (i - 0.4, 0.8),
                       facecolors='tab:green', label="infer" if i == 0 else "")
        # 通信阶段
        ax.broken_barh([(stage['comm_start'], stage['comm_end'] - stage['comm_start'])], (i - 0.4, 0.8),
                       facecolors='tab:red', label="commu" if i == 0 else "")
    # 加载阶段
    ax.broken_barh([(stage['comm_end'],last_time)], (i - 0.4, 0.8),
                   facecolors='tab:blue', label="load remain" if i == 0 else "")
    # 单设备恢复
    ax.broken_barh([(0, recovery_time_single_device)], (i +1 - 0.4, 0.8),
                   facecolors='tab:grey', label="single device" if i == 0 else "")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('device')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'device {i}' for i in range(n)])
    ax.grid(True)
    ax.legend()
    plt.show()


# 运行优化并记录时间
best_Mi, best_time, stages = simulated_annealing(M_total, n)
print(f"最佳子模型分配: {best_Mi}")
last_M = M_total - best_Mi[-1]
last_time = load_time(last_M)
recovery_time_single_device = load_time(M_total)
print(f"最小无感恢复时间: {best_time:.2f} 秒")
print(f"最小无感恢复时间: {best_time+last_time:.2f} 秒")
print(f"单设备恢复时间: {recovery_time_single_device:.2f} 秒")

# 打印每个设备的时间段信息
print_stage_info(stages)

# 画出流水线图
plot_pipeline(stages,last_time,recovery_time_single_device)
