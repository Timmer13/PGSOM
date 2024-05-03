import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define file names
file_names = [
    'eval_total.npy', 'eval_baseline_total.npy', 'eval_total_clip.npy', 'eval_total_entropy.npy',
    'eval_hessian_total.npy', 'eval_hessian_baseline_total.npy', 'eval_hessian_clip.npy', 'eval_hessian_entropy.npy',
    'eval_rk_total.npy', 'eval_rk_baseline_total.npy', 'eval_rk_clip.npy', 'eval_rk_entropy.npy'
]

# Load all the .npy files
eval_data = {}
for file_name in file_names:
    eval_data[file_name.split('.')[0]] = np.load(file_name)

# Calculate mean and standard deviation for each scenario
mean_data = {}
std_data = {}
for key, data in eval_data.items():
    mean_data[key] = np.mean(data, axis=0)
    std_data[key] = np.std(data, axis=0)

# Plotting
plt.figure(figsize=(10, 6))
for key, mean in mean_data.items():
    episodes = np.arange(0, len(mean), len(mean) // 20)
    line = sns.lineplot(x=episodes, y=mean, label=key, err_style=None)
    fill = plt.fill_between(episodes, mean - std_data[key], mean + std_data[key], alpha=0.2)
    
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Mean Evaluation Returns')
plt.title('Mean Evaluation Returns with Confidence Intervals')
plt.show()

for file_name, mean in mean_data.items():
    last_mean = mean[-1]
    print(f"{file_name} mean: {last_mean}")
    
print('\n')
for file_name, std in std_data.items():
    last_std = std[-1]
    print(f"{file_name} std: {last_std}")