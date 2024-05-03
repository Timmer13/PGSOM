import matplotlib.pyplot as plt


import numpy as np
from utils import *
import seaborn as sns

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


env = gym.make("CartPole-v1")
# env = gym.make("LunarLander-v2")
# env = gym.make("Walker2d")

# Initial parameters
learning_rate = 0.002
discount_factor = 0.99
N = 500

eval_total = []
eval_baseline_total = []
eval_hessian_total = []
eval_hessian_baseline_total = []
eval_rk_total = []
eval_rk_baseline_total = []

eval_total_clip = []
eval_hessian_clip = []
eval_rk_clip = []

eval_total_entropy = []
eval_hessian_entropy = []
eval_rk_entropy = []

repeats = 5  # Number of repeats for each exploration strategy

for _ in range(repeats):
    np.random.seed(_)
    _, evaluations = train(env, learning_rate, discount_factor, N)
    _, evaluations_baseline = train_with_baseline(env, learning_rate, discount_factor, N)
    _, evaluations_hessian = train_hessian(env, learning_rate, discount_factor, N)
    _, evaluations_hessian_baseline = train_hessian_baseline(env, learning_rate, discount_factor, N)
    _, evaluations_rk = train_rk(env, learning_rate, discount_factor, N)
    _, evaluations_rk_baseline = train_rk_baseline(env, learning_rate, discount_factor, N)
    
    
    _, evaluations_baseline_clip = train_with_baseline(env, 2*learning_rate, discount_factor, N, clip=True)
    _, evaluations_baseline_entropy = train_with_baseline(env, learning_rate, discount_factor, N, entropy=True)  
    _, evaluations_hessian_clip= train_hessian_baseline(env, 2*learning_rate, discount_factor, N, clip=True)
    _, evaluations_hessian_entropy = train_hessian_baseline(env, learning_rate, discount_factor, N, entropy=True)
    _, evaluations_rk_clip = train_rk_baseline(env, 2*learning_rate, discount_factor, N, clip=True)
    _, evaluations_rk_entropy = train_rk_baseline(env, learning_rate, discount_factor, N, entropy=True)
    
    eval_total.append(evaluations)
    eval_baseline_total.append(evaluations_baseline)
    eval_hessian_total.append(evaluations_hessian)
    eval_hessian_baseline_total.append(evaluations_hessian_baseline)
    eval_rk_total.append(evaluations_rk)
    eval_rk_baseline_total.append(evaluations_rk_baseline)
    
    
    eval_total_clip.append(evaluations_baseline_clip)
    eval_total_entropy.append(evaluations_baseline_entropy)
    eval_hessian_clip.append(evaluations_hessian_clip)
    eval_hessian_entropy.append(evaluations_hessian_entropy)
    eval_rk_clip.append(evaluations_rk_clip)
    eval_rk_entropy.append(evaluations_rk_entropy)

env.close()


np.save('eval_total.npy', eval_total)
np.save('eval_baseline_total.npy', eval_baseline_total)
np.save('eval_hessian_total.npy', eval_hessian_total)
np.save('eval_hessian_baseline_total.npy', eval_hessian_baseline_total)
np.save('eval_rk_total.npy', eval_rk_total)
np.save('eval_rk_baseline_total.npy', eval_rk_baseline_total)

# Save evaluation results with clipping
np.save('eval_total_clip.npy', np.array(eval_total_clip))
np.save('eval_hessian_clip.npy', np.array(eval_hessian_clip))
np.save('eval_rk_clip.npy', np.array(eval_rk_clip))

# Save evaluation results with entropy
np.save('eval_total_entropy.npy', np.array(eval_total_entropy))
np.save('eval_hessian_entropy.npy', np.array(eval_hessian_entropy))
np.save('eval_rk_entropy.npy', np.array(eval_rk_entropy))