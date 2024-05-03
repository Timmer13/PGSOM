import gym
import numpy as np
import matplotlib.pyplot as plt


def logistic(y):
    # Definition of logistic function
    return 1 / (1 + np.exp(-y))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def probabilities(x, y):
    # Returns probabilities of two actions
    y = np.dot(x, y)
    
    if y.shape == ():
        return [logistic(y), 1-logistic(y)]
    else:
        return softmax(y)


def act(x, y):
    # Sample an action in proportion to probabilities
    probs_array = probabilities(x, y)
    action = np.random.choice(len(probs_array), p=probs_array)
    return action, probs_array[action]



def gradient_log_probability(x, z):
    # Calculate gradient-log-probabilities
    y = np.dot(x, z)
    if y.shape == ():
        grad_log_p0 = x - x * logistic(y)
        grad_log_p1 = - x * logistic(y)
        return grad_log_p0, grad_log_p1
    else:
        probs = softmax(y)
        grad_log_probs = np.zeros((z.shape[1],z.shape[1],z.shape[0]))
        for i in range(len(probs)):
            for j in range(len(probs)):
                if i == j:
                    grad_log_probs[i,j] = x - x * probs[i]
                else:
                    grad_log_probs[i,j] = - x * probs[i]
        return grad_log_probs
        


def discount_rewards(rewards, discount_factor):
    # Calculate temporally adjusted, discounted rewards
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for i in reversed(range(0, len(rewards))):
        cumulative_rewards = cumulative_rewards * discount_factor + rewards[i]
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

def hes_log_probability(x, z):
    # Calculate gradient-log-probabilities
    y = np.dot(x, z)
    grad_log_p0 = x * x * logistic(y)* logistic(y)/np.exp(y)
    grad_log_p1 = x * x *  logistic(y)* logistic(y)/np.exp(y)
    return grad_log_p0, grad_log_p1

def hessian_vector_product(grad_log_p, actions, observations, parameters, damping=1e-3):

    hes = np.array([hes_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])
    grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])
    

    return hes + grad_log_p


def update_policy(parameters, learning_rate, grad_log_p, actions, discounted_rewards):
    # Gradient ascent on parameters
    dot = np.dot(grad_log_p.T, discounted_rewards)
    parameters += learning_rate * dot
    return parameters

def update_policy_c(parameters, learning_rate, grad_log_p, actions, discounted_rewards, clip_value=50):
    # Gradient ascent on parameters
    dot = np.dot(grad_log_p.T, discounted_rewards)
    parameters += learning_rate * np.clip(dot, -clip_value, clip_value)
    return parameters

def update_policy_e(parameters, learning_rate, grad_log_p, p, actions, discounted_rewards, entropy_coeff=0.01):
    # Gradient ascent on parameters
    dot = np.dot(grad_log_p.T, discounted_rewards)
    parameters += learning_rate * dot
    
    # Entropy regularization
    entropy = -np.sum(p * np.log(p + 1e-10))
    parameters += learning_rate * entropy_coeff * entropy
    
    return parameters

def update_policy_hessian(parameters, observations, learning_rate, grad_log_p, actions, discounted_rewards):
    # Hessian Aided Policy Gradient update
    hvp = hessian_vector_product(grad_log_p, actions, observations, parameters)+ grad_log_p
    hvp_dot = np.dot(hvp.T, discounted_rewards)
    parameters += learning_rate * hvp_dot
    return parameters

def update_policy_hessian_c(parameters, observations, learning_rate, grad_log_p, actions, discounted_rewards, clip_value=50):
    # Hessian Aided Policy Gradient update
    hvp = hessian_vector_product(grad_log_p, actions, observations, parameters)+ grad_log_p
    hvp_dot = np.dot(hvp.T, discounted_rewards)
    parameters += learning_rate * np.clip(hvp_dot, -clip_value, clip_value)
    return parameters

def update_policy_hessian_e(parameters, observations, learning_rate, grad_log_p, p, actions, discounted_rewards, entropy_coeff=0.01):
    # Hessian Aided Policy Gradient update
    hvp = hessian_vector_product(grad_log_p, actions, observations, parameters) + grad_log_p
    hvp_dot = np.dot(hvp.T, discounted_rewards)
    parameters += learning_rate * hvp_dot
    
    # Compute entropy regularization
    entropy = -np.sum(p * np.log(p + 1e-10))
    parameters += learning_rate * entropy_coeff * entropy

    return parameters


def evaluate_policy(policy, env, discount):
    quality = 0
    
    for _ in range(500):
        observation, info = env.reset(seed=1)
        terminated=False
        truncated=False
        i = 0
        while not (terminated or truncated):
            action,_ = act(observation, policy)
            observation, reward, terminated, truncated, info = env.step(action)
            i += 1
            quality += discount ** i * reward

            
    return quality / 500


def run_episode(environment, parameters, learning_rate, discount_factor, render=False):
    observation,_ = environment.reset()
    total_reward = 0

    observations = []
    actions = []
    rewards = []
    probabilities = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        observations.append(observation)

        action, prob = act(observation, parameters)
        observation, reward, terminated, truncated, info = environment.step(action)

        total_reward += reward
        rewards.append(reward)
        actions.append(action)
        probabilities.append(prob)

    return total_reward, np.array(rewards), np.array(observations), np.array(actions), np.array(probabilities)


def train(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Train until MAX_EPISODES
    for i in range(max_episodes):
        # Run a single episode
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        # Keep track of episode rewards
        episode_rewards.append(total_reward)
        
        
        
        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))

        # Calculate gradients for each action over all observations
        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        # Calculate temporally adjusted, discounted rewards
        discounted_rewards = discount_rewards(rewards, discount_factor)

        # Update policy
        
        if clip:
            parameters = update_policy_c(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
        elif entropy:
            parameters = update_policy_e(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
        else: 
            parameters = update_policy(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
    return episode_rewards, evaluation

def train_with_baseline(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Collect rewards from initial episodes to compute baseline
    initial_rewards = []
    for _ in range(100):
        obs,_ = env.reset()
        terminated = False
        truncated = False
    
        total_reward = 0
        while not (terminated or truncated):
            action,_ = act(obs, parameters)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = observation  # Update observation for the next step
        initial_rewards.append(total_reward)

    # Compute baseline as the average reward from initial episodes
    baseline = np.mean(initial_rewards)

    for i in range(max_episodes):
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        episode_rewards.append(total_reward)

        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))
            
        p = np.array([probabilities(obs, parameters)[action] for obs, action in zip(observations, actions)])
        
        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        discounted_rewards = discount_rewards(rewards, discount_factor)
        
        # Calculate advantages and subtract baseline
        advantages = discounted_rewards - baseline

        # Update policy
        if clip:
            parameters = update_policy_c(parameters, learning_rate, grad_log_p, actions, advantages)
        elif entropy:
            parameters = update_policy_e(parameters, learning_rate, grad_log_p, p, actions, advantages)
        else: 
            parameters = update_policy(parameters, learning_rate, grad_log_p, actions, advantages)
    return episode_rewards, evaluation

def train_hessian(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Train until MAX_EPISODES
    for i in range(max_episodes):
        # Run a single episode
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        # Keep track of episode rewards
        episode_rewards.append(total_reward)
        
        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))

        # Calculate gradients for each action over all observations
        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        # Calculate temporally adjusted, discounted rewards
        discounted_rewards = discount_rewards(rewards, discount_factor)

        # Update policy using HAPG
        if clip:
            parameters = update_policy_c(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
        elif entropy:
            parameters = update_policy_c(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
        else: 
            parameters = update_policy(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
    return episode_rewards, evaluation

def train_hessian_baseline(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Collect rewards from initial episodes to compute baseline
    initial_rewards = []
    for _ in range(100):
        obs,_ = env.reset()
        terminated = False
        truncated = False
    
        total_reward = 0
        while not (terminated or truncated):
            action,_ = act(obs, parameters)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = observation  # Update observation for the next step
        initial_rewards.append(total_reward)

    # Compute baseline as the average reward from initial episodes
    baseline = np.mean(initial_rewards)

    for i in range(max_episodes):
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        episode_rewards.append(total_reward)

        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))
            

        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        p = np.array([probabilities(obs, parameters)[action] for obs, action in zip(observations, actions)])
 
        discounted_rewards = discount_rewards(rewards, discount_factor)
        
        # Calculate advantages and subtract baseline
        advantages = discounted_rewards - baseline

        # Update policy
        if clip:
            parameters = update_policy_hessian_c(parameters, observations, learning_rate, grad_log_p, actions, advantages)
        elif entropy:
            parameters = update_policy_hessian_e(parameters, observations, learning_rate, grad_log_p, p, actions, advantages)
        else: 
            parameters = update_policy_hessian(parameters, observations, learning_rate, grad_log_p, actions, advantages)

    return episode_rewards, evaluation


def train_rk(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Train until MAX_EPISODES
    for i in range(max_episodes):
        # Run a single episode
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        # Keep track of episode rewards
        episode_rewards.append(total_reward)
        
        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))

        # Calculate gradients for each action over all observations
        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        # Calculate temporally adjusted, discounted rewards
        discounted_rewards = discount_rewards(rewards, discount_factor)

        # Update policy using HAPG
        
        K1 = np.dot(grad_log_p.T, discounted_rewards)
        parameters_hat = parameters + learning_rate*K1
        
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters_hat, learning_rate, discount_factor)
        
        discounted_rewards = discount_rewards(rewards, discount_factor)
        
        grad_log_p2 = np.array([gradient_log_probability(obs, parameters_hat)[action] for obs, action in zip(observations, actions)])
        
        K2 = np.dot(grad_log_p2.T, discounted_rewards)
        if clip:
            parameters += learning_rate*np.clip((K1+K2)/2,-0.5,0.5)
            
        elif entropy:
            entropy = -np.mean(np.sum(parameters * np.log(parameters + 1e-10), axis=1))
            entropy_grad = -np.mean(np.log(parameters + 1e-10) + 1)
        
            # Compute K3 with entropy regularization
            K3 = np.dot(entropy_grad.T, discounted_rewards)
        
            # Update parameters using RK method with entropy regularization
            parameters += learning_rate * ((K1 + K2) / 2 + 0.01 * K3)
        
        else:
            # Update parameters using the original RK method
            parameters += learning_rate * ((K1 + K2) / 2)
            
    
    return episode_rewards, evaluation


def train_rk_baseline(env, learning_rate, discount_factor, max_episodes=1000, clip=False, entropy=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a))
    episode_rewards = []
    evaluation = []

    # Collect rewards from initial episodes to compute baseline
    initial_rewards = []
    for _ in range(100):
        obs,_ = env.reset()
        terminated = False
        truncated = False
    
        total_reward = 0
        while not (terminated or truncated):
            action,_ = act(obs, parameters)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = observation  # Update observation for the next step
        initial_rewards.append(total_reward)

    # Compute baseline as the average reward from initial episodes
    baseline = np.mean(initial_rewards)

    for i in range(max_episodes):
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        episode_rewards.append(total_reward)

        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))

        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])
        
        p = np.array([probabilities(obs, parameters)[action] for obs, action in zip(observations, actions)])

        discounted_rewards = discount_rewards(rewards, discount_factor) - baseline
        
        
        K1 = np.dot(grad_log_p.T, discounted_rewards)
        parameters_hat = parameters + learning_rate*K1
        
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters_hat, learning_rate, discount_factor)
        
        discounted_rewards = discount_rewards(rewards, discount_factor) - baseline
        
        grad_log_p2 = np.array([gradient_log_probability(obs, parameters_hat)[action] for obs, action in zip(observations, actions)])
        
        K2 = np.dot(grad_log_p2.T, discounted_rewards)
        # Compute entropy and its gradient if entropy regularization is enabled
        if clip:
            parameters += learning_rate*np.clip((K1+K2)/2,-50,50)
            
        elif entropy:
        
            # Update parameters using RK method with entropy regularization
            parameters += learning_rate * ((K1 + K2) / 2 - 0.01 * np.sum(p * np.log(p + 1e-10)))
        
        else:
            # Update parameters using the original RK method
            parameters += learning_rate * ((K1 + K2) / 2)


    return episode_rewards, evaluation


def train_softmax(env, learning_rate, discount_factor, max_episodes=1000, evaluate=False):
    _a,_b = env.reset(seed=1) 
    parameters = np.random.rand(len(_a),env.action_space.n)
    episode_rewards = []
    evaluation = []

    # Train until MAX_EPISODES
    for i in range(max_episodes):
        # Run a single episode
        total_reward, rewards, observations, actions, _ = run_episode(env, parameters, learning_rate, discount_factor)

        # Keep track of episode rewards
        episode_rewards.append(total_reward)
        
        
        
        if i % (max_episodes//20) == 0:      
            value = evaluate_policy(parameters, env, 1.0 - 1E-3 )
            evaluation.append(value)
            print("Episode: " + str(i) + ' ' + str(value))

        # Calculate gradients for each action over all observations
        grad_log_p = np.array([gradient_log_probability(obs, parameters)[action] for obs, action in zip(observations, actions)])

        # Calculate temporally adjusted, discounted rewards
        discounted_rewards = discount_rewards(rewards, discount_factor)

        # Update policy
        parameters = update_policy(parameters, learning_rate, grad_log_p, actions, discounted_rewards)
        

    return episode_rewards, evaluation
