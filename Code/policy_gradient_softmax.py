import numpy as np
import gym

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def linear_policy_network(weights, state):
    logits = np.dot(weights, state)
    return logits

def policy_gradient_softmax(env, weights, num_episodes=1000, learning_rate=0.01, gamma=0.99):
    # Initialize variables
    episode_rewards = []
    gradients = []
    
    for episode in range(num_episodes):
        # Reset environment and initialize variables
        state,_ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_gradients = []
        
        while not (terminated or truncated):
            # Choose action based on policy network
            logits = linear_policy_network(weights, state)
            action_probs = softmax(logits)
            if np.any(np.isnan(action_probs)):
                # If probabilities contain NaN, choose a random action
                action = np.random.choice(len(action_probs))
            else:
                # Otherwise, sample action based on probabilities
                action = np.random.choice(len(action_probs), p=action_probs)
            
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Compute gradient
            dsoftmax = np.eye(len(action_probs))[action] - action_probs
            dlog = np.outer(state, dsoftmax)
            episode_gradients.append(dlog)
            
            # Update episode reward and state
            episode_reward += reward
            state = next_state
        
        # Update policy parameters after each episode
        episode_rewards.append(episode_reward)
        episode_gradients = np.array(episode_gradients)
        discounted_rewards = np.cumsum(episode_rewards[::-1])[::-1] * (gamma ** np.arange(len(episode_rewards)))
        normalized_discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
        gradients = np.sum(episode_gradients * normalized_discounted_rewards[:, np.newaxis, np.newaxis], axis=0)
        weights += learning_rate * gradients.T
        
    return episode_rewards

# Example usage
# Define environment
env = gym.make("LunarLander-v2")

# Initialize policy parameters
weights = np.random.rand(4, 8)

# Call the policy gradient function
episode_rewards = policy_gradient_softmax(env, weights)
