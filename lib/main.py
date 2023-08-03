import torch
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
import pickle
import random

from utils import to_device

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

def mkdir(path):
    import os
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        terminated = False
        truncated = False
        episode_reward = 0
        while not (terminated or truncated):
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, terminated, truncated, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = gym.make(env_name)
    # Set random seed
    env.reset(seed=seed)
    np.random.seed(seed)
    if seed:
        torch.manual_seed(seed)

    args.env_name = env_name
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    from datetime import datetime
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = 'runs/{}_{}/{}'.format(env_name, args.policy_dist, time_str)
    model_dir = run_dir + "/model"
    mkdir(model_dir)
    log_dir = run_dir + "/log"

    # Build a tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    # Save args
    argspath = run_dir + '/setting.txt'
    with open(argspath, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.use_gpu == 1 and torch.cuda.is_available():
        print("choose to use gpu...")
        args.device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)

        if args.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        args.device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    best_reward = - 1000
    save_best_flag = False

    while total_steps < args.max_train_steps:
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, terminated, truncated, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            # if terminated and episode_steps != args.max_episode_steps:
            #     dw = True
            # else:
            #     dw = False
            
            done = True if terminated or truncated else False
            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, terminated, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)

                # save model
                def save(checkpoint_path):
                    to_device(torch.device('cpu'), agent.actor, agent.critic)
                    model_checkpoint = \
                        {
                            'actor_dict': agent.actor.state_dict(),
                            'critic_dict': agent.critic.state_dict(),
                            'state_norm': state_norm,
                            'best_reward': best_reward,
                            'evaluate_num': evaluate_num
                        }
                    pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))
                    to_device(args.device, agent.actor, agent.critic)
                
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./runs/{}_{}/{}/number_{}_seed_{}.npy'.format(env_name, args.policy_dist, time_str, number, seed), np.array(evaluate_rewards))
                    save('%s/iter_%04d.p' % (model_dir, evaluate_num))

                if evaluate_reward > best_reward:
                    save_best_flag = True
                    best_reward = evaluate_reward
                
                if save_best_flag:
                    print("Saving the interval checkpoint with rewards: {:.2f}".format(best_reward))
                    save('%s/best.p' % model_dir)
                    save_best_flag = False

def str2bool(input_str):
    """Converts a string to a boolean value.

    Args:
        input_str (str): The string to be converted.

    Returns:
        bool: The boolean representation of the input string.
    """
    true_values = ["true", "yes", "1", "on", "y", "t"]
    false_values = ["false", "no", "0", "off", "n", "f"]

    lowercase_str = input_str.lower()
    if lowercase_str in true_values:
        return True
    elif lowercase_str in false_values:
        return False
    else:
        raise ValueError("Invalid input string. Could not convert to boolean.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=str2bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=str2bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=str2bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=str2bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=str2bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=str2bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=str2bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=str2bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=str2bool, default=True, help="Trick 10: tanh activation function")

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--env_index", type=int, required=True)

    args = parser.parse_args()

    env_name = ['Ant-v4', 
                'Hopper-v4', 
                'Walker2d-v4',
                'HalfCheetah-v4', 
                'Swimmer-v4']
    
    # 0: "Ant-v4"
    # 1: "Hopper-v4"
    # 2: "Walker2d-v4"
    # 3: "HalfCheetah-v4"
    # 4: "Swimmer-v4"

    seed = random.randint(0, 20)
    main(args, env_name=env_name[args.env_index], number=1, seed=seed)
