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

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

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
    env_evaluate = gym.make(env_name, render_mode="human")  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env_evaluate.reset(seed=seed)
    np.random.seed(seed)
    if seed:
        torch.manual_seed(seed)

    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    model_dir = args.run_dir + "model"
    log_dir = args.run_dir + "log"

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

    agent = PPO_continuous(args)

    # load model
    def load_checkpoint(checkpoint):
        if isinstance(checkpoint, int):
            checkpoint_path = '%s/iter_%04d.p' % (model_dir, checkpoint)
        else:
            assert isinstance(checkpoint, str)
            checkpoint_path = '%s/%s.p' % (model_dir, checkpoint)

        model_checkpoint = pickle.load(open(checkpoint_path, "rb"))
        print('Loading model from checkpoint: %s' % checkpoint_path)

        agent.actor.load_state_dict(model_checkpoint['actor_dict'])
        agent.critic.load_state_dict(model_checkpoint['critic_dict'])
        state_norm = model_checkpoint['state_norm']
        return state_norm

    state_norm = load_checkpoint(args.ckpt)
    evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
    print("average_evaluate_reward:{} \t".format(evaluate_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", default='best')

    args = parser.parse_args()
    run_dir = args.run_dir
    ckpt = int(args.ckpt) if args.ckpt.isdigit() else args.ckpt

    argspath = run_dir + "setting.txt"
    with open(argspath, 'r') as f:
        args.__dict__ = json.load(f)
    
    args.run_dir = run_dir
    args.ckpt = ckpt
    env_name = args.env_name
    main(args, env_name=env_name, number=1, seed=42)
