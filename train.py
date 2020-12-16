#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from src.evaluator import Evaluator
from src.util import *
import time

os.system('mkdir ./model')
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

def train(agent, env, evaluate):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    while step <= train_times:
        step += 1
        episode_steps += 1
        print(f"-------Step {step}------")
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation)    
        action = agent.select_action(observation)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    print('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    agent.save_model(step, output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > args.warmup:
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
            print('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}'.format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--k', default=3, type=int, help='number of strokes to generate at each step')
    parser.add_argument('--discount', default=0.95**3, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=10, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    parser.add_argument('--dataset_size', default='regular', type=str, help='the dataset size to use for training')
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    
    args = parser.parse_args()    
    args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from src.DDPG import DDPG
    from src.fastenv import FastEnv
    fenv = FastEnv(args.max_step, args.env_batch, args.dataset_size, args.k)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, args.k, \
                 None, args.resume)
    evaluate = Evaluator(args)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate)
