import os
import argparse
import warnings
import torch 
import distribute_rl_Pendulum as drl

from torch.distributions.categorical import Categorical
import numpy as np
import random


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore", category=DeprecationWarning)
#反对警告
warnings.simplefilter("ignore", category=ResourceWarning)
#资源警告


torch.manual_seed(42)
torch.cuda.manual_seed(42)
print(torch.cuda.is_available())

def main(args):

    if args.mode == "worker":
        worker_manager = drl.get_worker_manager(args)
        worker_manager.start_train()
    if args.mode == "learner":
        learner = drl.get_learner(args)
        learner.start()
    if args.mode == "selftest":
        pass
        


if __name__ == "__main__":
    # 创建path路径对象，并由此创建解析对象，添加命令行参数和选项
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="learner or worker", type=str, default="test")
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parser.add_argument('--seed', type=int, default=42, help='the random seeds')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers to collect samples')
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='the environment name')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size of updating')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate of the algorithm')
    parser.add_argument('--epoch', type=int, default=10, help='the epoch during training')
    parser.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parser.add_argument('--vloss_coef', type=float, default=0.5, help='the coefficient of value loss')
    parser.add_argument('--ent_coef', type=float, default=0, help='the entropy loss coefficient')
    parser.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parser.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parser.add_argument('--total_frames', type=int, default=int(2e6), help='the total frames for training')
    parser.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parser.add_argument('--clip', type=float, default=0.2, help='the ratio clip param')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='the folder to save models')
    parser.add_argument('--lr_decay', action='store_true', help='if using the learning rate decay during decay')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='grad norm')
    parser.add_argument('--display_interval', type=int, default=10, help='the interval that display log information')
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--redis_pass_word', type=str, default='Lab504redis')
    parser.add_argument('--redis_ip', type=str, default='172.18.232.25')
    parser.add_argument('--redis_port', type=int, default=6379)
    parser.add_argument("--update_interval", type=int, default=512)
    parser.add_argument("--log_interval", type=int, default = 100)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--max_episode", type = int, default=10000)
    args = parser.parse_args()
    print(args.mode)
    main(args)
