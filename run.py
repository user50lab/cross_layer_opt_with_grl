import os
import os.path as osp
import json
import yaml
from typing import Any, Optional
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace as SN
from functools import partial

import random
import numpy as np
import torch as th
import wandb

from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from envs import REGISTRY as env_REGISTRY
from envs.multi_agent_env import MultiAgentEnv

from policies import REGISTRY as policy_REGISTRY
from components.buffers import REGISTRY as buff_REGISTRY
from learners import REGISTRY as learn_REGISTRY, DETERMINISTIC_POLICY_GRADIENT_ALGOS
from runners import REGISTRY as run_REGISTRY

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(__file__)), 'results')


### 递归函数，用于合并两个字典。它将在字典d的基础上逐步更新或添加字典u中的键值对。
def recursive_dict_update(d, u):
    """Merges two dictionaries."""

    for k, v in u.items():
        if isinstance(v, Mapping):   # 检查v是否是另一个字典
            d[k] = recursive_dict_update(d.get(k, {}), v)   # d.get(k, {})尝试从字典d中获取键为k的值，如果没有找到就返回一个空字典。然后，这个值与v进行递归合并。
        else:
            d[k] = v
    return d


def config_copy(config):
    """Copies configuration."""

    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


### 对配置进行检查和调整，最后返回更新后的配置对象。
def check_args_sanity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Checks the feasibility of configuration."""

    # Setup correct device.
    if config['use_cuda'] and th.cuda.is_available():
        config['device'] = 'cuda:{}'.format(config['cuda_idx'])
    else:
        config['use_cuda'] = False
        config['device'] = 'cpu'
    print(f"Choose to use {config['device']}.")

    # Env specific requirements
    if config['env_id'] == 'mpe':
        assert config['obs'] == 'flat', "MPE only supports flat obs."
        if config['shared_obs'] is not None:
            assert config['shared_obs'] == 'flat', "MPE only supports flat shared obs."
    if config['state'] is not None:
        assert config['state'] == 'flat', f"Unsupported state format 's{config['state']}' is encountered."

    return config

### 根据传入的环境对象env更新一个叫作args的对象的属性。
def update_args_from_env(env: MultiAgentEnv, args):
    """Updates args from env."""

    env_info = env.get_env_info()
    args.n_agents = env_info['n_agents']

    if args.env_id.startswith('ad-hoc'):
        args.max_nbrs = getattr(env, 'max_nbrs', None)
        args.n_pow_opts = getattr(env, 'power_options', env.n_pow_lvs)
        args.khops = getattr(env, 'khops', 1)

    if args.runner == 'base':
        if args.rollout_len is None:
            args.rollout_len = env_info['episode_limit']
            print(f"`rollout_len` is set to `episode_limit` as {args.rollout_len}.")
        if args.data_chunk_len is None:
            args.data_chunk_len = env_info['episode_limit']
            print(f"`data_chunk_len` is set to `episode_limit` as {args.data_chunk_len}.")
        assert args.rollout_len is not None and args.data_chunk_len is not None, "Invalid rollout/data chunk length"
    elif args.runner == 'episode':
        args.data_chunk_len = env_info['episode_limit']
        print(f"`data_chunk_len` is set to `episode_limit` as {env_info['episode_limit']}.")
    else:
        raise KeyError("Unrecognized name of runner")

    # Assume that all agents share the same action space and retrieve action info.
    act_space = env.action_space[0]
    args.act_size = env_info['n_actions'][0]
    # Note that `act_size` specifies output layer of modules,
    # while `act_shape` indicates the shape of actions stored in buffers (which may be different from `act_size`).
    if isinstance(act_space, Discrete):
        args.is_discrete = True
        args.is_multi_discrete = False
        args.act_shape = 1 if args.learner not in DETERMINISTIC_POLICY_GRADIENT_ALGOS else args.act_size
    elif isinstance(act_space, MultiDiscrete):
        args.is_discrete = True  # Multi-discrete space is generalization of discrete space.
        args.is_multi_discrete = True
        args.nvec = act_space.nvec.tolist()  # Number of actions in each space
        args.act_shape = len(args.nvec) if args.learner not in DETERMINISTIC_POLICY_GRADIENT_ALGOS else args.act_size
    else:  # TODO: Continuous action use Box space.
        args.is_discrete = False
    # Discrete action selectors use available action mask.
    if args.is_discrete:
        args.pre_decision_fields.append('avail_actions')
    return args


def run(env_id: str, env_kwargs: Mapping[str, Any], seed: int = 0, algo_name: str = 'q',
        train_kwargs: Mapping[str, Any] = dict(), run_tag: Optional[str] = None,
        add_suffix: bool = False, suffix: Optional[str] = None) -> None:
    """Main function to run the training loop"""

    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Load the default configuration.
    with open(os.path.join(os.path.dirname(__file__), 'config', "default.yaml"), "r") as f:
        config = yaml.safe_load(f)
    # Load hyper-params of algo.
    with open(os.path.join(os.path.dirname(__file__), 'config', f"algos/{algo_name}.yaml"), "r") as f:
        algo_config = yaml.safe_load(f)
    config = recursive_dict_update(config, algo_config)
    # Load mac parameters of communicative agents.
    if config['agent'] == 'comm':
        assert config['comm'] is not None, "Absence of communication protocol for communicative agents!"
        with open(os.path.join(os.path.dirname(__file__), 'config', f"comm/{config['comm']}.yaml"), "r") as f:
            comm_config = yaml.safe_load(f)
        config = recursive_dict_update(config, comm_config)
    # Load preference from train_kwargs.
    config = recursive_dict_update(config, train_kwargs)
    # Add env id.
    config['env_id'] = env_id
    # Make sure the legitimacy of configuration.
    config = check_args_sanity(config)
    del algo_config, train_kwargs  # Delete redundant variables.
    '''
    例子：
    config = {'key1': 'value1', 'key2': 'value2'}
    args = SimpleNamespace(**config)
    现在args会拥有key1和key2作为它的属性，可以通过args.key1和args.key2来访问这些值。
    '''
    args = SN(**config)  # Convert to simple namespace.

    # Get directory to store models/results. 获取存储模型或结果文件的目录
    # Project identifier includes `env_id` and probably `scenario`.
    scenario = env_kwargs.get('scenario_name', None)  # 从env_kwargs字典中提取键为'scenario_name'的值。如果env_kwargs字典中没有'scenario_name'这个键，那么scenario变量将被赋值为None。
    if add_suffix:
        if suffix is not None:
            project_name = env_id + '_' + suffix
        elif scenario is not None:
            project_name = env_id + '_' + scenario
        else:
            raise Exception("Suffix of project is unavailable.")
    else:
        project_name = env_id
    # Multiple runs are distinguished by algo name and tag.
    run_name = run_tag if run_tag is not None else algo_name
    # Create a subdirectory to distinguish runs with different random seeds.
    args.run_dir = osp.join(DEFAULT_DATA_DIR, project_name, run_name + f"_seed{seed}")
    if not osp.exists(args.run_dir):
        os.makedirs(args.run_dir)
    print(f"Run '{run_name}' under directory '{args.run_dir}'.")

    if args.use_wandb:  # If W&B is used,
        # Runs with the same config except for rand seeds are grouped and their histories are plotted together.
        wandb.init(config=args, project=project_name, group=run_name, name=run_name + f"_seed{seed}", dir=args.run_dir,
                   reinit=True)
        args.wandb_run_dir = wandb.run.dir

    # Define env function.
    # functools.partial是Python的一个高阶函数，其作用是：基于一个函数创建一个新的可调用对象，把原函数的某些参数固定住（也就是设置默认值），返回一个新的函数。
    # env_REGISTRY是一个字典，保存了不同环境(env)的构造函数或者函数。env_id是这个字典中某个环境（此处是ad-hoc）的键（key）。
    # env_kwargs是一个包含了关键字参数的字典，**env_kwargs是一个关键字参数展开的语法，意味着它将env_kwargs这个字典中的所有项作为关键字参数传递给函数。
    env_fn = partial(env_REGISTRY[env_id], **env_kwargs)  # Env function   # 使用指定的env_id从env_REGISTRY这个字典中获取一个函数，并利用partial函数将env_kwargs字典中的参数作为默认值，创建一个新的函数对象并赋值给env_fn。
    test_env_fn = partial(env_REGISTRY[env_id], **env_kwargs)  # Test env function

    # Create runner holding instance(s) of env and get info.
    # runner是基于run_REGISTRY字典中的一个元素所创建的实例或者函数的结果。
    runner = run_REGISTRY[args.runner](env_fn, test_env_fn, args)
    args = update_args_from_env(runner.env, args)  # Adapt args to env.

    # Setup key components.
    env_info = runner.get_env_info()
    # 在default.yaml文件中定义了policy=='shared'
    # policy_REGISTRY是一个字典，通过shared索引关联到SharedPolicy 类
    # policy = policy_REGISTRY['shared'](env_info, args)：实例化 SharedPolicy 类，传入 env_info 和 args 参数。
    policy = policy_REGISTRY[args.policy](env_info, args)  # Policy making decisions
    policy.to(args.device)  # Move policy to device.

    buffer = buff_REGISTRY[args.buffer](args)  # Buffer holding experiences
    learner = learn_REGISTRY[args.learner](env_info, policy, args)  # Algorithm training policy
    runner.add_components(policy, buffer, learner)  # Add above components to runner.

    # Run the main loop of training.
    runner.run()
    # Clean-up after training.
    runner.cleanup()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--algo', type=str, default='q')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    # print(type(args))
    # a = dict(name='bob')
    # args = SN(**a)
    # args2 = argparse.Namespace(**a)
    # print(args)
    # print(vars(args))
    # print(type(args))
    # print(args2)
    # print(vars((args2)))
    # raise ValueError

    # Train UBS coverage.
    # env_id = 'ubs'
    # env_kwargs = dict(scenario_name='simple')

    # # Train Ad Hoc route.
    env_id = 'ad-hoc'
    env_kwargs = dict()   # 创建空字典

    # train_kwargs字典将这些参数存储为键值对，这将使得在函数或方法调用时更方便地传入训练参数。
    # 例如你可能会在代码中看到这样的调用：train_model(**train_kwargs)。在这种情况下，字典中存储的所有键值对都将被作为参数传递给train_model函数。
    train_kwargs = dict(use_cuda=True, cuda_idx=0, use_wandb=False, record_tests=True, rollout_len=10, data_chunk_len=5)
    run(env_id, env_kwargs, args.seed, algo_name=args.algo, train_kwargs=train_kwargs, run_tag=args.tag)
