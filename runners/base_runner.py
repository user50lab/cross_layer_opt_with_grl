import os
import os.path as osp
from typing import Any
import json

import numpy as np
from numpy import ndarray
import torch as th
from torch import Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb

from components.make_env import make_env
from components.misc import *
from learners import DETERMINISTIC_POLICY_GRADIENT_ALGOS
import time


class BaseRunner:
    """Base class of runners"""

    def __init__(self, env_fn, test_env_fn, args):

        self.env = make_env(env_fn, args)  # Env used for training
        self.test_env = make_env(test_env_fn, args)  # Separate copy of env used for evaluation

        self.args = args
        self.device = args.device
        self.run_dir = args.run_dir

        if not args.use_wandb:  # When W&B is not used,
            # Create a directory to save intermediate results as a Tensorboard event.
            self.log_dir = osp.join(self.run_dir, 'log')
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # Manually save config locally.
            # Note that wandb automatically stores config.
            config_path = osp.join(self.log_dir, 'run_config.json')
            with open(config_path, 'w') as f:
                f.write(json.dumps(vars(self.args), separators=(',', ':\t'), indent=4, sort_keys=True))

            # Setup SummaryWriter of Tensorboard.
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Create a directory to save model parameters.
        self.model_dir = osp.join(self.run_dir, 'model')
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Some empty attributes to be added later
        self.policy = None  # Policy to make decision from observations
        self.buffer = None  # Replay buffer holding experiences
        self.learner = None  # Algorithm to learn optimal policy

        self.start_time = None  # Record wall-clock time.
        self.t_warmup = None
        self.t_env = None  # Counter of env steps

    def add_components(self, policy, buffer, learner):
        """Adds key components of RL."""
        self.policy = policy
        self.buffer = buffer
        self.learner = learner

    def get_env_info(self):
        return self.env.get_env_info()

    def run(self):
        """Main function running the training loop."""

        # Prepare for training.
        self.start_time = time.time()  # Record the start time of training.
        rnn_states = self.warmup()
        self.test_agent()

        # Run the main loop of training.
        # print(f"self.args.update_after = {self.args.update_after}, self.args.batch_size = {self.args.batch_size}.")
        while self.t_env < self.args.total_env_steps:
            # Collect rollout with exploration.
            rnn_states = self.collect(rnn_states)
            # print(f"Finish collect at t = {self.t_env}")
            # print(f"(self.t_env >= self.args.update_after) = {self.t_env >= self.args.update_after} and "
            #       f"self.buffer.can_sample(self.args.batch_size) = {self.buffer.can_sample(self.args.batch_size)}")
            # print(f"len(self.buffer) = {len(self.buffer)}")
            # Perform update when feasible.
            if (self.t_env >= self.args.update_after) and self.buffer.can_sample(self.args.batch_size):
                diagnostic = self.learner.update(self.buffer, self.args.batch_size)
                self.log_metrics(diagnostic)

    def warmup(self) -> dict[str, Tensor]:
        """Prepares for training."""
        # Reset variables. 重置时间步计数器self.t_env
        self.t_env = 0  # Reset timestep counter.

        # Reset components. 重置环境self.env和学习算法self.learner。这通常涉及到将环境中的智能体位置回到初始状态，并且清除学习器中的累积数据，为新训练周期做准备。
        self.env.reset()  # Reset env.
        self.learner.reset()  # Reset learner.
        rnn_states = self.get_init_hidden()  # Initial RNN states 获取初始的递归神经网络（RNN）状态。

        # Let agents select random actions to fill replay buffer.
        self.policy.eval()  # Set policy to eval mode.
        self.learner.eval()  # Set modules held by learner to eval mode.
        self.t_warmup = 0  # Reset warm-up step counter.
        # 这是一个循环，将会持续到预热步数达到self.args.warmup_steps设定的值或者经验回放缓冲区有足够数据进行一次完整的批处理抽样。
        while (self.t_warmup < self.args.warmup_steps) or not self.buffer.can_sample(self.args.batch_size):
            rnn_states = self.interact(rnn_states, mode='rand')  # Random interaction 通过调用interact方法使用随机动作与环境互动，并逐步更新rnn_states。
            self.t_warmup += 1  # One frozen step is completed.

        return rnn_states  # Leave RNN states.

    ### 在固定数量的时间步长内探索环境以收集经验。
    def collect(self, rnn_states: dict[str, Tensor]) -> dict[str, Tensor]:
        """Explores a fixed number of timesteps to collect experiences."""
        self.policy.eval()  # Set policy to eval mode.
        self.learner.eval()  # Set modules held by learner to eval mode.

        for t in range(self.args.rollout_len):
            rnn_states = self.interact(rnn_states, mode='explore')
            self.t_env += 1  # Another env step is finished.

            # Regularly save checkpoints.
            if (self.t_env % self.args.save_interval == 0) or (self.t_env >= self.args.total_env_steps):
                save_path = osp.join(self.model_dir, f'checkpoint_t{self.t_env}.pt')
                self.learner.save_checkpoint(save_path)

            # Test the performance of trained models.
            if self.t_env % self.args.test_interval == 0:
                self.test_agent()

            # Handle the end of session.
            if self.t_env % self.args.steps_per_session == 0:
                session = self.t_env // self.args.steps_per_session  # Index of session
                print(f"Finish session {session} at step {self.t_env}/{self.args.total_env_steps} "
                      f"after {(time.time() - self.start_time):.1f}s.")
                # Call lr scheduler(s) if enabled.
                if self.args.anneal_lr:
                    self.learner.schedule_lr()
        return rnn_states  # Leave RNN states for next rollout.

    def interact(self, rnn_states: dict[str, Tensor], mode: str = 'explore'):
        """Completes an agent-env interaction loop of MDP."""

        # Build pre-decision data. 字典结合了从环境中获取的输入（通过self.get_inputs_from_env(self.env)方法）以及当前的rnn_states。
        pre_decision_data = dict(**self.get_inputs_from_env(self.env), **rnn_states)   # dict(**x, **y)是Python中的字典解包语法，它创建了一个新字典，包含了两个字典x和y中的所有键值对。如果两个字典中有相同的键，那么来自y的键值对将覆盖x中的键值对。

        # Select actions following epsilon-greedy strategy.
        # print(f"self.env.nbrs = {[nbr.nid for nbr in self.env.nbrs]}, "
        #       f"self.env.agent.is_connected = {self.env.agent.is_connected}")
        # print(f"avail_acts = \n{pre_decision_data['avail_actions']}")
        # 调用policy对象的act方法，从提供的观察（observation）、RNN状态和可用动作中选择动作。这个方法返回选定的动作和更新后的RNN隐藏状态。
        actions, h = self.policy.act(pre_decision_data['obs'], pre_decision_data['h'],
                                     pre_decision_data['avail_actions'], self.t_env, mode=mode)

        # Call environment step. 执行环境的step方法来将动作应用到环境中，并获得相应的奖励、是否终止了交互循环、以及其他信息。
        rewards, terminated, info = self.env.step(self.get_actions_to_env(actions, self.env))

        # When deterministic policy gradient is used, apply one-hot encoding to discrete actions.
        if (self.args.learner in DETERMINISTIC_POLICY_GRADIENT_ALGOS) and self.args.is_discrete:
            n_classes = self.args.nvec if self.args.is_multi_discrete else self.args.act_size
            actions = onehot_from_actions(actions, n_classes)

        # Call learner step to get required data (e.g., critic RNN states for off-policy AC algos).
        # 调用learner的step方法以获取所需的数据（例如，对于离策略算法，这可以是批评者的RNN状态），然后更新rnn_states字典。
        rnn_states = dict(h=h, **self.learner.step(pre_decision_data, actions))  # Next RNN states

        # Collect post-decision data.
        post_decision_data = {
            'actions': actions, 'rewards': rewards,
            'terminated': terminated != info.get('truncated', False),
        }
        # Save transition to replay buffer. 将经过处理的动作、奖励和终止信息，以及前决策和后决策数据一起存入回放缓冲区。
        self.cache(**pre_decision_data, **post_decision_data, filled=True)

        # Reach the end of an episode.
        if terminated:
            # # Log episode info.
            # if 'truncated' in info:  # Drop indicator of episode limit.
            #     del info['truncated']
            # self.log_metrics(info)

            # Append a pseudo-transition for correct bootstrapping.
            # This transition does not actually occur and only pre-decision data is used.
            last_inputs = self.get_inputs_from_env(self.env)
            # Let agents select last actions.
            actions, _ = self.policy.act(last_inputs['obs'], rnn_states['h'], last_inputs['avail_actions'],
                                         self.t_env + 1, mode=mode)
            if (self.args.learner in DETERMINISTIC_POLICY_GRADIENT_ALGOS) and self.args.is_discrete:
                n_classes = self.args.nvec if self.args.is_multi_discrete else self.args.act_size
                actions = onehot_from_actions(actions, n_classes)
            # Take empty rewards.
            empty_rewards = np.zeros(self.policy.n_agents, dtype=np.float32)
            # Forge the spurious transition and store to replay buffer. 创建了一个虚假的转移（pseudo-transition），包括了上文获取的最后动作、零奖励、终止信号以及是否填充的标志（伪造的转移在实际中并没有发生，因此填充标志为 False），同时还合并了之前获得的最后输入和RNN状态。
            pseudo_transition = dict(actions=actions, rewards=empty_rewards, terminated=True, filled=False,
                                     **last_inputs, **rnn_states)
            self.cache(**pseudo_transition)   # 将伪造的转移存储到缓存中，这通常用于后期的训练过程中，如利用回放缓冲区（replay buffer）进行经验回放。

            # Reset env and RNN states.
            self.env.reset()
            rnn_states = self.get_init_hidden()

        return rnn_states

    def get_init_hidden(self) -> dict[str, Tensor]:
        """Gets initial RNN states of policy and other modules."""
        h_policy = self.policy.init_hidden()  # RNN states of policy
        h_others = self.learner.init_hidden()  # Dict holding RNN states of other components
        rnn_states = dict(h=h_policy.to(self.device), **{k: v.to(self.device) for k, v in h_others.items()})
        return rnn_states

    def get_inputs_from_env(self, env, train_mode: bool = True) -> dict[str, Any]:
        """Gets inputs from env as part of pre-decision data."""
        # obs and avail_actions are required by all algorithms.
        obs = env.get_obs()
        obs = obs.to(self.device)
        avail_actions = env.get_avail_actions()
        if avail_actions is not None:
            avail_actions = th.tensor(avail_actions, dtype=th.float32, device=self.device)
        inputs_from_env = dict(obs=obs, avail_actions=avail_actions)

        # Note that following items can only be used in training rather than execution.
        if train_mode:
            # Shared observations and states are obtained only if they are specified in fields.
            if 'shared_obs' in self.args.pre_decision_fields:
                shared_obs = env.get_shared_obs()
                shared_obs = shared_obs.to(self.device)
                inputs_from_env['shared_obs'] = shared_obs
            if 'state' in self.args.pre_decision_fields:
                state = env.get_state().to(self.device)
                inputs_from_env['state'] = state
        return inputs_from_env

    ### 将策略产生的动作张量(actions)转换为环境(env)可以理解的动作列表格式。
    def get_actions_to_env(self, actions: Tensor, env) -> list:
        """Transforms actions from policy to a list."""
        acts = actions.cpu().numpy()  # Convert to ndarray.
        acts_per_agent = np.split(acts, env.n_agents, axis=0)  # Each entry correspond to action of an agent.   # 将 acts 数组分割成多个子数组的，每个子数组对应一个智能体（agent）的动作。
        if self.args.is_discrete:  # When discrete action is adopted,
            if self.args.is_multi_discrete:
                acts_per_agent = [act.squeeze().tolist() for act in acts_per_agent]  # Convert ndarray to list.
            else:
                acts_per_agent = [act.item() for act in acts_per_agent]  # Convert ndarray to scalar.
        return acts_per_agent

    def cache(self, obs, actions, rewards, terminated, avail_actions, filled=True, **kwargs):
        """Stores a transition to replay buffer."""
        # Reshape entries and hold them with a dict.
        transition = dict(obs=obs, avail_actions=avail_actions,
                          actions=actions.view(self.env.n_agents, self.args.act_shape),
                          rewards=th.tensor(rewards, dtype=th.float, device=self.device).reshape(1, self.env.n_agents),
                          terminated=th.tensor(terminated, dtype=th.int, device=self.device).reshape(1, 1),
                          filled=th.tensor(filled, dtype=th.int, device=self.device).reshape(1, 1))
        transition.update(**kwargs)
        # Insert transition to replay buffer.
        self.buffer.insert(transition)

    ### 测试训练好的智能体的性能。
    def test_agent(self):
        """Tests the performance of trained agent."""
        test_ep_rsts = {}   # 初始化一个空字典 test_ep_rsts，这个字典后续用于存储测试过程中的结果数据。
        self.policy.eval()  # Set policy to eval mode. 将智能体的策略模型设置为评估（eval）模式，这在深度学习框架中通常关闭训练时特有的行为，如dropout等。
        for j in range(self.args.n_test_episodes):   # n_test_episodes：要测试的回合数。
            self.test_env.reset()  # Reset test env.  重置测试环境 self.test_env 以便开始新的回合。
            h, terminated = self.policy.init_hidden().to(self.device), False  # Reset RNN states and terminated.

            # Run an episode.
            self.test_env.render()  # Render test env.   # 渲染测试环境，这通常用于可视化环境的当前状态。
            while not terminated:
                # Get observations and available actions.
                inputs = self.get_inputs_from_env(self.test_env, train_mode=False)   # 从测试环境中获取观测和可用的动作，train_mode=False表明是在测试模式下运行。
                # Take (quasi) deterministic actions.
                actions, h = self.policy.act(inputs['obs'], h, inputs['avail_actions'], mode='test')   # 智能体的策略模型基于观测inputs['obs']、当前的隐藏状态h和可用动作inputs['avail_actions']来决定动作actions，同时更新隐藏状态h。
                # Call test env step.
                _, terminated, info = self.test_env.step(self.get_actions_to_env(actions, self.test_env))   # 执行智能体决定的动作，更新环境状态，并获取反馈信息。terminated标志会更新以指示回合是否结束，info包含了回合的额外信息。
                # Render test env.
                self.test_env.render()   # 再次渲染测试环境，以显示智能体执行动作后的结果。

            # Save figure of rendered test env.
            if self.args.record_tests and j < 10:  # Save storage.   # 如果设置了记录测试并且当前回合数小于10，则执行下面的保存操作。
                self.test_env.save_replay(save_dir=osp.join(self.run_dir, f't{self.t_env}'), tag=f'ep{j}')   # 将渲染的测试环境保存下来，用于后续回放或分析。
            # Record episode info.
            for name, rst in info.items():
                if name != 'truncated':  # All entries other than episode limit are recorded
                    if name not in test_ep_rsts:   # 如果这个名称的信息还没有被记录在test_ep_rsts字典中，则初始化一个空列表。
                        test_ep_rsts[name] = []
                    test_ep_rsts[name].append(rst)

        # Log test results.
        self.log_metrics(test_ep_rsts, prefix='Test')

    def log_metrics(self, info: dict[str, Any], prefix: str = None) -> None:
        """Logs scalar metrics held in a dict."""
        for name, value in info.items():
            if prefix is not None:
                name = prefix + name
            # Transform vector metrics into scalars by averaging.
            if isinstance(value, list):
                value = np.array(value)
            if isinstance(value, ndarray) or isinstance(value, Tensor):
                value = value.mean()
            # Log given metrics.
            if not self.args.use_wandb:  # Tensorboard
                self.writer.add_scalar(name, value, self.t_env)
            else:  # wandb
                wandb.log({name: value}, step=self.t_env)
        
    def cleanup(self) -> None:
        """Terminates utils after training."""

        self.env.close()
        self.test_env.close()

        if not self.args.use_wandb:
            self.writer.flush()
            self.writer.close()
            print(f"Use command `tensorboard --logdir={self.run_dir}` to view results.")
        else:
            wandb.finish()
            # TODO: Export history to local .csv file.
            # Note that history data can be visited by using the Public API.
            # See https://docs.wandb.ai/guides/track/public-api-guide for detailed explanation.
