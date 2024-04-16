from typing import Any
import itertools
import os
import os.path as osp
from copy import deepcopy

import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from gym.spaces.discrete import Discrete
from gym.spaces.dict import Dict
from gym.spaces.box import Box

from envs.multi_agent_env import MultiAgentEnv
from envs.common import *
from envs.heatmap import heatmap, annotate_heatmap
from envs.chan_models import *
from envs.ad_hoc.ad_hoc_entities import *
from envs.ad_hoc.ad_hoc_layouts import SCENARIOS


class AdHocEnv(MultiAgentEnv):
    """Cross-Layer Optimization in Wireless Ad Hoc Networks"""

    LEGAL_ROUTERS = {'c2Dst', 'mSINR'}
    LEGAL_POWER_CONTROLLERS = {'Full', 'Rand'}

    def __init__(self,
                 scenario_name: str = '1flow',
                 mute_interference: bool = False,  # Whether to overlook interference.
                 learn_power_control: bool = True,  # Whether to learn power control.
                 enforce_connection: bool = True,  # Whether to enforce connection of path
                 benchmark_routers: tuple[str, ...] = ('c2Dst', 'mSINR'),  # Benchmark routers
                 benchmark_power_controllers: tuple[str, ...] = ('Full',),  # Benchmark power controllers
                 graph_khops: int = 1,
                 ):

        self._mute_inf = mute_interference
        self._learn_pc = learn_power_control
        self._force_cnct = enforce_connection

        self.bm_rts = benchmark_routers
        self.bm_pow_ctrls = benchmark_power_controllers

        self.khops = graph_khops

        # Get attributes from scenario.
        scenario = SCENARIOS[scenario_name]
        for k, v in scenario.__dict__.items():   # 将一个对象(scenario)的所有属性（键值对）复制到另一个对象(self)中。
            setattr(self, k, v)   # 如果scenario对象有一个属性name，其值为"example"，那么self.name也会被设置为"example"。
        self.set_entities = scenario.set_entities  # Callback to set initial attributes of entities

        # In single-channel case, each relay has to transmit/receive on the same spectrum.
        self._allow_full_duplex = True if self.n_chans == 1 else False

        # Get benchmarks.
        # self.n_pow_lvs > 1：这表示环境中是否有多于一个的功率水平可以选择。如果只有一个功率水平，那么所有的传输都将使用这个固定的功率水平，没有必要进行功率控制。
        # 'Rand' not in self.bm_pow_ctrls：这检查了self.bm_pow_ctrls（一个记录所有基准功率控制器名称的元组）是否不包括'Rand'。这里'Rand'代表随机选择功率水平的策略。
        if (self.n_pow_lvs > 1) and ('Rand' not in self.bm_pow_ctrls):  # Compare learnt policy and rand choice.
            self.bm_pow_ctrls += ('Rand',)   # 存在多个功率水平，并且随机功率控制策略还不在基准列表中，就会在环境中的基准功率控制器列表中添加一个随机策略，用于与学习到的策略进行性能比较。
        # 这行代码使用itertools.product函数创建了基准路由器(self.bm_rts)和基准功率控制器(self.bm_pow_ctrls)所有可能组合的一个列表。
        # itertools.product函数产生笛卡尔积，相当于嵌套的循环。
        # 例如，对于路由器['A', 'B'] 和功率控制器['X', 'Y', 'Z']，它将生成六个组合[('A', 'X'), ('A', 'Y'), ('A', 'Z'), ('B', 'X'), ('B', 'Y'), ('B', 'Z')]。
        benchmark_schemes = list(itertools.product(self.bm_rts, self.bm_pow_ctrls))   
        self.bm_perf = dict.fromkeys(benchmark_schemes)  # Performance of benchmark schemes
        self.bm_paths = dict()  # Copy of benchmark routes

        # Create entities in the env.
        self.nodes = [Node(m, self.n_chans) for m in range(self.n_nodes)]

        ## 构建离散的功率水平
        def builds_power_levels(n_levels, p_max):
            """Builds discrete power levels."""
            '''np.arange(n_levels, dtype=np.float32)：使用 numpy 库的 arange 函数生成从 0 开始的、长度为 n_levels（功率水平的数目），数据类型为 float32 的数组。例如，如果 n_levels 是 4，这个命令将生成一个数组 [0.0, 1.0, 2.0, 3.0]。'''
            '''(np.arange(n_levels, dtype=np.float32) + 1)：为数组中的每个元素加 1，使得创建的序列从 1 开始而不是从 0 开始，这样就不会有零功率的情况。对于 n_levels 是 4 的例子，现在数组会是 [1.0, 2.0, 3.0, 4.0]。'''
            '''(np.arange(n_levels, dtype=np.float32) + 1) / n_levels：上一步得到的数组的每个元素都除以 n_levels，对于4功率水平的例子，这将生成 [0.25, 0.5, 0.75, 1.0]。'''
            '''... * p_max：最后，把上一步的结果乘以 p_max，这将每个元素都按比例放大到 p_max 设定的最大功率值。如果 p_max 是 100，那么最终生成的功率水平就是 [25.0, 50.0, 75.0, 100.0]。'''
            return (np.arange(n_levels, dtype=np.float32) + 1) / n_levels * p_max

        # Ambient flows take fixed Tx power with unlimited budget,
        # while agent flows may select power levels based on their (constrained) budget.
        amb_flow_kwargs = dict(p_levels=builds_power_levels(1, self.p_amb), p_budget=float('inf'),
                               allow_full_duplex=self._allow_full_duplex)
        agt_flow_kwargs = dict(p_levels=builds_power_levels(self.n_pow_lvs, self.p_max), p_budget=self.p_bdg,
                               allow_full_duplex=self._allow_full_duplex)
        self.amb_flows = [Flow(i, self.n_chans, self.n_nodes, **amb_flow_kwargs) for i in range(self.n_amb_flows)]
        agt_flow_slice = range(self.n_amb_flows, self.n_amb_flows + self.n_agt_flows)
        self.agt_flows = [Flow(i, self.n_chans, self.n_nodes, **agt_flow_kwargs) for i in agt_flow_slice]

        self.chan = CHANNEL_MODELS[self.chan_name]()  # Wireless channel model
        self.bw = self.tot_bw / self.n_chans  # Bandwidth of each sub-channel (Hz)

        # Define time-varying attributes characterizing relations between entities.
        self.d_n2n = np.empty((self.n_nodes, self.n_nodes), dtype=np.float32)  # Distance between nodes
        self.d_n2dst = np.empty((self.n_nodes, self.n_flows),
                                dtype=np.float32)  # Distance between nodes and destinations
        self.chan_coef = np.empty((self.n_nodes, self.n_nodes, self.n_chans), dtype=np.float32)  # Channel coefficients

        self.p_rx = np.empty_like(self.chan_coef)
        self.p_inf = np.empty_like(self.chan_coef)
        self.link_sinr = np.empty_like(self.chan_coef)
        self.link_rates = np.empty_like(self.chan_coef)

        # Define MDP components.
        self.n_agents = 1  # Single-agent env
        self.episode_limit = self.max_hops  # Maximum number of timesteps
        self.agent: Flow = self.flows[-1]
        self.nbrs: list[Node] = []  # Neighbors around front node of agent

        self.all_actions = []   # 用于存储每个流程（flow）可能的所有动作（actions）。
        for flow in self.flows:
            action_tuples = [range(self.max_nbrs)]   # 表示从0到self.max_nbrs（一个表示最大邻居数的属性）的整数序列，代表可能的邻居选择。
            if self._learn_pc:
                action_tuples.append(range(flow.n_pow_lvs))
            all_agent_actions = list(itertools.product(*action_tuples))   # 使用itertools.product函数计算action_tuples列表中所有元素的笛卡尔积，生成所有可能的动作组合，然后将其转换为列表all_agent_actions。这个列表包含了所有可能的动作（邻居选择和功率级别组合）。
            all_agent_actions.append('no-op')   # 将字符串'no-op'追加到all_agent_actions列表中，这代表一个无操作（no operation）的动作，可能用于表示在当前时刻没有选择任何动作。
            self.all_actions.append(all_agent_actions)   # 将包含了当前流程所有可能动作的列表all_agent_actions追加到self.all_actions列表中。

        self.observation_space = []
        self.shared_observation_space = []
        self.action_space = []
        # 定义强化学习环境中的观测空间，即agent可以观察到的环境状态的范围。
        self.observation_space.append(
            Dict(spaces={
                'agent': Box(-np.inf, np.inf, shape=np.array([self.obs_own_feats_size])),
                'nbr': Box(-np.inf, np.inf, shape=np.array(self.obs_nbr_feats_size)),
            })
        )
        self.shared_observation_space.append(
            Dict(spaces={
                'agent': Box(-np.inf, np.inf, shape=np.array([self.shared_obs_own_feats_size])),
                'nbr': Box(-np.inf, np.inf, shape=np.array(self.shared_obs_nbr_feats_size)),
                'nbr2': Box(-np.inf, np.inf, shape=np.array(self.shared_obs_nbr2_feats_size)),
            })
        )
        self.action_space.append(Discrete(len(self.all_actions[-1])))

    @property
    def flows(self):
        """All data flows in a list"""
        return self.amb_flows + self.agt_flows

    @property
    def power_options(self):
        """Number of power options for RL algos"""
        return self.agent.n_pow_lvs if self._learn_pc else 1

    def reset(self):
        if self.agent != self.flows[-1]:  # Finished agent-flow is not the last one.   # 检查当前代理（self.agent）是否不是流列表（self.flows）中的最后一个元素。
            self.handover_agent(self.flows[self.flows.index(self.agent) + 1])  # Sequentially move to next flow.
        else:
            # Reset nodes and flows.
            pos_nodes, src_nids, dst_nids = self.set_entities()
            # print(f"src_nids = {src_nids}, dst_nids = {dst_nids}")
            for nid, node in enumerate(self.nodes):
                node.reset(pos_nodes[nid])
            for fid, flow in enumerate(self.flows):
                flow.reset(self.nodes[src_nids[fid]], self.nodes[dst_nids[fid]])

            # Compute the distance and channel coefficients between nodes.
            for m in range(self.n_nodes):
                for n in range(self.n_nodes):
                    # Distance between node-m and node-n.
                    self.d_n2n[m, n] = self.distance(self.nodes[m], self.nodes[n])
                for i in range(self.n_flows):
                    self.d_n2dst[m, i] = self.distance(self.nodes[m], self.flows[i].dst)
            d_n2n_copy = self.d_n2n.copy()
            d_n2n_copy[np.eye(self.n_nodes, dtype=bool)] = float('inf')
            self.chan_coef = self.chan.estimate_chan_gain(d_n2n_copy)
            self.chan_coef = np.tile(np.expand_dims(self.chan_coef, axis=-1), (1, 1, self.n_chans))
            self._update_per_link_rate()  # Compute achievable rate of all links.

            # Set route for ambient flows as warm-up.
            for flow in self.amb_flows:
                self.handover_agent(flow)
                terminated = False
                while not terminated:
                    action = self.get_heuristic_action(('c2Dst', 'Full'))
                    _, terminated, _ = self.step(action)
                # print(f"Ambient {flow} is set.")

            # Set heuristic routes for agent flows as benchmark.
            for rt in self.bm_rts:
                self.bm_paths[rt] = []
                for pc in self.bm_pow_ctrls:
                    for flow in self.agt_flows:
                        self.handover_agent(flow)
                        terminated = False
                        while not terminated:
                            action = self.get_heuristic_action((rt, pc))
                            _, terminated, _ = self.step(action)

                        # print(f"Agent {flow} is set by heuristic scheme ({rt}, {pc}).")
                        if len(self.bm_paths[rt]) < len(self.agt_flows):
                            self.bm_paths[rt].append(deepcopy(flow))

                    self.bm_perf[(rt, pc)] = self.evaluate_performance()

                    # Reset agent flows after evaluating performance.
                    for flow in self.agt_flows:
                        flow.clear_route()
                    self._update_per_link_rate()

            # Assign agent flow.
            self.handover_agent(self.agt_flows[0])

    def step(self, action) -> tuple[ndarray, bool, dict[str, Any]]:
        if isinstance(action, list):   # 判断 action 是否为一个列表。
            assert len(action) == self.n_agents, "Inconsistent numbers between actions and agents."   # 断言语句，用来确保 action 列表的长度和智能体（agents）的数量 self.n_agents 相等。
            action = action[0]   # 如果断言通过（即动作列表的长度等于智能体的数量），这行代码将 action 变量重新赋值为列表中的第一个元素。这是基于假设所有智能体都将执行相同的动作，或者这个环境仅仅需要一个代表性的动作。

        # Get selected nbr, power level.
        flow_action = self.all_actions[self.agent.fid][action]   # 据当前智能体的标识符 fid 和提供的 action 索引，从所有可能的动作集中选取了一个具体的动作
        # print(f"self.all_actions[self.agent.fid] = {self.all_actions[self.agent.fid]}")
        # print(f"action = {action}, flow_action = {flow_action}")
        if self._learn_pc:
            sel_nbr_idx, sel_pow_idx = flow_action
        else:
            sel_nbr_idx = flow_action[0]
            sel_pow_idx = self.allocate_power(self.nbrs[sel_nbr_idx], 'Rand')

        # Select the sub-channel with the least interference.
        next_node = self.nbrs[sel_nbr_idx]
        sel_chan_idx = self.allocate_channel(self.agent.front, next_node)

        # Enhance performance by eliminating unselected neighbors closer than selected one.
        front_nid = self.agent.front.nid
        for nbr in self.nbrs:
            is_closer_than_selected_nbr = self.d_n2n[nbr.nid, front_nid] < self.d_n2n[next_node.nid, front_nid]   # 计算当前邻居节点到前端节点的距离是否小于已选择的下一个节点到前端节点的距离。
            if is_closer_than_selected_nbr and (nbr is not self.agent.dst):
                self.agent.ban(nbr)

        # Apply action to agent.
        self.agent.add_hop(next_node, sel_chan_idx, sel_pow_idx)
        self._update_per_link_rate()
        self._find_neighbors()

        # Computes output signal from env.
        reward = self._get_reward()
        terminated = self._get_terminate()
        info = dict()

        # Evaluate performance of agent flows.
        agent_perf = self.evaluate_performance()
        for k, v in agent_perf.items():
            info['Agent' + k] = v
        # Provide performance of benchmark schemes on above flows .
        for sch, perf in self.bm_perf.items():
            rt, pc = sch
            if perf is not None:
                for k, v in perf.items():
                    info[rt + pc + k] = v

        return reward, terminated, info

    ### 将代理（agent）移交给数据流（flow）。
    def handover_agent(self, flow: Flow):
        """Hands-over agent to data flow."""
        self.agent = flow  # Assign flow to be agent.   # 将传入的flow参数分配给类的agent属性。
        self._find_neighbors()  # Since front node is altered, call update neighbors.

    @staticmethod
    def distance(node1: Node, node2: Node):
        """Computes distance between two nodes."""
        return np.linalg.norm(node1.pos - node2.pos)

    ### 为当前代理流的前端节点寻找符合条件的邻居节点。
    def _find_neighbors(self):
        """Finds qualified neighbors for front node of current agent flow."""
        # Check whether maximum hop is reached or battery is depleted.
        front_nid = self.agent.front.nid   # 获取当前代理流的前端节点的节点ID（nid），并将其存储在变量front_nid中。
        is_overtime = (self.agent.n_hops == self.max_hops - 1) and not self.agent.is_connected   # 检查是否达到最大跳数（max_hops）或代理流尚未连接（is_connected）。
        is_low_battery = (self.agent.p_rem < 2 * self.agent.p_lvs[0]) and not self.agent.is_connected   # 检查代理流的剩余功率（p_rem）是否小于两倍的最低生存功率（p_lvs[0]），同时代理流尚未连接。

        if self.agent.is_connected:  # When agent is connected:
            # `terminated` would be then activated to reset the env.
            nbrs = []  # No neighbor is available after termination of episode.   # 如果代理流已经连接，则将nbrs设为空列表，因为连接后不需要再寻找邻居。
        elif is_overtime or is_low_battery:  # When any failure occurs:
            if self._force_cnct:
                nbrs = [self.agent.dst]  # Destination is the only neighbor.   # 将目的节点作为唯一的邻居。
            else:
                nbrs = []
        else:   # 如果没有连接，也没有超时或电池电量不足的情况，将根据当前前端节点的距离对所有节点进行排序，并根据一定的条件收集邻居节点。
            # Sort all nodes in ascending order of distance to current front node.
            sorted_nids = np.argsort(self.d_n2n[self.agent.front.nid])   # 对所有节点按照与前端节点的距离进行排序。
            nbrs = []
            for nid in sorted_nids:
                # Add node to neighbors when all of following conditions are met: The neighbor
                # 1) lies within the sensing range, 2) Meets the qualification of current agent flow.   # （1）当前节点 nid 与前端节点 front_nid 之间的距离小于或等于传感范围 self.r_sns。（2）当前节点 nid 满足代理流（self.agent）的某些资格条件。
                if (self.d_n2n[nid, front_nid] <= self.r_sns) and self.agent.check(self.nodes[nid]):
                    nbrs.append(self.nodes[nid])
                # End when maximum number of neighbors are collected.
                if len(nbrs) >= self.max_nbrs:
                    break
            # Shuffle the order of neighbors.
            random.shuffle(nbrs)   # 对邻居列表进行随机排序。
            is_isolated = (len(nbrs) == 0) and not self.agent.is_connected   # 判断当前节点是否处于孤立状态。如果邻居列表为空（即 len(nbrs) == 0）并且代理流（self.agent）并未连接，则认为是孤立的。
            if is_isolated and self._force_cnct:   # 节点确实处于孤立状态并且存在一个强制连接的要求
                nbrs = [self.agent.dst]   # 邻居列表 nbrs 将被设置为仅包含代理的目的地节点
        # Assign neighbors.
        if (not self._force_cnct) and (not self.agent.is_connected):
            assert len(nbrs) > 0, "Empty neighbor set is found."
        self.nbrs = nbrs

    def _find_2hop_neighbors(self):
        """Finds neighbors of current neighbors (2nd-hop).

        Note that 2nd-hop neighbors are not directly added to route and thus qualification is not mandatory.
        """
        nbrs2_per_nbr = []
        for nbr in self.nbrs:
            sorted_nids = np.argsort(self.d_n2n[nbr.nid])
            nbrs2 = []
            for nid in sorted_nids:
                if self.nodes[nid] is not nbr:
                    nbrs2.append(self.nodes[nid])
                # End when maximum number of neighbors are collected.
                if len(nbrs2) >= self.max_nbrs:
                    break
            nbrs2_per_nbr.append(nbrs2)
        return nbrs2_per_nbr

    ### 计算网络中所有连接（links）的可实现传输速率。
    def _update_per_link_rate(self):
        """Computes achievable rate of all links."""
        p_tx = np.stack([node.p_tx for node in self.nodes])  # Tx power (Watt)   # 使用列表推导式获取 self.nodes 中每个节点的发送功率 p_tx，然后使用 NumPy 的 stack 函数将它们堆叠成一个数组。
        self.p_rx = self.chan_coef * (1 - np.expand_dims(np.eye(self.n_nodes), axis=-1)) * p_tx  # Rx power (Watt)   # 计算接收功率 p_rx。self.chan_coef 是一个与信道有关的系数矩阵，np.eye(self.n_nodes) 创建一个单位矩阵，表示节点不能接收到自己发送的功率，np.expand_dims(..., axis=-1) 用于增加一个维度以匹配 p_tx 的维度。最后与发送功率 p_tx 相乘以得到接收功率。
        self.p_inf = np.zeros_like(self.p_rx) if self._mute_inf else self.p_rx.sum(1, keepdims=True) - self.p_rx  # Interference (Watt)   # 计算干扰功率 p_inf。如果 self._mute_inf 为真，则干扰功率被设置为零。否则，计算总的接收功率（self.p_rx.sum(1, keepdims=True)），然后减去节点自己的接收功率，得到干扰功率。
        self.link_sinr = self.p_rx / (self.p_inf + self.bw * self.n0)  # Signal-to-interference-plus-noise ratio   # 计算每个连接的信噪比（SINR）。self.bw 是信道的带宽，self.n0 是噪声功率谱密度。将接收功率 p_rx 除以干扰功率加上噪声功率（self.bw * self.n0）得到 SINR。
        self.link_rates = self.bw * np.log2(1 + self.link_sinr) * 1e-6  # Achievable rates (Mbps)   # 使用夏农公式计算每个连接的可实现传输速率，并将结果转换为 Mbps（兆比特每秒）。

    def get_per_hop_rate(self, flow: Flow):
        """Returns the rate of each hop along a data flow."""
        rate_per_hop = []
        for link in flow.route:
            rate_per_hop.append(self.link_rates[link.rx.nid, link.tx.nid, link.chan_idx])
        return rate_per_hop

    def get_bottleneck_rate(self, flow: Flow):
        """Gets the bottleneck rate of a data flow."""
        rate_per_hop = self.get_per_hop_rate(flow)
        if len(rate_per_hop) > 0:
            bottleneck_rate = np.amin(rate_per_hop)
            bottleneck_idx = np.argmin(rate_per_hop)
            return bottleneck_rate.item(), bottleneck_idx.item()
        else:
            return 0.0, 0

    def get_bottleneck_sinr(self, flow: Flow):
        """Gets the bottleneck SINR of a data flow."""
        sinr_per_hop = []
        for link in flow.route:
            sinr_per_hop.append(self.link_sinr[link.rx.nid, link.tx.nid, link.chan_idx])
        return min(sinr_per_hop)

    ### 为发送节点和接收节点分配一个子信道。
    def allocate_channel(self, tx_node: Node, rx_node: Node):
        """Allocates sub-channel for a transceiver pair."""
        p_inf_per_chan = self.p_inf[rx_node.nid, tx_node.nid]   # 从一个名为 self.p_inf 的数据结构中获取了两个节点间所有子信道的干扰水平。p_inf 可能是一个二维数组或矩阵，其中 rx_node.nid 和 tx_node.nid 分别是接收节点和发送节点的节点ID，用于索引这两个节点之间的干扰信息。
        for chan_idx in np.argsort(p_inf_per_chan):  # Starting from channel with the lowest inf level:   # 对 p_inf_per_chan 中的干扰水平进行排序，并返回排序后的索引。这样就可以从干扰最小的信道开始遍历。
            if (tx_node.idle[chan_idx] or self._allow_full_duplex) and rx_node.idle[chan_idx]:   # 是否满足：发送节点在该信道上是空闲的，或者允许全双工通信（self._allow_full_duplex 为真），并且接收节点在该信道上也是空闲的。
                return chan_idx
        # This should not happen since it is prevented by `check` mechanism.
        raise Warning("No channel is available!")

    def allocate_power(self, next_node: Node, power_controller: str = 'Full'):
        """Allocates Tx power from current front node to next node"""
        can_afford_pow_lvs = self.get_affordable_power_levels(next_node)
        afford_pow_lv_idxes = np.flatnonzero(can_afford_pow_lvs)
        if power_controller == 'Full':
            sel_pow_idx = afford_pow_lv_idxes[-1]  # Maximum affordable power
        elif power_controller == 'Rand':
            rand_from_afford_pow_lv_idxes = random.sample(range(afford_pow_lv_idxes.size), 1)[0]
            sel_pow_idx = afford_pow_lv_idxes[rand_from_afford_pow_lv_idxes]  # Rand affordable power
        else:
            raise KeyError("Unrecognized power controller!")
        # print(f"Select p_idx = {sel_pow_idx} with p_rem = {self.agent.p_rem}.")
        return sel_pow_idx

    def get_affordable_power_levels(self, next_node: Node):
        """Returns affordable power levels from current agent front to next node"""
        p_rem = self.agent.p_rem  # Remaining power
        p_min = self.agent.p_lvs[0]  # Minute power
        if next_node is self.agent.dst:
            can_afford_pow_lvs = [(p < p_rem) for p in self.agent.p_lvs]
        else:
            can_afford_pow_lvs = [(p < p_rem - p_min) for p in self.agent.p_lvs]
        return can_afford_pow_lvs

    ### 根据启发式算法来返回智能体（agent）的动作。
    def get_heuristic_action(self, scheme: tuple[str, str]):
        """Returns agent action by heuristic AI."""
        # Interpret heuristic scheme.
        rt, pc = scheme   # 路由器策略和功率控制器策略。
        assert (rt in self.LEGAL_ROUTERS) and (pc in self.LEGAL_POWER_CONTROLLERS), \   # 断言，用来验证 rt 和 pc 是否是合法的路由器和功率控制器策略。
            f"Unrecognized heuristic router/power controller ({scheme}) is received."

        # Select next hop from neighbors.
        if rt == 'c2Dst':
            d_nbr2dst = [self.d_n2dst[nbr.nid, self.agent.fid] for nbr in self.nbrs]   # 使用列表推导式来计算每个邻居节点到目的地的距离，并将这些距离存储在列表 d_nbr2dst 中。
            sel_nbr_idx = np.argmin(d_nbr2dst)   # 找到到目的地距离最小的邻居节点的索引。
            # print(f"d_nbr2dst = {d_nbr2dst}, sel_nbr_idx = {sel_nbr_idx}")
        elif rt == 'mSINR':
            front_nid = self.agent.front.nid   # 获取当前智能体所在节点的前端节点ID。
            sinr_per_nbr = []   # 初始化列表，存储每个邻居节点的信噪比（SINR）。
            for nbr in self.nbrs:
                sinr = self.chan_coef[nbr.nid, front_nid] * self.agent.p_lvs[-1] / (self.n0 * self.bw + self.p_inf[nbr.nid, front_nid])   # 计算每个邻居节点的信噪比。
                sinr_per_nbr.append(sinr)   # 将计算出的信噪比添加到列表。
                sel_nbr_idx = np.argmax(sinr_per_nbr)   # 找出信噪比最大的邻居节点的索引。
        else:
            raise KeyError(f"Unrecognized heuristic scheme `{scheme}` to select next node.")

        # Allocate Tx power and get index of discrete action.
        if self._learn_pc:
            sel_pow_idx = self.allocate_power(self.nbrs[sel_nbr_idx], pc)
            act_idx = sel_nbr_idx * self.agent.n_pow_lvs + sel_pow_idx
        else:
            act_idx = sel_nbr_idx
        return act_idx

    def get_total_actions(self):
        return [self.action_space[i].n for i in range(self.n_agents)]

    def get_avail_actions(self):
        avail_actions = []

        no_op = True if self.agent.is_connected else False
        if self._learn_pc:
            avail_agent_actions = np.zeros((self.max_nbrs, self.agent.n_pow_lvs), dtype=bool)
            if not self.agent.is_connected:
                for nbr_idx, nbr in enumerate(self.nbrs):
                    can_afford_pow_lvs = self.get_affordable_power_levels(nbr)
                    avail_agent_actions[nbr_idx, can_afford_pow_lvs] = True
        else:
            avail_agent_actions = np.zeros(self.max_nbrs, dtype=bool)
            if not self.agent.is_connected:
                for nbr_idx in range(len(self.nbrs)):
                    avail_agent_actions[nbr_idx] = True
        avail_actions.append(avail_agent_actions.flatten().tolist() + [no_op])

        return avail_actions

    ### 获取代理（agent）和相邻节点（neighbor nodes）的特征，作为观察结果（observation）返回。
    def get_obs(self):
        obs = []   # 初始化一个空列表obs，用于存放最终构建的观察结果。

        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)   # 用来存储代理自身的特征。
        nbr_feats = np.zeros(self.obs_nbr_feats_size, dtype=np.float32)   # 用来存储邻居节点的特征。

        # Get features of agent flow
        ind = 0   # 索引变量ind，用于在数组中逐步填充特征。
        own_feats[ind:ind + self.dim_pos] = self.agent.front.pos / self.range_pos  # Position of front node   # 将代理前节点的位置（self.agent.front.pos）除以位置范围（self.range_pos）后，存入own_feats数组的相应位置。这是归一化后的前节点位置特征。
        ind += self.dim_pos   # 更新索引ind，以便下一个特征能够被放置在正确的位置。
        own_feats[ind:ind + self.dim_pos] = (self.agent.dst.pos - self.agent.front.pos) / self.range_pos  # Distance to destination   # 计算智能体目的地（dst）与其前方节点（front）之间的位置差，并且将这个差值进行标准化，然后存储到智能体的特征数组 own_feats 中的特定位置。
        ind += self.dim_pos   # 再次更新索引ind。
        if self.agent.p_bdg < float('inf'):  # If power budget is limited:   # 检查代理的能量预算是否有限，即不是无穷大。
            own_feats[ind] = self.agent.p_rem / self.agent.p_bdg  # Remaining power   # 如果能量预算有限，将代理剩余能量（self.agent.p_rem）除以能量预算（self.agent.p_bdg），存储在own_feats数组的当前索引位置。
            ind += 1

        # Get features of neighbor nodes.
        front_nid = self.agent.front.nid   # 获取代理前节点的全局节点ID。
        p_max = self.agent.p_lvs[-1]   # 获取代理可用功率级别的最大值。
        for m, nbr in enumerate(self.nbrs):   # 遍历邻居节点列表self.nbrs，m是邻居的索引，nbr是邻居节点对象。
            ind = 0   # 重置索引ind，用于填充邻居节点的特征。
            # Availability
            nbr_feats[m, ind] = 1   # 将邻居的可用性（Availability）设置为1（表示可用），存储在邻居特征数组nbr_feats的当前位置。
            ind += 1
            # Relative distance to front node of agent
            nbr_feats[m, ind:ind + self.dim_pos] = (nbr.pos - self.agent.front.pos) / self.range_pos   # 计算邻居节点位置与代理前节点位置的相对距离，并进行归一化，存储在nbr_feats数组的相应位置。
            ind += self.dim_pos
            # Relative distance to destination of agent
            nbr_feats[m, ind:ind + self.dim_pos] = (nbr.pos - self.agent.dst.pos) / self.range_pos   # 计算邻居节点位置与代理目的地位置的相对距离，并进行归一化，存储在nbr_feats数组的相应位置。
            ind += self.dim_pos
            # SINR in dB
            sinr_per_chan = self.chan_coef[nbr.nid, front_nid] * p_max / (self.n0 * self.bw + self.p_inf[nbr.nid, front_nid])   # 计算信噪比（SINR）：通过将信道系数（self.chan_coef[nbr.nid, front_nid]）乘以最大功率（p_max），然后除以噪声功率（由噪声功率谱密度self.n0乘以带宽self.bw）加上干扰功率（self.p_inf[nbr.nid, front_nid]）得到的。
            # self.chan_coef[nbr.nid, front_nid]：这部分表示从邻居节点 nbr 到前方节点 front_nid 的信道系数（channel coefficient）。
            # self.p_inf[nbr.nid, front_nid]：这部分表示从邻居节点到前方节点的干扰功率（Interference Power）。
            nbr_feats[m, ind] = np.log10(max(np.max(sinr_per_chan), 1e-10))  # Avoid extreme value of SINR in dB.   # 将计算出的SINR值转换为分贝（dB）单位，并确保不会出现极端值。这个值存储在nbr_feats数组的当前索引位置。

        obs.append(dict(agent=own_feats, nbr=nbr_feats))
        return obs

    @property
    def obs_own_feats_size(self):
        nf_own = self.dim_pos + self.dim_pos
        if self.agent.p_bdg < float('inf'):
            nf_own += 1  # Scalar to show remaining energy
        return nf_own

    @property
    def obs_nbr_feats_size(self):   # 返回两个值：最大邻居数和每个邻居的特征维度数。
        nf_nbr = 1 + self.dim_pos + self.dim_pos  # Availability, distance to agent, distance to destination
        nf_nbr += 1  # Cumulative interference
        return self.max_nbrs, nf_nbr

    def get_obs_size(self):
        return [self.obs_own_feats_size + np.prod(self.obs_nbr_feats_size)] * self.n_agents

    def get_graph_inputs(self):
        k = self.khops

        # ============ Count number of nodes of each type. ============
        num_nodes_dict = dict(agent=1, nbr=self.n_nodes)

        # ============ Get node features. ============
        own_feats = np.zeros(self.obs_own_feats_size, dtype=np.float32)
        # Get features of agent flow
        ind = 0
        own_feats[ind:ind + self.dim_pos] = self.agent.front.pos / self.range_pos  # Position of front node
        ind += self.dim_pos
        own_feats[ind:ind + self.dim_pos] = (self.agent.dst.pos - self.agent.front.pos) / self.range_pos  # Distance to destination
        ind += self.dim_pos
        if self.agent.p_bdg < float('inf'):  # If power budget is limited:
            own_feats[ind] = self.agent.p_rem / self.agent.p_bdg  # Remaining power
            ind += 1

        nbr_feats = np.zeros((self.n_nodes, self.graph_nbr_feats))
        for nid, node in enumerate(self.nodes):
            ind = 0
            nbr_feats[nid, ind:ind + self.dim_pos] = (self.agent.dst.pos - node.pos) / self.range_pos
            ind += self.dim_pos

        node_feats = {'agent': np.expand_dims(own_feats, 0), 'nbr': nbr_feats}

        # ============ Define edges and their features. ============
        graph_data = {('nbr', '1hop', 'agent'): ([], [])}
        edge_feats = {'1hop': []}
        if k == 2:
            graph_data.update({('nbr', '2hop', 'nbr'): ([], [])})
            edge_feats.update({'2hop': []})

        nbrs2_per_nbr = self._find_2hop_neighbors()
        front_nid = self.agent.front.nid
        p_max = self.agent.p_lvs[-1]

        for nbr_idx, nbr in enumerate(self.nbrs):
            # Agent and its neighbors defines 1-hop relations.
            graph_data[('nbr', '1hop', 'agent')][0].append(nbr.nid)
            graph_data[('nbr', '1hop', 'agent')][1].append(0)

            h1_feats = np.zeros(self.graph_hop_feats, dtype=np.float32)
            ind = 0
            h1_feats[ind:ind + self.dim_pos] = (nbr.pos - self.agent.front.pos) / self.range_pos
            ind += self.dim_pos
            # SINR in dB
            sinr_per_chan = self.chan_coef[nbr.nid, front_nid] * p_max / (self.n0 * self.bw + self.p_inf[nbr.nid, front_nid])
            h1_feats[ind] = np.log10(max(np.max(sinr_per_chan), 1e-10))  # Avoid extreme value of SINR in dB.

            edge_feats['1hop'].append(h1_feats)

            if k == 2:
                for nbr2_idx, nbr2 in enumerate(nbrs2_per_nbr[nbr_idx]):
                    # Neighbors and their neighbors defines 2-hop relations.
                    graph_data[('nbr', '2hop', 'nbr')][0].append(nbr2.nid)
                    graph_data[('nbr', '2hop', 'nbr')][1].append(nbr.nid)

                    h2_feats = np.zeros(self.graph_hop_feats, dtype=np.float32)
                    ind = 0
                    h2_feats[ind:ind + self.dim_pos] = (nbr2.pos - nbr.pos) / self.range_pos
                    ind += self.dim_pos
                    # SINR in dB
                    sinr_per_chan = self.chan_coef[nbr2.nid, nbr.nid] * p_max / (self.n0 * self.bw + self.p_inf[nbr2.nid, nbr.nid])
                    h2_feats[ind] = np.log10(max(np.max(sinr_per_chan), 1e-10))  # Avoid extreme value of SINR in dB.

                    edge_feats['2hop'].append(h2_feats)

        for etype, edata in edge_feats.items():
            if len(edata) == 0:
                edge_feats[etype] = np.zeros((0, self.graph_feats['hop']), dtype=np.float32)
            else:
                edge_feats[etype] = np.stack(edata)

        graph_inputs = {
            'graph_data': graph_data,  # Define edges
            'num_nodes_dict': num_nodes_dict,  # Number of nodes
            'ndata': node_feats,  # Node features
            'edata': edge_feats,  # Edge features
        }
        return graph_inputs

    @property
    def graph_feats(self, ):
        return {
            'agent': self.graph_own_feats,
            'nbr': self.graph_nbr_feats,
            'hop': self.graph_hop_feats,
        }

    @property
    def graph_own_feats(self):
        return self.obs_own_feats_size

    @property
    def graph_nbr_feats(self):
        return self.dim_pos

    @property
    def graph_hop_feats(self):
        return self.dim_pos + 1

    def get_shared_obs(self):

        # Get local observations.
        local_obs = self.get_obs()

        nbrs2_per_nbr = self._find_2hop_neighbors()
        nbr2_feats = np.zeros(self.shared_obs_nbr2_feats_size, dtype=np.float32)
        p_max = self.agent.p_lvs[-1]
        for nbr_idx, nbr in enumerate(self.nbrs):
            nbrs2 = nbrs2_per_nbr[nbr_idx]
            for nbr2_idx, nbr2 in enumerate(nbrs2):
                ind = 0
                nbr2_feats[nbr_idx, nbr2_idx, ind] = 1
                ind += 1
                nbr2_feats[nbr_idx, nbr2_idx, ind:ind + self.dim_pos] = (nbr2.pos - nbr.pos) / self.range_pos
                ind += self.dim_pos
                nbr2_feats[nbr_idx, nbr2_idx, ind:ind + self.dim_pos] = (nbr2.pos - self.agent.dst.pos) / self.range_pos
                ind += self.dim_pos
                sinr_per_chan = self.chan_coef[nbr2.nid, nbr.nid] * p_max / (self.n0 * self.bw + self.p_inf[nbr2.nid, nbr.nid])
                nbr2_feats[nbr_idx, nbr2_idx, ind] = np.log10(max(np.max(sinr_per_chan), 1e-10))  # Avoid extreme value of SINR in dB.

        shared_obs = [dict(nbr2=nbr2_feats, **local_obs_agent) for local_obs_agent in local_obs]
        return shared_obs

    def get_shared_obs_size(self):
        return [self.shared_obs_own_feats_size + np.prod(self.shared_obs_nbr_feats_size) + np.prod(self.shared_obs_nbr2_feats_size)] * self.n_agents

    @property
    def shared_obs_own_feats_size(self):
        return self.obs_own_feats_size

    @property
    def shared_obs_nbr_feats_size(self):
        return self.max_nbrs, self.obs_nbr_feats_size[-1]

    @property
    def shared_obs_nbr2_feats_size(self):
        return self.max_nbrs, self.max_nbrs, self.obs_nbr_feats_size[-1]

    def _get_reward(self):
        # Reward SINR of bottleneck link.
        bias = 5
        bottleneck_sinr = np.log10(self.get_bottleneck_sinr(self.agent)) + bias
        reward = self.agent.is_connected * max(bottleneck_sinr, 0)
        return np.array([reward], dtype=np.float32)

    def _get_terminate(self):
        """Returns termination of episode."""
        if self.agent.is_connected or (len(self.nbrs) == 0):
            return True
        else:
            return False

    def measure_link_distance(self, link: Link):
        tx_nid, rx_nid = link.tx.nid, link.rx.nid
        return self.d_n2n[rx_nid, tx_nid]

    def evaluate_performance(self):
        """Evaluates the overall performance of agent flows."""
        perf_ind_dict = {
            'BottleneckRate': [self.get_bottleneck_rate(flow)[0] for flow in self.agt_flows],
            'Hops': [flow.n_hops for flow in self.agt_flows],
        }

        # When power budget is finite, record the total amount of power consumption.
        if self.p_bdg < float('inf'):
            perf_ind_dict['TotalPowCost'] = [flow.p_tot for flow in self.agt_flows]

        # Check whether the data is connected autonomously.
        def check_autonomous_connection(flow: Flow):
            if not flow.is_connected:
                return 0
            else:
                for link in flow.route:
                    if self.measure_link_distance(link) > self.r_sns:
                        return 0
                return 1

        perf_ind_dict['ConnectionProb'] = [check_autonomous_connection(flow) for flow in self.agt_flows]

        return perf_ind_dict

    def render(self):
        pass

    def save_replay(self, show_img: bool = False, save_dir: str = None, tag: str = None):
        self.visualize_policy(show_img, save_dir, tag)
        self.visualize_route(show_img, save_dir, tag)
        self.visualize_power(show_img, save_dir, tag)

    def add_boundary(self, ax):
        """Adds boundary to ax."""
        boundary = plot_boundary(self.range_pos)
        ax.plot(*boundary, color='black')

        ax.axis([-0.1 * self.range_pos, 1.1 * self.range_pos, -0.1 * self.range_pos, 1.1 * self.range_pos])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    def plot_nodes(self, ax, ignore_idle_nodes: bool = True):
        """Plots nodes to ax."""
        # Styles of nodes according to their role
        node_styles = {
            'idle': {'marker': 'o', 'alpha': 0.5, 's': 70, 'color': 'grey'},
            'src': {'marker': 'v', 'alpha': 0.75, 's': 100, 'color': 'tomato'},
            'dst': {'marker': 's', 'alpha': 0.75, 's': 100, 'color': 'tomato'},
            'rly': {'marker': 'o', 'alpha': 0.75, 's': 85, 'color': 'lightskyblue'},
        }
        txt_offset = 7.5  # Offset of text
        font_size = 6  # Font size

        for node in self.nodes:
            # Determine the role played by node.
            role = 'idle' if node.idle.all() else node.role
            txt_offset_bias = 0 if node.nid < 10 else txt_offset
            if role == 'idle':
                if not ignore_idle_nodes:  # Idle nodes are plotted only if specially requested.
                    ax.scatter(node.pos[0], node.pos[1], **node_styles['idle'])
                    ax.text(node.pos[0] - txt_offset - txt_offset_bias, node.pos[1] - txt_offset, f"{node.nid}",
                            fontsize=font_size, alpha=0.5, weight='light')
            else:
                ax.scatter(node.pos[0], node.pos[1], **node_styles[role])
                ax.text(node.pos[0] - txt_offset - txt_offset_bias, node.pos[1] - txt_offset, f"{node.nid}",
                        fontsize=font_size)

    def visualize_policy(self, show_img: bool = False, save_dir: str = None, tag: str = None, **kwargs):
        """Shows the routes/resource allocation decisions of policy."""
        dpi = 200  # Resolution of figures
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(7, 3.25), layout="constrained", dpi=dpi)
        fig.suptitle('Route and Resource Allocation in Wireless Ad Hoc Network', fontsize=12)

        # ============ Sub-figure 1: Plot route. ============
        ax1 = axs[0]
        ax1.set_aspect('equal')

        # Style of different heuristic routers
        rt_styles = {
            'c2Dst': {'color': 'pink', 'linewidth': 2.5, 'alpha': 0.5},
            'mSINR': {'color': 'yellowgreen', 'linewidth': 2.5, 'alpha': 0.5},
        }

        from matplotlib.collections import LineCollection
        import matplotlib.lines as mlines

        # Plot routes set by benchmark routers.
        rt_labels = []
        for rt, flows in self.bm_paths.items():
            xs, ys = [], []
            for flow in flows:
                for link in flow.route:
                    xs.append(np.linspace(link.tx.pos[0], link.rx.pos[0], 25))
                    ys.append(np.linspace(link.tx.pos[1], link.rx.pos[1], 25))
            segs = [np.stack((x, y)).T for x, y in zip(xs, ys)]
            ax1.add_collection(LineCollection(segs, zorder=-1, **rt_styles[rt]))
            rt_labels.append(mlines.Line2D([], [], label=rt, **rt_styles[rt]))
        ax1.legend(handles=rt_labels, loc='lower right', prop={'size': 6})

        # Get the range of rates.
        min_rate, max_rate = float('inf'), 0
        for flow in self.flows:
            if flow.n_hops > 0:
                rate_per_hop = self.get_per_hop_rate(flow)
                min_rate = min(min_rate, min(rate_per_hop))
                max_rate = max(max_rate, max(rate_per_hop))
        max_rate += 1e-2
        min_rate -= 1e-2
        # Plot routes of all flows with color specifying link rates.
        cmap = plt.colormaps["magma"]
        for flow in self.flows:
            for link in flow.route:
                link_rate = self.link_rates[link.rx.nid, link.tx.nid, link.chan_idx]
                norm_r = (link_rate - min_rate) / (max_rate - min_rate)
                ax1.arrow(link.tx.pos[0], link.tx.pos[1],
                          link.rx.pos[0] - link.tx.pos[0], link.rx.pos[1] - link.tx.pos[1],
                          shape='full', length_includes_head=True, width=5, color=cmap(norm_r),
                          alpha=norm_r * 0.8 + 0.1)

        # Plot all nodes.
        self.plot_nodes(ax1, ignore_idle_nodes=False)
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(min_rate, max_rate), cmap=cmap), ax=ax1,
                     fraction=0.05, pad=0.05, label="Rate (Mbps)")

        # Plot the boundary of legal region.
        self.add_boundary(ax1)
        ax1.set_title("Route")

        # ============ Sub-figure 2: Plot resource allocation. ============
        ax2 = axs[1]

        # Extract busy nodes
        busy_nids, p_tx = [], []
        for node in self.nodes:
            if not node.idle.all():
                p_tx.append(node.p_tx)
                busy_nids.append(node.nid)

        if len(busy_nids) > 0:
            # Stack Tx power of nodes.
            p_tx = np.stack(p_tx)
            # Create heatmap.
            im, cbar = heatmap(p_tx, busy_nids, range(self.n_chans), ax=ax2,
                               cmap="magma", cbarlabel="Tx power (Watt)")
            # Annotate heatmap.
            texts = annotate_heatmap(im, valfmt="{x:.2f}", size=7, threshold=0.04,
                                     textcolors=("white", "black"))

            ax2.set_title("Resource Allocation")
            ax2.set_xlabel('Chan Idx')
            ax2.set_ylabel('NID')

        # ============ Show/save figure. ============

        # Display the image.
        if show_img:
            plt.show()

        # Write results to disk.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig_name = 'rt_ra.pdf'
            if tag is not None:
                fig_name = tag + '_' + fig_name
            fig_path = osp.join(save_dir, fig_name)
            plt.savefig(fig_path)
        plt.close()

    def visualize_route(self, show_img: bool = False, save_dir: str = None, tag: str = None, **kwargs):
        """Shows the routes/resource allocation decisions of policy."""
        dpi = 200  # Resolution of figures
        fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(3.5, 3.25), layout="constrained", dpi=dpi)

        # ============ Sub-figure 1: Plot route. ============
        ax1.set_aspect('equal')

        # Style of different heuristic routers
        rt_styles = {
            'c2Dst': {'color': 'pink', 'linewidth': 2.5, 'alpha': 0.5},
            'mSINR': {'color': 'yellowgreen', 'linewidth': 2.5, 'alpha': 0.5},
        }

        from matplotlib.collections import LineCollection
        import matplotlib.lines as mlines

        # Plot routes set by benchmark routers.
        rt_labels = []
        for rt, flows in self.bm_paths.items():
            xs, ys = [], []
            for flow in flows:
                for link in flow.route:
                    if self.measure_link_distance(link) <= self.r_sns:
                        xs.append(np.linspace(link.tx.pos[0], link.rx.pos[0], 25))
                        ys.append(np.linspace(link.tx.pos[1], link.rx.pos[1], 25))
            segs = [np.stack((x, y)).T for x, y in zip(xs, ys)]
            ax1.add_collection(LineCollection(segs, zorder=-1, **rt_styles[rt]))
            rt_labels.append(mlines.Line2D([], [], label=rt, **rt_styles[rt]))
        ax1.legend(handles=rt_labels, loc='lower right', prop={'size': 6})

        # Get the range of rates.
        min_rate, max_rate = float('inf'), 0
        for flow in self.flows:
            if flow.n_hops > 0:
                rate_per_hop = self.get_per_hop_rate(flow)
                min_rate = min(min_rate, min(rate_per_hop))
                max_rate = max(max_rate, max(rate_per_hop))
        max_rate += 1e-2
        min_rate -= 1e-2
        # Plot routes of all flows with color specifying link rates.
        cmap = plt.colormaps["magma"]
        for flow in self.flows:
            for link in flow.route:
                if self.measure_link_distance(link) <= self.r_sns:
                    link_rate = self.link_rates[link.rx.nid, link.tx.nid, link.chan_idx]
                    norm_r = (link_rate - min_rate) / (max_rate - min_rate)
                    ax1.arrow(link.tx.pos[0], link.tx.pos[1],
                              link.rx.pos[0] - link.tx.pos[0], link.rx.pos[1] - link.tx.pos[1],
                              shape='full', length_includes_head=True, width=5, color=cmap(norm_r),
                              alpha=norm_r * 0.75 + 0.25)

        # Plot all nodes.
        self.plot_nodes(ax1, ignore_idle_nodes=False)
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(min_rate, max_rate), cmap=cmap), ax=ax1,
                     fraction=0.05, pad=0.05, label="Rate (Mbit/s)")

        # Plot the boundary of legal region.
        self.add_boundary(ax1)

        # ============ Show/save figure. ============

        # Display the image.
        if show_img:
            plt.show()

        # Write results to disk.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig_name = 'rt.pdf'
            if tag is not None:
                fig_name = tag + '_' + fig_name
            fig_path = osp.join(save_dir, fig_name)
            plt.savefig(fig_path)
        plt.close()

    def visualize_power(self, show_img: bool = False, save_dir: str = None, tag: str = None):
        """Shows power relation between nodes."""
        dpi = 200  # Resolution of figures
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(7, 3.25), layout="constrained", dpi=dpi)
        fig.suptitle('Received Power', fontsize=12)

        snr_min, snr_max = -10, 40  # Min/Max value of power/N (dB)
        cmap = plt.colormaps["viridis"]  # Colormap

        def normalize_snr(p):
            snr_db = 10 * np.log10(max(p / (self.n0 * self.bw), 1e-8))
            rescaled_snr_db = (np.clip(snr_db, snr_min, snr_max) - snr_min) / (snr_max - snr_min)
            return rescaled_snr_db

        # ============ Sub-figure 1: Plot direct signals. ============
        ax1 = axs[0]
        ax1.set_aspect('equal')

        for flow in self.flows:
            _, bottleneck_idx = self.get_bottleneck_rate(flow)
            for l_idx, link in enumerate(flow.route):
                tx, rx = link.tx, link.rx
                # Accentuate bottleneck link.
                if l_idx == bottleneck_idx:
                    ax1.plot(np.linspace(tx.pos[0], rx.pos[0], 25), np.linspace(tx.pos[1], rx.pos[1], 25),
                             alpha=0.25, color='grey', linewidth=7.5)
                snr_val = normalize_snr(self.p_rx[rx.nid, tx.nid, link.chan_idx])
                ax1.arrow(tx.pos[0], tx.pos[1], rx.pos[0] - tx.pos[0], rx.pos[1] - link.tx.pos[1],
                          shape='full', length_includes_head=True, color=cmap(snr_val), alpha=snr_val, width=5)

        # Plot nodes that are not idle.
        self.plot_nodes(ax1)
        # Plot the boundary of legal region.
        self.add_boundary(ax1)
        ax1.set_title("SNR")

        # ============ Sub-figure 2: Plot interference signals. ============
        ax2 = axs[1]
        ax2.set_aspect('equal')

        for flow in self.flows:
            for link in flow.route:
                tx, rx = link.tx, link.rx
                for inf_node in self.nodes:
                    if (inf_node not in {tx, rx}) and inf_node.p_tx[link.chan_idx] > 0:
                        if self.p_rx[rx.nid, inf_node.nid, link.chan_idx] > self.n0 * self.bw:
                            inr_val = normalize_snr(self.p_rx[rx.nid, inf_node.nid, link.chan_idx])
                            ax2.arrow(inf_node.pos[0], inf_node.pos[1],
                                      rx.pos[0] - inf_node.pos[0], rx.pos[1] - inf_node.pos[1],
                                      shape='left', length_includes_head=True, width=5,
                                      alpha=inr_val, color=cmap(inr_val))

        # Plot nodes that are not idle.
        self.plot_nodes(ax2)
        # Plot the boundary of legal region.
        self.add_boundary(ax2)
        ax2.set_title("INR")

        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(snr_min, snr_max), cmap=cmap), ax=ax2,
                     fraction=0.05, pad=0.05, label="SNR/INR in dB")

        # ============ Show/save figure. ============
        # Display the image.
        if show_img:
            plt.show()

        # Write results to disk.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig_name = 'snr_inr.pdf'
            if tag is not None:
                fig_name = tag + '_' + fig_name
            fig_path = osp.join(save_dir, fig_name)
            plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    from components.misc import get_random_actions
    env = AdHocEnv('cls-500')
    env.reset()
    terminated = False
    while not terminated:
        avail_actions = env.get_avail_actions()
        print(f"At n{env.agent.front.nid}, nbrs: {[nbr.nid for nbr in env.nbrs]}")
        rand_action = get_random_actions(avail_actions)
        _, terminated, _ = env.step(rand_action)
    print(env.agent)
    env.save_replay(save_dir='./')

    # shared_obs = env.get_shared_obs_relations()
    # for k, v in shared_obs.items():
    #     print(f"shared_obs[{k}] = \n{v}.")
