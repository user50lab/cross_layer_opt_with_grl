import random
from collections import deque

from components.misc import *


class ReplayBuffer:
    """Replay buffer storing sequences of transitions."""

    ### 初始化经验缓冲区，用于在强化学习中存储智能体与环境交互的经验。
    def __init__(self, args):

        # 创建集合
        self.pre_decision_fields = set(args.pre_decision_fields)  # Field names before agent decision   # 创建了集合 self.pre_decision_fields，包含所有在代理（agent）做出决策之前需要考虑的字段名称。它使用了 set 函数来确保字段唯一。
        self.post_decision_fields = set(args.post_decision_fields)  # Field names after agent decision   # 创建了集合 self.post_decision_fields，包含所有在代理做出决策之后需要考虑的字段。
        self.fields = self.pre_decision_fields.union(self.post_decision_fields)  # Overall fields   # 将前决策和后决策的字段合并在一起，形成一个总的字段集合 self.fields。
        self.fields.add('filled')   # 然后，加入一个叫做 'filled' 的字段到这个总集合中。

        self.capacity = args.buffer_size  # Total number of data sequences that can be held by memory   # 内存能够容纳的数据序列的总数。
        self.memory = deque(maxlen=self.capacity)  # Memory holding samples   # 双端队列，用于存储样本，maxlen 参数确保队列的最大长度不会超过容量。
        self.data_chunk_len = args.data_chunk_len  # Maximum length of data sequences   # 设置每个数据序列可容纳的最大长度 self.data_chunk_len

        self.sequence = None  # Data sequence holding up-to-date transitions   # 用来持有最新的转换（transitions）。
        self.ptr = None  # Recorder of data sequence length   # 用来记录数据序列长度的记录器。
        self._reset_sequence()   # 重置数据序列，准备接收新的数据。

    def _reset_sequence(self) -> None:
        """cleans up the data sequence."""
        self.sequence = {k: [] for k in self.fields}   # 字典的键是由 self.fields 集合中的元素定义的，而每个键对应的值都是一个空列表。
        self.ptr = 0

    ### 将一系列的变换（transitions）存储到内存（或者说缓冲区）中。
    def insert(self, transition):
        """Stores a transition into memory. A transition is first held by data sequence.
        When maximum length is reached, contents of data sequence is stored to memory.
        """

        # When maximum sequence length is reached,
        if self.ptr == self.data_chunk_len:   # 检查当前数据序列中的transition个数（由 self.ptr 记录）是否已达到设定的最大长度 self.data_chunk_len。
            # Append the pre-decision data beyond the last timestep to data sequence.
            for k in self.pre_decision_fields:   # 当数据序列达到最大长度后，此循环遍历所有决策前的字段（预决策字段集合 self.pre_decision_fields）。
                self.sequence[k].append(transition.get(k, ''))   # 对于 self.pre_decision_fields 中的每一个字段 k，该行代码将尝试从当前的变换（transition 字典）中获取该字段的值并添加到数据序列 self.sequence 的相应字段列表中。如果 transition 中不存在字段 k，则默认添加空字符串 ''。
            # Move data sequence to memory.
            self.memory.append(self.sequence)   # 将已满的数据序列加入到内存（self.memory）中，它是一个双端队列。
            # Clear the sequence and reset pointer.
            self._reset_sequence()   # 调用内部方法以清除当前的数据序列并重置指针，为下一组数据序列做准备。
            # Pseudo transition is no longer added to the beginning of next sequence.
            if not transition.get('filled'):   # 如果当前的变换（transition）在 'filled' 字段中的值为 False 或不存在，那么直接返回，不会执行后续添加变换到序列的操作。
                return

        # Store data specified by fields.
        # Note that pseudo transition is stored if not appended to the end of sequence.
        for k, v in transition.items():   # 对于变换（transition）中的每个字段 k 和值 v，都会尝试将它添加到数据序列中。
            if k in self.fields:   # 如果字段 k 在 self.fields 集合中，这表示它是有效的字段，需要被保存。
                self.sequence[k].append(v)   # 将值 v 添加到数据序列 self.sequence 中相应字段 k 的列表中。
        self.ptr += 1  # A complete transition is stored.   # 更新指针 self.ptr，自增1表示一个完整的变换已被存储到数据序列中。

    def recall(self, batch_size: int):
        """Selects a random batch of samples."""
        assert len(self) >= batch_size, "Samples are insufficient."
        samples = random.sample(self.memory, batch_size)  # List of samples
        batched_samples = {k: [] for k in self.fields}  # Dict holding batch of samples.

        # Construct input sequences.
        for t in range(self.data_chunk_len):
            for k in self.fields:
                batched_samples[k].append(cat([samples[idx][k][t] for idx in range(batch_size)]))

        # Add pre-decision data beyond the last timestep for bootstrapping.
        for k in self.pre_decision_fields:
            batched_samples[k].append(cat([samples[idx][k][self.data_chunk_len] for idx in range(batch_size)]))

        return batched_samples

    def can_sample(self, batch_size: int) -> bool:
        """Whether sufficient samples are available."""
        return batch_size <= len(self)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return f"ReplayBuffer, holding {len(self)}/{self.capacity} sequences."


if __name__ == '__main__':
    a = {'apple'}
    b = {'pear', 'banana', 'apple'}
    buffer = ReplayBuffer(a, b, 10, 10)

    a = [1, 2]
    from typing import Iterable
    print(isinstance(a, Iterable))
