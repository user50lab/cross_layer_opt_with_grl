r"""Code blocks shared across envs"""

from abc import abstractmethod

from random import sample
from itertools import product, repeat

import numpy as np
from numpy import ndarray

from envs.multi_agent_env import MultiAgentEnv

# --- Math ---


def get_one_hot(x: int, n: int) -> ndarray:
    """Gets one-hot encoding of integer."""
    v = np.zeros(n, dtype=np.float32)
    v[x] = 1
    return v

### 在一个n维的盒子中非重复地选择元素。
def select_from_box(n_els: int, min_val: int, max_val: int, n_dims: int) -> ndarray:   # 接受四个参数：n_els是要选择的元素的数量，min_val是盒子中每个维度的最小值，max_val是盒子中每个维度的最大值，n_dims是盒子的维数。函数将返回一个NumPy数组(ndarray)。
    """Selects non-repetitive elements from a box."""
    legal_points = list(product(*list(repeat(np.arange(min_val, max_val), n_dims))))
    '''
    np.arange(min_val, max_val)：创建一个数组，包含从min_val（包括）到max_val（不包括）的整数序列。
    repeat(np.arange(min_val, max_val), n_dims)：重复上面创建的数组n_dims次，为每个维度创建一个相同的范围。
    product(*list(...))：计算笛卡尔积，为n_dims维度的盒子中的每个点创建一个坐标。*是解包操作符，它将列表中的元素作为单独的参数传递给product函数。
    list(...)：将笛卡尔积结果转换成一个列表，这个列表包含了盒子中所有可能的点的坐标。
    '''
    return np.array(sample(legal_points, n_els), dtype=np.float32)
    '''
    sample(legal_points, n_els)：从legal_points列表中随机选择n_els个不重复的元素。这使用了random.sample函数。
    np.array(..., dtype=np.float32)：将选择的坐标点转换成一个NumPy数组，并设置数组的数据类型为float32。
    '''


def compute_jain_fairness_index(x: ndarray) -> float:
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1


def build_discrete_moves(n_dirs: int, v_max: float, v_levels: int) -> ndarray:
    """Builds discrete moves from kronecker product of directions and velocities."""
    move_amounts = []
    v = v_max
    for i in range(v_levels):
        move_amounts.append(v)
        v /= 2
    move_amounts = np.array(move_amounts).reshape(-1, 1)  # Amounts of movement in each timestep (m)
    ang = 2 * np.pi * np.arange(n_dirs) / n_dirs  # Possible flying angles
    move_dirs = np.stack([np.cos(ang), np.sin(ang)]).T  # Moving directions of UBSs
    avail_moves = np.concatenate((np.zeros((1, 2)), np.kron(move_amounts, move_dirs)))  # Available moves
    return avail_moves


# --- Recorder ---

class Recorder(object):
    """Tool to record condition of env at each timestep."""

    def __init__(self, env: MultiAgentEnv, variables) -> None:
        self.env = env
        self.variables = variables
        self.film = {k: [] for k in self.variables}  # A dict to hold records of variables

    def __getattr__(self, item):
        if item == '__setstate__':
            raise AttributeError(item)
        else:
            return getattr(self.env, item)

    def reload(self) -> None:
        """Clears the film to prepare for new episode."""
        self.film = {k: [] for k in self.variables}

    def click(self) -> None:
        """Takes the snapshot at each timestep."""
        for k in self.variables:
            v = self.__getattr__(k)
            if isinstance(v, ndarray):
                self.film[k].append(v.copy())
            else:
                self.film[k].append(v)

    @ abstractmethod
    def replay(self, **kwargs):
        """Replays the entire episode from recording."""
        raise NotImplementedError


# --- Functions to plot simple objects ---


def plot_line(a: ndarray, b: ndarray) -> list[ndarray]:
    """Plots a line from point a to point b."""
    assert a.shape == b.shape, "Inconsistent dimension between a and b."
    return [np.linspace(a[d], b[d], 50) for d in range(a.size)]


def plot_circ(x_o: ndarray, y_o: ndarray, r: float) -> tuple[ndarray, ndarray]:
    """Plots a circle centered at given origin (x_0, y_o) with radius r."""
    assert x_o.shape == x_o.shape, "Inconsistent dimension between x_o and y_o."
    t = np.linspace(0, 2 * np.pi, 100)
    x_data, y_data = r * np.cos(t), r * np.sin(t)
    return x_o + x_data, y_o + y_data


def plot_segments(points: list[ndarray]):
    """Plots segments between points."""
    for p in points:
        assert p.ndim == 1 and p.size == 2, "Invalid shape of points."

    x, y = [], []
    for s in range(len(points) - 1):
        seg = plot_line(points[s], points[s + 1])
        x.append(seg[0])
        y.append(seg[1])
    return np.concatenate(x), np.concatenate(y)


def plot_boundary(range_pos: float, symmetric: bool = False):
    if not symmetric:
        b = plot_segments([np.array([0, 0]), np.array([range_pos, 0]), np.array([range_pos, range_pos]),
                           np.array([0, range_pos]), np.array([0, 0])])
    else:
        b = plot_segments([np.array([-range_pos, -range_pos]), np.array([-range_pos, range_pos]),
                           np.array([range_pos, range_pos]), np.array([range_pos, -range_pos]),
                           np.array([-range_pos, -range_pos])])
    return b
