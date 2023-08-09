# -*- coding:utf-8  -*-
# Time  : 2021/12/8 下午3:45
# Author: Yahui Cui


from abc import ABCMeta, abstractmethod
from functools import reduce
from pathlib import Path

# from gfootball.env import player_base
from typing import Dict, Any, Tuple, Callable, List, Sequence
from gym import spaces
# from gfootball.env import football_action_set


import functools
import os
import pickle
import numpy as np
import torch
import copy
import gym
import torch.nn as nn
import torch.nn.functional as F
import operator
import warnings

DataTransferType = np.ndarray
ModelConfig = Dict[str, Any]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def _get_batched(data: Any):
    """Get batch dim, nested data must be numpy array like"""

    res = []
    if isinstance(data, Dict):
        for k, v in data.items():
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i > len(res):
                    res[i] = {}
                res[i][k] = e
    elif isinstance(data, Sequence):
        for v in data:
            cleaned_v = _get_batched(v)
            for i, e in enumerate(cleaned_v):
                if i > len(res):
                    res[i] = []
                res[i].append(e)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unexpected nested data type: {type(data)}")


class Preprocessor(metaclass=ABCMeta):
    def __init__(self, space: spaces.Space):
        self._original_space = space

    @abstractmethod
    def transform(self, data, nested=False) -> DataTransferType:
        """Transform original data to feet the preprocessed shape. Nested works for nested array."""
        pass

    @abstractmethod
    def write(self, array: DataTransferType, offset: int, data: Any):
        pass

    @property
    def size(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            self.shape,
            dtype=np.float32,
        )


class DictFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Dict):
        assert isinstance(space, spaces.Dict), space
        super(DictFlattenPreprocessor, self).__init__(space)
        self._preprocessors = {}

        for k, _space in space.spaces.items():
            self._preprocessors[k] = get_preprocessor(_space)(_space)

        self._size = sum([prep.size for prep in self._preprocessors.values()])

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return self._size

    def transform(self, data, nested=False) -> DataTransferType:
        """Transform support multi-instance input"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, Dict):
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        elif isinstance(data, Sequence):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, dict):
            for k, _data in sorted(data.items()):
                size = self._preprocessors[k].size
                array[offset: offset + size] = self._preprocessors[k].transform(_data)
                offset += size
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class TupleFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Tuple):
        assert isinstance(space, spaces.Tuple), space
        super(TupleFlattenPreprocessor, self).__init__(space)
        self._preprocessors = []
        for k, _space in space.spaces:
            self._preprocessors.append(get_preprocessor(_space)(_space))
        self._size = sum([prep.size for prep in self._preprocessors])

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self.size,)

    def transform(self, data, nested=False) -> DataTransferType:
        if nested:
            data = _get_batched(data)

        if isinstance(data, List):
            array = np.zeros((len(data),) + self.shape)
            for i in range(len(array)):
                self.write(array[i], 0, data[i])
        else:
            array = np.zeros(self.shape)
            self.write(array, 0, data)
        return array

    def write(self, array: DataTransferType, offset: int, data: Any):
        if isinstance(data, Tuple):
            for _data, prep in zip(data, self._preprocessors):
                array[offset: offset + prep.size] = prep.transform(_data)
        else:
            raise TypeError(f"Unexpected type: {type(data)}")


class BoxFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Box):
        super(BoxFlattenPreprocessor, self).__init__(space)
        self._size = reduce(operator.mul, space.shape)

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        if nested:
            data = _get_batched(data)

        if isinstance(data, list):
            array = np.stack(data)
            array = array.reshape((len(array), -1))
            return array
        else:
            array = np.asarray(data).reshape((-1,))
            return array

    def write(self, array, offset, data):
        pass


class DiscreteFlattenPreprocessor(Preprocessor):
    def __init__(self, space: spaces.Discrete):
        super(DiscreteFlattenPreprocessor, self).__init__(space)
        self._size = space.n

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def transform(self, data, nested=False) -> np.ndarray:
        """Transform to one hot"""

        if nested:
            data = _get_batched(data)

        if isinstance(data, int):
            array = np.zeros(self.size, dtype=np.int32)
            array[data] = 1
            return array
        else:
            raise TypeError(f"Unexpected type: {type(data)}")

    def write(self, array, offset, data):
        pass


class Mode:
    FLATTEN = "flatten"
    STACK = "stack"


def get_preprocessor(space: spaces.Space, mode: str = Mode.FLATTEN):
    if mode == Mode.FLATTEN:
        if isinstance(space, spaces.Dict):
            # logger.debug("Use DictFlattenPreprocessor")
            return DictFlattenPreprocessor
        elif isinstance(space, spaces.Tuple):
            # logger.debug("Use TupleFlattenPreprocessor")
            return TupleFlattenPreprocessor
        elif isinstance(space, spaces.Box):
            # logger.debug("Use BoxFlattenPreprocessor")
            return BoxFlattenPreprocessor
        elif isinstance(space, spaces.Discrete):
            return DiscreteFlattenPreprocessor
        else:
            raise TypeError(f"Unexpected space type: {type(space)}")
    elif mode == Mode.STACK:
        raise NotImplementedError
    else:
        raise ValueError(f"Unexpected mode: {mode}")


class Error(Exception):
    pass


class RepeatedAssignError(Error):
    """Raised when repeated assign value to a not-None dict"""

    pass


class SimpleObject:
    def __init__(self, obj, name):
        assert hasattr(obj, name), f"Object: {obj} has no such attribute named `{name}`"
        self.obj = obj
        self.name = name

    def load_state_dict(self, v):
        setattr(self.obj, self.name, v)

    def state_dict(self):
        value = getattr(self.obj, self.name)
        return value


DEFAULT_MODEL_CONFIG = {
    "actor": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": "Softmax"},
    },
    "critic": {
        "network": "mlp",
        "layers": [
            {"units": 64, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ],
        "output": {"activation": False},
    },
}


class Policy(metaclass=ABCMeta):
    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: ModelConfig = None,
            custom_config: Dict[str, Any] = None,
    ):
        """Create a policy instance.

        :param str registered_name: Registered policy name.
        :param gym.spaces.Space observation_space: Raw observation space of related environment agent(s), determines
            the model input space.
        :param gym.spaces.Space action_space: Raw action space of related environment agent(s).
        :param Dict[str,Any] model_config: Model configuration to construct models. Default to None.
        :param Dict[str,Any] custom_config: Custom configuration, includes some hyper-parameters. Default to None.
        """

        self.registered_name = registered_name
        self.observation_space = observation_space
        self.action_space = action_space

        self.custom_config = {
            "gamma": 0.98,
            "use_cuda": False,
            "use_dueling": False,
            "preprocess_mode": Mode.FLATTEN,
        }
        self.model_config = DEFAULT_MODEL_CONFIG

        if custom_config is None:
            custom_config = {}
        self.custom_config.update(custom_config)

        # FIXME(ming): use deep update rule
        if model_config is None:
            model_config = {}
        self.model_config.update(model_config)

        self.preprocessor = get_preprocessor(
            observation_space, self.custom_config["preprocess_mode"]
        )(observation_space)

        self._state_handler_dict = {}
        self._actor = None
        self._critic = None
        self._exploration_callback = None

    @property
    def exploration_callback(self) -> Callable:
        return self._exploration_callback

    def register_state(self, obj: Any, name: str) -> None:
        """Register state of obj. Called in init function to register model states.

        Example:
            >>> class CustomPolicy(Policy):
            ...     def __init__(
            ...         self,
            ...         registered_name,
            ...         observation_space,
            ...         action_space,
            ...         model_config,
            ...         custom_config
            ...     ):
            ...     # ...
            ...     actor = MLP(...)
            ...     self.register_state(actor, "actor")

        :param Any obj: Any object, for non `torch.nn.Module`, it will be wrapped as a `Simpleobject`.
        :param str name: Humanreadable name, to identify states.
        :raise: malib.utils.errors.RepeatedAssign
        :return: None
        """

        if not isinstance(obj, nn.Module):
            obj = SimpleObject(self, name)
        if self._state_handler_dict.get(name, None) is not None:
            raise RepeatedAssignError(
                f"state handler named with {name} is not None."
            )
        self._state_handler_dict[name] = obj

    def deregister_state(self, name: str):
        if self._state_handler_dict.get(name) is None:
            print(f"No such state tagged with: {name}")
        else:
            self._state_handler_dict.pop(name)
            print(f"Deregister state tagged with: {name}")

    @property
    def description(self):
        """Return a dict of basic attributes to identify policy.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }

    @abstractmethod
    def compute_actions(
            self, observation: DataTransferType, **kwargs
    ) -> DataTransferType:
        """Compute batched actions for the current policy with given inputs.

        Legal keys in kwargs:

        - behavior_mode: behavior mode used to distinguish different behavior of compute actions.
        - action_mask: action mask.
        """

        pass

    @abstractmethod
    def compute_action(
            self, observation: DataTransferType, **kwargs
    ) -> Tuple[Any, Any, Any]:
        """Compute single action when rollout at each step, return 3 elements:
        action, None, extra_info['actions_prob']
        """

        pass

    def state_dict(self):
        """Return state dict in real time"""

        res = {
            k: copy.deepcopy(v).cpu().state_dict()
            if isinstance(v, nn.Module)
            else v.state_dict()
            for k, v in self._state_handler_dict.items()
        }
        return res

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict outside.

        Note that the keys in `state_dict` should be existed in state handler.

        :param state_dict: Dict[str, Any], A dict of state dict
        :raise: KeyError
        """

        for k, v in state_dict.items():
            self._state_handler_dict[k].load_state_dict(v)

    def set_weights(self, parameters: Dict[str, Any]):
        """Set parameter weights.

        :param parameters: Dict[str, Any], A dict of parameters.
        :return:
        """

        for k, v in parameters.items():
            # FIXME(ming): strict mode for parameter reload
            self._state_handler_dict[k].load_state_dict(v)

    def set_actor(self, actor) -> None:
        """Set actor. Note repeated assign will raise a warning

        :raise RuntimeWarning, repeated assign.
        """

        # if self._actor is not None:
        #     raise RuntimeWarning("repeated actor assign")
        self._actor = actor

    def set_critic(self, critic):
        """Set critic"""

        # if self._critic is not None:
        #     raise RuntimeWarning("repeated critic assign")
        self._critic = critic

    def set_meta(self, meta) -> None:
        """Set actor. Note repeated assign will raise a warning

        :raise RuntimeWarning, repeated assign.
        """

        # if self._actor is not None:
        #     raise RuntimeWarning("repeated actor assign")
        self._meta = meta

    @property
    def actor(self) -> Any:
        """Return policy, cannot be None"""

        return self._actor

    @property
    def critic(self) -> Any:
        """Return critic, can be None"""

        return self._critic

    @property
    def meta(self) -> Any:
        """Return critic, can be None"""

        return self._meta

    @deprecated
    def train(self):
        pass

    @deprecated
    def eval(self):
        pass


def mlp(layers_config):
    layers = []
    for j in range(len(layers_config) - 1):
        tmp = [nn.Linear(layers_config[j]["units"], layers_config[j + 1]["units"])]
        if layers_config[j + 1].get("activation"):
            tmp.append(getattr(torch.nn, layers_config[j + 1]["activation"])())
        layers += tmp
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: Dict[str, Any],
    ):
        super(MLP, self).__init__()

        obs_dim = get_preprocessor(observation_space)(observation_space).size
        act_dim = get_preprocessor(action_space)(action_space).size
        layers_config: list = (
            self._default_layers()
            if model_config.get("layers") is None
            else model_config["layers"]
        )
        layers_config.insert(0, {"units": obs_dim})
        layers_config.append(
            {"units": act_dim, "activation": model_config["output"]["activation"]}
        )
        self.net = mlp(layers_config)

    def _default_layers(self):
        return [
            {"units": 256, "activation": "ReLU"},
            {"units": 64, "activation": "ReLU"},
        ]

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        pi = self.net(obs)
        return pi


class RNN(nn.Module):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: Dict[str, Any],
    ):
        super(RNN, self).__init__()
        self.hidden_dims = (
            64 if model_config is None else model_config.get("rnn_hidden_dim", 64)
        )

        # default by flatten
        obs_dim = get_preprocessor(observation_space)(observation_space).size()
        act_dim = get_preprocessor(action_space)(action_space).size()

        self.fc1 = nn.Linear(obs_dim, self.hidden_dims)
        self.rnn = nn.GRUCell(self.hidden_dims, self.hidden_dims)
        self.fc2 = nn.Linear(self.hidden_dims, act_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dims).zero_()

    def forward(self, obs, hidden_state):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.hidden_dims)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


def get_model(model_config: Dict[str, Any]):
    model_type = model_config["network"]

    if model_type == "mlp":
        handler = MLP
    elif model_type == "rnn":
        handler = RNN
    elif model_type == "cnn":
        raise NotImplementedError
    elif model_type == "rcnn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    def builder(observation_space, action_space, use_cuda=False):
        model = handler(observation_space, action_space, copy.deepcopy(model_config))
        if use_cuda:
            model.cuda()
        return model

    return builder


def init_fc_weights(m, init_method, gain=1.0):
    init_method(m.weight.data, gain=gain)
    nn.init.constant_(m.bias.data, 0)


class MAPPO(Policy):
    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            model_config: Dict[str, Any] = None,
            custom_config: Dict[str, Any] = None,
    ):
        super(MAPPO, self).__init__(
            registered_name=registered_name,
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.opt_cnt = 0

        # Todo (Linghui): hypothesize we had only one reward factor
        self.reward_weights_num = 1

        self.register_state(self.opt_cnt, "opt_cnt")

        self._use_q_head = custom_config["use_q_head"]

        actor = get_model(self.model_config["actor"])(
            observation_space, action_space, custom_config.get("use_cuda", False)
        )

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )

        global_observation_space = custom_config["global_state_space"]
        critic = get_model(self.model_config["critic"])(
            global_observation_space,
            action_space if self._use_q_head else gym.spaces.Discrete(1),
            custom_config.get("use_cuda", False),
        )

        meta_network = get_model(self.model_config["actor"])(
            observation_space, gym.spaces.Discrete(5), custom_config.get("use_cuda", False)
        )  # todo: the output units num = 8 which is the reward factor

        use_orthogonal = model_config["initialization"]["use_orthogonal"]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_weights(m):
            if type(m) == nn.Linear:
                init_fc_weights(m, init_method, model_config["initialization"]["gain"])

        actor.apply(init_weights)
        critic.apply(init_weights)
        meta_network.apply(init_weights)

        # register state handler
        self.set_actor(actor)
        self.set_critic(critic)
        self.set_meta(meta_network)

        self.register_state(self._actor, "actor")
        self.register_state(self._critic, "critic")
        self.register_state(self._meta, "meta")

    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy._actor = self_copy._actor.to(device)
        self_copy._critic = self_copy._critic.to(device)
        self_copy._meta = self_copy._meta.to(device)
        return self_copy

    def compute_actions(self, observation, **kwargs):
        raise RuntimeError("Shouldn't use it currently")

    def compute_action(self, observation, **kwargs):
        logits = self.actor(observation)
        illegal_action_mask = torch.FloatTensor(
            1 - observation[..., : logits.shape[-1]]
        ).to(logits.device)
        assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
            illegal_action_mask.max(),
            illegal_action_mask.min(),
        )
        logits = logits - 1e10 * illegal_action_mask
        if "action_mask" in kwargs:
            raise NotImplementedError
        dist = torch.distributions.Categorical(logits=logits)
        extra_info = {}
        action_prob = dist.probs.detach().numpy()  # num_action

        # if observation is not None:
        #     for n in range(logits.shape[0]):
        #         action_prob[n][observation[n][:logits.shape[1]]==0] = 0.
        extra_info["action_probs"] = action_prob
        action = dist.sample().numpy()
        if "share_obs" in kwargs and kwargs["share_obs"] is not None:
            extra_info["value"] = self.critic(kwargs["share_obs"]).detach().numpy()
        return action, action_prob, extra_info

    def compute_reward_weights(self, observation, **kwargs):
        logits = self.meta(observation)
        dist = torch.distributions.Categorical(logits=logits)
        reward_weights = dist.probs.detach().numpy()

        return reward_weights

    def train(self):
        pass

    def eval(self):
        pass

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    # def dump(self, dump_dir):
    #     torch.save(self._actor, os.path.join(dump_dir, "actor.pt"))
    #     torch.save(self._critic, os.path.join(dump_dir, "critic.pt"))
    #     pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))
    #
    # @staticmethod
    # def load(dump_dir):
    #     with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
    #         desc_pkl = pickle.load(f)
    #
    #     res = MAPPO(
    #         desc_pkl["registered_name"],
    #         desc_pkl["observation_space"],
    #         desc_pkl["action_space"],
    #         desc_pkl["model_config"],
    #         desc_pkl["custom_config"]
    #     )
    #
    #     actor = torch.load(os.path.join(dump_dir, "actor.pt"), res.device)
    #     critic = torch.load(os.path.join(dump_dir, "critic.pt"), res.device)
    #
    #     hard_update(res._actor, actor)
    #     hard_update(res._critic, critic)
    #     return res


def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0

    def get_feature_dims(self):
        dims = {
            "player": 29,
            "ball": 18,
            "left_team": 7,
            "left_team_closest": 7,
            "right_team": 7,
            "right_team_closest": 7,
        }
        return dims

    def encode(self, obs):
        player_num = obs["active"]

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail_new(obs, ball_distance)
        # avail = self._get_avail(obs, ball_distance)
        player_state = np.concatenate(
            (
                # avail[2:],
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array(obs["ball_direction"]) * 20,
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )

        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_relative = obs_left_team
        left_team_distance = np.linalg.norm(
            left_team_relative - obs["left_team"][player_num], axis=1, keepdims=True
        )
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(
            obs["left_team_tired_factor"], player_num, axis=0
        ).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                left_team_relative * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance * 2,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        obs_right_team = np.array(obs["right_team"])
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance * 2,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
            "avail": avail,
        }

        return state_dict

    def _get_avail(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            MOVE,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
                -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _get_avail_new(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.03:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
                -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)


class PlayerBase(object):
    """Base player class."""

    def __init__(self, player_config=None):
        self._num_left_controlled_players = 1
        self._num_right_controlled_players = 0
        self._can_play_right = False
        if player_config:
            self._num_left_controlled_players = int(player_config['left_players'])
            self._num_right_controlled_players = int(player_config['right_players'])

    def num_controlled_left_players(self):
        return self._num_left_controlled_players

    def num_controlled_right_players(self):
        return self._num_right_controlled_players

    def num_controlled_players(self):
        return (self._num_left_controlled_players +
                self._num_right_controlled_players)

    def reset(self):
        pass

    def can_play_right(self):
        return self._can_play_right


class Player(PlayerBase):
    """An agent loaded from torch model."""

    def __init__(self, player_config, env_config):
        PlayerBase.__init__(self, player_config)

        self._action_set = (env_config['action_set']
                            if 'action_set' in env_config else 'default')
        self._actor = load_model(player_config['checkpoint'])

    def take_action(self, observation):
        assert len(observation) == 1, 'Multiple players control is not supported'

        feature_encoder = FeatureEncoder()
        observation = feature_encoder.encode(observation[0])
        observation = concate_observation_from_raw(observation)
        # print("-"*20, observation)
        logits = self._actor(observation)
        illegal_action_mask = torch.FloatTensor(
            1 - observation[..., : logits.shape[-1]]
        ).to(logits.device)
        assert illegal_action_mask.max() == 1 and illegal_action_mask.min() == 0, (
            illegal_action_mask.max(),
            illegal_action_mask.min(),
        )
        logits = logits - 1e10 * illegal_action_mask
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().numpy()
        # actions = [football_action_set.action_set_dict[self._action_set][action]]
        return action


def load_model(load_path):
    with open(os.path.join(load_path, "desc.pkl"), "rb") as f:
        desc_pkl = pickle.load(f)
    res = MAPPO(
        desc_pkl["registered_name"],
        desc_pkl["observation_space"],
        desc_pkl["action_space"],
        desc_pkl["model_config"],
        desc_pkl["custom_config"]
    )
    actor = copy.deepcopy(res._actor)
    actor.load_state_dict(torch.load(os.path.join(load_path, "actor_state_dict.pt"), res.device))
    hard_update(res._actor, actor)
    return actor


def concate_observation_from_raw(obs):
    obs_cat = np.hstack(
        [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
    )
    return obs_cat


left_player_config = {'index': 0, 'left_players': 1, 'right_players': 0,
                      'checkpoint': str(Path(__file__).resolve().parent)}
right_player_config = {'index': 0, 'left_players': 0, 'right_players': 1,
                       'checkpoint': str(Path(__file__).resolve().parent)}
env_config = {'action_set': 'full'}



def action_to_list(a):
    if isinstance(a, np.ndarray):
        return a.tolist()
    if not isinstance(a, list):
        return [a]
    return a


# action_set = football_action_set.get_action_set(env_config)
# action_dict = {}
# for action_id, action in enumerate(action_set):
#     action_dict[str(action)] = action_id


def my_controller(observation, action_space, is_act_continuous=False):
    players = []
    for i in range(8):
        if i < 4:
            players.append(Player(left_player_config, env_config))
        else:
            players.append(Player(right_player_config, env_config))

    obs = [observation]
    player_id = observation['controlled_player_index']
    action = players[player_id].take_action(obs)
    # action = action_to_list(action)
    # action_num = action_dict[str(action[0])]
    action_final = [[0] * 19]
    action_final[0][action] = 1
    return action_final
