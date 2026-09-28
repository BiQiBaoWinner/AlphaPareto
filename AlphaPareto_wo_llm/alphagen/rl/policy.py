import gymnasium as gym
import json
import math
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn, Tensor
from typing import Any, Dict, List, Optional

from alphagen.data.expression import *
from alphagen.rl.llm_features import get_llm_embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore


class TransformerSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head,
                dim_feedforward=d_ffn, dropout=dropout,
                activation=lambda x: F.leaky_relu(x),               # type: ignore
                batch_first=True, device=device
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=1e-5, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        pad_mask = obs == 0
        src = self._pos_enc(self._token_emb(obs))
        res = self._transformer(src, src_key_padding_mask=pad_mask)
        return res.mean(dim=1)


class LSTMSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs: Tensor) -> Tensor:
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:,:real_len])[0]
        return res.mean(dim=1)

class Enhanced_LSTMSharedNet(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        dropout: float,
        device: torch.device,
        # llm_dim: Optional[int] = None,
        ff_hidden: int = 256,
        llm_model: str = "deepseek-67b-chat",
        llm_endpoint: Optional[str] = None,
        llm_backend: Optional[str] = None,
    ):
        super().__init__(observation_space, d_model)

        if isinstance(observation_space, gym.spaces.Dict):
            lstm_space = observation_space.spaces["original_obs"]
        else:
            lstm_space = observation_space
        assert isinstance(lstm_space, gym.spaces.Box)

        llm_dim_dict = {
            "deepseek-r1:1.5b":1536,
            "deepseek-r1:7b":3584,
            "deepseek-llm-7b-chat":4096,
            "deepseek-r1:14b":5120,
            "deepseek-r1:32b":5120,
            "deepseek-r1:70b":8192
        }
        
        self.device = device
        # self.llm_dim = llm_dim
        if llm_model in llm_dim_dict:
            llm_dim = llm_dim_dict[llm_model]
        else:
            raise ValueError(f"Unknown LLM model for embeddings dimension: {llm_model}")
        self.llm_model = llm_model
        self.llm_endpoint = llm_endpoint
        self.llm_backend = llm_backend
        self._llm_cache: Dict[str, torch.Tensor] = {}
        self._prev_pool_json: Optional[List[str]] = None
        self._prev_llm_batch: Optional[torch.Tensor] = None

        self.lstm_net = LSTMSharedNet(
            observation_space=lstm_space,
            n_layers=n_layers,
            d_model=d_model,
            dropout=dropout,
            device=device,
        )
        # if llm_dim is None:
        #     self.llm_FFN = nn.Sequential(
        #         nn.LazyLinear(ff_hidden),
        #         nn.ReLU(),
        #         nn.Linear(ff_hidden, ff_hidden)
        #     )
        # else:
        #     self.llm_FFN = nn.Sequential(
        #         nn.Linear(llm_dim, ff_hidden),
        #         nn.ReLU(),
        #         nn.Linear(ff_hidden, ff_hidden)
        #     )
        self.llm_FFN = nn.Sequential( # input: 4096 dims
            nn.Linear(llm_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, ff_hidden)
        )
        self.lstm_FFN = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, ff_hidden)
        )
        self.final_FFN = nn.Sequential(
            nn.Linear(ff_hidden, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )

    def _ensure_tensor(self, pool_json: str) -> torch.Tensor:
        if pool_json not in self._llm_cache:
            features = get_llm_embedding(
                pool_json,
                endpoint=self.llm_endpoint,
                model=self.llm_model,
                backend=self.llm_backend,
            )
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
            self._llm_cache[pool_json] = tensor
        return self._llm_cache[pool_json]

    def _decode_pool_batch(self, pool_tensor: torch.Tensor) -> List[str]:
        pool_cpu = pool_tensor.to(torch.uint8).detach().cpu()
        decoded: List[str] = []
        for row in pool_cpu:
            length = int.from_bytes(row[:4].tolist(), byteorder="little")
            if length <= 0:
                decoded.append("{}")
                continue
            payload = bytes(row[4:4 + length].tolist())
            decoded.append(payload.decode("utf-8"))
        return decoded

    def forward(self, observations: Dict[str, Any]) -> Tensor:
        pool_tensor = observations["pool_state"]
        if not isinstance(pool_tensor, torch.Tensor):
            pool_tensor = torch.as_tensor(pool_tensor, device=self.device)

        pool_json_list = self._decode_pool_batch(pool_tensor)

        if (
            self._prev_pool_json is not None
            and self._prev_llm_batch is not None
            and pool_json_list == self._prev_pool_json
        ):
            llm_batch = self._prev_llm_batch
        else:
            llm_batch = torch.stack(
                [self._ensure_tensor(pool_json) for pool_json in pool_json_list],
                dim=0
            ).to(self.device)
            self._prev_pool_json = pool_json_list
            self._prev_llm_batch = llm_batch

        lstm_out = self.lstm_net(observations["original_obs"])
        llm_ffn_out = self.llm_FFN(llm_batch)
        lstm_ffn_out = self.lstm_FFN(lstm_out)

        combined = llm_ffn_out * lstm_ffn_out
        return self.final_FFN(combined)


class Decoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0] + 1                   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        # Actually an encoder for now
        self._decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, device=device
            ),
            n_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        batch_size = obs.size(0)
        begins = torch.full(size=(batch_size, 1), fill_value=self._n_actions,
                            dtype=torch.long, device=obs.device)
        obs = torch.cat((begins, obs.type(torch.long)), dim=1)      # (bs, len)
        pad_mask = obs == 0
        res = self._token_emb(obs)                                  # (bs, len, d_model)
        res = self._pos_enc(res)                                    # (bs, len, d_model)
        res = self._decoder(res, src_key_padding_mask=pad_mask)     # (bs, len, d_model)
        return res.mean(dim=1)                                      # (bs, d_model)
