import os
import pickle
from typing import Sequence, Tuple, TypeVar, Optional, List
from langchain.storage import RedisStore

# Assuming RedisStore is already imported
K = TypeVar("K")
V = TypeVar("V")


class VF_Redis(RedisStore):
    def __init__(self, redis_url: str):
        # Initialize the RedisStore with the provided URL
        super().__init__(redis_url=redis_url)

    def mget(self, keys: Sequence[K]) -> List[Optional[V]]:
        pickled_values = super().mget(keys)
        return [pickle.loads(v) if v is not None else None for v in pickled_values]

    def mset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        pickled_key_value_pairs = [(k, pickle.dumps(v)) for k, v in key_value_pairs]
        super().mset(pickled_key_value_pairs)

    def mdelete(self, keys: Sequence[K]) -> None:
        super().mdelete(keys)
