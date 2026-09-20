import re
import zlib

import numpy as np
import pytest

from ragchat.store import Store

DIM = 256


def fake_embed(texts, kind="document"):
    """Bag-of-words hashed into a fixed size vector: texts that share words end up close together."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


class FakeLLM:
    def __init__(self, reply="The answer is 42 [1]."):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return {"role": "assistant", "content": self.reply}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "index")


@pytest.fixture
def llm():
    return FakeLLM()
