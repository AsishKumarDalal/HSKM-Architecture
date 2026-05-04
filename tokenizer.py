"""
BPE Tokenizer using tiktoken (GPT-2 compatible).
Provides byte-pair encoding for generalized language modeling.
"""

import tiktoken
from typing import List, Optional


class BPETokenizer:
    """
    BPE Tokenizer using the 'gpt2' encoding from tiktoken.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.special_tokens = {
            "<pad>": self.enc.max_token_value + 1,
            "<bos>": self.enc.max_token_value + 2,
            "<eos>": self.enc.eot_token,
        }
        # In tiktoken, we can use the existing eot_token for EOS
        self._pad_id = self.special_tokens["<pad>"]
        self._bos_id = self.special_tokens["<bos>"]
        self._eos_id = self.special_tokens["<eos>"]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        ids = self.enc.encode(text)
        if add_bos:
            ids = [self._bos_id] + ids
        if add_eos:
            ids = ids + [self._eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        # Filter out special tokens manually if they aren't in the base encoding
        specials = {self._pad_id, self._bos_id, self._eos_id}
        filtered_ids = [idx for idx in ids if idx not in specials] if skip_special else ids
        return self.enc.decode(filtered_ids)

    @property
    def vocab_size(self) -> int:
        # GPT-2 base is 50257. Plus our custom specials.
        return self.enc.n_vocab + 2 

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def save(self, path: str):
        # tiktoken encodings are usually fixed/downloaded, 
        # but we can save a dummy or metadata if needed.
        pass

    def load(self, path: str):
        pass
