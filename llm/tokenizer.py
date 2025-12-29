# llm/tokenizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
import json
import os

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

@dataclass
class CharTokenizer:
    stoi: Dict[str, int]          # string -> id
    itos: List[str]               # id -> string
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        extra_chars: Optional[Iterable[str]] = None,
        keep_whitespace: bool = True,
    ) -> "CharTokenizer":
        """
        Build a character-level vocabulary from a collection of strings.

        - keep_whitespace=True keeps spaces/newlines/tabs in the vocab (recommended).
        - extra_chars lets you force-include characters that may be rare but important.
        """
        char_set = set()
        for t in texts:
            if not keep_whitespace:
                t = " ".join(t.split())
            char_set.update(list(t))

        if extra_chars is not None:
            char_set.update(list(extra_chars))

        # Deterministic ordering: special tokens first, then sorted chars
        itos = list(SPECIAL_TOKENS) + sorted(char_set)
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_files(
        cls,
        paths: List[str],
        encoding: str = "utf-8",
        extra_chars: Optional[Iterable[str]] = None,
        keep_whitespace: bool = True,
        max_chars_per_file: Optional[int] = None,
    ) -> "CharTokenizer":
        """
        Build vocab from text files. max_chars_per_file can cap reading for speed.
        """
        texts = []
        for p in paths:
            with open(p, "r", encoding=encoding, errors="replace") as f:
                if max_chars_per_file is None:
                    texts.append(f.read())
                else:
                    texts.append(f.read(max_chars_per_file))
        return cls.from_texts(texts, extra_chars=extra_chars, keep_whitespace=keep_whitespace)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.stoi[self.bos_token])
        unk_id = self.stoi[self.unk_token]
        for ch in text:
            ids.append(self.stoi.get(ch, unk_id))
        if add_eos:
            ids.append(self.stoi[self.eos_token])
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def token_id(self, token: str) -> int:
        return self.stoi[token]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "itos": self.itos,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        itos = payload["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_token=payload.get("pad_token", "<pad>"),
            bos_token=payload.get("bos_token", "<bos>"),
            eos_token=payload.get("eos_token", "<eos>"),
            unk_token=payload.get("unk_token", "<unk>"),
        )
