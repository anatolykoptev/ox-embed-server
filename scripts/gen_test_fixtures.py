#!/usr/bin/env python3
"""Regenerate the tiny ONNX fixtures under tests/fixtures/tiny/.

The weights are random (fixed seed) and the vocabulary is ~100 word pieces, so
every model is well under 1 MB and the whole fixture set can be committed. The
graphs are real BERT encoders run by the real ONNX Runtime, which is what the
worker path needs; they carry no meaning, so tests assert shape, determinism
and batch-independence, not relevance.

Layout matches what each loader expects:
  embed/   model.onnx            input_ids, attention_mask, token_type_ids -> last_hidden_state
  rerank/  model_quantized.onnx  input_ids, attention_mask                 -> logits [b, 1]
  splade/  model.onnx            input_ids, attention_mask                 -> logits [b, seq, vocab]
Each directory also gets a tokenizer.json (WordPiece, [CLS] ... [SEP]).

Usage (pinned versions in scripts/test-fixtures.requirements.txt):
  python3 -I -B scripts/gen_test_fixtures.py tests/fixtures/tiny
"""
import hashlib
import pathlib
import sys

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = (
    "a an and are as at be by for from has hello in inference is it learning "
    "machine models of on onnx open paris passage query runtime the to what "
    "world accelerates capital format france rerank sparse sentence text"
).split()
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789")
MAX_POS = 128


def vocab():
    toks = SPECIALS + sorted(set(WORDS)) + CHARS + ["##" + c for c in CHARS]
    return {t: i for i, t in enumerate(dict.fromkeys(toks))}


def write_tokenizer(path):
    v = vocab()
    tok = Tokenizer(models.WordPiece(v, unk_token="[UNK]", max_input_chars_per_word=100))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", v["[CLS]"]), ("[SEP]", v["[SEP]"])],
    )
    tok.save(str(path))
    return len(v)


def cfg(n_vocab):
    return BertConfig(
        vocab_size=n_vocab,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=MAX_POS,
        initializer_range=0.5,  # wide enough that outputs differ per input
        pad_token_id=0,
        attn_implementation="eager",
    )


class Embed(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state


class NoTti(torch.nn.Module):
    """The reranker and SPLADE loaders feed only input_ids + attention_mask."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def export(module, in_names, out_name, out_dyn, out_path, with_tti):
    ids = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 8, 9, 3]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    args = (ids, mask, torch.zeros_like(ids)) if with_tti else (ids, mask)
    dyn = {n: {0: "batch", 1: "seq"} for n in in_names}
    dyn[out_name] = out_dyn
    torch.onnx.export(
        module.eval(),
        args,
        str(out_path),
        input_names=in_names,
        output_names=[out_name],
        dynamic_axes=dyn,
        opset_version=17,
        dynamo=False,
    )


def main(root):
    root = pathlib.Path(root)
    n_vocab = None
    for sub in ("embed", "rerank", "splade"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        n_vocab = write_tokenizer(root / sub / "tokenizer.json")
    c = cfg(n_vocab)

    torch.manual_seed(1)
    export(Embed(BertModel(c, add_pooling_layer=False)),
           ["input_ids", "attention_mask", "token_type_ids"], "last_hidden_state",
           {0: "batch", 1: "seq"}, root / "embed" / "model.onnx", True)

    torch.manual_seed(2)
    rc = cfg(n_vocab)
    rc.num_labels = 1
    export(NoTti(BertForSequenceClassification(rc)),
           ["input_ids", "attention_mask"], "logits",
           {0: "batch"}, root / "rerank" / "model_quantized.onnx", False)

    torch.manual_seed(3)
    export(NoTti(BertForMaskedLM(cfg(n_vocab))),
           ["input_ids", "attention_mask"], "logits",
           {0: "batch", 1: "seq"}, root / "splade" / "model.onnx", False)

    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".onnx", ".json"):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
