import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Tiny CPU models: a few threads is fastest. On containers (e.g. RunPod) torch can see every host core and
# oversubscribe the CPUs it is actually allowed, making these tests crawl.
torch.set_num_threads(min(4, torch.get_num_threads()))

VOCAB = 50
EOS = VOCAB - 1
PAD = EOS


def make_tiny_lm(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=EOS, eos_token_id=EOS,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture
def tiny_lm():
    return make_tiny_lm(0)


@pytest.fixture
def tiny_lm_pair():
    return make_tiny_lm(0), make_tiny_lm(1)
