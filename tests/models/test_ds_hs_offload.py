from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import transformers

from lm_eval.api.registry import get_model
from lm_eval.models import MODEL_MAPPING
from lm_eval.models.ds_hs_offload import HSOffloadHFLM
import lm_eval.models.ds_hs_offload as ds_hs_offload_module
import lm_eval.models.huggingface as huggingface_module


def _make_fake_model():
    config = transformers.GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=32)
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.name_or_path = "fake-model"
    return model


def _make_fake_tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "hello": 3,
            },
            unk_token="<pad>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.name_or_path = "fake-tokenizer"
    return tokenizer


def test_hs_offload_model_is_registered():
    expected_target = "lm_eval.models.ds_hs_offload:HSOffloadHFLM"
    assert MODEL_MAPPING["hs-offload-hf"] == expected_target
    assert MODEL_MAPPING["ds_hs_offload"] == expected_target
    assert get_model("hs-offload-hf") is HSOffloadHFLM
    assert get_model("ds_hs_offload") is HSOffloadHFLM


def test_hs_offload_init_uses_custom_build_path(monkeypatch):
    build_calls = {}
    fake_model = _make_fake_model()
    fake_engine = SimpleNamespace(module=fake_model, hs_runtime=SimpleNamespace(reset_profiler=lambda: None))
    fake_hf_ds_config = object()
    fake_hs_config = object()
    fake_tokenizer = _make_fake_tokenizer()

    def fake_build(cls, pretrained, **kwargs):
        build_calls["pretrained"] = pretrained
        build_calls["kwargs"] = kwargs
        return fake_engine, fake_model, fake_hf_ds_config, fake_hs_config

    def fail_create_model(*args, **kwargs):
        raise AssertionError("HSOffloadHFLM should not call HFLM._create_model().")

    monkeypatch.setattr(
        HSOffloadHFLM,
        "_build_hs_offload_model",
        classmethod(fake_build),
    )
    monkeypatch.setattr(huggingface_module.HFLM, "_create_model", fail_create_model)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fake_tokenizer,
    )
    monkeypatch.setattr(
        huggingface_module,
        "configure_pad_token",
        lambda tokenizer, model_config=None: tokenizer,
    )

    lm = HSOffloadHFLM(
        pretrained="unit/test-model",
        tokenizer="unit/test-tokenizer",
        device="cpu",
        dtype="float16",
        batch_size=1,
        hs_offload=True,
        hs_max_gen_len=2048,
        hs_device="cpu",
        hs_codec_name="int8",
        ds_batch_size=2,
        cpu_offload=True,
        offload_dir="/tmp/hs-offload-test",
        pin_memory=True,
        quant_bits=8,
        quant_group_size=128,
    )

    assert build_calls["pretrained"] == "unit/test-model"
    assert build_calls["kwargs"]["hs_max_gen_len"] == 2048
    assert build_calls["kwargs"]["hs_device"] == "cpu"
    assert build_calls["kwargs"]["hs_codec_name"] == "int8"
    assert build_calls["kwargs"]["ds_batch_size"] == 2
    assert build_calls["kwargs"]["cpu_offload"] is True
    assert build_calls["kwargs"]["offload_dir"] == "/tmp/hs-offload-test"
    assert build_calls["kwargs"]["pin_memory"] is True
    assert build_calls["kwargs"]["quant_bits"] == 8
    assert build_calls["kwargs"]["quant_group_size"] == 128
    assert lm._ds_engine is fake_engine
    assert lm._hf_ds_config is fake_hf_ds_config
    assert lm._hs_config is fake_hs_config
    assert lm._hs_runtime is fake_engine.hs_runtime
    assert lm._model is fake_engine.module
    assert lm.model is fake_engine.module
    assert lm.tokenizer is fake_tokenizer
    assert lm.config is fake_model.config
    assert lm.pretrained == "unit/test-model"
    assert lm.device == torch.device("cpu")


def test_hs_offload_adapter_does_not_import_run_model_script():
    source = inspect.getsource(ds_hs_offload_module)
    assert "zero_inference.run_model" not in source
    assert "run_model.py" not in source


def test_hs_offload_deepspeed_config_supports_quant_bits():
    ds_config = HSOffloadHFLM._build_deepspeed_config(
        hidden_size=8,
        dtype=torch.float16,
        batch_size=1,
        cpu_offload=False,
        disk_offload=False,
        offload_dir=None,
        pin_memory=False,
        quant_bits=8,
        quant_group_size=128,
    )

    quant_init = ds_config["weight_quantization"]["quantized_initialization"]
    assert quant_init["num_bits"] == 8
    assert quant_init["group_size"] == 128


def test_hs_offload_kv_offload_setup_uses_generation_budget():
    calls = []
    fake_model = SimpleNamespace(
        set_kv_cache_offload=lambda *args: calls.append(args),
    )
    lm = object.__new__(HSOffloadHFLM)
    lm._model = fake_model
    lm._kv_offload_enabled = True
    lm._pin_kv_cache = True
    lm._async_kv_offload = True

    lm._maybe_enable_kv_offload(
        max_length=12,
        context=torch.zeros(1, 5, dtype=torch.long),
    )

    assert calls == [(True, 7, True, True)]
