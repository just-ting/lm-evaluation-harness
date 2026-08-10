from __future__ import annotations
import atexit
import logging
import os
from contextlib import nullcontext
from typing import Any, Literal
import torch
import transformers
from deepspeed.runtime.hs_offload import profiler as hs_profiler
from packaging.version import parse as vparse
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import get_dtype
from transformers import LogitsProcessor, LogitsProcessorList

class DynamicEOSObserver(LogitsProcessor):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        top_k = 5
        top_logits, top_indices = torch.topk(scores[0], top_k)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices)
        print(f'\n--- Generation Step {self.step} ---')
        for i in range(top_k):
            print(f'Rank {i + 1} | Token: {top_tokens[i]:<10} | ID: {top_indices[i]:<5} | Logit: {top_logits[i]:.4f}')
        return scores
eval_logger = logging.getLogger(__name__)

def _as_bool(value: Any, default: bool=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off', 'none', 'null', ''}:
        return False
    return bool(default)

def _optional_int(value: Any, default: int | None=None) -> int | None:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {'', 'none', 'null', 'na'}:
        return default
    return int(value)

def _hs_profile_enabled() -> bool:
    return bool(hs_profiler.enabled())

def _hs_profile_section(name: str, sync_cuda: bool=False):
    if not hs_profiler.enabled():
        return nullcontext()
    return hs_profiler.section(name, sync_cuda=sync_cuda)

def _hs_profile_set(name: str, value: Any) -> None:
    hs_profiler.set_value(name, value)

def _hs_profile_add(name: str, value: float) -> None:
    hs_profiler.add_value(name, value)

def _hs_profile_count(name: str, amount: float=1.0) -> None:
    hs_profiler.add_counter(name, amount)

def _hs_profile_dump(path: str | None=None) -> None:
    hs_profiler.dump(path)

@register_model('hs-offload-hf', 'ds_hs_offload')
class HSOffloadHFLM(HFLM):
    """HFLM adapter that builds the underlying HF model via DeepSpeed HS offload."""

    def __init__(self, pretrained: str | transformers.PreTrainedModel, backend: Literal['default', 'causal', 'seq2seq']='default', revision: str | None='main', subfolder: str='', tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None=None, truncation: bool | None=False, logits_cache: bool=True, max_length: int | None=None, device: str | None='cuda', dtype: str | torch.dtype | None='auto', softmax_dtype: str | torch.dtype | None=None, mixed_precision_dtype: str | torch.dtype | None=None, batch_size: int | str | None=1, max_batch_size: int | None=64, trust_remote_code: bool | None=False, use_fast_tokenizer: bool | None=True, add_bos_token: bool | None=None, prefix_token_id: int | None=None, parallelize: bool | None=False, max_memory_per_gpu: int | str | None=None, max_cpu_memory: int | str | None=None, offload_folder: str | os.PathLike | None='./offload', peft: str | None=None, delta: str | None=None, autogptq: bool | str | None=False, gptqmodel: bool | None=False, gguf_file: str | None=None, think_end_token: str | int | None=None, enable_thinking: bool | None=None, chat_template_args: dict[str, Any] | None=None, hs_offload: bool=True, hs_max_gen_len: int | None=None, hs_device: str='cpu', hs_codec_name: str='fp16', hs_qbi_stride: int=4, hs_qbi_group_size: int | None=None, hs_qbi_quant_group_size: int=64, hs_qbclerp_stride: int=3, hs_qbclerp_prefix_layers: int=4, hs_qbclerp_quant_group_size: int=32, hs_qbclerp_block_tokens: int=64, hs_qbi_exact_layer: str | None=None, hs_qbi_exact_layers: str | None=None, hs_qbi_rel_l2_threshold: float | None=None, hs_decode_hidden_prefetch_overlap: bool=True, hs_async_hidden_offload: bool=True, hs_async_compression: bool=False, hs_stage_slots: int=2, hs_triton_chunk_size: int=128, hs_block_size: int=64, ds_batch_size: int | None=None, cpu_offload: bool=False, disk_offload: bool=False, offload_dir: str | None=None, pin_memory: bool=False, quant_bits: int=16, quant_group_size: int=64, kv_offload: bool=False, pin_kv_cache: bool=False, async_kv_offload: bool=False, **kwargs) -> None:
        if hs_offload and kv_offload:
            raise ValueError('hs_offload and standard kv_offload cannot be enabled at the same time.')
        self._ds_engine = None
        self._hf_ds_config = None
        self._hs_runtime = None
        self._hs_config = None
        self._hs_profile_dump_registered = False
        self._qbclerp_sample_calibration_info = {'enabled': False}
        self._kv_offload_enabled = _as_bool(kv_offload)
        self._pin_kv_cache = _as_bool(pin_kv_cache)
        self._async_kv_offload = _as_bool(async_kv_offload)
        hs_offload = _as_bool(hs_offload, True)
        hs_decode_hidden_prefetch_overlap = _as_bool(hs_decode_hidden_prefetch_overlap, True)
        hs_async_hidden_offload = _as_bool(hs_async_hidden_offload, True)
        hs_async_compression = _as_bool(hs_async_compression, False)
        hs_stage_slots = int(hs_stage_slots)
        hs_triton_chunk_size = int(hs_triton_chunk_size)
        hs_block_size = int(hs_block_size)
        hs_codec_name = str(hs_codec_name).strip().lower()
        if hs_codec_name not in {'fp16', 'qbi', 'qbi_adaptive', 'qbclerp'}:
            raise ValueError(
                'hs_codec_name must be one of: fp16, qbi, qbi_adaptive, qbclerp.'
            )
        if hs_qbi_group_size not in (None, ''):
            hs_qbi_stride = hs_qbi_group_size
        hs_qbi_stride = int(hs_qbi_stride)
        if hs_qbi_stride < 1:
            raise ValueError('hs_qbi_stride must be >= 1.')
        hs_qbi_quant_group_size = int(hs_qbi_quant_group_size)
        hs_qbclerp_stride = int(hs_qbclerp_stride)
        hs_qbclerp_prefix_layers = int(hs_qbclerp_prefix_layers)
        hs_qbclerp_quant_group_size = int(hs_qbclerp_quant_group_size)
        hs_qbclerp_block_tokens = int(hs_qbclerp_block_tokens)
        if hs_qbclerp_stride < 1:
            raise ValueError('hs_qbclerp_stride must be >= 1.')
        if hs_qbclerp_prefix_layers < 0:
            raise ValueError('hs_qbclerp_prefix_layers must be non-negative.')
        if hs_qbclerp_quant_group_size <= 0:
            raise ValueError('hs_qbclerp_quant_group_size must be positive.')
        if hs_qbclerp_block_tokens <= 0:
            raise ValueError('hs_qbclerp_block_tokens must be positive.')
        hs_qbi_exact_layer = None if hs_qbi_exact_layer in (None, '') else str(hs_qbi_exact_layer)
        hs_qbi_exact_layers = None if hs_qbi_exact_layers in (None, '') else str(hs_qbi_exact_layers)
        if hs_qbi_exact_layer is not None:
            hs_qbi_exact_layers = hs_qbi_exact_layer if hs_qbi_exact_layers is None else f'{hs_qbi_exact_layer}:{hs_qbi_exact_layers}'
        hs_qbi_rel_l2_threshold = None if hs_qbi_rel_l2_threshold in (None, '') else float(hs_qbi_rel_l2_threshold)
        quant_bits = int(quant_bits)
        quant_group_size = int(quant_group_size)
        if isinstance(pretrained, str):
            _hs_profile_set('adapter.pretrained', pretrained)
            _hs_profile_set('adapter.hs_codec_name', hs_codec_name)
            _hs_profile_set('adapter.hs_qbi_stride', int(hs_qbi_stride))
            _hs_profile_set('adapter.hs_qbi_quant_group_size', int(hs_qbi_quant_group_size))
            if str(hs_codec_name).lower() == 'qbclerp':
                _hs_profile_set('adapter.hs_qbclerp_stride', int(hs_qbclerp_stride))
                _hs_profile_set('adapter.hs_qbclerp_prefix_layers', int(hs_qbclerp_prefix_layers))
                _hs_profile_set('adapter.hs_qbclerp_quant_group_size', int(hs_qbclerp_quant_group_size))
                _hs_profile_set('adapter.hs_qbclerp_block_tokens', int(hs_qbclerp_block_tokens))
                _hs_profile_set('adapter.hs_qbclerp_calibration_sample_tokens', int(os.environ.get('HS_QBCLERP_CALIBRATION_SAMPLE_TOKENS', '0') or '0'))
            _hs_profile_set('adapter.hs_qbi_exact_layer', hs_qbi_exact_layer)
            _hs_profile_set('adapter.hs_qbi_exact_layers', hs_qbi_exact_layers)
            _hs_profile_set('adapter.hs_qbi_rel_l2_threshold', hs_qbi_rel_l2_threshold)
            _hs_profile_set('adapter.hs_device', hs_device)
            _hs_profile_set('adapter.quant_bits', int(quant_bits))
            _hs_profile_set('adapter.quant_group_size', int(quant_group_size))
            _hs_profile_set('adapter.hs_decode_hidden_prefetch_overlap', bool(hs_decode_hidden_prefetch_overlap))
            _hs_profile_set('adapter.hs_async_hidden_offload', bool(hs_async_hidden_offload))
            _hs_profile_set('adapter.hs_async_compression', bool(hs_async_compression))
            _hs_profile_set('adapter.hs_stage_slots', int(hs_stage_slots))
            _hs_profile_set('adapter.hs_triton_chunk_size', int(hs_triton_chunk_size))
            _hs_profile_set('adapter.hs_block_size', int(hs_block_size))
            _hs_profile_set('adapter.kv_offload', bool(self._kv_offload_enabled))
            _hs_profile_set('adapter.pin_kv_cache', bool(self._pin_kv_cache))
            _hs_profile_set('adapter.async_kv_offload', bool(self._async_kv_offload))
            with _hs_profile_section('adapter.tokenizer_load'):
                tokenizer_obj = self._load_tokenizer(pretrained=pretrained, tokenizer=tokenizer, revision=revision, trust_remote_code=trust_remote_code, use_fast_tokenizer=use_fast_tokenizer, add_bos_token=add_bos_token, subfolder=subfolder)
            self._validate_hs_load_args(backend=backend, parallelize=parallelize, peft=peft, delta=delta, autogptq=autogptq, gptqmodel=gptqmodel, gguf_file=gguf_file, max_memory_per_gpu=max_memory_per_gpu, max_cpu_memory=max_cpu_memory)
            self._ds_engine, model, self._hf_ds_config, self._hs_config = self._build_hs_offload_model(pretrained, backend=backend, revision=revision, subfolder=subfolder, device=device, dtype=dtype, trust_remote_code=trust_remote_code, batch_size=batch_size, ds_batch_size=ds_batch_size, cpu_offload=cpu_offload, disk_offload=disk_offload, offload_dir=offload_dir, pin_memory=pin_memory, quant_bits=quant_bits, quant_group_size=quant_group_size, hs_offload=hs_offload, hs_max_gen_len=hs_max_gen_len, hs_device=hs_device, hs_codec_name=hs_codec_name, hs_qbi_stride=hs_qbi_stride, hs_qbi_quant_group_size=hs_qbi_quant_group_size, hs_qbclerp_stride=hs_qbclerp_stride, hs_qbclerp_prefix_layers=hs_qbclerp_prefix_layers, hs_qbclerp_quant_group_size=hs_qbclerp_quant_group_size, hs_qbclerp_block_tokens=hs_qbclerp_block_tokens, hs_qbi_exact_layers=hs_qbi_exact_layers, hs_qbi_rel_l2_threshold=hs_qbi_rel_l2_threshold, hs_decode_hidden_prefetch_overlap=hs_decode_hidden_prefetch_overlap, hs_async_hidden_offload=hs_async_hidden_offload, hs_async_compression=hs_async_compression, hs_stage_slots=hs_stage_slots, hs_triton_chunk_size=hs_triton_chunk_size, hs_block_size=hs_block_size, **kwargs)
            self._hs_runtime = getattr(self._ds_engine, 'hs_runtime', None)
            if self._hs_runtime is not None:
                self._hs_runtime.reset_profiler()
            with _hs_profile_section('adapter.hflm_super_init'):
                super().__init__(pretrained=model, backend=backend, revision=revision, subfolder=subfolder, tokenizer=tokenizer_obj, truncation=truncation, logits_cache=logits_cache, max_length=max_length, device=device, dtype=dtype, softmax_dtype=softmax_dtype, mixed_precision_dtype=mixed_precision_dtype, batch_size=batch_size, max_batch_size=max_batch_size, trust_remote_code=trust_remote_code, use_fast_tokenizer=use_fast_tokenizer, add_bos_token=add_bos_token, prefix_token_id=prefix_token_id, parallelize=False, max_memory_per_gpu=max_memory_per_gpu, max_cpu_memory=max_cpu_memory, offload_folder=offload_folder, peft=peft, delta=delta, autogptq=autogptq, gptqmodel=gptqmodel, gguf_file=gguf_file, think_end_token=think_end_token, enable_thinking=enable_thinking, chat_template_args=chat_template_args, **kwargs)
            self.pretrained = pretrained
            self._device = self._resolve_runtime_device(device)
            self._sync_distributed_state()
            self._register_hs_profile_dump()
            return
        super().__init__(pretrained=pretrained, backend=backend, revision=revision, subfolder=subfolder, tokenizer=tokenizer, truncation=truncation, logits_cache=logits_cache, max_length=max_length, device=device, dtype=dtype, softmax_dtype=softmax_dtype, mixed_precision_dtype=mixed_precision_dtype, batch_size=batch_size, max_batch_size=max_batch_size, trust_remote_code=trust_remote_code, use_fast_tokenizer=use_fast_tokenizer, add_bos_token=add_bos_token, prefix_token_id=prefix_token_id, parallelize=parallelize, max_memory_per_gpu=max_memory_per_gpu, max_cpu_memory=max_cpu_memory, offload_folder=offload_folder, peft=peft, delta=delta, autogptq=autogptq, gptqmodel=gptqmodel, gguf_file=gguf_file, think_end_token=think_end_token, enable_thinking=enable_thinking, chat_template_args=chat_template_args, **kwargs)

    @staticmethod
    def _load_tokenizer(*, pretrained: str, tokenizer: str | transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None, revision: str | None='main', trust_remote_code: bool | None=False, use_fast_tokenizer: bool | None=True, add_bos_token: bool | None=None, subfolder: str=''):
        kwargs = {'revision': revision, 'trust_remote_code': trust_remote_code, 'use_fast': use_fast_tokenizer}
        if add_bos_token is not None:
            kwargs['add_bos_token'] = add_bos_token
        if subfolder:
            kwargs['subfolder'] = subfolder
        if tokenizer is None:
            return transformers.AutoTokenizer.from_pretrained(pretrained, **kwargs)
        if isinstance(tokenizer, str):
            return transformers.AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        return tokenizer

    @classmethod
    def _build_hs_offload_model(cls, pretrained: str, *, backend: Literal['default', 'causal', 'seq2seq']='default', revision: str | None='main', subfolder: str='', device: str | None='cuda', dtype: str | torch.dtype | None='auto', trust_remote_code: bool | None=False, batch_size: int | str | None=1, ds_batch_size: int | None=None, cpu_offload: bool=False, disk_offload: bool=False, offload_dir: str | None=None, pin_memory: bool=False, quant_bits: int=16, quant_group_size: int=64, hs_offload: bool=True, hs_max_gen_len: int | None=None, hs_device: str='cpu', hs_codec_name: str='fp16', hs_qbi_stride: int=4, hs_qbi_group_size: int | None=None, hs_qbi_quant_group_size: int=64, hs_qbclerp_stride: int=3, hs_qbclerp_prefix_layers: int=4, hs_qbclerp_quant_group_size: int=32, hs_qbclerp_block_tokens: int=64, hs_qbi_exact_layers: str | None=None, hs_qbi_rel_l2_threshold: float | None=None, hs_decode_hidden_prefetch_overlap: bool=True, hs_async_hidden_offload: bool=True, hs_async_compression: bool=False, hs_stage_slots: int=2, hs_triton_chunk_size: int=128, hs_block_size: int=64, **model_kwargs):
        import deepspeed
        from deepspeed.runtime.hs_offload.config import HSOffloadConfig
        from transformers.deepspeed import HfDeepSpeedConfig
        with _hs_profile_section('adapter.distributed_init'):
            cls._maybe_init_distributed(deepspeed=deepspeed, device=device)
        with _hs_profile_section('adapter.config_load'):
            config = cls._load_model_config(pretrained, revision=revision, trust_remote_code=trust_remote_code, subfolder=subfolder)
        cls._ensure_causal_backend(backend=backend, config=config)
        hidden_size = cls._get_hidden_size(config)
        num_layers = cls._get_num_layers(config)
        effective_hs_max_gen_len = cls._resolve_hs_staging_capacity(config=config, hs_max_gen_len=hs_max_gen_len)
        resolved_dtype = cls._resolve_model_dtype(dtype=dtype, config=config)
        resolved_batch_size = cls._resolve_ds_batch_size(batch_size=batch_size, ds_batch_size=ds_batch_size)
        ds_config = cls._build_deepspeed_config(hidden_size=hidden_size, dtype=resolved_dtype, batch_size=resolved_batch_size, cpu_offload=cpu_offload, disk_offload=disk_offload, offload_dir=offload_dir, pin_memory=pin_memory, quant_bits=quant_bits, quant_group_size=quant_group_size)
        with _hs_profile_section('adapter.hf_deepspeed_config'):
            hf_ds_config = HfDeepSpeedConfig(ds_config)
        dtype_arg = 'dtype' if vparse(transformers.__version__) >= vparse('4.56.0') else 'torch_dtype'
        load_kwargs = dict(model_kwargs)
        load_kwargs.update({'revision': revision, 'trust_remote_code': trust_remote_code, dtype_arg: resolved_dtype})
        if subfolder:
            load_kwargs['subfolder'] = subfolder
        with _hs_profile_section('adapter.model_from_pretrained', sync_cuda=True):
            model = transformers.AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs).eval()
        with _hs_profile_section('adapter.deepspeed_initialize', sync_cuda=True):
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
        ds_engine.module.eval()
        if hs_qbi_group_size not in (None, ''):
            hs_qbi_stride = hs_qbi_group_size
        hs_qbi_stride = int(hs_qbi_stride)
        if hs_qbi_stride < 1:
            raise ValueError('hs_qbi_stride must be >= 1.')
        hs_config = None
        if hs_offload:
            hs_config = HSOffloadConfig(
                enabled=True,
                max_gen_len=effective_hs_max_gen_len,
                device=hs_device,
                codec_name=hs_codec_name,
                qbi_stride=int(hs_qbi_stride),
                hs_qbi_stride=int(hs_qbi_stride),
                qbi_quant_group_size=int(hs_qbi_quant_group_size),
                qbi_exact_layers=hs_qbi_exact_layers,
                qbi_rel_l2_threshold=hs_qbi_rel_l2_threshold,
                hs_qbclerp_stride=int(hs_qbclerp_stride),
                hs_qbclerp_prefix_layers=int(hs_qbclerp_prefix_layers),
                hs_qbclerp_quant_group_size=int(hs_qbclerp_quant_group_size),
                hs_qbclerp_block_tokens=int(hs_qbclerp_block_tokens),
                hidden_dim=hidden_size,
                num_layers=num_layers,
                num_attention_heads=getattr(config, 'num_attention_heads', None),
                num_key_value_heads=getattr(
                    config,
                    'num_key_value_heads',
                    getattr(config, 'num_attention_heads', None),
                ),
                block_size=int(hs_block_size),
                async_store=bool(hs_async_hidden_offload),
                async_compression=bool(hs_async_compression),
                stage_slots=int(hs_stage_slots),
                decode_hidden_prefetch_overlap=bool(hs_decode_hidden_prefetch_overlap),
                triton_chunk_size=int(hs_triton_chunk_size),
            )
            with _hs_profile_section('adapter.set_hidden_state_offload', sync_cuda=True):
                ds_engine.set_hidden_state_offload(enabled=True, config=hs_config)
        return (ds_engine, ds_engine.module, hf_ds_config, hs_config)

    def _register_hs_profile_dump(self) -> None:
        if self._hs_profile_dump_registered or not _hs_profile_enabled():
            return
        atexit.register(self._write_hs_profile_snapshot)
        self._hs_profile_dump_registered = True

    def _write_hs_profile_snapshot(self) -> None:
        if not _hs_profile_enabled():
            return
        if self._hs_runtime is not None:
            _hs_profile_set('hs_runtime', self._hs_runtime.profiler_summary())
            _hs_profile_set('hs_runtime.decode_backend_stats', self._hs_runtime.decode_backend_stats())
        _hs_profile_set('qbclerp_sample_calibration', self._qbclerp_sample_calibration_info)
        if self._hs_config is not None:
            hs_config_profile = {'codec_name': getattr(self._hs_config, 'codec_name', None), 'num_layers': getattr(self._hs_config, 'num_layers', None), 'num_attention_heads': getattr(self._hs_config, 'num_attention_heads', None), 'num_key_value_heads': getattr(self._hs_config, 'num_key_value_heads', None), 'max_gen_len': getattr(self._hs_config, 'max_gen_len', None), 'device': getattr(self._hs_config, 'device', None), 'hidden_dim': getattr(self._hs_config, 'hidden_dim', None), 'block_size': getattr(self._hs_config, 'block_size', None), 'stage_slots': getattr(self._hs_config, 'stage_slots', None)}
            if str(getattr(self._hs_config, 'codec_name', '')).lower() == 'qbclerp':
                hs_config_profile.update({'hs_qbclerp_stride': getattr(self._hs_config, 'hs_qbclerp_stride', None), 'hs_qbclerp_prefix_layers': getattr(self._hs_config, 'hs_qbclerp_prefix_layers', None), 'hs_qbclerp_quant_group_size': getattr(self._hs_config, 'hs_qbclerp_quant_group_size', None), 'hs_qbclerp_block_tokens': getattr(self._hs_config, 'hs_qbclerp_block_tokens', None)})
            else:
                hs_config_profile.update({})
            _hs_profile_set('hs_config', hs_config_profile)
        _hs_profile_dump()

    def tok_batch_encode(self, strings: list[str], padding_side: str='left', left_truncate_len: int | None=None, truncation: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        with _hs_profile_section('lm_eval.tok_batch_encode'):
            input_ids, attention_mask = super().tok_batch_encode(strings=strings, padding_side=padding_side, left_truncate_len=left_truncate_len, truncation=truncation)
        _hs_profile_count('lm_eval.tok_batch_encode.calls')
        _hs_profile_add('lm_eval.tok_batch_encode.batch_size_total', float(len(strings)))
        _hs_profile_add('lm_eval.tok_batch_encode.input_token_slots', float(input_ids.numel()))
        _hs_profile_add('lm_eval.tok_batch_encode.attention_tokens', float(attention_mask.sum().item()))
        return (input_ids, attention_mask)

    @staticmethod
    def _restore_env_value(name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _qbclerp_calibration_sample_tokens(self) -> int:
        return int(os.environ.get('HS_QBCLERP_CALIBRATION_SAMPLE_TOKENS', '0') or '0')

    def _slice_qbclerp_calibration_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, sample_tokens: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        sample_tokens = int(sample_tokens)
        if sample_tokens <= 0:
            raise ValueError('sample_tokens must be positive for QBCLERP calibration.')
        if attention_mask is None:
            effective_len = min(sample_tokens, int(input_ids.size(1)))
            return (input_ids[:, :effective_len].contiguous(), None, int(effective_len))
        rows: list[torch.Tensor] = []
        max_len = 0
        attention_mask = attention_mask.to(device=input_ids.device)
        for row, mask in zip(input_ids, attention_mask, strict=True):
            active = row[mask.to(dtype=torch.bool)]
            if active.numel() == 0:
                active = row[:1]
            sample = active[:sample_tokens].contiguous()
            rows.append(sample)
            max_len = max(max_len, int(sample.numel()))
        max_len = max(max_len, 1)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eot_token_id
        sliced_ids = torch.full((len(rows), max_len), int(pad_id), dtype=input_ids.dtype, device=input_ids.device)
        sliced_mask = torch.zeros((len(rows), max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        for idx, sample in enumerate(rows):
            length = int(sample.numel())
            sliced_ids[idx, max_len - length:] = sample
            sliced_mask[idx, max_len - length:] = 1
        return (sliced_ids.contiguous(), sliced_mask.contiguous(), int(max_len))

    def _record_qbclerp_sample_calibration(self, info: dict[str, Any]) -> None:
        previous = self._qbclerp_sample_calibration_info or {'enabled': False}
        count = int(previous.get('count', 0)) + 1
        elapsed_total_ms = float(previous.get('elapsed_total_ms', 0.0)) + float(info.get('elapsed_ms', 0.0))
        aggregate = dict(info)
        aggregate.update({'enabled': True, 'count': count, 'elapsed_total_ms': elapsed_total_ms, 'elapsed_avg_ms': elapsed_total_ms / max(count, 1), 'last': info})
        self._qbclerp_sample_calibration_info = aggregate
        _hs_profile_set('qbclerp_sample_calibration', aggregate)

    def _maybe_run_qbclerp_sample_calibration(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None, *, source: str) -> None:
        sample_tokens = self._qbclerp_calibration_sample_tokens()
        if sample_tokens <= 0:
            return
        if self._hs_runtime is None or self._hs_config is None:
            raise ValueError('HS_QBCLERP_CALIBRATION_SAMPLE_TOKENS requires HS offload.')
        if str(getattr(self._hs_config, 'codec_name', '')).lower() != 'qbclerp':
            raise ValueError('HS_QBCLERP_CALIBRATION_SAMPLE_TOKENS requires hs_codec_name=qbclerp.')
        sample_input_ids, sample_attention_mask, effective_len = self._slice_qbclerp_calibration_inputs(input_ids=input_ids, attention_mask=attention_mask, sample_tokens=sample_tokens)
        env_names = ('HS_QBCLERP_CUSTOM_ANCHORS', 'HS_QBCLERP_RUNTIME_REL_L2', 'HS_QBCLERP_RUNTIME_CURVATURE', 'HS_QBCLERP_RUNTIME_SAMPLE_TOKENS')
        saved_env = {name: os.environ.get(name) for name in env_names}
        start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        import time
        wall_start = time.perf_counter()
        try:
            os.environ.pop('HS_QBCLERP_CUSTOM_ANCHORS', None)
            os.environ['HS_QBCLERP_RUNTIME_REL_L2'] = '1'
            os.environ['HS_QBCLERP_RUNTIME_CURVATURE'] = '1'
            os.environ['HS_QBCLERP_RUNTIME_SAMPLE_TOKENS'] = str(effective_len)
            kwargs = {'input_ids': sample_input_ids, 'use_cache': True, 'return_dict': True}
            if sample_attention_mask is not None:
                kwargs['attention_mask'] = sample_attention_mask
            if start_event is not None:
                start_event.record(torch.cuda.current_stream())
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype, enabled=self.mixed_precision_dtype is not None):
                _ = self.model(**kwargs)
            if end_event is not None:
                end_event.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            cache = getattr(self._hs_runtime.store, 'qbclerp_cache', None)
            if cache is None:
                raise RuntimeError('[QBCLERPCalibration] Calibration did not create a QBCLERP cache.')
            anchors = [int(x) for x in getattr(cache, 'anchor_layer_ids', [])]
            if not anchors:
                raise RuntimeError('[QBCLERPCalibration] Calibration produced an empty anchor schedule.')
            anchor_text = ','.join((str(x) for x in anchors))
            try:
                from deepspeed.runtime.hs_offload.qbclerp import qbclerp_segment_specs
                segment_specs = qbclerp_segment_specs(getattr(cache, 'segments', []))
            except Exception:
                segment_specs = None
            elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
            gpu_elapsed_ms = float(start_event.elapsed_time(end_event)) if start_event is not None and end_event is not None else None
            info = {'source': str(source), 'requested_sample_tokens': int(sample_tokens), 'effective_sample_tokens': int(effective_len), 'runtime_n': float(os.environ.get('HS_QBCLERP_RUNTIME_N', '5')), 'runtime_budget_pct': float(os.environ.get('HS_QBCLERP_RUNTIME_BUDGET_PCT', '70')), 'runtime_outlier_pct': float(os.environ.get('HS_QBCLERP_RUNTIME_OUTLIER_PCT', '99')), 'runtime_force_prefix': int(os.environ.get('HS_QBCLERP_RUNTIME_FORCE_PREFIX', '0')), 'anchor_count': int(len(anchors)), 'anchor_layer_ids': anchor_text, 'segment_specs': segment_specs, 'applied_custom_anchors': True, 'runtime_rel_l2_disabled_for_request': True, 'elapsed_ms': float(elapsed_ms), 'gpu_elapsed_ms': gpu_elapsed_ms}
            self._record_qbclerp_sample_calibration(info)
            if getattr(self, 'rank', 0) == 0:
                print(f'[QBCLERPCalibration] source={source}, sample_tokens={effective_len}, anchors={anchor_text}, elapsed_ms={elapsed_ms:.3f}', flush=True)
            self._hs_runtime.reset()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            for name, value in saved_env.items():
                self._restore_env_value(name, value)
            os.environ['HS_QBCLERP_RUNTIME_REL_L2'] = '0'
            os.environ['HS_QBCLERP_RUNTIME_CURVATURE'] = '0'
            os.environ['HS_QBCLERP_CUSTOM_ANCHORS'] = anchor_text
        except Exception:
            try:
                self._hs_runtime.reset()
            finally:
                for name, value in saved_env.items():
                    self._restore_env_value(name, value)
            raise

    def generate_until(self, requests, disable_tqdm: bool=False) -> list[str]:
        _hs_profile_count('lm_eval.generate_until.calls')
        _hs_profile_add('lm_eval.generate_until.request_count', float(len(requests)))
        with _hs_profile_section('lm_eval.generate_until_total', sync_cuda=True):
            return super().generate_until(requests=requests, disable_tqdm=disable_tqdm)

    def _model_call(self, inps: torch.Tensor, attn_mask: torch.Tensor | None=None, labels: torch.Tensor | None=None) -> torch.Tensor:
        _hs_profile_count('lm_eval.model_call.calls')
        _hs_profile_add('lm_eval.model_call.input_token_slots', float(inps.numel()))
        self._maybe_run_qbclerp_sample_calibration(input_ids=inps, attention_mask=attn_mask, source='model_call')
        return super()._model_call(inps=inps, attn_mask=attn_mask, labels=labels)

    def _model_generate(self, context, max_length: int, stop: list[str], **generation_kwargs) -> torch.Tensor:
        _hs_profile_count('lm_eval.model_generate.calls')
        _hs_profile_add('lm_eval.model_generate.context_token_slots', float(context.numel()))
        _hs_profile_add('lm_eval.model_generate.max_new_token_budget', float(max(max_length - context.shape[1], 0)))
        if os.environ.get('HS_DEBUG_GENERATION_TOPK', '0') == '1':
            if 'logits_processor' not in generation_kwargs:
                generation_kwargs['logits_processor'] = LogitsProcessorList()
            observer = DynamicEOSObserver(tokenizer=self.tokenizer)
            generation_kwargs['logits_processor'].append(observer)
        with _hs_profile_section('lm_eval.model_generate', sync_cuda=True):
            self._maybe_run_qbclerp_sample_calibration(input_ids=context, attention_mask=generation_kwargs.get('attention_mask'), source='model_generate')
            self._maybe_enable_kv_offload(max_length=max_length, context=context)
            output = super()._model_generate(context=context, max_length=max_length, stop=stop, **generation_kwargs)
        generated_tokens = max(int(output.shape[1]) - int(context.shape[1]), 0)
        _hs_profile_add('lm_eval.model_generate.generated_tokens', float(generated_tokens))
        _hs_profile_set('lm_eval.model_generate.last_generated_tokens', float(generated_tokens))
        _hs_profile_set('lm_eval.model_generate.last_context_len', float(context.shape[1]))
        _hs_profile_set('lm_eval.model_generate.last_output_len', float(output.shape[1]))
        return output

    def _maybe_enable_kv_offload(self, *, max_length: int, context: torch.Tensor) -> None:
        if not self._kv_offload_enabled:
            return
        set_kv_cache_offload = getattr(self.model, 'set_kv_cache_offload', None)
        if set_kv_cache_offload is None:
            raise ValueError('kv_offload=True requires the loaded model to expose set_kv_cache_offload().')
        gen_len = max(int(max_length) - int(context.shape[1]), 0)
        set_kv_cache_offload(True, gen_len, self._pin_kv_cache, self._async_kv_offload)

    @staticmethod
    def _validate_hs_load_args(*, backend: Literal['default', 'causal', 'seq2seq'], parallelize: bool | None, peft: str | None, delta: str | None, autogptq: bool | str | None, gptqmodel: bool | None, gguf_file: str | None, max_memory_per_gpu: int | str | None, max_cpu_memory: int | str | None) -> None:
        if backend == 'seq2seq':
            raise NotImplementedError('HSOffloadHFLM currently supports causal decoder-only models only.')
        unsupported = {'parallelize': parallelize, 'peft': peft, 'delta': delta, 'autogptq': autogptq, 'gptqmodel': gptqmodel, 'gguf_file': gguf_file, 'max_memory_per_gpu': max_memory_per_gpu, 'max_cpu_memory': max_cpu_memory}
        enabled = [name for name, value in unsupported.items() if value not in (None, False)]
        if enabled:
            raise ValueError('HSOffloadHFLM does not support these HFLM load-time options: ' + ', '.join(enabled))

    @staticmethod
    def _load_model_config(pretrained: str, *, revision: str | None='main', trust_remote_code: bool | None=False, subfolder: str='') -> transformers.PretrainedConfig:
        kwargs = {'revision': revision, 'trust_remote_code': trust_remote_code}
        if subfolder:
            kwargs['subfolder'] = subfolder
        return transformers.AutoConfig.from_pretrained(pretrained, **kwargs)

    @staticmethod
    def _ensure_causal_backend(*, backend: Literal['default', 'causal', 'seq2seq'], config: transformers.PretrainedConfig) -> None:
        if backend == 'seq2seq':
            raise NotImplementedError('HSOffloadHFLM currently supports causal decoder-only models only.')
        model_type = getattr(config, 'model_type', None)
        if model_type in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES:
            raise NotImplementedError(f"HSOffloadHFLM does not support seq2seq backend for model type '{model_type}'.")
        if backend == 'default' and model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            eval_logger.info("Model type '%s' is not explicitly listed as causal in transformers; continuing with AutoModelForCausalLM for HS offload.", model_type)

    @staticmethod
    def _get_hidden_size(config: transformers.PretrainedConfig) -> int:
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None and hasattr(config, 'text_config'):
            hidden_size = getattr(config.text_config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError('Unable to infer hidden_size from model config for DeepSpeed HS offload.')
        return int(hidden_size)

    @staticmethod
    def _get_num_layers(config: transformers.PretrainedConfig) -> int | None:
        for attr in ('num_hidden_layers', 'n_layer', 'num_layers'):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        text_config = getattr(config, 'text_config', None)
        if text_config is not None:
            for attr in ('num_hidden_layers', 'n_layer', 'num_layers'):
                value = getattr(text_config, attr, None)
                if value is not None:
                    return int(value)
        return None

    @staticmethod
    def _resolve_model_dtype(*, dtype: str | torch.dtype | None, config: transformers.PretrainedConfig) -> torch.dtype:
        if dtype in (None, 'auto'):
            config_dtype = getattr(config, 'torch_dtype', None)
            if config_dtype is None:
                return torch.float16
            return get_dtype(config_dtype)
        return get_dtype(dtype)

    @staticmethod
    def _resolve_hs_staging_capacity(*, config: transformers.PretrainedConfig, hs_max_gen_len: int | None) -> int:
        candidate_lengths = [getattr(config, 'max_position_embeddings', None), getattr(config, 'n_positions', None), getattr(config, 'max_sequence_length', None), getattr(config, 'seq_length', None)]
        if hasattr(config, 'text_config'):
            candidate_lengths.extend([getattr(config.text_config, 'max_position_embeddings', None), getattr(config.text_config, 'n_positions', None), getattr(config.text_config, 'max_sequence_length', None), getattr(config.text_config, 'seq_length', None)])
        supported_len = max([int(length) for length in candidate_lengths if length is not None], default=1024)
        if hs_max_gen_len is None:
            eval_logger.info('hs_max_gen_len was not provided; using model sequence capacity %s for HS staging.', supported_len)
            return supported_len
        configured_len = int(hs_max_gen_len)
        effective_len = max(configured_len, supported_len)
        if effective_len != configured_len:
            eval_logger.info('Increasing HS staging capacity from %s to %s to cover the model sequence length for generate_until workloads.', configured_len, effective_len)
        return effective_len

    @staticmethod
    def _resolve_ds_batch_size(*, batch_size: int | str | None, ds_batch_size: int | None) -> int:
        if ds_batch_size is not None:
            return int(ds_batch_size)
        if isinstance(batch_size, int):
            return batch_size
        eval_logger.info("Using DeepSpeed train_batch_size=1 because lm-eval batch_size was set to '%s'.", batch_size)
        return 1

    @staticmethod
    def _build_deepspeed_config(*, hidden_size: int, dtype: torch.dtype, batch_size: int, cpu_offload: bool, disk_offload: bool, offload_dir: str | None, pin_memory: bool, quant_bits: int=16, quant_group_size: int=64) -> dict[str, Any]:
        if cpu_offload and disk_offload:
            raise ValueError('cpu_offload and disk_offload cannot both be enabled.')
        quant_bits = int(quant_bits)
        quant_group_size = int(quant_group_size)
        if quant_bits not in (4, 8, 16):
            raise ValueError('quant_bits must be one of 4, 8, or 16.')
        ds_config: dict[str, Any] = {'fp16': {'enabled': dtype == torch.float16}, 'bf16': {'enabled': dtype == torch.bfloat16}, 'zero_optimization': {'stage': 3, 'stage3_prefetch_bucket_size': 2 * hidden_size * hidden_size, 'stage3_param_persistence_threshold': hidden_size, 'stage3_max_live_parameters': 2 * hidden_size * hidden_size}, 'steps_per_print': 2000, 'train_batch_size': int(batch_size), 'wall_clock_breakdown': False}
        if quant_bits != 16:
            ds_config['weight_quantization'] = {'quantized_initialization': {'num_bits': quant_bits, 'group_size': quant_group_size, 'group_dim': 1, 'symmetric': False}}
        if cpu_offload:
            ds_config['zero_optimization']['offload_param'] = {'device': 'cpu', 'pin_memory': bool(pin_memory)}
        if disk_offload:
            if not offload_dir:
                raise ValueError('offload_dir must be provided when disk_offload is enabled.')
            resolved_offload_dir = os.path.abspath(os.path.expanduser(offload_dir))
            ds_config['zero_optimization']['offload_param'] = {'device': 'nvme', 'pin_memory': bool(pin_memory), 'nvme_path': resolved_offload_dir, 'buffer_count': 5, 'buffer_size': 2 * 1024 * 1024 * 1024}
            ds_config['aio'] = {'block_size': 1048576 * 16, 'queue_depth': 64, 'thread_count': 8, 'single_submit': False, 'overlap_events': True}
        return ds_config

    @staticmethod
    def _maybe_init_distributed(*, deepspeed, device: str | None) -> None:
        if not torch.distributed.is_available():
            return
        if torch.distributed.is_initialized():
            return
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size <= 1:
            return
        backend = 'nccl'
        if device is not None and str(device).startswith('cpu'):
            backend = 'gloo'
        deepspeed.init_distributed(backend)

    @staticmethod
    def _resolve_runtime_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _sync_distributed_state(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

    def all_gather(self, tensor):
        if self.world_size <= 1:
            return tensor
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        tensor = tensor.contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, tensor)
        if tensor.ndim == 0:
            return torch.stack(gathered)
        return torch.cat(gathered, dim=0)

    def gather_object(self, obj, dst=0):
        if self.world_size <= 1:
            return [obj]
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [obj]
        result = [None] * self.world_size if self.rank == dst else None
        torch.distributed.gather_object(obj=obj, object_gather_list=result, dst=dst)
        return result

    def barrier(self) -> None:
        if self.world_size > 1 and torch.distributed.is_available():
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def get_model_info(self) -> dict:
        model_info = super().get_model_info()
        hs_compression = self._hs_compression_result_info()
        if hs_compression:
            model_info['hs_compression'] = hs_compression
        return model_info

    def _hs_compression_result_info(self) -> dict[str, Any]:
        if self._hs_runtime is None:
            return {}
        summary = self._hs_runtime.compression_summary() or {}
        return {
            'codec_name': getattr(self._hs_config, 'codec_name', None),
            'num_layers': getattr(self._hs_config, 'num_layers', None),
            'num_attention_heads': getattr(self._hs_config, 'num_attention_heads', None),
            'num_key_value_heads': getattr(self._hs_config, 'num_key_value_heads', None),
            'kv_input_bytes': float(summary.get('kv_input_bytes', 0.0)),
            'hidden_input_bytes': float(summary.get('hidden_input_bytes', 0.0)),
            'codec_input_bytes': float(summary.get('codec_input_bytes', 0.0)),
            'stored_output_bytes': float(summary.get('stored_output_bytes', 0.0)),
            'compressed_block_count': float(summary.get('compressed_block_count', 0.0)),
            'physical_stored_layer_count': float(summary.get('physical_stored_layer_count', 0.0)),
            'compression_ratio': float(summary.get('compression_ratio', 0.0)),
            'compressed_size_ratio': float(summary.get('compressed_size_ratio', 0.0)),
            'hidden_storage_ratio': float(summary.get('hidden_storage_ratio', 0.0)),
            'codec_compression_ratio': float(summary.get('codec_compression_ratio', 0.0)),
        }
