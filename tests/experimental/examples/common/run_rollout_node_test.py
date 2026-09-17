# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for run_rollout_node CLI parsing, dict decoding, and vLLM sampler creation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock
import unittest
import yaml

# Dynamically import run_rollout_node with mocks for unavailable TPU/JAX dependencies
try:
  from tunix.experimental.examples.common import run_rollout_node
except Exception:
  class _MockModule(types.ModuleType):
    def __getattr__(self, name):
      val = mock.MagicMock()
      setattr(self, name, val)
      return val

  class _MockLoader:
    def create_module(self, spec):
      mod = _MockModule(spec.name)
      mod.__path__ = []
      mod.__file__ = spec.name + ".py"
      return mod

    def exec_module(self, module):
      pass

  class _AutoMockFinder:
    def find_spec(self, fullname, path, target=None):
      stdlib = {
          "sys",
          "os",
          "yaml",
          "json",
          "ast",
          "argparse",
          "typing",
          "unittest",
          "re",
          "importlib",
          "builtins",
          "collections",
          "pathlib",
          "itertools",
          "functools",
      }
      if fullname.split(".")[0] in sys.builtin_module_names or fullname.split(".")[0] in stdlib:
        return None
      from importlib.machinery import ModuleSpec
      return ModuleSpec(fullname, _MockLoader())

  sys.meta_path.insert(0, _AutoMockFinder())
  _module_path = (
      Path(__file__).resolve().parents[4]
      / "tunix"
      / "experimental"
      / "examples"
      / "common"
      / "run_rollout_node.py"
  )
  _spec = importlib.util.spec_from_file_location(
      "tunix.experimental.examples.common.run_rollout_node", str(_module_path)
  )
  run_rollout_node = importlib.util.module_from_spec(_spec)
  sys.modules["tunix.experimental.examples.common.run_rollout_node"] = run_rollout_node
  _spec.loader.exec_module(run_rollout_node)


class RunRolloutNodeTest(unittest.TestCase):

  def test_parse_dict_arg_json(self):
    json_str = '{"sharding": {"expert_parallelism": 8}, "multiplier": 16}'
    result = run_rollout_node._parse_dict_arg(json_str)
    self.assertEqual(result, {"sharding": {"expert_parallelism": 8}, "multiplier": 16})

  def test_parse_dict_arg_yaml_with_tabs_and_unquoted(self):
    raw_user_yaml = """{
sharding: {
sharding_strategy: {
\texpert_parallelism: 8,
\ttensor_parallelism: 1,
enable_dp_attention: true
}}, 
custom_mamba_cache_multiplier: 16, 
# maxtext configs
maxtext_config: {
model_name: Qwen/Qwen3.5-397B-A17B, 
load_parameters_path:gs://maxtext-model-checkpoints/qwen3.5-397b-a17b/unscanned/0/items,
scan_layers: false, 
attention: vllm_rpa, 
enable_nnx: true, 
pure_nnx_decoder: true, 
allow_split_physical_axes: true, 
use_multimodal: false, 
prefuse_moe_weights: true}}
}"""
    result = run_rollout_node._parse_dict_arg(raw_user_yaml)
    self.assertIsInstance(result, dict)
    self.assertEqual(result.get("custom_mamba_cache_multiplier"), 16)
    self.assertEqual(
        result["sharding"]["sharding_strategy"]["expert_parallelism"], 8
    )
    self.assertEqual(
        result["sharding"]["sharding_strategy"]["tensor_parallelism"], 1
    )
    self.assertTrue(
        result["sharding"]["sharding_strategy"]["enable_dp_attention"]
    )
    maxtext_cfg = result["maxtext_config"]
    self.assertEqual(maxtext_cfg["model_name"], "Qwen/Qwen3.5-397B-A17B")
    self.assertFalse(maxtext_cfg["scan_layers"])
    self.assertEqual(maxtext_cfg["attention"], "vllm_rpa")
    self.assertTrue(maxtext_cfg["enable_nnx"])
    self.assertTrue(maxtext_cfg["pure_nnx_decoder"])
    self.assertTrue(maxtext_cfg["allow_split_physical_axes"])
    self.assertFalse(maxtext_cfg["use_multimodal"])
    self.assertTrue(maxtext_cfg["prefuse_moe_weights"])

  def test_parse_dict_arg_python_literal(self):
    literal_str = "{'a': 1, 'b': [2, 3], 'c': True}"
    result = run_rollout_node._parse_dict_arg(literal_str)
    self.assertEqual(result, {"a": 1, "b": [2, 3], "c": True})

  def test_parse_dict_arg_from_file(self):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
      tmp.write("key1: value1\nkey2:\n  nested: 42\n")
      tmp_path = tmp.name

    try:
      result = run_rollout_node._parse_dict_arg(tmp_path)
      self.assertEqual(result, {"key1": "value1", "key2": {"nested": 42}})
    finally:
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  def test_parse_dict_arg_empty_and_invalid(self):
    self.assertEqual(run_rollout_node._parse_dict_arg(None), {})
    self.assertEqual(run_rollout_node._parse_dict_arg(""), {})
    self.assertEqual(run_rollout_node._parse_dict_arg("   "), {})

    with self.assertRaises(ValueError):
      run_rollout_node._parse_dict_arg("just a plain string without mapping")

    with self.assertRaises(ValueError):
      run_rollout_node._parse_dict_arg("[1, 2, 3]")

  def test_deep_merge_dicts(self):
    base = {
        "maxtext_config": {
            "model_name": "qwen3-0.6b",
            "scan_layers": True,
            "attention": "dot_product",
        },
        "default_key": "foo",
    }
    override = {
        "maxtext_config": {
            "attention": "vllm_rpa",
            "enable_nnx": True,
        },
        "custom_mamba_cache_multiplier": 16,
    }
    merged = run_rollout_node._deep_merge_dicts(base, override)
    self.assertEqual(merged["default_key"], "foo")
    self.assertEqual(merged["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(merged["maxtext_config"]["model_name"], "qwen3-0.6b")
    self.assertTrue(merged["maxtext_config"]["scan_layers"])
    self.assertEqual(merged["maxtext_config"]["attention"], "vllm_rpa")
    self.assertTrue(merged["maxtext_config"]["enable_nnx"])

  def test_parse_args_all_user_flags_kebab_case(self):
    argv = [
        "--max-model-len=65536",
        "--max-num-batched-tokens=2048",
        "--max-num-seqs=8",
        "--gpu-memory-utilization=0.9",
        "--data-parallel-size=1",
        "--enable-expert-parallel",
        "--additional-config",
        '{"custom_mamba_cache_multiplier": 16}',
        "--enable-prefix-caching",
        "--prefix-cache-retention-interval",
        "256",
        "--kv-cache-dtype=bfloat16",
        "--block-size=256",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser=hermes",
        "--reasoning-parser=deepseek_r1",
        "--default-chat-template-kwargs",
        '{"enable_thinking": true}',
        "--limit-mm-per-prompt",
        '{"image": 4}',
    ]
    args = run_rollout_node._parse_args(argv)
    self.assertEqual(args.max_model_len, 65536)
    self.assertEqual(args.max_num_batched_tokens, 2048)
    self.assertEqual(args.max_num_seqs, 8)
    self.assertEqual(args.gpu_memory_utilization, 0.9)
    self.assertEqual(args.data_parallel_size, 1)
    self.assertTrue(args.enable_expert_parallel)
    self.assertEqual(
        args.additional_config, {"custom_mamba_cache_multiplier": 16}
    )
    self.assertTrue(args.enable_prefix_caching)
    self.assertEqual(args.prefix_cache_retention_interval, 256)
    self.assertEqual(args.kv_cache_dtype, "bfloat16")
    self.assertEqual(args.block_size, 256)
    self.assertTrue(args.async_scheduling)
    self.assertTrue(args.enable_chunked_prefill)
    self.assertTrue(args.language_model_only)
    self.assertTrue(args.enable_auto_tool_choice)
    self.assertEqual(args.tool_call_parser, "hermes")
    self.assertEqual(args.reasoning_parser, "deepseek_r1")
    self.assertEqual(
        args.default_chat_template_kwargs, {"enable_thinking": True}
    )
    self.assertEqual(args.limit_mm_per_prompt, {"image": 4})

  def test_parse_args_snake_case(self):
    argv = [
        "--max_model_len=32768",
        "--max_num_batched_tokens=1024",
        "--max_num_seqs=4",
        "--gpu_memory_utilization=0.85",
        "--data_parallel_size=2",
        "--enable_expert_parallel",
        "--prefix_cache_retention_interval=128",
        "--kv_cache_dtype=float8",
        "--block_size=128",
        "--async_scheduling",
        "--enable_chunked_prefill",
        "--language_model_only",
        "--enable_auto_tool_choice",
        "--tool_call_parser=mistral",
        "--reasoning_parser=default",
    ]
    args = run_rollout_node._parse_args(argv)
    self.assertEqual(args.max_model_len, 32768)
    self.assertEqual(args.max_num_batched_tokens, 1024)
    self.assertEqual(args.max_num_seqs, 4)
    self.assertEqual(args.gpu_memory_utilization, 0.85)
    self.assertEqual(args.data_parallel_size, 2)
    self.assertTrue(args.enable_expert_parallel)
    self.assertEqual(args.prefix_cache_retention_interval, 128)
    self.assertEqual(args.kv_cache_dtype, "float8")
    self.assertEqual(args.block_size, 128)
    self.assertTrue(args.async_scheduling)
    self.assertTrue(args.enable_chunked_prefill)
    self.assertTrue(args.language_model_only)
    self.assertTrue(args.enable_auto_tool_choice)
    self.assertEqual(args.tool_call_parser, "mistral")
    self.assertEqual(args.reasoning_parser, "default")

  def test_parse_args_extra_unknown_vllm_kwargs(self):
    argv = [
        "--seed=42",
        "--scheduling-policy=priority",
        "--trust-remote-code",
        "--some-future-vllm-flag=123",
    ]
    args = run_rollout_node._parse_args(argv)
    self.assertEqual(args.extra_vllm_kwargs["seed"], 42)
    self.assertEqual(args.extra_vllm_kwargs["scheduling_policy"], "priority")
    self.assertTrue(args.extra_vllm_kwargs["trust_remote_code"])
    self.assertEqual(args.extra_vllm_kwargs["some_future_vllm_flag"], 123)

  def test_create_inprocess_vllm_sampler_wires_arguments(self):
    mock_vllm_sampler = mock.MagicMock()
    mock_vllm_config = mock.MagicMock()
    mock_vllm_sampler.VllmConfig = mock_vllm_config

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        data_parallel_size=1,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        block_size=256,
        async_scheduling=True,
        enable_chunked_prefill=True,
        language_model_only=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        default_chat_template_kwargs=None,
        limit_mm_per_prompt=None,
        prefix_cache_retention_interval=256,
        kv_cache_dtype="bfloat16",
        additional_config={"custom_mamba_cache_multiplier": 16},
        maxtext_model_name="",
        maxtext_attention="",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        extra_vllm_kwargs={},
    )

    with mock.patch.object(
        run_rollout_node, "_import_vllm_sampler", return_value=mock_vllm_sampler
    ), mock.patch.object(
        run_rollout_node, "_create_rollout_mesh", return_value=None
    ):
      run_rollout_node._create_inprocess_vllm_sampler(args, mock_tokenizer)

    mock_vllm_config.assert_called_once()
    _, kwargs = mock_vllm_config.call_args
    self.assertEqual(kwargs["data_parallel_size"], 1)
    self.assertEqual(kwargs["hbm_utilization"], 0.9)
    self.assertEqual(
        kwargs["additional_config"], {"custom_mamba_cache_multiplier": 16}
    )
    engine_kwargs = kwargs["engine_kwargs"]
    self.assertEqual(engine_kwargs["max_model_len"], 65536)
    self.assertEqual(engine_kwargs["max_num_batched_tokens"], 2048)
    self.assertEqual(engine_kwargs["max_num_seqs"], 8)
    self.assertEqual(engine_kwargs["block_size"], 256)
    self.assertTrue(engine_kwargs["enable_prefix_caching"])
    self.assertEqual(engine_kwargs["prefix_cache_retention_interval"], 256)
    self.assertEqual(engine_kwargs["kv_cache_dtype"], "bfloat16")
    self.assertTrue(engine_kwargs["async_scheduling"])
    self.assertTrue(engine_kwargs["enable_chunked_prefill"])
    # Reserved keys must NOT be present in engine_kwargs
    self.assertNotIn("data_parallel_size", engine_kwargs)
    self.assertNotIn("tensor_parallel_size", engine_kwargs)
    self.assertNotIn("gpu_memory_utilization", engine_kwargs)

  def test_create_vllm_sampler_wires_arguments(self):
    mock_async_engine_args = mock.MagicMock()
    mock_vllm_adapter = mock.MagicMock()

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        data_parallel_size=1,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        enable_expert_parallel=True,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        block_size=256,
        async_scheduling=True,
        enable_chunked_prefill=True,
        language_model_only=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        default_chat_template_kwargs=None,
        limit_mm_per_prompt=None,
        prefix_cache_retention_interval=256,
        kv_cache_dtype="bfloat16",
        additional_config={"custom_mamba_cache_multiplier": 16},
        maxtext_model_name="",
        maxtext_attention="",
        prefuse_moe_weights=True,
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        extra_vllm_kwargs={},
    )

    with mock.patch(
        "vllm.engine.arg_utils.AsyncEngineArgs", mock_async_engine_args
    ):
      run_rollout_node._create_vllm_sampler(args, mock_tokenizer)

    mock_async_engine_args.assert_called_once()
    _, kwargs = mock_async_engine_args.call_args
    self.assertEqual(kwargs["max_model_len"], 65536)
    self.assertEqual(kwargs["max_num_batched_tokens"], 2048)
    self.assertEqual(kwargs["max_num_seqs"], 8)
    self.assertEqual(kwargs["gpu_memory_utilization"], 0.9)
    self.assertEqual(kwargs["data_parallel_size"], 1)
    self.assertTrue(kwargs["enable_expert_parallel"])
    self.assertTrue(kwargs["enable_prefix_caching"])
    self.assertEqual(kwargs["prefix_cache_retention_interval"], 256)
    self.assertEqual(kwargs["kv_cache_dtype"], "bfloat16")
    self.assertEqual(kwargs["block_size"], 256)
    self.assertTrue(kwargs["async_scheduling"])
    self.assertTrue(kwargs["enable_chunked_prefill"])
  def test_inprocess_vllm_sampler_with_maxtext_and_user_merging(self):
    mock_vllm_sampler = mock.MagicMock()
    mock_vllm_config = mock.MagicMock()
    mock_vllm_sampler.VllmConfig = mock_vllm_config

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        data_parallel_size=1,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        maxtext_model_name="Qwen/Qwen3.5-397B-A17B",
        maxtext_attention="vllm_rpa",
        prefuse_moe_weights=True,
        additional_config={
            "custom_mamba_cache_multiplier": 16,
            "maxtext_config": {"scan_layers": False},
        },
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        extra_vllm_kwargs={"scheduling_policy": "priority", "tensor_parallel_size": 99},
    )

    fake_built_maxtext = {
        "maxtext_config": {
            "model_name": "qwen3-0.6b",
            "attention": "vllm_rpa",
            "scan_layers": True,
        }
    }
    with mock.patch.object(
        run_rollout_node, "_import_vllm_sampler", return_value=mock_vllm_sampler
    ), mock.patch.object(
        run_rollout_node, "_create_rollout_mesh", return_value=None
    ), mock.patch.object(
        run_rollout_node.maxtext_utils,
        "build_vllm_maxtext_additional_config",
        return_value=fake_built_maxtext,
    ):
      run_rollout_node._create_inprocess_vllm_sampler(args, mock_tokenizer)

    mock_vllm_config.assert_called_once()
    _, kwargs = mock_vllm_config.call_args
    cfg = kwargs["additional_config"]
    self.assertEqual(cfg["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(cfg["maxtext_config"]["model_name"], "qwen3-0.6b")
    self.assertEqual(cfg["maxtext_config"]["attention"], "vllm_rpa")
    # User override took effect
    self.assertFalse(cfg["maxtext_config"]["scan_layers"])
    # extra_vllm_kwargs forwarded but reserved keys filtered
    self.assertEqual(kwargs["engine_kwargs"]["scheduling_policy"], "priority")
    self.assertEqual(kwargs["tensor_parallel_size"], 1)

  def test_vllm_sampler_with_maxtext_and_user_merging(self):
    mock_async_engine_args = mock.MagicMock()
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.encode.return_value = [101]

    args = argparse.Namespace(
        worker_id="worker_0",
        model_name="Qwen/Qwen3.5-397B-A17B",
        model_id="Qwen/Qwen3.5-397B-A17B",
        model_dir="",
        tokenizer_path=None,
        max_prompt_length=512,
        max_response_length=512,
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        data_parallel_size=1,
        mesh_tp=1,
        mesh_fsdp=1,
        sampler_mesh_tp=None,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        enable_expert_parallel=True,
        maxtext_model_name="Qwen/Qwen3.5-397B-A17B",
        maxtext_attention="vllm_rpa",
        prefuse_moe_weights=True,
        additional_config={
            "custom_mamba_cache_multiplier": 16,
            "maxtext_config": {"scan_layers": False},
        },
        use_lora=False,
        lora_rank=16,
        weight_sync_mode="none",
        env_name="test_env",
        agent_name="test_agent",
        agent_config_json="{}",
        eos_tokens="",
        extra_vllm_kwargs={"scheduling_policy": "priority"},
    )

    fake_built_maxtext = {
        "maxtext_config": {
            "model_name": "qwen3-0.6b",
            "scan_layers": True,
        }
    }
    with mock.patch(
        "vllm.engine.arg_utils.AsyncEngineArgs", mock_async_engine_args
    ), mock.patch.object(
        run_rollout_node.maxtext_utils,
        "build_vllm_maxtext_additional_config",
        return_value=fake_built_maxtext,
    ):
      run_rollout_node._create_vllm_sampler(args, mock_tokenizer)

    mock_async_engine_args.assert_called_once()
    _, kwargs = mock_async_engine_args.call_args
    cfg = kwargs["additional_config"]
    self.assertEqual(cfg["custom_mamba_cache_multiplier"], 16)
    self.assertEqual(cfg["maxtext_config"]["model_name"], "qwen3-0.6b")
    self.assertFalse(cfg["maxtext_config"]["scan_layers"])
    self.assertEqual(kwargs["scheduling_policy"], "priority")


if __name__ == "__main__":
  unittest.main()
