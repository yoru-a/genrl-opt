import gc
import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM


from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP

def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    ref_model = deepcopy(model)
    for param in model.parameters():
        param.requires_grad = False
    return ref_model.eval()

def detect_cpu_vendor():
    vendor = "other"
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read()
        if 'GenuineIntel' in cpuinfo:
            vendor = "intel"
        elif 'AuthenticAMD' in cpuinfo:
            vendor = "amd"
        elif 'ARM' in cpuinfo or 'aarch64' in cpuinfo:
            vendor = "arm"
    except Exception:
        pass
    return vendor

def detect_cuda_environment():
    """Detect CUDA version and GPU compute capability."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have an NVIDIA GPU and CUDA drivers installed.")
    cuda_version = torch.version.cuda
    device_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    capability = torch.cuda.get_device_capability(device_idx)
    return {
        "cuda_version": cuda_version,
        "device_name": device_name,
        "compute_capability": capability,
        "device_idx": device_idx
    }

def check_vllm_bnb_compatibility(selected_quantization):
    if selected_quantization in ("bnb_4bit", "bnb_8bit", "vllm"):
        env = detect_cuda_environment()
        major, minor = env["compute_capability"]
        if major < 7:
            raise RuntimeError(
                f"Your GPU ({env['device_name']}, compute capability {major}.{minor}) does not support vLLM/bitsandbytes quantization (requires compute capability >= 7.0)."
            )
        return env
    return None

@dataclass
class GRPOTrainerConfig:
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    temperature: float = 1.0
    dtype: str = "float32"
    enable_gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    max_tokens: int = 256
    num_generations: int = 2
    learning_rate: float = 1e-5
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    num_iterations: int = 1
    optimizer: str = "Adam"
    quantization: str = "none"  # one of: "none", "bnb_4bit", "bnb_8bit", "onnx_int8", "openvino_int8"
    use_vllm: bool = False
    device_map: Optional[str] = "auto"
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    force_cpu: bool = False # force CPU even if GPU present
    gpu_memory_utilization: float = 0.9
    
class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    GRPO trainer supporting vLLM, bitsandbytes (4/8bit), CPU int8, and standard full-precision.
    Supports dtype selection for GPU paths (float32/float16).
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")
        self.args = config
        self.quantization = getattr(self.args, "quantization", "none")
        self.use_vllm = getattr(self.args, "use_vllm", False)
        self.device_map = getattr(self.args, "device_map", "auto")
        self.trust_remote_code = getattr(self.args, "trust_remote_code", False)
        self.low_cpu_mem_usage = getattr(self.args, "low_cpu_mem_usage", True)
        self.optimizer_type = getattr(self.args, "optimizer", "Adam")
        self.force_cpu = getattr(self.args, "force_cpu", False)
        self.dtype_config = getattr(self.args, "dtype", "float32") # "float32" or "float16"

        # Hardware detection
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available()

        # --- DType selection (enforce float32 for CPU, allow config for GPU) ---
        # - For force_cpu or cpu_int8 or CPU-only: float32
        # - For GPU vLLM/bnb/none: user config (float32/float16)
        if self.force_cpu or self.quantization == "cpu_int8" or (not has_cuda and not has_mps):
            dtype_str = "float32"
        else:
            dtype_str = self.dtype_config
        self.dtype = DTYPE_MAP.get(dtype_str, torch.float32)
        self.dtype_str = dtype_str  # for vLLM which takes a string

        # --- Device selection ---
        # Priority:
        # 1. If force_cpu: always use CPU (no vLLM or bnb allowed)
        # 2. If quantization == cpu_int8: use CPU
        # 3. If CUDA and not force_cpu: use CUDA
        # 4. If MPS and not force_cpu: use MPS
        # 5. Else: use CPU
        if self.force_cpu:
            self.device = torch.device("cpu")
            if self.use_vllm:
                raise RuntimeError("vLLM cannot be used with force_cpu=True.")
            if self.quantization in ("bnb_4bit", "bnb_8bit"):
                raise RuntimeError("bitsandbytes quantization cannot be used with force_cpu=True.")
        elif self.quantization in ("onnx_int8", "openvino_int8"):
            self.device = torch.device("cpu")
        elif has_cuda and not self.force_cpu:
            self.device = torch.device("cuda")
        elif has_mps and not self.force_cpu:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # --- Backend selection logic ---
        # vLLM only allowed if on GPU and not force_cpu
        if self.use_vllm:
            if self.force_cpu or not has_cuda:
                raise RuntimeError("vLLM requires a CUDA GPU and cannot be used with force_cpu=True.")
            check_vllm_bnb_compatibility(self.quantization)
        if self.quantization in ("bnb_4bit", "bnb_8bit"):
            if self.force_cpu or not has_cuda:
                raise RuntimeError(f"bitsandbytes {self.quantization} quantization requires a CUDA GPU and cannot be used with force_cpu=True.")
            check_vllm_bnb_compatibility(self.quantization)
        # int8 quantization only allowed on CPU
        if self.quantization in ("onnx_int8", "openvino_int8") and self.device.type != "cpu":
            raise RuntimeError("int8 quantization is only for CPU hosts. Please use quantization: none or bnb_4bit/bnb_8bit for GPU.")

        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.callbacks = kwargs.get("callbacks", [])
        self.global_step = 0
        self._total_train_tokens = 0

        # --- Model backend setup ---
        self.model = models[0]
        self.inference_model = None
        self.vllm_engine = None

        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)

        # --- Core initializations ---
        self._initialize_model(self.args.enable_gradient_checkpointing)
        self._initialize_optimizer()
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self._initialize_inference_backend()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))


        assert self.args.num_generations > 1, f"For GRPO training, number of generations must be > 1, got {self.args.num_generations}"

    def _initialize_optimizer(self):
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
            
    def _initialize_model(self, enable_gradient_checkpointing):
        # Optimum ONNX/OpenVINO int8 support
        if self.quantization in ("onnx_int8", "openvino_int8"):
            model_name_or_path = getattr(getattr(self.model, "config", None), "_name_or_path", None)
            cpu_vendor = detect_cpu_vendor()
            # Diagnostic logging
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path)
            print(f"[DEBUG] Model name/path: {model_name_or_path}")
            print(f"[DEBUG] Model type: {config.model_type}")
            print(f"[DEBUG] Architectures: {config.architectures}")
            # OpenVINO for Intel, ONNX for others
            if self.quantization == "openvino_int8" or (self.quantization == "onnx_int8" and cpu_vendor == "intel"):
                if OVModelForCausalLM is None:
                    raise ImportError("Optimum OpenVINO not installed. Run 'pip install optimum[openvino]'")
                print("[INFO] Loading model with Optimum OpenVINO backend (int8)...")
                ov_model = OVModelForCausalLM.from_pretrained(model_name_or_path)
                self.model = ov_model
            else:
                if ORTModelForCausalLM is None:
                    raise ImportError("Optimum ONNX Runtime not installed. Run 'pip install optimum[onnxruntime]'")
                print("[INFO] Loading model with Optimum ONNX Runtime backend (int8)...")
                print("[DEBUG] About to call ORTModelForCausalLM.from_pretrained (ONNX export).")
                ort_model = ORTModelForCausalLM.from_pretrained(model_name_or_path)
                self.model = ort_model
        else:
            # Standard fallback path: float32/float16/other, not quantized
            try:
                self.model = self.model.to(device=self.device, dtype=self.dtype)
            except Exception:
                self.model = self.model.to(device=self.device)
        if enable_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass
        if getattr(self.args, "beta", 0.0) == 0.0:
            self.ref_model = None
        else:
            try:
                self.ref_model = create_reference_model(self.model).to(device=self.device, dtype=self.dtype)
            except Exception:
                self.ref_model = create_reference_model(self.model).to(device=self.device)

    def _initialize_inference_backend(self):
        model_name_or_path = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        # vLLM (GPU only, not force_cpu)
        if self.use_vllm:
            try:
                from vllm import LLM
            except Exception:
                raise ImportError("vLLM backend requested but vllm is not installed. Run `pip install vllm`.")
            if model_name_or_path is None:
                raise ValueError("vLLM requires model.config._name_or_path to be set.")
            self.vllm_engine = LLM(
                model=model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype_str,
                gpu_memory_utilization=self.args.gpu_memory_utilization
            )
            return
        # bitsandbytes (GPU only, not force_cpu)
        if self.quantization in ("bnb_4bit", "bnb_8bit"):
            if model_name_or_path is None:
                raise ValueError("bitsandbytes quantization requires model.config._name_or_path to be set.")
            check_model_bnb_compatibility(model_name_or_path, self.quantization)
            quant_config_kwargs = {}
            if self.quantization == "bnb_4bit":
                quant_config_kwargs = dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.quantization == "bnb_8bit":
                quant_config_kwargs = dict(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True
                )
            quant_config = BitsAndBytesConfig(**quant_config_kwargs)
            inf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=quant_config,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=self.low_cpu_mem_usage
            )
            self.inference_model = inf_model
            
    def _initialize_tokenizers(self):
        if self.processing_class is None:
            model_name_or_path = getattr(getattr(self.model, "config", None), "_name_or_path", None)
            if model_name_or_path is None:
                raise ValueError("Tokenizer requires model.config._name_or_path to be set when processing_class not provided")
            self.processing_class = AutoTokenizer.from_pretrained(
                model_name_or_path, padding_side="left"
            )

    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _initialize_generation_config(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            pad_token_id=self.processing_class.pad_token_id,
            bos_token_id=self.processing_class.bos_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]
        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.args.num_generations):
                        templated_prompts.append(
                            self.processing_class.apply_chat_template(
                                item["prompt"], tokenize=False, add_generation_prompt=True
                            )
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(
                        item["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    for item in inputs
                ]
        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] for item in inputs]
        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    def _build_prompts(self, inputs: Any, for_training: bool = False, with_template: bool = True) -> list[str]:
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]
        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.args.num_generations):
                        templated_prompts.append(
                            self.processing_class.apply_chat_template(
                                item["prompt"], tokenize=False, add_generation_prompt=True
                            )
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(
                        item["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    for item in inputs
                ]
        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] for item in inputs]
        return templated_prompts

    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0) -> Any:
        if self.vllm_engine is not None:
            if return_completion_ids:
                raise NotImplementedError("return_completion_ids is not supported with vLLM backend")
            try:
                from vllm import SamplingParams
            except Exception as exc:
                raise RuntimeError("vLLM is not installed but requested.") from exc
            prompts = self._build_prompts(inputs, for_training=False, with_template=True)
            sampling_params = SamplingParams(
                n=self.args.num_generations,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k if self.args.top_k is not None else -1,
                min_p=self.args.min_p if self.args.min_p is not None else 0,
                max_tokens=self.args.max_new_tokens,
                repetition_penalty=self.args.repetition_penalty,
            )
            request_outputs = self.vllm_engine.generate(prompts, sampling_params)
            rollout = []
            for req in request_outputs:
                texts = [out.text for out in req.outputs]
                rollout.append(texts)
            return rollout

        inference_model = self.inference_model or self.model
        input_tokens = self._process_inputs(inputs)
        rollout = []
        rollout_ids = []
        for _ in range(self.args.num_generations):
            with torch.no_grad():
                if self.inference_model is not None and self.quantization in {"bnb_4bit", "bnb_8bit"}:
                    device = self.inference_model.device
                    outputs = inference_model.generate(
                     input_ids=input_tokens.input_ids.to(device),
                     attention_mask=input_tokens.attention_mask.to(device),
                     generation_config=self.generation_config,
                    )
                elif self.inference_model is not None and self.quantization == "cpu_int8":
                    outputs = inference_model.generate(
                        input_ids=input_tokens.input_ids.to("cpu"),
                        attention_mask=input_tokens.attention_mask.to("cpu"),
                        generation_config=self.generation_config,
                    )
                else:
                    device = inference_model.device
                    outputs = inference_model.generate(
                        input_ids=input_tokens.input_ids.to(device),
                        attention_mask=input_tokens.attention_mask.to(device, dtype=self.dtype),
                        generation_config=self.generation_config,
                    )
            prompt_length = input_tokens.input_ids.size(1)
            completion_ids = outputs[:, prompt_length:]
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])
        if return_completion_ids:
            return rollout, rollout_ids
        return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]
        loss_mask = (
            attention_mask[:, -logits_to_keep:].to(device=logits.device, dtype=logits.dtype).contiguous()
        )
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()
        logits = logits / self.args.temperature
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits.shape[0], logits.shape[1])
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        )
        return token_log_probs

    def compute_loss(self, model, inputs, mode="train", return_metrics=False):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device, dtype=self.dtype
        )
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        if getattr(self.args, "beta", 0.0) != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                ref_per_token_logps = per_token_logps.clone()
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.args.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.epsilon,
            1 + self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon,
        )
        advantages = advantages.unsqueeze(dim=-1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if getattr(self.args, "beta", 0.0) != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        if getattr(self.args, "beta", 0.0) != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(mean_kl.item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())
        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if getattr(self.args, "beta", 0.0) != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
        }
        if return_metrics:
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """
        Train the model using the given game state and reward manager.

        Args:
            game_state: The current game state.
            reward_manager: The reward manager to use for computing rewards.
        """
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1
        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"
        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"], model_inputs["prompt_mask"] = (
            processed_inputs.input_ids.to(self.model.device),
            processed_inputs.attention_mask.to(self.model.device),
        )
        processed_outputs = self._process_inputs(
            stage_outputs, with_template=False, for_training=True
        )
        model_inputs["completion_ids"], model_inputs["completion_mask"] = (
            processed_outputs.input_ids.to(self.model.device),
            processed_outputs.attention_mask.to(self.model.device),
        )
        rewards = reward_manager[stage]
        rewards = [
            rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards)
        with torch.no_grad():
            advantages = rewards - rewards.mean(dim=1, keepdim=True)
            if rewards.shape[1] > 1:
                advantages /= rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = torch.flatten(advantages).to(self.model.device, dtype=self.dtype)
        model_inputs["advantages"] = advantages.squeeze(dim=-1)
        model_inputs["old_per_token_logps"] = None
        loss = self.compute_loss(self.model, model_inputs)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        metrics = {"train/loss": loss.cpu().mean().item()}
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)
        self.cleanup_step()
        return global_step

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        pass

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        try:
            self.model.save_pretrained(save_dir)
        except Exception:
            pass
        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config,
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        trainer = cls([model], GRPOTrainerConfig())
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"))
        trainer._metrics = trainer_state.get("metrics", {"train": defaultdict(list), "eval": defaultdict(list)})
        trainer._total_train_tokens = trainer_state.get("total_train_tokens", 0)
        trainer.generation_config = trainer_state.get("generation_config", trainer.generation_config)
        return trainer
        
    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
