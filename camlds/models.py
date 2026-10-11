"""Model registry and lazy-loaded inference backends."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import subprocess

MODELS = {
    'qwen1b': dict(repo='ThomasBaruzier/Qwen2.5-1.5B-Instruct-GGUF', file='Qwen2.5-1.5B-Instruct-IQ1_M.gguf', tokenizer='Qwen/Qwen2.5-1.5B-Instruct', context=16384, native=32768, vram_gib=5),
    'hermes': dict(repo='mradermacher/OpenHermes-Qwen1.5-1.8B-i1-GGUF', file='OpenHermes-Qwen1.5-1.8B.i1-IQ1_M.gguf', tokenizer='vicgalle/OpenHermes-Qwen1.5-1.8B', context=16384, native=32768, vram_gib=7),
    'bitnet2b': dict(repo='microsoft/bitnet-b1.58-2B-4T', tokenizer='microsoft/bitnet-b1.58-2B-4T', context=4096, native=4096, vram_gib=9, backend='transformers'),
    'llama3b': dict(repo='bartowski/Llama-3.2-3B-Instruct-GGUF', file='Llama-3.2-3B-Instruct-IQ3_M.gguf', tokenizer='meta-llama/Llama-3.2-3B-Instruct', context=16384, native=131072, vram_gib=7),
    'mistral7b': dict(repo='bartowski/Mistral-7B-Instruct-v0.3-GGUF', file='Mistral-7B-Instruct-v0.3-IQ1_M.gguf', tokenizer='mistralai/Mistral-7B-Instruct-v0.3', context=20480, native=32768, vram_gib=13),
    'qwen7b': dict(repo='ThomasBaruzier/Qwen2.5-7B-Instruct-GGUF', file='Qwen2.5-7B-Instruct-IQ1_M.gguf', tokenizer='Qwen/Qwen2.5-7B-Instruct', context=16384, native=32768, vram_gib=10),
}
MODELS['qwen1b_q4'] = {**MODELS['qwen1b'], 'file': 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf', 'vram_gib': 6}
DEFAULT_MODELS = list(MODELS)[:-1]


def gpu_info():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free,compute_cap', '--format=csv,noheader,nounits'], text=True)
        name, memory, capability = output.splitlines()[0].rsplit(',', 2)
        return dict(name=name.strip(), free_gib=float(memory)/1024, capability=capability.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return dict(name='CPU', free_gib=0, capability='')


def gguf_profiles(on_gpu, free_gib, low_memory=False):
    if not on_gpu:
        return [(0, 128, 32, False)]
    if low_memory:
        # A new worker must actually avoid the previous flash-attention path.
        return [(-1, 128, 32, False), (8, 128, 32, False), (0, 128, 32, False)]
    batch, micro = (2048, 512) if free_gib >= 35 else (512, 128)
    return [(-1, batch, micro, True), (-1, 256, 64, False), (8, 128, 32, False), (0, 128, 32, False)]


class GGUFBackend:
    def __init__(self, spec, revision, cache, context=None, low_memory=False):
        # Set before importing llama.cpp: the backend reads this at initialization.
        # Keep GPU offload; bypass the graph capture/replay path implicated by
        # the user's A100 worker log. No rebuild or notebook restart is required.
        os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama, llama_supports_gpu_offload
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        path = hf_hub_download(spec['repo'], spec['file'], revision=revision, cache_dir=cache)
        self.context = context or spec['context']
        gpu = gpu_info()
        on_gpu = gpu['free_gib'] > 0 and llama_supports_gpu_offload()
        self.model = None
        errors = []
        # Try full offload, conservative full offload, partial offload, then CPU.
        profiles = gguf_profiles(on_gpu, gpu['free_gib'], low_memory)
        for layers, batch, micro, flash in profiles:
            try:
                print(f'GGUF load: layers={layers}, batch={batch}, microbatch={micro}, flash={flash}, CUDA graphs=disabled', flush=True)
                self.model = Llama(model_path=path, n_ctx=self.context, n_gpu_layers=layers,
                                   n_batch=batch, n_ubatch=micro, flash_attn=flash,
                                   n_threads=min(8, os.cpu_count() or 1), seed=42,
                                   verbose=True, logits_all=False)
                self.profile = dict(layers=layers, batch=batch, microbatch=micro, flash=flash, cuda_graphs=False)
                break
            except (RuntimeError, ValueError, MemoryError) as error:
                errors.append(str(error))
                gc.collect()
        if self.model is None:
            raise RuntimeError('All load profiles failed: ' + '; '.join(errors))
        try:
            metadata = self.model.metadata
            template = metadata.get('tokenizer.chat_template')
            if not template:
                raise ValueError('GGUF has no chat template; refusing an implicit fallback.')
            native = [int(v) for k, v in metadata.items() if k.endswith('.context_length')]
            if native and self.context > min(native):
                raise ValueError(f'Configured context {self.context} exceeds GGUF native limit {min(native)}.')
            token_text = lambda token: self.model.detokenize([token], special=True).decode('utf-8') if token >= 0 else ''
            self.formatter = Jinja2ChatFormatter(template=template,
                eos_token=token_text(self.model.token_eos()), bos_token=token_text(self.model.token_bos()),
                add_generation_prompt=True)
        except BaseException:
            self.close()
            raise

    def formatted(self, prompt):
        formatted = self.formatter(messages=[{'role': 'user', 'content': prompt}])
        # Use this exact token list for both counting and generation: no hidden
        # second chat template or doubled BOS token can change the budget.
        return self.model.tokenize(formatted.prompt.encode('utf-8'), add_bos=False, special=True), formatted.stop

    def count(self, prompt):
        return len(self.formatted(prompt)[0])

    def generate(self, prompt, cap):
        tokens, stop = self.formatted(prompt)
        if len(tokens) + cap + 16 > self.context:
            raise ValueError('Prompt plus reserved output exceeds context.')
        self.model.reset()
        result = self.model.create_completion(prompt=tokens, stop=stop,
            max_tokens=cap, temperature=0.0, repeat_penalty=1.0,
            top_k=0, top_p=1.0, min_p=0.0, seed=42)
        choice = result['choices'][0]
        return dict(raw=choice['text'], finish=choice['finish_reason'],
                    input_tokens=result['usage']['prompt_tokens'], output_tokens=result['usage']['completion_tokens'])

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
        gc.collect()


class BitNetBackend:
    def __init__(self, spec, revision, cache, context=None, low_memory=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.context = context or spec['context']
        self.tokenizer = AutoTokenizer.from_pretrained(spec['repo'], revision=revision, cache_dir=cache)
        # Packed official weights, NOT the BF16 master/training checkpoint.
        # No custom CUDA extension, flash-attn install, or trust_remote_code.
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == 'cuda:0' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(spec['repo'], revision=revision,
            cache_dir=cache, torch_dtype=dtype, device_map=device, attn_implementation='eager').eval()
        self.context = min(self.context, self.model.config.max_position_embeddings)
        self.profile = dict(device=device, dtype=str(dtype), attention='eager')

    def formatted(self, prompt):
        return self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True)

    def count(self, prompt):
        return len(self.formatted(prompt))

    def generate(self, prompt, cap):
        from transformers import GenerationConfig
        torch = self.torch
        tokens = self.formatted(prompt)
        if len(tokens) + cap + 16 > self.context:
            raise ValueError('Prompt plus output exceeds BitNet context.')
        inputs = torch.tensor([tokens], device=self.model.device)
        # Explicit neutral settings; do not inherit repository repetition/format rules.
        settings = GenerationConfig(max_new_tokens=cap, do_sample=False,
            num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
        torch.manual_seed(42)
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         generation_config=settings)
        generated = output[0, len(tokens):]
        ended = len(generated) > 0 and generated[-1].item() == self.tokenizer.eos_token_id
        # Remove only the terminating EOS marker, preserve all other generated text.
        text_tokens = generated[:-1] if ended else generated
        return dict(raw=self.tokenizer.decode(text_tokens, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False),
                    finish='stop' if ended else 'length',
                    input_tokens=len(tokens), output_tokens=len(generated))

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def load_backend(spec, revision, cache, context=None, low_memory=False):
    cls = BitNetBackend if spec.get('backend') == 'transformers' else GGUFBackend
    return cls(spec, revision, cache, context, low_memory)
