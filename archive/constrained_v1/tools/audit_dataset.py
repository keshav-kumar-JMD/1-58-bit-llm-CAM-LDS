"""Offline token audit after downloading tokenizer files only (no model weights)."""
import json
from pathlib import Path
import sys
from functools import lru_cache
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.cache' / 'audit-packages'))
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment
from camlds.core import clean_prompt, load_queries, plan_chunks
from camlds.models import MODELS


def main():
    records = load_queries(ROOT / 'data' / 'llm_queries.json')
    report = {'records': len(records), 'changed_by_cleanup': sum(clean_prompt(r['prompt']) != r['prompt'] for r in records), 'models': {}}
    for key, spec in MODELS.items():
        if key == 'qwen1b_q4':
            continue
        # Meta repository is gated; BitNet publishes the same Llama-3 vocabulary.
        # The authoritative GGUF tokenizer/template audit runs in each GPU worker.
        repo = 'microsoft/bitnet-b1.58-2B-4T' if key == 'llama3b' else spec['tokenizer']
        directory = ROOT / '.cache' / 'tokenizers' / key
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for filename in ['tokenizer.json', 'tokenizer_config.json', 'config.json']:
                destination = directory / filename
                if not destination.exists():
                    with urlopen(f'https://huggingface.co/{repo}/resolve/main/{filename}', timeout=120) as response:
                        destination.write_bytes(response.read())
            config = json.loads((directory / 'tokenizer_config.json').read_text(encoding='utf-8'))
            model_config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
            template = config['chat_template']
            if isinstance(template, list):
                template = next(t['template'] for t in template if t['name'] == 'default')
            environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            environment.globals['raise_exception'] = lambda message: (_ for _ in ()).throw(ValueError(message))
            render = environment.from_string(template)
            @lru_cache(maxsize=20000)
            def count(prompt):
                text = render.render(messages=[{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                    bos_token=config.get('bos_token', ''), eos_token=config.get('eos_token', ''))
                return len(tokenizer.encode(text, add_special_tokens=False).ids)
            lengths = [count(clean_prompt(r['prompt'])) for r in records]
            chunks = [plan_chunks(clean_prompt(r['prompt']), count, spec['context']) for r in records]
            sorted_lengths = sorted(lengths)
            report['models'][key] = dict(tokenizer_repo=repo, tokenizer_source_context=model_config.get('max_position_embeddings'), model_native_limit=spec['native'],
                caveat='Llama-3 vocabulary proxy; GGUF worker audit authoritative' if key == 'llama3b' else 'HF template; GGUF worker verifies its own template',
                minimum=min(lengths), median=sorted_lengths[len(lengths)//2], p95=sorted_lengths[int(len(lengths)*.95)],
                maximum=max(lengths), context=spec['context'], output_reserve=1024,
                chunked_rows=sum(len(c) > 1 for c in chunks), max_chunks=max(map(len, chunks)),
                max_chunk_tokens=max(count(c) for row in chunks for c in row),
                former_failures={str(i): lengths[i-1] for i in [2, 89, 172]})
            print(key, report['models'][key], flush=True)
        except Exception as error:
            report['models'][key] = {'error': f'{type(error).__name__}: {error}'}
            print(key, report['models'][key], flush=True)
    (ROOT / 'review').mkdir(exist_ok=True)
    (ROOT / 'review' / 'dataset_token_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
