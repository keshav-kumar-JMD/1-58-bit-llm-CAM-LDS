"""Idempotent, GPU-specific Linux worker environment. No kernel imports."""
from __future__ import annotations
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import venv
from .core import atomic_write, digest
from .models import gpu_info

PACKAGES = ['transformers==4.57.1', 'accelerate==1.11.0',
            'huggingface-hub==0.36.0', 'filelock==3.20.0', 'sentencepiece==0.2.1',
            'protobuf>=4.25', 'jinja2>=3.1', 'numpy>=1.26', 'diskcache>=5.6.3']
LLAMA_VERSION = '0.3.36'


def create_worker_environment(environment, executable):
    """Bootstrap through the working parent pip, without distro ensurepip.

    Colab's Python can have pip but lack a working ensurepip installation.
    Reuse a partial environment without clearing its packages or other caches.
    --ignore-installed ensures pip lives in the venv, even with system-site access.
    """
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    log_path = Path(environment) / 'bootstrap.log'
    command = [sys.executable, '-m', 'pip', '--python', str(executable),
               'install', '--ignore-installed', 'pip']
    print(f'Bootstrapping worker pip; full output: {log_path}', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-6000:]
        raise RuntimeError(f'Worker pip bootstrap failed. Full log: {log_path}\n{tail}')
    subprocess.run([str(executable), '-c',
        'import pip, sys; from pathlib import Path; '
        'assert Path(pip.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), '
        '"pip was not installed inside the worker environment"; '
        'print("Worker pip:", pip.__version__)'], check=True)


def prepare(project):
    """Reuse a versioned venv + cached wheel matching Python/CUDA/GPU capability.

    The CUDA toolkit bundled by Colab is used to compile llama.cpp. A CPU-only
    fallback keeps evaluation possible when the compiler cannot target the GPU.
    A different GPU selects another environment without unloading native code.
    """
    if platform.system() != 'Linux':
        raise RuntimeError('Run GPU setup on the Colab Linux kernel. Local data/CSV tools work on Windows.')
    project = Path(project).resolve()
    cache = project / '.cache'
    cache.mkdir(exist_ok=True)
    gpu = gpu_info()
    nvcc = shutil.which('nvcc')
    cuda_version = subprocess.check_output([nvcc, '--version'], text=True) if nvcc else 'none'
    architecture = gpu['capability'].replace('.', '')
    use_cuda = bool(nvcc and architecture)
    if use_cuda:
        supported = subprocess.check_output([nvcc, '--list-gpu-code'], text=True)
        use_cuda = f'sm_{architecture}' in supported.split()
    try:
        host_torch = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        host_torch = 'missing'
    signature = dict(python=sys.version.split()[0], machine=platform.machine(), host_torch=host_torch,
                     cuda=cuda_version, arch=architecture if use_cuda else 'cpu',
                     llama=LLAMA_VERSION, packages=PACKAGES)
    key = digest(signature)[:16]
    environment = cache / 'envs' / key
    executable = environment / 'bin' / 'python'
    ready = environment / 'ready.json'
    if ready.exists():
        print(f'Reusing worker environment: {environment}', flush=True)
        return executable
    # Removing the unused format-enforcer dependency does not invalidate an
    # otherwise identical compiled environment from the previous notebook.
    legacy_packages = PACKAGES[:2] + ['lm-format-enforcer==0.11.3'] + PACKAGES[2:]
    legacy_key = digest({**signature, 'packages': legacy_packages})[:16]
    legacy_environment = cache / 'envs' / legacy_key
    if (legacy_environment / 'ready.json').exists():
        print(f'Reusing compatible worker environment: {legacy_environment}', flush=True)
        return legacy_environment / 'bin' / 'python'
    create_worker_environment(environment, executable)
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = str(cache / 'pip')
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(min(4, os.cpu_count() or 1))
    env['TORCHDYNAMO_DISABLE'] = '1'
    def run(*args):
        subprocess.run([str(executable), *args], env=env, check=True)
    run('-m', 'pip', 'install', '--upgrade', 'pip', 'cmake', 'ninja', *PACKAGES)
    # Reuse Colab's torch if it actually executes on the assigned architecture.
    probe = 'import torch; print(torch.__version__); x=torch.ones(8,device="cuda"); print((x@x).item())' if gpu['capability'] else 'import torch; print(torch.__version__)'
    if subprocess.run([str(executable), '-c', probe], env=env).returncode:
        # CUDA 12.8 supports Blackwell as well as the usual Colab T4/A100/H100.
        run('-m', 'pip', 'install', '--upgrade', 'torch==2.9.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
        run('-c', probe)
    wheels = cache / 'wheels' / key
    wheels.mkdir(parents=True, exist_ok=True)
    existing = list(wheels.glob('llama_cpp_python-*.whl'))
    if not existing:
        env['CMAKE_ARGS'] = (f'-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={architecture} -DGGML_NATIVE=OFF'
                             if use_cuda else '-DGGML_CUDA=OFF -DGGML_NATIVE=OFF')
        print(f'Building llama.cpp for {gpu["name"]}, architecture={architecture}, CUDA={use_cuda}', flush=True)
        try:
            run('-m', 'pip', 'wheel', f'llama-cpp-python=={LLAMA_VERSION}', '--no-deps',
                '--no-binary=llama-cpp-python', '--no-cache-dir', '--wheel-dir', str(wheels))
        except subprocess.CalledProcessError:
            if not use_cuda:
                raise
            print('CUDA compilation failed; building a CPU fallback in the worker environment.', flush=True)
            use_cuda = False
            env['CMAKE_ARGS'] = '-DGGML_CUDA=OFF -DGGML_NATIVE=OFF'
            run('-m', 'pip', 'wheel', f'llama-cpp-python=={LLAMA_VERSION}', '--no-deps',
                '--no-binary=llama-cpp-python', '--no-cache-dir', '--wheel-dir', str(wheels))
        existing = list(wheels.glob('llama_cpp_python-*.whl'))
    run('-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(existing[0]))
    run('-c', 'import llama_cpp; from transformers import BitNetForCausalLM; print("Worker imports verified; GPU offload:", llama_cpp.llama_supports_gpu_offload())')
    atomic_write(ready, json.dumps({**signature, 'built_with_cuda': use_cuda}, indent=2))
    if not use_cuda:
        print('llama.cpp CPU fallback: CUDA compiler unavailable or cannot target this card. Torch BitNet can still use a compatible GPU.', flush=True)
    return executable
