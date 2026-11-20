# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_all,
    copy_metadata,
)

block_cipher = None

datas = [
    ('config.yaml', '.'),
]
binaries = []
hiddenimports = [
    'pkg_resources',
    'pkg_resources.extern',
    'Levenshtein',
    'sacrebleu',
    'yaml',
    'openpyxl',
    'torch',
    'torchvision',
    'torchaudio',
    'torchmetrics',
    'pytorch_lightning',
    'lightning_utilities',
    'bert_score',
    'comet',
    'sentence_transformers',
    'matplotlib.backends.backend_qtagg',
]

# 1. Recollida completa de sentencepiece (dades, binaris .pyd/.so/.dylib i submòduls)
try:
    sp_datas, sp_binaries, sp_hidden = collect_all('sentencepiece')
    datas += sp_datas
    binaries += sp_binaries
    hiddenimports += sp_hidden
except Exception:
    pass

# 2. Còpia explícita de les DLLs de Visual C++ Runtime NOMÉS a Windows
if sys.platform == 'win32':
    sys32 = os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32')
    vc_dlls = ['msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140_1.dll']
    for dll in vc_dlls:
        dll_path = os.path.join(sys32, dll)
        if os.path.exists(dll_path):
            binaries.append((dll_path, '.'))

# 3. Fitxers de dades addicionals (incloent version.info de pytorch_lightning i lightning_utilities)
for pkg_data in [
    'sacrebleu',
    'transformers',
    'sentence_transformers',
    'pytorch_lightning',
    'lightning_utilities',
    'torchmetrics',
    'comet',
]:
    try:
        datas += collect_data_files(pkg_data)
    except Exception:
        pass

# 4. Metadades dels paquets (evita errors de importlib.metadata a qualsevol SO)
for pkg in [
    'sentencepiece',
    'transformers',
    'tokenizers',
    'huggingface-hub',
    'safetensors',
    'torch',
    'pytorch-lightning',
    'lightning-utilities',
    'torchmetrics',
    'unbabel-comet',
    'tqdm',
    'regex',
    'requests',
    'packaging',
    'filelock',
]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# 5. Submòduls ocults
hiddenimports += collect_submodules('sacrebleu')
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('sentence_transformers')
hiddenimports += collect_submodules('torchmetrics')
hiddenimports += collect_submodules('pytorch_lightning')
hiddenimports += collect_submodules('lightning_utilities')
hiddenimports += collect_submodules('comet')
hiddenimports += collect_submodules('bert_score')

a = Analysis(
    ['MTUOC-eval.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'tensorboard',
        'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MTUOC-eval',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MTUOC-eval',
)

# 6. Generació automàtica del paquet .app només en macOS
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MTUOC-eval.app',
        icon=None,
        bundle_identifier='org.mtuoc.eval',
    )
