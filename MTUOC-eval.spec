# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Recollida de fitxers de dades necessaris
datas = [
    ('config.yaml', '.'),
]
datas += collect_data_files('sacrebleu')
datas += collect_data_files('transformers')
datas += collect_data_files('sentence_transformers')

# Importacions ocultes per evitar errors de mòduls no trobats en temps d'execució
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
    'bert_score',
    'comet',
    'sentence_transformers',
    'matplotlib.backends.backend_qtagg',
]
hiddenimports += collect_submodules('sacrebleu')
hiddenimports += collect_submodules('transformers')
hiddenimports += collect_submodules('sentence_transformers')
hiddenimports += collect_submodules('torchmetrics')
hiddenimports += collect_submodules('pytorch_lightning')
hiddenimports += collect_submodules('comet')
hiddenimports += collect_submodules('bert_score')

a = Analysis(
    ['MTUOC-eval.py'],
    pathex=[],
    binaries=[],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='MTUOC-eval',
)
