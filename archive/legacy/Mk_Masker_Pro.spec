# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('checkpoints/sam2_hiera_small.pt', 'checkpoints'), ('sam2_hiera_s.yaml', '.'), ('logo.png', '.')]
binaries = [('ffmpeg', '.')]
hiddenimports = ['torchvision', 'PIL', 'cv2']
datas += copy_metadata('torch')
datas += copy_metadata('tqdm')
datas += copy_metadata('numpy')
tmp_ret = collect_all('sam2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Mk_Masker_Pro',
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
    icon=['logo.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Mk_Masker_Pro',
)
app = BUNDLE(
    coll,
    name='Mk_Masker_Pro.app',
    icon='logo.icns',
    bundle_identifier=None,
)
