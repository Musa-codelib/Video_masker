# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('checkpoints/sam2.1_hiera_tiny.pt', 'checkpoints'), ('configs/sam2.1_hiera_t.yaml', '.'), ('assets/logo.png', '.'), ('assets/logo.icns', '.')]
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
    name='Mk Masker Pro 1.2 Lite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    # After build, run: codesign --sign - --force --deep <app_path>
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/logo.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Mk Masker Pro 1.2 Lite',
)
app = BUNDLE(
    coll,
    name='Mk Masker Pro 1.2 Lite.app',
    icon='assets/logo.icns',
    bundle_identifier=None,
)
