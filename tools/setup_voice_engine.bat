@echo off
REM ── Installs the native Khmer voice engine (Seed-VC voice conversion) ──
REM Needs: Python 3.11 on PATH, an NVIDIA GPU with a recent driver, ~4 GB free disk, internet.
REM Everything goes into this "tools" folder — your main Python install is not touched.
setlocal
cd /d "%~dp0"

if not exist vc-env\Scripts\python.exe (
  echo Creating voice engine environment...
  python -m venv vc-env || goto :fail
)
vc-env\Scripts\python.exe -m pip install --upgrade pip
echo Installing PyTorch for NVIDIA GPU (about 3.5 GB, takes a while)...
vc-env\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail
vc-env\Scripts\python.exe -m pip install "transformers==4.46.3" "librosa==0.10.2" munch einops matplotlib soundfile pyyaml "huggingface_hub<1.0" scipy tqdm pyworld || goto :fail

if not exist models mkdir models
if not exist models\bigvgan mkdir models\bigvgan
if not exist models\whisper-small mkdir models\whisper-small
set HF=https://huggingface.co
echo Downloading voice models (about 0.9 GB)...
curl -L -C - -o models\config_dit_mel_seed_uvit_whisper_small_wavenet.yml %HF%/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_whisper_small_wavenet.yml
curl -L -C - -o models\DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth %HF%/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth
curl -L -C - -o models\campplus_cn_common.bin %HF%/funasr/campplus/resolve/main/campplus_cn_common.bin
curl -L -C - -o models\bigvgan\config.json %HF%/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/config.json
curl -L -C - -o models\bigvgan\bigvgan_generator.pt %HF%/nvidia/bigvgan_v2_22khz_80band_256x/resolve/main/bigvgan_generator.pt
echo Downloading the lively 44 kHz model that keeps the speech melody (about 1.5 GB)...
if not exist models\bigvgan44k mkdir models\bigvgan44k
curl -L -C - -o models\config_dit_mel_seed_uvit_whisper_base_f0_44k.yml %HF%/Plachta/Seed-VC/resolve/main/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml
curl -L -C - -o models\DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth %HF%/Plachta/Seed-VC/resolve/main/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth
curl -L -C - -o models\rmvpe.pt %HF%/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
curl -L -C - -o models\bigvgan44k\config.json %HF%/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/config.json
curl -L -C - -o models\bigvgan44k\bigvgan_generator.pt %HF%/nvidia/bigvgan_v2_44khz_128band_512x/resolve/main/bigvgan_generator.pt
curl -L -C - -o models\whisper-small\config.json %HF%/openai/whisper-small/resolve/main/config.json
curl -L -C - -o models\whisper-small\preprocessor_config.json %HF%/openai/whisper-small/resolve/main/preprocessor_config.json

if not exist models\whisper-small\pytorch_model.bin (
  if exist "%USERPROFILE%\.cache\whisper\small.pt" (
    echo Converting your existing Whisper small model...
    vc-env\Scripts\python.exe convert_whisper_small.py || goto :fail
  ) else (
    echo Downloading Whisper small encoder ^(about 1 GB^)...
    curl -L -C - -o models\whisper-small\model.safetensors %HF%/openai/whisper-small/resolve/main/model.safetensors
  )
)
echo.
echo Voice engine ready. Restart the app.
exit /b 0

:fail
echo.
echo Setup failed — see the message above.
exit /b 1
