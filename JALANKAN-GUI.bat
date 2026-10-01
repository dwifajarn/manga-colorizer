@echo off
REM ============================================================
REM  Manga Colorizer - GUI (versi jendela, sekali klik)
REM
REM  1) Klik dua kali file ini
REM  2) Pilih folder input & output, atur opsi
REM  3) Klik "Start"
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- Cek model manga-colorization-v2 (engine default) ---
if not exist "models\mangacolv2\generator.zip" goto :need_models
if not exist "models\mangacolv2\net_rgb.pth" goto :need_models
goto :models_ok

:need_models
echo [!] Model belum lengkap. Mengunduh sekarang...
python scripts\download_model.py --engine mangacolv2
if errorlevel 1 goto :dl_fail
echo.

:models_ok
REM --- Jalankan GUI (pythonw = tanpa jendela konsol hitam) ---
start "" pythonw src\gui.py
if errorlevel 1 (
    REM Fallback kalau pythonw tidak tersedia.
    python src\gui.py
)
exit /b 0

:dl_fail
echo.
echo [X] Gagal mengunduh model. Periksa koneksi internet.
pause
exit /b 1
