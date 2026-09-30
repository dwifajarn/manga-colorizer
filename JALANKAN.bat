@echo off
REM ============================================================
REM  Manga Colorizer - sekali klik
REM
REM  1) Taruh gambar manga (hitam-putih) di folder "input"
REM  2) Klik dua kali file ini
REM  3) Hasil berwarna + CBZ + PDF muncul di folder "output"
REM
REM  Pipeline: pre-process -> colorize (manga-colorization-v2)
REM            -> line overlay -> upscale 2x (Real-ESRGAN) -> CBZ/PDF
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   MANGA COLORIZER
echo ============================================================
echo.

REM --- Cek model manga-colorization-v2 (engine default) ---
if not exist "models\mangacolv2\generator.zip" goto :need_models
if not exist "models\mangacolv2\net_rgb.pth" goto :need_models
REM --- Cek model upscaler ---
if not exist "models\upscale\RealESRGAN_x2plus.onnx" goto :need_models
goto :models_ok

:need_models
echo [!] Model belum lengkap. Mengunduh sekarang...
python scripts\download_model.py --engine mangacolv2
if errorlevel 1 goto :dl_fail
python scripts\download_model.py --engine upscale
if errorlevel 1 goto :dl_fail
echo.

:models_ok
REM --- Cek apakah ada gambar ATAU file .cbz/.zip di folder input ---
set "adafile="
for %%F in (input\*.png input\*.jpg input\*.jpeg input\*.webp input\*.bmp input\*.cbz input\*.zip) do set "adafile=1"
if not defined adafile (
    echo [!] Folder "input" kosong.
    echo     Taruh gambar manga ATAU file komik .cbz di: %cd%\input
    echo     Lalu jalankan file ini lagi.
    pause
    exit /b 0
)

REM --- Menu pilih format output ---
echo Pilih format output yang ingin dibuat:
echo   [1] Gambar berwarna (PNG)
echo   [2] Komik CBZ sebagai satu file
echo   [3] Dokumen PDF
echo   [4] Semua (PNG + CBZ + PDF)
echo.
set "pilihan="
set /p "pilihan=Masukkan angka (boleh lebih dari satu, pisah koma. Enter = semua): "

if "!pilihan!"=="" set "pilihan=4"

REM Terjemahkan pilihan angka -> daftar format
set "format="
echo !pilihan! | findstr /c:"1" >nul && set "format=!format!image,"
echo !pilihan! | findstr /c:"4" >nul && set "format=all" 
if not "!format!"=="all" (
    echo !pilihan! | findstr /c:"2" >nul && set "format=!format!cbz,"
    echo !pilihan! | findstr /c:"3" >nul && set "format=!format!pdf,"
)
if not "!format!"=="all" (
    REM buang koma terakhir
    set "format=!format:~0,-1!"
)
if "!format!"=="" set "format=all"

echo.
echo [*] Format dipilih: !format!
echo [*] Memproses isi folder "input" ...
echo     (gambar dan/atau .cbz -^> engine: mangacolv2 + upscale x2, CPU)
echo.
python src\cli.py chapter input\ output\ --out "!format!"
echo.
if errorlevel 1 (
    echo [X] Terjadi kesalahan saat memproses. Lihat pesan di atas.
    pause
    exit /b 1
)

echo ============================================================
echo   SELESAI! Format yang dibuat: !format!
echo   Lihat hasilnya di: %cd%\output
echo ============================================================
echo.
pause
exit /b 0

:dl_fail
echo.
echo [X] Gagal mengunduh model. Periksa koneksi internet.
pause
exit /b 1
