# 🎨 Manga Colorizer

Otomatis mewarnai **panel manga/komik hitam-putih** — cukup jalankan di **CPU**
(tanpa GPU CUDA). Dirancang untuk hardware sederhana (mis. laptop AMD/Intel).

> Automatically colorize **black-and-white manga/comic panels** on **CPU** —
> no CUDA GPU required.

Proyek pendamping [`Manga-merger`](../Manga-merger): gabungkan chapter, lalu
warnai di sini.

---

## ✨ Fitur / Features

- **3 engine** colorize: `mangacolv2` (default, terbaik untuk manga), `comicnet`,
  `zhang`.
- **Input fleksibel**: gambar lepas (PNG/JPG) **atau file komik `.cbz`/`.zip`**.
- **Pilih format output** (seperti checkbox): gambar (PNG), CBZ, PDF — bisa
  dipilih satu, beberapa, atau semua.
- Proses **1 gambar**, **satu folder/chapter**, satu **file `.cbz`**, atau
  perintah **`chapter`** sekali-jalan.
- **Pipeline lengkap**: pre-process (bersihkan screentone) → colorize →
  line-overlay (garis hitam dipertajam) → **upscale 2×** (Real-ESRGAN).
- Urutan halaman natural (`page_2` sebelum `page_10`).
- Semua model jalan **offline di CPU**.

> ⚠️ **Warna = tebakan model.** Model memprediksi warna *masuk akal* (kulit,
> rambut, baju) tapi **tidak tahu** warna kanonik karakter. Karakter yang sama
> bisa beda warna antar halaman. Ini wajar untuk pipeline otomatis tanpa
> referensi.

---

## 🧠 Engine

| Engine       | Model                                       | Cocok untuk            | Ukuran  |
| ------------ | ------------------------------------------- | ---------------------- | ------- |
| **mangacolv2** | qweasdd manga-colorization-v2 (GAN)       | **manga/anime (default)** | ~127 MB |
| `comicnet`   | ColorComicNet (`1plus1/MangaColorization`)  | manga (alternatif)     | ~32 MB  |
| `zhang`      | Zhang et al. ECCV16                         | foto (opsional)        | ~123 MB |

Pipeline juga memakai **Real-ESRGAN x2plus (ONNX)** untuk upscale — ~34 MB.

---

## 🚀 Instalasi / Setup

### 1. Dependensi
- **Python 3.8+** (diuji di 3.10)
- PyTorch CPU (Direkomendasikan):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- Sisanya:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Unduh model / Download the models
```bash
python scripts/download_model.py                   # semua (mangacolv2 + comicnet + zhang + upscale)
python scripts/download_model.py --engine mangacolv2
python scripts/download_model.py --engine upscale
```

### 3. Cek / Verify
```bash
python src/cli.py check
```

---

## 🖥️ Cara Pakai / Usage

### Termudah (Windows)
Taruh gambar **atau file `.cbz`** di folder `input/`, lalu **klik dua kali
`JALANKAN.bat`**. Anda akan ditanya format output:

```
Pilih format output yang ingin dibuat:
  [1] Gambar berwarna (PNG)
  [2] Komik CBZ sebagai satu file
  [3] Dokumen PDF
  [4] Semua (PNG + CBZ + PDF)
Masukkan angka (boleh lebih dari satu, pisah koma. Enter = semua):
```

Hasil muncul di `output/`.

### Memilih format lewat terminal
Gunakan `--out` (daftar dipisah koma, atau `all`):
```bash
python src/cli.py chapter input/ output/ --out image      # hanya PNG
python src/cli.py chapter input/ output/ --out cbz,pdf    # CBZ + PDF (tanpa PNG)
python src/cli.py chapter input/ output/ --out all        # semua
```
Jika `image` tidak dipilih, PNG perantara dihapus otomatis (atur
`"keep_images": true` di settings untuk tetap menyimpannya).

### Lewat terminal

**Satu gambar:**
```bash
python src/cli.py colorize input/page_01.png output/page_01.png
```

**Satu folder / chapter (gambar lepas):**
```bash
python src/cli.py batch input/ output/ --out all
```

**Satu file komik `.cbz` → `.cbz` berwarna:**
```bash
python src/cli.py cbz input/manga.cbz output/
# menghasilkan output/manga_colorized.cbz
```

**Folder campuran (gambar + .cbz) + CBZ + PDF gabungan (rekomendasi):**
```bash
python src/cli.py chapter input/ output/ --out all --title "Chapter 1"
```

**Pilih engine:**
```bash
python src/cli.py batch input/ output/ --engine mangacolv2
python src/cli.py batch input/ output/ --engine comicnet
```

**Atur upscale:**
```bash
python src/cli.py batch input/ output/ --upscale      # paksa aktif
python src/cli.py batch input/ output/ --no-upscale   # matikan
```
Default mengikuti `settings.json` (`"upscale": true`).

---

## 📦 Output

Pilih sendiri format yang ingin dibuat lewat `--out` / menu `JALANKAN.bat`:

- **image** — halaman berwarna per halaman (sudah di-upscale 2×).
- **cbz** — satu komik berwarna (`<nama>_colorized.cbz`). ZIP halaman urut baca +
  `ComicInfo.xml`. Bisa dibuka di Tachiyomi, CDisplayEx, YACReader, dsb.
- **pdf** — satu dokumen berwarna (`<nama>_colorized.pdf`, satu halaman per gambar).

Layout output:

- **Input satu `.cbz`** (mis. `input/Chapter 1 END_e29f39.cbz`):
  ```
  output/
  ├── Chapter 1 END_e29f39/                  ← halaman berwarna (kalau "image" dipilih)
  │   ├── 0001_colored.webp
  │   └── ...
  ├── Chapter 1 END_e29f39_colorized.cbz      ← di root output/
  └── Chapter 1 END_e29f39_colorized.pdf
  ```
  (kalau `image` tidak dipilih, folder `Chapter 1 END_e29f39/` dihapus otomatis)

- **Input beberapa `.cbz`** (`input/a.cbz`, `input/b.cbz`): **tiap arsip diisolasi**,
  jadi halaman tidak saling menimpa. CBZ/PDF tiap arsip diletakkan langsung di
  `output/`:
  ```
  output/
  ├── a/            ← halaman a (kalau "image" dipilih)
  ├── a_colorized.cbz
  ├── a_colorized.pdf
  ├── b/
  ├── b_colorized.cbz
  └── b_colorized.pdf
  ```

- **Input gambar lepas** (mis. `input/*.png`): gambar digabung jadi **satu**
  `output/<nama>_colorized.cbz` + `.pdf` (perilaku seperti sebelumnya).

Nama file mengikuti **nama input** (+ akhiran `_colorized`). Untuk banyak arsip,
tiap file mengikuti nama arsipnya masing-masing.

### Input CBZ / ZIP
Taruh file `.cbz` (atau `.zip`) langsung di folder `input/`. Pipeline akan:
1. mengekstrak halaman ke folder **sementara** (otomatis dihapus setelah selesai),
2. mewarnai tiap halaman ke subfolder `output/<nama>/`,
3. membungkus ulang menjadi `output/<nama>_colorized.cbz` (+ `.pdf` bila dipilih).

Atau lewat CLI:
```bash
python src/cli.py cbz input/manga.cbz output/
```

> Catatan: **CBR/RAR tidak didukung** (butuh tool tambahan) — konversi dulu ke
> `.cbz`/`.zip` (ekstensi ZIP biasa juga diterima).

Paket terpisah:
```bash
python src/package.py output/ out/mycomic.cbz
python src/package.py output/ out/mycomic.pdf
```

---

## ⚙️ Konfigurasi — `config/settings.json`

| Key | Default | Arti |
| --- | --- | --- |
| `engine` | `"mangacolv2"` | `mangacolv2`, `comicnet`, atau `zhang` |
| `outputs` | `["image","cbz","pdf"]` | Format default yang dibuat |
| `keep_images` | `false` | Simpan PNG walau `image` tak dipilih |
| `device` | `"cpu"` | Torch device (`cpu`/`cuda`) |
| `mangacolv2_size` | `576` | Ukuran sisi terpanjang saat colorize |
| `mangacolv2_denoise` | `true` | Denoise FFDNet sebelum colorize |
| `preprocess` | `true` | Normalisasi kontras + bersihkan screentone |
| `pre_screentone_ksize` | `3` | Blur untuk merapikan titik screentone |
| `postprocess` | `true` | Line-overlay + restore kertas putih |
| `post_paper_lo/hi` | `228`/`250` | Ambang area "kertas" yang diputihkan |
| `upscale` | `true` | Aktifkan upscale Real-ESRGAN |
| `upscale_factor` | `2` | Faktor upscale |
| `upscale_tile` | `256` | Ukuran tile (batasi pemakaian RAM) |
| `output_suffix` | `"_colored"` | Suffix nama file output |

Tips: matikan `"upscale": false` untuk proses lebih cepat; naikkan
`mangacolv2_size` untuk detail lebih (lebih lambat).

---

## 📁 Struktur Project

```
manga-colorizer/
├── JALANKAN.bat                 # runner sekali-klik (Windows)
├── config/settings.json         # pengaturan
├── models/
│   ├── mangacolv2/              # bobot engine default (gitignored)
│   │   ├── generator.zip
│   │   └── net_rgb.pth
│   ├── upscale/RealESRGAN_x2plus.onnx
│   ├── comicnet/colorizer.pth   # opsional
│   └── colorization_release_v2.pth  # opsional (zhang)
├── input/                       # taruh gambar hitam-putih ATAU .cbz di sini
├── output/                      # hasil + CBZ/PDF
├── scripts/download_model.py    # unduh bobot
└── src/
    ├── mangacolv2_engine.py     # engine default (GAN)
    ├── mangacolv2_repo/         # kode vendored qweasdd
    ├── comicnet_engine.py       # engine ColorComicNet
    ├── comicnet/                # kode vendored ColorComicNet
    ├── colorize.py              # engine Zhang ECCV16
    ├── enhance.py               # pre/post-processing
    ├── cbz.py                   # baca/tulis arsip CBZ
    ├── upscale.py               # upscale Real-ESRGAN (ONNX)
    ├── batch.py                 # proses folder/chapter/CBZ
    ├── package.py               # pembuat CBZ/PDF
    └── cli.py                   # antarmuka command-line
```

---

## 🧠 Cara Kerja / How it works

```
input (gambar HBP dan/atau .cbz)
   │
   ├─ 0) Jika .cbz: ekstrak halaman (cbz.extract_cbz)
   │
   ├─ 1) Pra-proses (enhance.preprocess_gray)
   │      normalisasi kontras + bersihkan titik screentone
   │
   ├─ 2) Colorize (mangacolv2_engine)
   │      grayscale -> GAN manga-colorization-v2 -> RGB
   │      (denoise FFDNet opsional sebelum masuk)
   │
   ├─ 3) Line overlay (enhance._line_overlay)
   │      garis hitam dipertajam + area kertas diputihkan
   │
   ├─ 4) Upscale (upscale.Upscaler)
   │      Real-ESRGAN x2plus ONNX, tiled -> 2x resolusi
   │
   └─ 5) Paket (package.py / cbz.py)
          PNG + CBZ (+ .cbz berwarna) + PDF
```

**Mengapa pipeline, bukan sekadar model?** Berdasarkan riset, untuk CPU tidak
ada model otomatis yang jelas lebih baik dari manga-colorization-v2. Kenaikan
kualitas terbesar datang dari **tahapan pipeline** (bersihkan screentone,
re-ink garis, upscale) — bukan dari mengganti model.

---

## ⚠️ Catatan / Disclaimer

Untuk **penggunaan pribadi/pembelajaran**. Hormati hak cipta: warnai manga
public-domain atau karya Anda sendiri. Jangan menyebarkan ulang hasil berwarna
dari karya berlisensi.

Model: *manga-colorization-v2* (qweasdd), *ColorComicNet*
(`1plus1/MangaColorization`), *Colorful Image Colorization* (Zhang, Isola &
Efros, ECCV 2016), dan *Real-ESRGAN* (Wang et al., 2021).
