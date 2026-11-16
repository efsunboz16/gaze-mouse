# gaze-mouse

Webcam ile bakışla imleç kontrolü ve göz kapağı hareketleriyle tıklama.

- Mimari ve literatür: [`docs/mimari-raporu.md`](docs/mimari-raporu.md)
- Claude Code için proje bağlamı: [`CLAUDE.md`](CLAUDE.md)

## Kurulum (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
```

### L2CS gaze model

The V2 appearance model uses an exported L2CS-Net Gaze360 checkpoint. Obtain
`L2CSNet_gaze360.pkl` from the official L2CS-Net release, install the exporter
dependencies and the L2CS-Net package, then run:

```powershell
pip install torch torchvision
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
python -m tools.export_l2cs models/L2CSNet_gaze360.pkl models/gaze.onnx
```

The default configuration expects the resulting file at `models/gaze.onnx`.

## V2 implementation status

V2 replaces the original direct iris-feature-to-screen regression path with
separate model, geometry, and calibration stages:

- camera intrinsics and metric eye-center estimation;
- 3D gaze directions and ray/screen-plane intersection;
- affine face normalization and an ONNX Runtime model adapter;
- L2CS-Net ONNX logits decoding, including the L2CS `-Z` direction convention;
- pose-diverse calibration for upright, normal, and leaned-back laptop poses;
- per-user kappa calibration and V2 session recording.

The live V2 calibration command is:

```powershell
python -m tools.calibrate_v2 --config config/default.yaml --user efsun
```

It records a pose-diverse NPZ session under `profiles/`. Camera intrinsics are
machine-specific and are generated with `tools/calibrate_camera.py`; they are
not committed. The resulting kappa JSON is also user-specific and is not
committed.

### Known limitations

- L2CS-Net was trained for face bounding-box crops, not the earlier eye-based
  normalized patch. The runtime therefore uses a landmark-derived face crop.
- L2CS-Net does not provide per-eye uncertainty; the adapter currently emits
  zero uncertainty and duplicates the single-face prediction for both eyes.
- Screen-plane geometry currently assumes the laptop display is approximately
  parallel to the camera plane. A full camera-to-screen pose calibration is
  still needed for robust laptop-lid angle handling.
- V2 must be validated with a fresh camera session after changing the model
  input or screen geometry. Do not enable `--live` mouse control until the
  dry-run target error is acceptable.

The V2 unit and regression suite currently passes with:

```powershell
python -m pytest
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`, sonra aynı adımlar.

## Durum

Faz 0 ve Faz 1 kapatıldı. Faz 1 çıkış ölçütü karşılanmadı: 300 px düğmelerde son canlı testler 2/12 ve 8/12 oldu, geçerli kalibrasyon ve sabit başla toplam isabet %75. Winkler güvenilir çalışıyor; bakış doğruluğu Faz 3'te ele alınacak. Faz listesi ve notlar `CLAUDE.md` içinde.

## Kullanım

```powershell
python -m tools.preview                        # kamera ve landmark önizlemesi
python -m tools.calibrate --user efsun         # kalibrasyon, kullanımdan hemen önce yapılmalı
python -m tools.target_test --user efsun       # 300 px düğme testi (dry-run)
python -m gaze_mouse.app --user efsun --live   # gerçek fare olayları
```

Varsayılan dry-run'dır; gerçek fare olayları yalnızca `--live` ile gönderilir. `Ctrl+Alt+Q` programı kapatır, fareyi ekran köşesine götürmek duraklatır.
