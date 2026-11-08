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
