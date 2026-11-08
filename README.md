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

Faz 0 tamamlandı: canlı önizleme (`python -m tools.preview`) ve oturum kaydı (`python -m tools.record --save`) çalışıyor. Faz listesi ve çıkış ölçütleri `CLAUDE.md` içinde.
