# gaze-mouse — Webcam ile bakış ve göz kapağıyla bilgisayar kontrolü

Bu dosya, projede çalışan Claude Code oturumları için kalıcı bağlamdır. Her oturumun başında oku.

## Proje ne yapıyor

Dizüstü/masaüstü webcam'i ile kullanıcının ekranda nereye baktığını kestirip imleci oraya götürür; göz kapağı hareketleriyle (wink, uzun kapama) sol/sağ tık, çift tık, sürükleme, kaydırma ve duraklatma yapar. Ek donanım yok, tüm işleme cihaz üstünde.

Mimari ve tüm tasarım kararlarının gerekçesi (literatür, eşikler, doğruluk hesapları): **`docs/mimari-raporu.md`**. Bir kararı değiştirmeden önce oradaki gerekçeyi oku; rapordan sapmak gerekiyorsa önce kullanıcıya sor.

## Kullanıcı ve iletişim

- Kullanıcı Efsun, Bilgisayar Mühendisliği öğrencisi, görüntü işleme / AI odaklı. Projeyi aynı zamanda akademik çalışma (makale, değerlendirme deneyleri) olarak da düşünüyor.
- **Yanıtları Türkçe ver**; teknik terimler (landmark, fixation, blendshape, throughput vb.) İngilizce kalabilir.
- Kod yorumları ve docstring'ler İngilizce, kısa.

## Temel fikir (değişmez ilkeler)

1. Webcam bakış doğruluğu kalibrasyondan sonra bile ~1,4–4°; 15,6" FHD ekranda 60 cm'de bu ~80–230 px demek. **Bakış imleci kabaca götürür, ince ayar başka kanaldan gelir** (hedefe yapışma, baş hareketi, büyüteç).
2. **Konum ve komut ayrı kanallar:** bakış hattı = imleç nerede; göz kapağı hattı = hangi komut. İkisi durum makinesinde birleşir (Midas touch'a yapısal çözüm).
3. Doğal kırpma iki gözde birlikte ve 100–400 ms sürer → komut sinyali olarak **tek göz wink** ve **uzun kapama** kullanılır; doğal kırpma yok sayılır.
4. Göz kapanırken bakış tahmini bozulur → **kapanma başlarken imleç dondurulur**, tık kapanma öncesi fixation noktasına gider.
5. Tüm eşikler kişiye göre kalibre edilir; hiçbir eşik koda gömülmez → `config/default.yaml` + kullanıcı profili.

## Ortam

- Geliştirme makinesi **Windows 11 Home 64-bit, build 26300** (2026-11-08'de doğrulandı); hedef platform Windows 10/11. macOS'ta fare kontrolü için Erişilebilirlik ve Kamera izinleri gerekir.
- **Python 3.12.10** (`E:\python\python.exe`, `py -3.12` ile seçilir); proje 3.11 ve 3.12'yi hedefler. Proje kökünde `.venv`. Komutlar: `py -3.12 -m venv .venv`, `.venv\Scripts\activate`, sonra `pip install -r requirements.txt`.
- PATH'te MSYS2 Python'ları da var (`E:\mys32\...`, `E:\msys2\...`); venv'i bunlarla kurma. Komutları venv etkin değilse `.venv\Scripts\python.exe` ile çalıştır.
- **MediaPipe 1.x kullanılıyor: yalnız Tasks API** (`mediapipe.tasks.python.vision.FaceLandmarker`). Eski `mp.solutions.face_mesh` API'si 1.x'te YOK; internetteki eski örnekleri kopyalama.
- MediaPipe, `opencv-contrib-python`'ı bağımlılık olarak kurar; ayrıca `opencv-python` kurma (çakışır).
- Model dosyası: `python scripts/download_model.py` → `models/face_landmarker.task`
  (kaynak: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`).
- FaceLandmarker ayarları: `running_mode=VIDEO` (veya `LIVE_STREAM`), `num_faces=1`, `output_face_blendshapes=True`, `output_facial_transformation_matrixes=True`. VIDEO modunda zaman damgası milisaniye cinsinden ve **kesin artan** olmalı.

## Hedef klasör yapısı

```
gaze_mouse/
  capture/camera.py          # M1  kamera thread'i, halka tampon, pozlama/odak kilidi
  perception/landmarks.py    # M2  FaceLandmarker sarmalayıcı, baş pozu, iris tabanlı mesafe
  gaze/normalize.py          # M3  sanal kamera normalizasyonu (Faz 3)
  gaze/features.py           # M4-A geometrik öznitelikler (iris–köşe ofsetleri, poz)
  gaze/model.py              # M4-B görünüm tabanlı model, ONNX çıkarımı (Faz 3)
  gaze/screen.py             # M5  ışın–ekran kesişimi, DPI, çoklu monitör
  calibration/               # M6  nokta kalibrasyonu, ridge başlık, örtük kalibrasyon, kırpma kalibrasyonu
  filtering/one_euro.py      # M7  1€ filtresi
  filtering/fixation.py      # M7  I-DT fixation tespiti + kırpma maskesi
  cursor/controller.py       # M8  imleç, hedefe yapışma, baş ince ayarı, büyüteç
  eyelid/features.py         # M9  EAR, blendshape okuma, kişisel taban
  eyelid/classifier.py       # M9  histerezisli göz durumu + olay segmentasyonu
  control/state_machine.py   # M10 Duraklatıldı / İşaretleme / Sürükleme / Kaydırma / Kalibrasyon
  os_io/mouse.py             # M11 fare olayları (pynput)
  ui/overlay.py              # M11 PyQt6 şeffaf üst pencere: bakış noktası, ilerleme halkası, büyüteç
  app.py                     # ana döngü
tests/                       # pytest; saf mantık modüllerinin testleri
tools/                       # kayıt aracı, önizleme, değerlendirme scriptleri
config/default.yaml          # tüm parametreler
```

Bu yapı bir hedef; dosyaları ihtiyaç oldukça oluştur, boş iskelet doldurma.

## Kodlama kuralları

- **Saf mantık ile donanımı ayır.** Filtreler, fixation, kalibrasyon regresyonu, göz kapağı sınıflandırıcı ve durum makinesi kamera/OS'e dokunmaz; zaman damgalı veri alır, sonuç döner. Bunlar sentetik veriyle **pytest** üzerinden test edilir (ör. sentetik EAR dizisinden wink / doğal kırpma ayrımı).
- Parametreler `config/default.yaml`'dan okunur; kişisel değerler `profiles/<kullanıcı>.yaml`'a yazılır.
- Her kare tek bir veri yapısıyla taşınır (ör. `FrameResult`: zaman damgası, landmark'lar, blendshape'ler, poz, bakış, göz durumu). Modüller arası sözleşme buradan okunur.
- Gecikme bütçesi: kare → imleç < 100 ms, hedef < 80 ms. Ana döngüde aşama sürelerini ölçen basit bir profilleyici tut.
- Küçük, sık commit (git). Her faz sonunda çalışır durumda bir commit.

## Güvenlik (fare kontrolü tehlikeli olabilir)

- **Varsayılan mod dry-run:** imleç hareket ettirilmez, tıklama gönderilmez; tahmin ve olaylar yalnız overlay/önizlemede gösterilir. Gerçek kontrol yalnız `--live` bayrağıyla açılır.
- Her zaman çalışan bir **acil durdurma**: global kısayol (ör. `Ctrl+Alt+Q`) ve fare ekranın bir köşesine giderse otomatik duraklatma.
- Yüz kaybolursa veya güven düşükse olay üretme.
- Testlerde gerçek tıklama gönderme; `os_io/mouse.py` için sahte (mock) arka uç kullan.

## Gizlilik

- Ham video varsayılan olarak diske yazılmaz. Kayıt aracı yalnız açık bir bayrakla `data/` altına yazar; `data/`, `profiles/` ve `models/` git'e girmez.

## Doğrulanması gereken ayrıntılar (varsayma, ölç)

- **Landmark indeksleri:** EAR için yaygın başlangıç: bir göz `[33, 160, 158, 133, 153, 144]`, diğer göz `[362, 385, 387, 263, 373, 380]`; iris merkezleri 468 ve 473. Hangisinin kullanıcının sol/sağ gözü olduğunu kare üzerine çizerek doğrula.
- **Blendshape sol/sağ yönü:** `eyeBlinkLeft/Right` görüntü aynalamasına göre ters görünebilir. Kırpma kalibrasyonunda kullanıcıya "sol gözünü kırp" dedirtip hangi sinyalin tepki verdiğini kaydet; eşlemeyi profile yaz.
- **DPI:** Windows'ta süreci DPI-aware yap; hesap fiziksel pikselde, OS'e giden koordinat doğru ölçekte olmalı.
- **Kamera:** doğru kamera indeksini ve gerçek fps'i ölç; pozlama/odak kilidi her kamerada desteklenmeyebilir.

## Fazlar ve çıkış ölçütleri

Her fazın başında plan yap ve kullanıcıya onaylat. Faz bitince ilgili kutuyu işaretle ve kısa not düş.

- [x] **Faz 0 — Altyapı:** kamera thread'i, FaceLandmarker, canlı önizleme (landmark'lar, iris merkezleri, EAR sol/sağ, eyeBlink skorları, fps ve gecikme), kayıt aracı (kare özellikleri + ekran hedefi + zaman damgası). Çıkış: 30 fps'te kararlı akış ve kaydedilmiş ilk oturum. *Tamamlandı 2026-11-08: `data/session_20261108_112018`, 826 kare, 29,9 fps, %100 yüz, 1 düşen kare; iris ofseti–hedef korelasyonu x −0,989, y 0,943. Pozlama kilidi hâlâ tutarsız.*
- [x] **Faz 1 — MVP:** geometrik bakış (Kademe A) + 9 nokta kalibrasyon + doğrulama ekranı (hata ° ve px), 1€ filtresi, EAR tabanlı sol/sağ wink, overlay, `--live` ile fare olayları. Çıkış: ≥ 300 px düğmelere bakış + wink ile tıklama. *Hedef 2026-11-08'de kullanıcı kararıyla 150 px'ten 300 px'e çekildi: beş kalibrasyonda 150 px tık noktası isabeti %22–83 arasında oynadı, 300 px'te her çalıştırma ≥ %80 (toplam %94). Kalibrasyon başka oturuşa taşınmıyor (%0–21), her oturuşta yeniden yapılmalı. Geometrik baş düzeltmesi denendi, yeni veride tutarlı kazanç vermedi, kaldırıldı. Daha küçük hedefler Faz 3–4'e kaldı. Wink kuralı 2026-11-08'de kullanıcı onayıyla rapordan ayrıldı: wink sırasında öbür göz de kısıldığı için "öbür göz > %80" kuralı 10 winkin 8'ini iptal ediyordu; yerine en derin anda asimetri ≥ 0,15 geldi (yeni veride 10/10 wink, 0/5 kırpma). Kısa wink üst sınırı Faz 1 için 1500 ms; sürükleme için Faz 2'de yeniden ele alınacak. Regresör poly2 oldu (7 kalibrasyonda kapıyı geçen 7/7, ridge 6/7). **Kapatıldı 2026-11-08, kullanıcı kararıyla, çıkış ölçütü (`--live` ≥ 10/12) karşılanmadan:** `--live` 2/12 (kalibrasyondan sonra baş oynamış olabilir, ölçülmedi) ve 8/12 (kalibrasyon hemen öncesi, baş sabit); dry-run 10/12. Geçerli kalibrasyon ve sabit başla toplam 18/24 (%75). Sistem uçtan uca çalışıyor, winkler güvenilir (son turda 12/12 tık); kalan sorun bakış doğruluğu (tık hatası eksen başına ~80–90 px) ve Faz 3'e bırakıldı. Kalibrasyon kullanımdan hemen önce yapılmalı; doğrulama kapısı (%99 çıktığı turda test %67) gerçek kullanımı olduğundan iyi gösteriyor.*
- [ ] **Faz 2 — Göz kapağı katmanı:** kişisel kırpma kalibrasyonu, olay segmentasyonu, durum makinesi, kapanmada imleç dondurma, ilerleme halkası. Çıkış: 30 dk okuma/video oturumunda saatte ≤ 2 yanlış tık.
- [ ] **Faz 3 — Görünüm tabanlı model:** normalizasyon, ETH-XGaze / L2CS / UniGaze ön eğitimli model, ONNX Runtime çıkarımı, kalibrasyon başlığı. Çıkış: ekran merkezinde ≤ 2° doğruluk.
- [ ] **Faz 4 — Hassasiyet:** erişilebilirlik ağacından hedefe yapışma (Windows `uiautomation`), baş hareketiyle ince ayar, büyüteç. Çıkış: 32 px ikonları ≤ 2 adımda seçebilme.
- [ ] **Faz 5 — Sağlamlaştırma ve deney:** örtük yeniden kalibrasyon, kullanıcı profilleri, ISO 9241-411 Fitts testi ve görev deneyleri. Çıkış: rapordaki değerlendirme tablosunun tüm satırları ölçülmüş.

## Çalıştırma

Komutlar proje kökünden, venv etkinken:

- Önizleme: `python -m tools.preview` (`--duration N` süre sonunda özet yazar; `--no-window`, `--camera N`)
- Oturum kaydı: `python -m tools.record --save` (tam ekran hedefler, `data/session_*/` altına `frames.npz` + `session.json`; `--save` olmadan hiçbir şey yazmaz, ham video hiç yazılmaz)
- Testler: `python -m pytest`
- Kalibrasyon ve doğrulama: `python -m tools.calibrate --user efsun` (`profiles/<kullanıcı>.yaml` yazar; imleç oynatmaz)
- Uygulama: `python -m gaze_mouse.app --user efsun` (varsayılan dry-run; gerçek fare için `--live`; çıkış `Ctrl+Alt+Q`, fareyi ekran köşesine götürmek duraklatır)
- Çıkış testi (300 px düğmeler): `python -m tools.target_test --user efsun` (varsayılan dry-run, gerçek tıklar için `--live`)
- Her oturuşta önce kalibrasyon yapılmalı; profil başka oturuşa taşınmıyor.

Kamera ölçümü (HD Webcam, 2026-11-08, 30 sn): DSHOW + MJPG ile 1280×720, kamera 29,9 fps, işleme 29,0 fps; çıkarım p50 7,8 ms, kare → ekran p50 15,4 / p95 45,5 ms (sensör gecikmesi hariç). Pozlama kilidi tutarsız: `CAP_PROP_AUTO_EXPOSURE` ile manuel mod istendiğinde iki çalıştırmada parlaklık ~68'e düştü ve kilit geri alındı (119,9 ve 195,9'dan), ilk kayıtta ise değişmedi (203,0 → 203,1, `requested_unverified`). Kilidin gerçekten tutup tutmadığı bilinmiyor; her kaydın durumu `session.json` içindeki `camera` alanında. Otomatik odak kontrolü desteklenmiyor.

Sol/sağ eşlemesi (2026-11-08, Efsun'un gerçek wink denemesi, aynalı önizleme): `default.yaml`'daki `ear_left`/`iris_left` ve `eyeBlinkLeft` kullanıcının anatomik sol gözüne karşılık geliyor; sağ taraf için de aynısı. Başka kullanıcılar için kırpma kalibrasyonundaki kontrol yine yapılmalı.
