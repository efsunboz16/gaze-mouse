# Webcam ile Göz Takibi ve Göz Kırpmasıyla Bilgisayar Kontrolü

Nov 7, 2026 · @efsun boz

## Yönetici özeti

Sadece bir webcam ile kalibrasyon sonrası gerçekçi bakış doğruluğu 1,4° ile 4° arasıdır; bu, 15,6" Full HD bir dizüstünde 60 cm mesafede yaklaşık 80–230 piksel hata demektir. 16–32 piksellik bir ikonu yalnızca bakışla seçmek bu yüzden mümkün değil. Önerilen mimari bu gerçeği merkeze alıyor: **bakış kaba konumu verir, ince ayar başka bir kanaldan gelir, tıklamalar ise istemli göz kapağı hareketleriyle ve niyet doğrulamasıyla tetiklenir.**

- **Algılama:** MediaPipe Face Landmarker tek geçişte 478 yüz/iris noktası, 52 blendshape skoru (`eyeBlinkLeft/Right` dahil) ve baş pozu matrisi verir ([kaynak](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)).
- **Bakış tahmini:** Görüntü normalizasyonu + ETH-XGaze/UniGaze üzerinde ön eğitilmiş görünüm tabanlı model + kişiye özel kalibrasyon. MVP için önce geometrik (iris–göz köşesi) regresyonla başlanır.
- **İmleç:** 1€ filtresi ve fixation tespiti ile titreme bastırılır; küçük hedefler için hedefe yapışma, baş hareketiyle ince ayar ve gerektiğinde büyüteç kullanılır.
- **Tıklama:** Doğal kırpmalar iki gözde birlikte gerçekleşir ve 100–400 ms sürer ([Soukupová ve Čech](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf)); bu yüzden sol/sağ tık için tek göz kırpma (wink), alternatif olarak uzun çift göz kapama kullanılır. Kapak kapanırken imleç dondurulur ve tık, kapanmadan önceki bakış noktasına gönderilir.
- **Başarı ölçütü:** ISO 9241-411 Fitts testiyle throughput (bit/s), açısal hata (°) ve saatlik yanlış tıklama sayısı.

En kritik iki risk doğruluk–piksel açığı ve istemsiz tetiklemedir (Midas touch); mimarinin her katmanı bu ikisinden birini çözmek için var.

## Problem tanımı ve gereksinimler

Asıl darboğaz açısal doğruluğun kendisi değil, onun ekrandaki piksel karşılığı ve istenmeyen tetiklemelerdir. Ekranda d mesafesinden θ açısal hata, piksel aralığı p olan bir ekranda yaklaşık şu kadar piksele denk gelir:

```latex
\text{hata}_{px} \approx \frac{2\,d\,\tan(\theta/2)}{p}
```

| Açısal hata | 15,6" FHD dizüstü, 60 cm (px) | 24" FHD monitör, 65 cm (px) |
| --- | --- | --- |
| 1° | 58 | 41 |
| 2° | 116 | 82 |
| 3° | 175 | 123 |
| 4° | 233 | 164 |

Hesap: 15,6" FHD için piksel aralığı 0,180 mm, 24" FHD için 0,277 mm. Göz takibi literatürü gaze etkileşimi için en az yaklaşık 1° görsel açılık hedef boyutu önerir ([Penkar vd., 2012](https://www.cs.auckland.ac.nz/~lutteroth/publications/PenkarLutterothWeber2012-EyeDwellParameters.pdf)); masaüstü arayüzleri bunun çok altında.

**Midas touch problemi:** Göz hem bakmak hem komut vermek için kullanıldığında her bakış bir komut gibi algılanabilir. Dwell (bakarak bekleme) seçimi aşırı istemsiz tıklamaya yol açar ([Penkar vd.](https://www.cs.auckland.ac.nz/~lutteroth/publications/PenkarLutterothWeber2012-EyeDwellParameters.pdf)); kırpmalar da çoğunlukla bilinçsizdir ([Zhai vd., MAGIC](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Zhai/magic.pdf)). Bu yüzden komut sinyalleri doğal göz davranışından açıkça ayrışmalıdır.

| Kod | Gereksinim | Hedef (öneri) |
| --- | --- | --- |
| F1 | İmleci bakılan noktaya götürme; küçük hedeflerde ince ayar | Büyük hedefe tek bakışta, küçük hedefe ≤ 2 adımda ulaşma |
| F2 | Sol tık | Sol göz wink veya uzun çift göz kapama |
| F3 | Sağ tık | Sağ göz wink |
| F4 | Çift tık, sürükle-bırak, kaydırma | Wink kombinasyonları ve mod geçişleri |
| F5 | Duraklatma / kilitleme | Gözler ≥ 2 s kapalı veya ekran dışına bakış |
| F6 | Kalibrasyon ve yeniden kalibrasyon | 9 nokta ≤ 30 s; arka planda örtük güncelleme |
| NF1 | Uçtan uca gecikme (kare → imleç) | < 100 ms |
| NF2 | Kare hızı | ≥ 30 fps, yalnız CPU ile |
| NF3 | Bakış doğruluğu (kalibrasyon sonrası, ekran merkezi) | ≤ 2° |
| NF4 | İstemsiz kırpma reddi | ≥ %98; okuma/video izlerken saatte ≤ 2 yanlış tık |
| NF5 | Gizlilik | Görüntü cihazdan çıkmaz, yalnız öznitelik saklanır |

## Literatür: bakış tahmini

Webcam için en uygun yol, normalizasyonlu görünüm tabanlı (appearance-based) derin öğrenme ile az örnekli kişiselleştirmenin birleşimidir. Kişiden bağımsız modeller kontrollü veri setlerinde bile 4–5° hata verir; kalan farkı kişiye özel kalibrasyon kapatır.

**Model tabanlı yöntemler** iris merkezi, göz köşeleri ve 3B göz küresi geometrisinden bakış ışını hesaplar. Hafiftir ve yorumlanabilir, ama webcam'de iris yalnızca birkaç düzine piksel olduğu için gürültülüdür. Yine de bu ailenin bir parçası mimaride kalır: MediaPipe Iris, yatay iris çapının insanlarda 11,7 ± 0,5 mm civarında sabit olmasından yararlanarak kameraya uzaklığı %4,3 ortalama bağıl hatayla kestirir ([Google Research, 2020](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/)). Bu, bakış ışınını ekran düzlemiyle kesiştirmek için gereken derinliği verir.

**Görünüm tabanlı yöntemler** göz/yüz görüntüsünden doğrudan bakış vektörünü (pitch, yaw) regresyonla öğrenir. Öne çıkan çalışmalar:

| Çalışma | Yıl / yer | Katkı | Rapor edilen hata |
| --- | --- | --- | --- |
| [UniGaze](https://arxiv.org/abs/2502.02307v2) | WACV 2026 | Büyük ölçekli öz-denetimli ön eğitim; alanlar arası genelleme, kod ve model açık | Çapraz veri seti protokollerinde belirgin iyileşme |
| [Cheng vd., inceleme ve benchmark](https://arxiv.org/abs/2104.12668v2) | IEEE TPAMI 2024 | GazeHub ortak kod tabanı, normalizasyon ve kalibrasyon taksonomisi | MPIIFaceGaze'de klasik modeller 4,06–4,93° |
| [L2CS-Net](https://arxiv.org/abs/2203.03339) | 2022 | Pitch ve yaw için ayrı kayıp; hafif CNN | MPIIGaze 3,92°, Gaze360 10,41° |
| [ETH-XGaze](https://arxiv.org/abs/2007.15837) | ECCV 2020 | 110 kişi, 1 milyondan fazla yüksek çözünürlüklü görüntü, 18 DSLR kamera, aşırı baş pozları | Gaze360 modeli bu sette 4,46° ([benchmark](https://arxiv.org/html/2104.12668v2)) |
| [FAZE](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html) | ICCV 2019 | Meta-öğrenmeyle ≤ 9 kalibrasyon örneğinden kişiselleştirme; 3 örnekte bile kazanç | GazeCapture'da 3,18° |
| [Gaze360](https://openaccess.thecvf.com/content_ICCV_2019/html/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.html) | ICCV 2019 | 238 kişi, iç/dış mekân; zamansal model ve doğrudan belirsizlik çıktısı | Gaze360'ta 11,04° ([benchmark](https://arxiv.org/html/2104.12668v2)) |
| [GazeCapture / iTracker](https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html) | CVPR 2016 | 1.450+ kişi, \~2,5 milyon kare, telefon/tablet | Telefon 1,71 cm (kalibrasyonla 1,34), tablet 2,53 cm (2,12) |
| [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild) | CVPR 2015 | 15 kişi, 213.659 dizüstü webcam görüntüsü, 3+ ay doğal kullanım | Kişi dışı test 6,3° |

**Normalizasyon** bu ailenin en önemli ön işleme adımıdır. Sanal bir kamera, z ekseni yüz merkezine bakacak şekilde döndürülür, baş yuvarlanması (roll) kaldırılır ve görüntü perspektif dönüşümle sabit mesafeye taşınır; bakış etiketi de aynı rotasyonla dönüştürülür ([Cheng vd.](https://arxiv.org/html/2104.12668v2)). Zhang vd. ölçek çarpanını bakış etiketinden çıkararak değerlendirme koşullarına göre %9,5–32,7 iyileşme bildirmiştir ([ETRA 2018](https://perceptualui.org/publications/zhang18_etra/)).

**Kişisel kalibrasyon** için literatürdeki seçenekler: son katmanı SVM ile değiştirme, üçüncü dereceden polinom eşleme, kişiye özgü sapma (bias) kestirimi, meta-öğrenme ve kullanıcı farkında olmadan veri toplama, örneğin fare tıklaması sırasında bakışın imleçte olduğunu varsayma ([Cheng vd.](https://arxiv.org/html/2104.12668v2)). Bu son fikir, mimarideki örtük yeniden kalibrasyonun temelidir.

## Literatür: webcam tabanlı sistemler ve ürünler

En iyi güncel webcam sistemi yaklaşık 1,4° doğruluk ve 30 Hz örnekleme raporluyor; bu, laboratuvar takipçisinin yalnızca \~0,5° gerisinde ama ikon düzeyinde seçim için yine yetersiz. Ürünlerin hiçbiri yalnız bakışla ince işaretleme yapmıyor: ya dwell kullanıyor ya da imleci baş hareketiyle sürüyor.

| Sistem | Tür | Nasıl çalışıyor | Doğruluk / not |
| --- | --- | --- | --- |
| [Labvanced webcam ET (Kaduk vd.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017) | Araştırma, Behavior Research Methods 2023 | EyeLink 1000 ile eşzamanlı kayıt, cihaz üstü işleme | Doğruluk \~1,4°, kesinlik \~1,1°, \~30 Hz; sakkad/mikrosakkad için yetersiz |
| [GaVe (Zeng vd.)](https://arxiv.org/pdf/2201.05533) | Araştırma, JEMR 2023 | Tek nokta kalibrasyon, iki aşamalı dwell, seçim okları ekran kenarında | En iyi dwell 0,8–1,0 s, tercih edilen mesafe 45–55 cm; 12 öğe arasında ortalama 6,76 s seçim |
| [WebGazer.js (Papoutsaki vd.)](https://www.ijcai.org/Abstract/16/540) | Açık kaynak, IJCAI 2016 | Pupil dedektörü + kullanıcı etkileşimlerinden kendini kalibre eden regresyon | Yazarlara göre yaklaşık bakış için yeterli; önceki webcam çalışmaları 2–6° aralığı bildiriyor ([GaVe](https://arxiv.org/pdf/2201.05533)) |
| [Apple Eye Tracking](https://www.macrumors.com/2024/05/15/ios-18-accessibility-features) | Ürün, iOS/iPadOS 18 (2024) | Ön kamera, cihaz üstü ML, saniyeler süren kalibrasyon, Dwell Control ile etkinleştirme | Ek donanım yok; veriler cihazda kalıyor |
| [Project GameFace (Google)](https://github.com/google/project-gameface) | Açık kaynak, Apache-2.0 | MediaPipe Face Landmarker; imleç baş hareketiyle, eylemler yüz jestleriyle | Windows ve Android; Eylül 2025'te arşivlendi (salt okunur) |
| [OptiKey](https://github.com/OptiKey/OptiKey) | Açık kaynak, GPL-3.0 | Windows ekran klavyesi; göz takipçisi, fare veya webcam ile | MND hastaları için tasarlandı; arayüz tasarımı için iyi referans |
| [Tobii Eye Tracker 5](https://help.tobii.com/hc/en-us/articles/360012483818-Specifications-for-Eye-Tracker-5) | Donanım (karşılaştırma) | 850 nm NIR aydınlatma, 133 Hz görüntü, 33 Hz bakış çıktısı | 15–27" 16:9 ekranlar |

**Etkileşim tekniği çalışmaları** asıl tasarım dersini veriyor. MAGIC pointing, bakışı imleci hedefe yaklaştırmak, eli ise ince işaretleme için kullanır; donanım takipçiyle bile \~1° hata nedeniyle yalnızca yaklaşık 0,5 inçlik hedeflerin güvenilir olduğunu belirtir ([Zhai vd.](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Zhai/magic.pdf)). BimodalGaze, doğal baş hareketini bilinçli (jest niteliğindeki) baş hareketinden ayırıp ikincisini bakış imlecini inceltmek için kullanır ([Sidenmark vd., ETRA 2020](https://researchportal.bath.ac.uk/en/publications/bimodalgaze-seamlessly-refined-pointing-with-gaze-and-filtered-ge/)). Bu projede el yerine baş hareketi kullanmak, ikisini birleştirmek anlamına geliyor: bakış kaba, baş ince.

## Literatür: göz kapağı ve göz kırpma tespiti

Kırpmanın varlığını tespit etmek çözülmüş bir problem; asıl iş kırpmanın istemli olup olmadığını ayırmak. Literatürdeki güvenilir ayrıştırıcılar süre, tek göz / çift göz ayrımı ve zamansal desendir; öğrenilmiş sınıflandırıcılar tek başına yetmiyor.

**Eye Aspect Ratio (EAR):** Göz başına altı landmark üzerinden hesaplanır; göz kapandıkça sıfıra yaklaşır ([Soukupová ve Čech, CVWW 2016](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf)):

```latex
\mathrm{EAR} = \frac{\lVert p_2 - p_6 \rVert + \lVert p_3 - p_5 \rVert}{2\,\lVert p_1 - p_4 \rVert}
```

Yazarlar tek eşik yerine 30 fps'te ±6 karelik (13 kare, \~430 ms) EAR penceresini SVM ile sınıflandırır. Eyeblink8 setinde kesinlik/duyarlılık eşikte %77,9/%77,9 iken SVM'de %94,3/%96,2'dir. Kırpmalar genellikle 100–400 ms, ölçülen ortalama \~250 ms sürer.

**Kişiye özel eşik:** Sabit 0,2 veya 0,3 eşik yerine kişi/video başına yeniden hesaplanan eşikler 0,21 ile 0,29 arasında değişmiştir ([PeerJ CS, 2022](https://peerj.com/articles/cs-943)). Sonuç: EAR eşiği kalibrasyonda kişiye göre ölçülmeli, sabit kodlanmamalı.

**Blendshape skorları:** MediaPipe'ın 52 blendshape'i arasında `eyeBlinkLeft` ve `eyeBlinkRight` doğrudan kapanma skoru verir ([MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)). EAR geometrik, blendshape öğrenilmiş bir sinyal olduğu için ikisi birlikte daha sağlamdır.

**Tıklamaya eşleme çalışmaları:**

- **BlinkLink** ([Grauman ve Betke, UAIS 2003](https://www.cs.bu.edu/fac/betke/papers/GraumanBetke-UAIS-2003.pdf)): Uzun istemli kırpmalar tık üretir, kısa istemsiz kırpmalar yok sayılır; minimum süre kullanıcı tarafından ayarlanabilir. 15 kişide kırpma tespiti %95,6, uzun/kısa sınıflandırma %93,0, \~28 fps.
- **Gaze+Blink** ([Rolff vd., 2025](https://arxiv.org/pdf/2501.11540)): Bakışla hedefleme, çift göz kırpmayla seçim, tek göz kapama + baş dönüşüyle kaydırma/sürükleme. İstemsiz kırpmalar yüzünden seçim hatası %8,7 (pinch'te %3,0). 19.699 kırpmayla eğitilen derin sınıflandırıcı 0,76 doğruluğa ulaştı ama kullanıcı çalışmasında hata oranını düşürmedi.

**Mimari için çıkarımlar:**

1. Doğal kırpmalar iki gözde birlikte gerçekleşir, bu yüzden **tek göz kırpma (wink)** çift göz kırpmadan çok daha ayırt edicidir; sol/sağ tık için birincil sinyal o olmalı.
2. Wink yapamayan kullanıcılar için **uzun çift göz kapama** (doğal kırpmanın 400 ms üst sınırının belirgin üstü) yedek yol olmalı.
3. Kapak kapanırken iris landmark'ları aşağı kayar ve bakış tahmini bozulur; **imleç kapanma başlarken dondurulmalı** ve tık, kapanmadan önceki fixation noktasına gönderilmeli.
4. Her eylem için **görsel ilerleme geri bildirimi** (imleç etrafında dolan halka) verilmeli; kullanıcı eşiği geçmeden gözünü açarak iptal edebilmeli.

## Önerilen mimari

Sistem iki paralel hat üzerine kurulu: bakış hattı imlecin nereye gideceğini, göz kapağı hattı hangi komutun ne zaman verileceğini belirler; ikisi durum makinesinde birleşir. Konum ve komutu ayrı kanallara bölmek Midas touch probleminin yapısal çözümüdür.

> Diyagram (rapor dokümanında çizili): M1 Kamera → M2 MediaPipe → iki paralel hat. Bakış hattı: M3 Normalizasyon → M4 Bakış modeli → M5 Ekran geometrisi → M6 Kişisel kalibrasyon → M7 Filtre/fixation → M8 İmleç ve hassasiyet. Göz kapağı hattı: Öznitelikler (EAR, blendshape, baş hızı) → Göz başı durum (histerezis) → M9 Olay sınıflandırıcı. İki hat M10 Durum makinesi → M11 İşletim sistemi ile birleşir. Geri beslemeler: M9 kapanma başlayınca M7'yi dondurur (tık kapanma öncesi noktaya gider); M11'deki her onaylı tık M6'ya örtük kalibrasyon örneği olarak döner.

Kesikli çizgiler iki geri besleme döngüsüdür: göz kapanırken bakış hattı dondurulur ve tık kapanma öncesi noktaya gider; onaylanan her tık da kalibrasyona yeni örnek olarak döner, böylece sistem kullanıldıkça kendini düzeltir.

## Modül detayları

On bir modülün her biri tek bir şeyi çözer; aşağıdaki eşik ve parametreler başlangıç değerleridir ve kendi kayıtlarınla ayarlanmalıdır.

### M1 — Kamera yakalama

- 1280×720, 30 fps (destekleniyorsa 60). Ayrı thread, 2–3 karelik halka tampon; eski kareler düşürülür, her kareye monotonik zaman damgası verilir.
- **Otomatik pozlama ve odak kilitlenir.** Pozlama dalgalanması EAR ve iris tespitini doğrudan bozar.
- Kamera iç parametreleri (fx, fy, cx, cy, distorsiyon) bir kez OpenCV dama tahtası kalibrasyonuyla ölçülür; yoksa fx ≈ görüntü genişliği yaklaşık değeri kullanılır.

### M2 — Yüz, landmark ve baş pozu

- MediaPipe Face Landmarker (`VIDEO`/`LIVE_STREAM` modu): 478 3B landmark (iris dahil), 52 blendshape, 4×4 yüz dönüşüm matrisi.
- Baş pozu (R, t) bu matristen veya 3B kanonik yüz modeliyle `solvePnP`'den alınır.
- Kameraya uzaklık iris çapından (\~11,7 mm) çapraz kontrol edilir.
- Çıktı: göz köşeleri, iris merkezleri, kapak noktaları, R, t, blendshape vektörü.

### M3 — Normalizasyon

- Sanal kamera yüz merkezine döndürülür, roll kaldırılır, sabit mesafeye (ör. 600 mm) ve sabit odak uzaklığına taşınır.
- Çıktı: 224×224 normalize yüz ve isteğe bağlı 60×36 göz yamaları; histogram eşitleme.
- Normalizasyon rotasyonu saklanır; model çıktısı ters rotasyonla kamera koordinatına geri döndürülür.

### M4 — Bakış modeli (iki kademe)

- **Kademe A, geometrik (MVP):** Öznitelik vektörü = göz genişliğine bölünmüş iris–göz köşesi ofsetleri (her iki göz) + baş pozu açıları + iris tabanlı mesafe. Ekran koordinatına ridge veya 2. derece polinom regresyonla eşlenir (WebGazer yaklaşımına benzer). Bir haftada çalışır hale gelir.
- **Kademe B, görünüm tabanlı:** ETH-XGaze veya UniGaze ağırlıklarıyla başlatılmış ResNet-50/ViT omurga; pitch ve yaw için L2CS tarzı bin sınıflandırma + regresyon başlığı; Gaze360 tarzı belirsizlik (σ) çıktısı. Eğitim PyTorch, çıkarım ONNX Runtime.
- Kademe B'nin çıktısı, Kademe A öznitelikleriyle birlikte kişisel kalibrasyon başlığına girer; böylece iki kaynak birbirini tamamlar.

### M5 — Bakıştan ekrana geometri

- Bakış ışını: başlangıç = 3B göz merkezi (landmark + derinlik), yön = kamera koordinatındaki bakış vektörü.
- Işın ekran düzlemiyle kesiştirilir. Dizüstünde kamera üst çerçevenin ortasındadır; ekranın fiziksel boyutu (EDID veya kullanıcı girişi) ve çerçeve ofseti gerekir. Ekran eğimi ve küçük geometri hataları kalibrasyon başlığı tarafından emilir.
- Windows DPI ölçeklendirmesine dikkat: hesap fiziksel pikselde yapılır, OS'e mantıksal piksel gönderilir. Çoklu monitörde her monitör ayrı düzlem olarak tanımlanır.

### M6 — Kalibrasyon ve kişiselleştirme

- **Açık kalibrasyon:** 9 nokta (iyi sonuç için 13–16), nokta başına 1,5 s; ilk 500 ms atılır (sakkad ve oturma süresi), kalan karelerin medyanı alınır. Ardından 5–10 s yavaş hareket eden bir hedefle (smooth pursuit) yoğun örnek toplanır.
- **Doğrulama:** Kalibrasyondan bağımsız 4–5 noktada hata ° ve px olarak raporlanır; eşik üstündeyse tekrar istenir.
- **Model uyarlama:** Ridge regresyon başlığı veya son katmanın az örnekli ince ayarı (FAZE, ≤ 9 örnekle kazanç gösterdi).
- **Örtük yeniden kalibrasyon:** Her onaylanmış tıkta bakışın hedef öğenin merkezinde olduğu varsayılır; örnek kayan pencereli tampona eklenir ve başlık periyodik olarak güncellenir. Bu, baş pozu değiştikçe oluşan kaymayı düzeltir.
- **Kırpma kalibrasyonu:** 5 s açık göz (taban EAR), 5 doğal kırpma, 5 sol wink, 5 sağ wink, 3 uzun kapama kaydedilir; kişisel eşikler ve sınıflandırıcı bundan çıkarılır.
- Profiller kullanıcı ve aydınlatma koşulu başına saklanır.

### M7 — Filtreleme ve fixation tespiti

- **1€ filtresi** ekran koordinatlarına uygulanır: yavaş harekette çok yumuşatma, hızlı harekette az gecikme ([Casiez vd., CHI 2012](https://cristal.univ-lille.fr/~casiez/1euro)). Ayar: β = 0 ve fcmin = 1 Hz ile başla; sabit bakarken titreme gidene kadar fcmin'i düşür; sonra hızlı bakış geçişlerinde gecikme gidene kadar β'yı 10'ar kat artır.
- **I-DT fixation tespiti:** Dağılım eşiği 0,5–1° (60 cm'de dizüstünde \~30–60 px), süre eşiği 100–200 ms ([Salvucci ve Goldberg, ETRA 2000](https://www.cs.drexel.edu/~dds26/publications/Salvucci-ETRA00.pdf)). Fixation sırasında imleç fixation merkezinde sabitlenir, sakkadda yeni noktaya atlar.
- **Kırpma maskesi:** Göz açıklığı kişisel tabanın %70'inin altına indiği andan, açıldıktan \~100 ms sonrasına kadar bakış örnekleri geçersiz sayılır.

### M8 — İmleç kontrolü ve hassasiyet stratejisi

2° hata dizüstünde \~116 px demek; standart ikonlar 16–32 px. Bu açığı kapatmak için üç teknik birlikte kullanılır:

1. **Hedefe yapışma:** OS erişilebilirlik ağacından (Windows UI Automation, Linux AT-SPI) tıklanabilir öğelerin dikdörtgenleri alınır. İmleç, bakış belirsizliği yarıçapı içindeki en olası öğeye çekilir; olasılık mesafe ve öğe boyutuyla ağırlıklandırılır.
2. **Baş hareketiyle ince ayar:** Fixation kararlıyken küçük baş dönüşleri imleci düşük kazançla (ör. 1° baş dönüşü → 10–20 px) göreli olarak kaydırır. Yeni büyük bir sakkad ince ayarı sıfırlar. Bu, MAGIC'teki el kanalının yerini alır.
3. **Büyüteç:** Yarıçap içinde birden fazla aday varsa fixation çevresinde 2–3 kat büyütülmüş bir mercek açılır; mercek içindeki ikinci fixation seçimi yapar.

Varsayılan akış 1 + 2'dir; 3 yalnız belirsizlikte devreye girer.

### M9 — Göz kapağı olay sınıflandırıcı

- **Kare başı öznitelikler:** Kişisel tabana bölünmüş EAR\_sol, EAR\_sağ; `eyeBlinkLeft/Right`, `eyeSquintLeft/Right` blendshape'leri; baş pozu hızı (baş hareketi sırasında EAR yanıltıcıdır).
- **Göz başı durum:** Histerezisli eşik: kapanma < tabanın %60'ı, açılma > tabanın %80'i. Bu, eşiğin çevresinde titremeyi önler.
- **Olay segmentasyonu:** Başlangıç, bitiş, süre, tek/çift göz ve asimetri (|EAR\_sol − EAR\_sağ|) çıkarılır. İsteğe bağlı ikinci aşama: Soukupová tarzı 13 karelik pencere üzerinde SVM.

| Göz kapağı hareketi | Tanım (başlangıç eşikleri) | Komut |
| --- | --- | --- |
| Doğal kırpma | İki göz birlikte, < 400 ms | Yok sayılır |
| Sol göz wink | Sol kapalı ≥ 200 ms, sağ göz açık (> tabanın %80'i) | Sol tık |
| Sağ göz wink | Sağ kapalı ≥ 200 ms, sol göz açık | Sağ tık |
| Çift sol wink | 600 ms içinde iki sol wink | Çift tık |
| Uzun sol wink | Sol kapalı ≥ 1 s | Sürüklemeyi başlat / bırak |
| Uzun çift göz kapama | İki göz 600–1500 ms | Sol tık (wink yapamayanlar için) |
| Çok uzun çift göz kapama | İki göz ≥ 2 s | Sistemi duraklat / devam ettir |
| Kaydırma modu | Sağ wink ≥ 1 s ile aç; bakış ekranın üst/alt %15'inde | Yukarı/aşağı kaydırma |

- **Karar anı:** 200 ms–1 s arası sol wink göz açıldığında sol tık olur; 1 s'ye ulaşıldığında göz hâlâ kapalıyken sürükleme başlar. Çift tık için ayrıca beklemeye gerek yok: art arda iki tık işletim sisteminin çift tık süresi içinde kalırsa OS bunu zaten çift tık sayar, böylece ilk tık gecikmez.

* **Tık konumu:** Kapanma başlangıcından önceki son 150–300 ms'lik fixation merkezi kullanılır (bakış tamponu bu yüzden saklanır).
* **Geri bildirim:** Kapanma süresi ilerledikçe imleç çevresinde halka dolar; eylem eşiği geçildiğinde ses çalar. Eşikten önce göz açılırsa eylem iptal olur.

### M10 — Olay eşleme ve durum makinesi

Durumlar: Duraklatıldı, İşaretleme, Sürükleme, Kaydırma, Kalibrasyon. Her komut yalnız ilgili durumda geçerlidir; örneğin Sürükleme durumunda sağ wink iptal anlamına gelir. Bu, tek bir göz hareketinin bağlama göre güvenli biçimde yeniden kullanılmasını sağlar.

> Diyagram (rapor dokümanında çizili): Merkez durum İşaretleme (sol wink → sol tık, sağ wink → sağ tık, çift sol wink → çift tık). İşaretleme ↔ Duraklatıldı: iki göz ≥ 2 s. İşaretleme ↔ Kalibrasyon: kısayolla gir, bitince dön. İşaretleme ↔ Sürükleme: uzun sol wink (≥ 1 s) başlat/bırak; sağ wink iptal. İşaretleme ↔ Kaydırma: uzun sağ wink (≥ 1 s) aç/kapat; üst/alt kenara bakış kaydırır.

Vurgulu kutu varsayılan durumdur; yan durumlara giriş çıkış aynı uzun hareketle yapılır, böylece kullanıcının ezberlemesi gereken hareket sayısı az kalır.

### M11 — İşletim sistemi entegrasyonu

- Fare olayları: `pynput` veya `pyautogui` (Windows'ta SendInput); ileride düşük gecikme için doğrudan Win32 API.
- Üstte duran şeffaf PyQt penceresi: bakış noktası, ilerleme halkası, büyüteç, durum göstergesi.
- Güvenlik: klavye kısayoluyla anında devre dışı bırakma; yüz kaybolursa imleç olay üretmeyi durdurur.

### Gecikme bütçesi (öneri)

| Aşama | Hedef süre (ms) |
| --- | --- |
| Kare yakalama ve tampon | ≤ 33 (30 fps kare aralığı) |
| Landmark + blendshape | ≤ 15 |
| Normalizasyon + bakış modeli | ≤ 20 |
| Kalibrasyon, filtre, fixation | ≤ 5 |
| Olay sınıflandırma + OS çağrısı | ≤ 5 |
| Toplam (kare → imleç) | < 80 |

## Değerlendirme metodolojisi

Sistem dört katmanda ölçülmeli: bakış doğruluğu, kırpma sınıflandırma, işaretleme performansı ve gerçek görev başarısı. Literatürle karşılaştırılabilir olmak için açısal hata (°) ve ISO 9241-411 throughput (bit/s) mutlaka raporlanmalı.

| Katman | Metrik | Protokol |
| --- | --- | --- |
| Bakış | Doğruluk (ortalama açısal hata, ° ve px), kesinlik (RMS ve SD) | Kalibrasyondan bağımsız 9–16 noktada; merkez ve kenar ayrı raporlanır ([Kaduk vd.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017) gibi) |
| Dayanıklılık | Koşul başına hata artışı | 3 aydınlatma seviyesi, gözlüklü/gözlüksüz, 45/60/75 cm mesafe, serbest baş hareketi |
| Kırpma | Olay başına kesinlik, duyarlılık, F1; saatlik yanlış tık; kapanma bitişinden tıka gecikme (ms) | Yönlendirilmiş wink oturumu + 30 dk okuma/video izleme (yalnız doğal kırpma) |
| İşaretleme | Throughput, hata oranı, hareket süresi | ISO 9241-411 çok yönlü Fitts görevi; fare ile karşılaştırma |
| Görev | Tamamlama süresi, hata; SUS ve NASA-TLX anketleri | Klasör açma, pencere kapatma, metin seçme, tarayıcıda gezinme |

**Throughput hesabı** ([MacKenzie](https://www.yorku.ca/mack/UCLAN2018/FittsLaw.pdf)): etkin genişlik We = 4,133 × SDx, etkin zorluk IDe = log2(Ae / We + 1), throughput TP = IDe / MT. Fare için tipik değer \~4,5 bit/s'dir; MAGIC donanım takipçiyle 4,5–4,7 bit/s raporlamıştır ([Zhai vd.](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Zhai/magic.pdf)).

**Ablasyonlar:** Normalizasyon açık/kapalı; 5, 9 ve 16 nokta kalibrasyon; Kademe A ve B; 1€ ve Kalman filtresi; yalnız bakış, bakış + yapışma, bakış + yapışma + baş ince ayarı. Katılımcı içi (within-subject) tasarımla 10–15 kişi makale için makul bir başlangıçtır.

## Uygulama yol haritası ve teknoloji yığını

İlk hedef, geometrik bakış ve wink ile uçtan uca çalışan bir fare; derin model ve hassasiyet teknikleri üstüne katman katman eklenir. Her faz bir çıkış ölçütüyle kapanır.

1. **Faz 0 — Altyapı:** Kamera thread'i, MediaPipe entegrasyonu, canlı görselleştirme ve kayıt aracı (kare + landmark + ekran hedefi + zaman damgası). Kendi veri setin buradan doğar.
   - Çıkış: 30 fps'te kararlı landmark akışı ve kaydedilmiş ilk oturum.
2. **Faz 1 — MVP:** Kademe A geometrik bakış, 9 nokta kalibrasyon, 1€ filtresi, EAR tabanlı sol/sağ wink ve `pynput` ile fare olayları.
   - Çıkış: büyük düğmelere (≥ 150 px) bakış + wink ile tıklama.
3. **Faz 2 — Göz kapağı katmanı:** Kişisel kırpma kalibrasyonu, olay segmentasyonu, durum makinesi, kapanmada imleç dondurma, ilerleme halkası.
   - Çıkış: 30 dk okuma oturumunda NF4 hedefinin tutturulması.
4. **Faz 3 — Görünüm tabanlı model:** Normalizasyon, ETH-XGaze/UniGaze ön eğitimli model, ONNX çıkarımı, kalibrasyon başlığı.
   - Çıkış: merkezde ≤ 2° doğruluk (NF3).
5. **Faz 4 — Hassasiyet:** Erişilebilirlik ağacından hedefe yapışma, baş hareketiyle ince ayar, büyüteç.
   - Çıkış: 32 px ikonları ≤ 2 adımda seçebilme.
6. **Faz 5 — Sağlamlaştırma ve deney:** Örtük yeniden kalibrasyon, kullanıcı profilleri, Fitts ve görev deneyleri, makale.
   - Çıkış: değerlendirme tablosunun tüm satırları doldurulmuş.

| Katman | Seçim | Neden |
| --- | --- | --- |
| Dil | Python 3.11 | Hızlı prototip; darboğaz çıkarsa kritik kısım C++'a taşınır |
| Görüntü | OpenCV | Yakalama, kalibrasyon, `solvePnP`, perspektif dönüşüm |
| Landmark | MediaPipe Tasks, Face Landmarker | 478 nokta + 52 blendshape + poz, CPU'da gerçek zamanlı |
| Derin öğrenme | PyTorch (eğitim), ONNX Runtime (çıkarım) | Ön eğitimli ağırlıklarla uyum, GPU'suz çıkarım |
| Ön eğitimli bakış | ETH-XGaze, L2CS-Net, [UniGaze](https://arxiv.org/abs/2502.02307v2) | Kod ve ağırlıklar açık; ince ayar için başlangıç noktası |
| Eğitim verisi | MPIIFaceGaze, ETH-XGaze + kendi kayıtların | Dizüstü webcam koşullarına en yakın veri |
| Filtre | [OneEuroFilter](https://github.com/casiez/OneEuroFilter) | Referans uygulama, tek dosya |
| OS girdisi | `pynput` / `pyautogui`; Windows için `uiautomation` | Fare olayları ve erişilebilirlik ağacı |
| Arayüz | PyQt6 şeffaf üst pencere | Bakış noktası, halka, büyüteç, kalibrasyon ekranı |

## Riskler ve önlemler

En olası başarısızlık kaynağı model değil, ortam ve kullanıcı çeşitliliği: ışık, gözlük, baş hareketi ve wink yapamayan kullanıcılar.

| Risk | Etkisi | Önlem |
| --- | --- | --- |
| Arka ışık, değişken aydınlatma | Landmark ve iris gürültüsü, EAR sapması | Pozlama kilidi, histogram eşitleme, ışık kalitesi uyarısı, koşul başına profil |
| Gözlük yansıması | İris tespit hatası; MediaPipe derinlik hatası %4,3'ten %4,8'e çıkar ([Google](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/)) | Tüm yüz bağlamını kullanan Kademe B model, gözlüklü veriyle ince ayar |
| Kalibrasyondan sonra baş pozu değişimi | Sistematik kayma | Normalizasyon, baş pozu özniteliği, örtük yeniden kalibrasyon |
| Wink yapamayan kullanıcı | Sol/sağ tık kullanılamaz | Uzun çift göz kapama, dwell yedeği, tık türü menüsü |
| Göz yorgunluğu | Uzun kullanımda hata artışı | Dwell alternatifi, duraklatma modu, oturum süresi uyarısı |
| Kapak kapanırken imleç kayması | Tık yanlış yere gider | İmleç dondurma ve kapanma öncesi fixation noktasına tık |
| 30 Hz örnekleme | Sakkadlar kaba yakalanır ([Kaduk vd.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017)) | İmleç için yeterli; mümkünse 60 fps kamera |
| DPI ölçekleme, çoklu monitör | Yanlış piksel eşlemesi | Fiziksel/mantıksal piksel dönüşümü, monitör başına düzlem |
| Gizlilik | Yüz görüntüsünün sızması | Tüm işleme cihaz üstünde; diske yalnız öznitelik yazılır |

## Kaynaklar

**Bakış tahmini**

- [Cheng, Wang, Bao, Lu — Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/html/2104.12668v2)
- [UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training](https://arxiv.org/abs/2502.02307v2)
- [Abdelrahman vd. — L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments](https://arxiv.org/abs/2203.03339)
- [Zhang vd. — ETH-XGaze](https://arxiv.org/abs/2007.15837)
- [Park vd. — Few-Shot Adaptive Gaze Estimation (FAZE)](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.html)
- [Kellnhofer vd. — Gaze360](https://openaccess.thecvf.com/content_ICCV_2019/html/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.html)
- [Krafka vd. — Eye Tracking for Everyone (GazeCapture, iTracker)](https://openaccess.thecvf.com/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html)
- [Zhang vd. — Appearance-based Gaze Estimation in the Wild (MPIIGaze)](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild)
- [Zhang, Sugano, Bulling — Revisiting Data Normalization for Appearance-Based Gaze Estimation](https://perceptualui.org/publications/zhang18_etra/)
- [Google Research — MediaPipe Iris: Real-time Iris Tracking & Depth Estimation](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/)
- [Google — MediaPipe Face Landmarker dokümantasyonu](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

**Webcam sistemleri ve ürünler**

- [Kaduk vd. — Webcam eye tracking close to laboratory standards](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017)
- [Papoutsaki vd. — WebGazer: Scalable Webcam Eye Tracking Using User Interactions](https://www.ijcai.org/Abstract/16/540)
- [Zeng vd. — GaVe: A Webcam-Based Gaze Vending Interface Using One-Point Calibration](https://arxiv.org/pdf/2201.05533)
- [Apple Eye Tracking (iOS 18) — MacRumors](https://www.macrumors.com/2024/05/15/ios-18-accessibility-features)
- [Google — Project GameFace](https://github.com/google/project-gameface)
- [OptiKey](https://github.com/OptiKey/OptiKey)
- [Tobii Eye Tracker 5 teknik özellikleri](https://help.tobii.com/hc/en-us/articles/360012483818-Specifications-for-Eye-Tracker-5)

**Göz kırpma**

- [Soukupová, Čech — Real-Time Eye Blink Detection using Facial Landmarks](https://cmp.felk.cvut.cz/ftp/articles/cech/Soukupova-TR-2016-05.pdf)
- [Adjusting eye aspect ratio for strong eye blink detection based on facial landmarks (PeerJ CS)](https://peerj.com/articles/cs-943)
- [Grauman, Betke vd. — Communication via eye blinks (BlinkLink)](https://www.cs.bu.edu/fac/betke/papers/GraumanBetke-UAIS-2003.pdf)
- [Rolff vd. — Gaze and Blink Input with Blink Prediction for Extended Reality](https://arxiv.org/pdf/2501.11540)

**Etkileşim, filtreleme ve değerlendirme**

- [Zhai, Morimoto, Ihde — MAGIC Pointing](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Zhai/magic.pdf)
- [Sidenmark vd. — BimodalGaze](https://researchportal.bath.ac.uk/en/publications/bimodalgaze-seamlessly-refined-pointing-with-gaze-and-filtered-ge/)
- [Penkar, Lutteroth, Weber — Designing for the Eye: Design Parameters for Dwell](https://www.cs.auckland.ac.nz/~lutteroth/publications/PenkarLutterothWeber2012-EyeDwellParameters.pdf)
- [Casiez, Roussel, Vogel — 1€ Filter](https://cristal.univ-lille.fr/~casiez/1euro)
- [Salvucci, Goldberg — Identifying Fixations and Saccades in Eye-Tracking Protocols](https://www.cs.drexel.edu/~dds26/publications/Salvucci-ETRA00.pdf)
- [MacKenzie — Fitts' Law (throughput hesabı)](https://www.yorku.ca/mack/UCLAN2018/FittsLaw.pdf)
