# YOLOv8 For Indonesian Plate Reader

<img width="1619" alt="image" src="https://github.com/user-attachments/assets/260f82cb-3119-443f-b6a6-d829c155d8c2" />

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Latar Belakang**

Identifikasi kendaraan secara otomatis menjadi kebutuhan penting dalam berbagai sistem modern, seperti parkir otomatis, sistem tilang elektronik (ETLE), hingga pemantauan lalu lintas berbasis kamera. Salah satu elemen kunci dari sistem ini adalah kemampuan untuk mendeteksi dan mengenali plat nomor kendaraan secara akurat dan efisien.

Proyek ini bertujuan untuk membangun sistem yang mampu mendeteksi digit atau karakter pada plat nomor motor menggunakan pendekatan *object detection* berbasis deep learning. Dengan memanfaatkan algoritma *You Only Look Once* versi 8 (YOLOv8), sistem dibagi menjadi dua tahap utama, yaitu deteksi posisi plat nomor dan deteksi karakter (huruf dan angka) di dalam plat tersebut. Model dilatih pada dataset khusus yang terdiri dari berbagai citra kendaraan bermotor dan karakter plat, sehingga mampu mengenali format umum plat nomor Indonesia.

Hasil pelatihan dan pengujian menunjukkan bahwa pendekatan ini mampu memberikan akurasi tinggi serta kecepatan inferensi yang baik, menjadikannya cocok untuk implementasi real-time. Dengan sistem ini, diharapkan deteksi digit plat nomor motor dapat dilakukan secara otomatis, efisien, dan andal dalam berbagai kondisi pencahayaan maupun sudut pandang kamera.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Business Understanding**

#### Problem Statements

* Identifikasi plat nomor kendaraan secara manual atau dengan sistem konvensional masih kurang efisien dan rawan kesalahan, terutama dalam skenario real-time.
* Sistem pengawasan lalu lintas dan parkir otomatis membutuhkan deteksi karakter plat nomor yang akurat dan cepat untuk mendukung pengambilan keputusan otomatis.
* Tantangan teknis muncul dalam kondisi pencahayaan rendah, sudut pandang tidak ideal, atau kualitas gambar yang buruk, yang menyebabkan kesalahan deteksi karakter.

#### Goals

* Mengembangkan sistem berbasis deep learning yang mampu mendeteksi dan mengenali digit (huruf dan angka) pada plat nomor motor secara otomatis.
* Mencapai performa tinggi dari segi akurasi (precision, recall, mAP) dan kecepatan inferensi agar sistem layak untuk aplikasi real-time.
* Menyediakan model yang dapat diintegrasikan ke dalam aplikasi seperti parkir otomatis, sistem tilang elektronik (ETLE), dan pemantauan lalu lintas.

#### Solution Statements

* Mengimplementasikan dua tahap model deteksi menggunakan algoritma YOLOv8: satu model untuk mendeteksi lokasi plat nomor pada gambar kendaraan, dan satu model untuk mendeteksi karakter di dalam plat tersebut.
* Melatih model menggunakan dataset gambar plat nomor motor yang telah diberi anotasi sesuai format YOLO, mencakup karakter A–Z dan angka 0–9.
* Melakukan evaluasi model berdasarkan metrik seperti precision, recall, mAP\@50, dan mAP\@50–95 pada data uji untuk memastikan kemampuan generalisasi model terhadap kondisi nyata.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Data Understanding

### Sumber Data Deteksi Object plat motor
Sumber dataset yang digunakan dalam proyek ini berasal dari Roboflow.
| **Komponen**        | **Deskripsi**                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Nama Dataset**    | Plat Nomor Kendaraan Indonesia                                                                                        |
| **Sumber**          | [Roboflow - Plat Nomor Kendaraan Indonesia](https://universe.roboflow.com/denny-pmzkg/plat-nomor-kendaraan-indonesia) |
| **Pemilik Dataset** | `denny-pmzkg` (Workspace di Roboflow)                                                                                 |
| **Versi Dataset**   | Versi 2                                                                                                               |
| **Format Anotasi**  | YOLOv8                                                                                                                |
| **Jenis Data**      | Gambar plat nomor kendaraan (termasuk mobil dan motor) dengan anotasi bounding box                                    |
| **Jumlah Kelas**    | 1 kelas (plat nomor) atau multi-kelas jika termasuk karakter A-Z dan 0-9                                              |
| **Tipe File**       | Gambar (.jpg/.png) dan label (.txt sesuai format YOLO)                                                                |

### Sumber Data License Plate Character Recognition
Sumber dataset yang digunakan dalam proyek ini berasal dari Roboflow.
| **Komponen**        | **Deskripsi**                                                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nama Dataset**    | Indonesia License Plate Character Recognition                                                                                                      |
| **Sumber**          | [Roboflow - Indonesia License Plate Character Recognition](https://universe.roboflow.com/test-lt9f2/indonesia-license-plate-character-recognition) |
| **Pemilik Dataset** | `test-lt9f2` (Workspace di Roboflow)                                                                                                               |
| **Versi Dataset**   | Versi 4                                                                                                                                            |
| **Format Anotasi**  | YOLOv5                                                                                                                                             |
| **Jenis Data**      | Gambar karakter plat nomor kendaraan (huruf A–Z dan angka 0–9)                                                                                     |
| **Jumlah Kelas**    | 36 kelas (26 huruf A–Z + 10 angka 0–9)                                                                                                             |
| **Tipe File**       | Gambar (.jpg/.png) dan label (.txt sesuai format YOLO)                                                                                             |

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Data Preprocessing
### Konversi Label Segmentasi ke Bounding Box

Pada tahap preprocessing, dilakukan konversi format label dari **polygon (segmentasi)** ke **bounding box (YOLO format)**. Hal ini diperlukan karena model deteksi objek (seperti YOLOv5 atau YOLOv8) menggunakan format label berbasis bounding box, bukan koordinat segmentasi.

### Tujuan

* Menyesuaikan format anotasi dataset agar kompatibel dengan model deteksi karakter plat nomor berbasis YOLO.
* Mengubah koordinat bentuk polygon (segmentasi) menjadi format YOLO:
  `(class_id, x_center, y_center, width, height)`
  dengan semua nilai ter-normalisasi terhadap dimensi gambar.

### Proses

Skrip Python di bawah ini digunakan untuk membaca setiap file label `.txt` dalam folder input, menghitung bounding box dari koordinat segmentasi, dan menyimpan hasilnya dalam format YOLO bounding box ke folder output.

```python
def convert_polygon_to_bbox(input_label_folder, output_label_folder):
    """
    Mengonversi label polygon YOLO ke format bbox (x_center, y_center, width, height)
    """
    os.makedirs(output_label_folder, exist_ok=True)

    for filename in os.listdir(input_label_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_label_folder, filename)
            output_path = os.path.join(output_label_folder, filename)

            with open(input_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                cls = parts[0]
                coords = list(map(float, parts[1:]))

                points = np.array(coords).reshape(-1, 2)

                x_min = np.min(points[:, 0])
                x_max = np.max(points[:, 0])
                y_min = np.min(points[:, 1])
                y_max = np.max(points[:, 1])

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                new_line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                new_lines.append(new_line)

            with open(output_path, 'w') as f:
                for nl in new_lines:
                    f.write(nl + '\n')
```

### Hasil

* Label baru disimpan di folder output dalam format `.txt` yang sesuai dengan standar YOLOv5/YOLOv8.
* Setiap file berisi informasi kelas dan koordinat bounding box yang telah terkonversi dari bentuk polygon.


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Modeling
#### Arsitektur Model untuk proyek deteksi plat nomor kendaraan

Model yang digunakan adalah **YOLOv8n** (nano) dari pustaka **Ultralytics**, yang merupakan salah satu arsitektur deteksi objek modern yang ringan, cepat, dan cocok untuk implementasi real-time dengan resource terbatas. Model ini sangat ideal untuk eksperimen awal sebelum beralih ke varian lebih besar seperti `yolov8s`, `yolov8m`, atau `yolov8l`.

```python
from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="Plat-nomor-kendaraan-Indonesia-2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo8-plat"
)
```

#### Konfigurasi Pelatihan

| Parameter     | Nilai                                        |
| ------------- | -------------------------------------------- |
| Dataset       | `Plat-nomor-kendaraan-Indonesia-2/data.yaml` |
| Jumlah Epoch  | 100                                          |
| Ukuran Gambar | 640 x 640                                    |
| Batch Size    | 16                                           |
| Model         | `yolov8n.pt` (YOLOv8 Nano)                   |
| Output Folder | `runs/detect/yolo8-plat`                     |

#### Hasil Pelatihan (Training Results)

| Metrik        | Nilai  |
| ------------- | ------ |
| Precision     | 98.29% |
| Recall        | 95.78% |
| mAP\@0.5      | 98.74% |
| mAP\@0.5:0.95 | 95.34% |
| Fitness Score | 95.67% |

* **Kelas yang Dideteksi**: `plat_nomor`
* **Jumlah Sampel per Kelas**: 120 instance
* **Kecepatan Inferensi**: \~2.62 ms/gambar
* **Preprocessing Time**: \~0.42 ms/gambar
* **Postprocessing Time**: \~4.59 ms/gambar

#### Output

Model dan log pelatihan disimpan otomatis di:

```
runs/detect/yolo8-plat/
```

#### Kesimpulan

Model YOLOv8n menunjukkan performa sangat baik dalam mendeteksi plat nomor kendaraan dengan precision dan mAP yang tinggi. Kecepatan inferensi yang cepat menjadikannya cocok untuk aplikasi real-time seperti pengawasan lalu lintas, sistem parkir otomatis, dan sistem pemantauan kendaraan lainnya.

Berikut adalah dokumentasi bagian **Modeling** untuk **deteksi karakter plat nomor** menggunakan YOLOv8, berdasarkan kode dan hasil pelatihan yang kamu berikan:

---

#### **Modeling – Deteksi Karakter Plat Nomor**

#### Arsitektur Model

Model yang digunakan adalah **YOLOv8n (Nano)** dari pustaka Ultralytics. Model ini dipilih karena ringan, cepat, dan efisien untuk eksperimen awal serta cocok untuk sistem real-time yang dijalankan di perangkat dengan sumber daya terbatas.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Menggunakan model YOLOv8 Nano

model.train(
    data="indonesia-license-plate-character-recognition-4/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo8-plat"
)
```

#### Konfigurasi Pelatihan

| Parameter     | Nilai                                                       |
| ------------- | ----------------------------------------------------------- |
| Dataset       | `indonesia-license-plate-character-recognition-4/data.yaml` |
| Jumlah Epoch  | 100                                                         |
| Ukuran Gambar | 640 x 640                                                   |
| Batch Size    | 16                                                          |
| Model         | `yolov8n.pt` (YOLOv8 Nano)                                  |
| Output Folder | `runs/detect/yolo8-plat2`                                   |
| Jumlah Kelas  | 36 kelas (0–9 dan A–Z)                                      |

#### Kelas yang Dideteksi

`{'0', '1', ..., '9', 'A', 'B', ..., 'Z'}`
Total: 36 kelas (karakter alfanumerik pada plat nomor)


#### Hasil Pelatihan (Training Results)

| Metrik        | Nilai  |
| ------------- | ------ |
| Precision     | 96.92% |
| Recall        | 92.51% |
| mAP\@0.5      | 96.36% |
| mAP\@0.5:0.95 | 84.93% |
| Fitness Score | 86.08% |

> Terdapat variasi nilai mAP antar kelas, berkisar antara \~73% hingga \~92%, menunjukkan bahwa beberapa karakter (terutama huruf dengan data sedikit) memerlukan penambahan sampel atau augmentasi.

#### Kecepatan Proses

| Tahap          | Waktu Rata-rata per Gambar |
| -------------- | -------------------------- |
| Preprocessing  | \~0.30 ms                  |
| Inference      | \~2.18 ms                  |
| Postprocessing | \~2.71 ms                  |
| Total          | \~5.19 ms                  |


#### Kesimpulan

Model deteksi karakter plat nomor menggunakan YOLOv8n menunjukkan performa yang baik, dengan akurasi tinggi dan kecepatan inferensi yang mendukung aplikasi real-time. Namun, distribusi data yang tidak seimbang pada beberapa kelas huruf (seperti I, Q, X, Z) menyebabkan performa kelas tersebut lebih rendah. Untuk peningkatan lebih lanjut, disarankan:

* Melakukan **augmentasi data** pada kelas minoritas
* Menambahkan lebih banyak data latih untuk karakter yang jarang muncul
* Bereksperimen dengan model varian lebih besar (misal: `yolov8s.pt` atau `yolov8m.pt`)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Evaluasi

### 1. Evaluasi Model deteksi object plat motor
**Kesimpulan Hasil Training**

Model YOLOv8 telah dilatih untuk mendeteksi plat nomor kendaraan dengan hasil yang sangat baik pada dataset validasi.
Hasil metrik utama selama training adalah sebagai berikut:

* **Precision (B)**: 98.29% → Model mampu meminimalkan false positive dengan sangat baik.
* **Recall (B)**: 95.78% → Hampir seluruh plat nomor pada data validasi berhasil dideteksi.
* **mAP50 (B)**: 98.74% → Model memiliki akurasi tinggi dalam mendeteksi objek pada IoU ≥ 50%.
* **mAP50-95 (B)**: 95.34% → Akurasi rata-rata pada berbagai tingkat IoU tetap tinggi.
* **Fitness**: 95.68% → Skor gabungan yang menunjukkan kualitas model secara keseluruhan.

Model disimpan di direktori:

```
runs/detect/yolo8-plat/weights/best.pt
```

**Kesimpulan Hasil Testing**

Model kemudian diuji pada dataset uji (test set) untuk mengevaluasi kemampuan generalisasi terhadap data baru.
Hasil metrik pada test set adalah:

* **Precision (B)**: 98.27% → Model tetap presisi pada data baru.
* **Recall (B)**: 100% → Semua plat nomor pada test set berhasil dideteksi tanpa ada yang terlewat.
* **mAP50 (B)**: 98.98% → Akurasi deteksi bounding box sangat tinggi.
* **mAP50-95 (B)**: 93.93% → Model tetap akurat pada berbagai tingkat IoU.
* **Fitness**: 94.43% → Skor gabungan yang menunjukkan performa sangat baik pada test set.

Model diuji pada:

* **Jumlah objek (plat\_nomor)**: 62
* **Jumlah rata-rata objek per gambar**: 59

**Kesimpulan Akhir**

Model YOLOv8 yang dilatih berhasil mendeteksi plat nomor kendaraan dengan **tingkat presisi dan recall yang sangat tinggi**, baik pada data validasi maupun data uji. Model menunjukkan kemampuan **generalize** dengan baik, sehingga layak digunakan untuk deployment pada sistem deteksi plat nomor secara real-time.

### 2. Evaluasi Model deteksi karakter object plat motor

**Kesimpulan Final Model Deteksi Karakter**

Model deteksi karakter yang dikembangkan menggunakan YOLO menunjukkan performa yang sangat baik baik pada tahap *training* maupun *testing*. Model berhasil mencapai keseimbangan antara akurasi deteksi, presisi, dan kemampuan generalisasi pada data baru.

#### Hasil Training

* **Fitness**: 0.8608
* **Precision (B)**: 96,92%
* **Recall (B)**: 92,51%
* **mAP\@0.5 (B)**: 96,37%
* **mAP\@0.5-0.95 (B)**: 84,94%

#### Hasil Testing

* **Fitness**: 0.8770
* **Precision (B)**: 92,67%
* **Recall (B)**: 95,40%
* **mAP\@0.5 (B)**: 96,15%
* **mAP\@0.5-0.95 (B)**: 86,76%

#### Analisis

* Model mendeteksi karakter dengan presisi dan recall yang tinggi, menunjukkan konsistensi performa pada data pelatihan dan pengujian.
* Kelas dengan kinerja terbaik mencakup **'U' (94,72% mAP)**, **'R' (93,13% mAP)**, dan **'N' (93,69% mAP)**.
* Beberapa kelas dengan jumlah sampel terbatas seperti **'V' (77,21% mAP)**, **'X' (75,84% mAP)**, dan **'Z' (76,67% mAP)** menunjukkan potensi perbaikan pada dataset atau augmentasi data.
* Kecepatan inferensi rata-rata:

  * **Training**: 2,18 ms (inference), 2,71 ms (postprocess)
  * **Testing**: 9,15 ms (inference), 2,96 ms (postprocess)

#### Kesimpulan Akhir

Model dinilai **layak untuk diimplementasikan pada aplikasi nyata** dalam deteksi karakter, dengan tingkat akurasi tinggi dan waktu inferensi yang efisien. Diperlukan optimasi tambahan pada kelas minoritas dan evaluasi lebih lanjut pada lingkungan nyata untuk memastikan robustness model.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Kesimpulan
### **Penjelasan Proyek Deteksi Digit Plat Nomor Motor Menggunakan Dua Model**

Proyek ini bertujuan untuk membangun sistem deteksi digit pada plat nomor motor dengan memanfaatkan dua tahap deteksi berbasis *You Only Look Once* (YOLO). Sistem ini dirancang agar mampu mendeteksi area plat nomor terlebih dahulu, kemudian mengenali digit atau karakter di dalam plat tersebut secara terpisah, sehingga meningkatkan akurasi dan keandalan deteksi.

<img width="697" alt="image" src="https://github.com/user-attachments/assets/de3225f5-7aa0-45ec-ad91-dc1f99b1f161" />


#### **Tahap 1 — Deteksi Plat Nomor**

Pada tahap awal, sistem menggunakan model YOLOv8 yang dilatih khusus untuk mendeteksi posisi plat nomor pada gambar kendaraan bermotor. Model ini akan menghasilkan bounding box (kotak deteksi) yang membatasi area plat nomor.
Output dari tahap ini berupa koordinat lokasi plat nomor yang kemudian diekstraksi untuk proses selanjutnya.

#### **Tahap 2 — Deteksi Karakter / Digit**

Setelah plat nomor terdeteksi dan dipotong (crop), gambar plat nomor tersebut menjadi masukan untuk model YOLOv8 kedua yang didesain untuk mendeteksi digit dan huruf satu per satu. Model ini mendeteksi posisi setiap karakter pada plat motor secara detail, sekaligus mengklasifikasikannya ke dalam label 0–9 atau A–Z.
Hasil dari tahap ini berupa urutan digit/huruf yang membentuk nomor plat kendaraan.

#### **Alur Sistem**

* Gambar kendaraan → **Model 1**: Deteksi plat nomor → Crop plat nomor
* Crop plat nomor → **Model 2**: Deteksi karakter → Ekstraksi digit/huruf

#### **Keunggulan Pendekatan Dua Model**

**Lebih presisi**: Pemisahan tugas deteksi plat dan karakter mengurangi noise dari lingkungan sekitar kendaraan.
**Fleksibel**: Sistem dapat dikembangkan untuk plat nomor dengan format berbeda.
**Mudah dioptimasi**: Kedua model dapat dilatih dan dioptimasi secara terpisah sesuai kebutuhan data.

#### **Implementasi**

Sistem ini dapat diterapkan untuk:

* Gerbang parkir otomatis
* Sistem tilang elektronik (ETLE)
* Aplikasi monitoring kendaraan di area tertentu
