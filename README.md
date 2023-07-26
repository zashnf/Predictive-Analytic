# Predictive Analysis - Zashika Hanifa

## Domain Proyek
Diabetes merupakan penyakit yang disebabkan oleh gagalnya organ pankreas memproduksi jumlah hormon insulin secara memadai sehingga menyebabkan peningkatan kadar glukosa dalam darah. Diabetes telah menjadi penyakit kronis yang sangat mengancam kesehatan masyarakat. Kebanyakan orang tidak menyadari gejala awal dari penyakit ini, hingga konsekuensi jangka panjang dari diabetes, pasien berada dalam situasi kritis.[1]
 
Ada 537 juta orang yang hidup dengan diabetes saat ini. Diperkirakan pada tahun 2045, 700 juta orang akan menderita penyakit ini di seluruh dunia. Diabetes memiliki efek merusak pada individu, dan menyebabkan lebih dari 4 juta kematian per tahun.[2]
 
Untuk meminimalisir angka kematian dari penyakit Diabetes ini, para pakar kesehatan harus melakukan pendiagnosaan penyakit sedini mungkin. Oleh karena itu, diperlukan suatu model pembelajaran mesin yang dapat membantu mendeteksi kemungkinan penyakit diabetes dengan tingkat analisis yang akurat, sehingga para pakar kesehatan dapat menangani pasien lebih dini untuk mengurangi jumlah penderita dan kematian.
 
 
## Business Understanding

### Problem Statement
Berdasarkan penjelasan yang telah disampaikan sebelumnya, maka _problem statement_ (rumusan masalah), yaitu sebagai berikut.
Faktor-faktor apa saja yang dapat mempengaruhi kemungkinan penyakit diabetes.
 
### Goals
Tujuan yang ingin dicapai dari pembuatan _machine learning_ prediksi diabetes antara lainnya adalah:
Mengetahui faktor apa saja yang dapat berpengaruh dalam penyakit diabetes,mencari model _machine learning_ yang  memiliki  nilai _accuracy_ terbaik dalam mendeteksi penyakit diabetes. 
 
### Solution statements
Solusi yang dapat dilakukan untuk menangani permasalahan sebagaimana terdapat dalam _problem statements_, yaitu dengan membuat model prediksi diabetes. Adapun aplikasi tersebut dibuat dengan menerapkan teknologi _machine learning_ serta bahasa pemrograman python. Algoritma _machine learning_ yang akan digunakan, yaitu _Logistic Regression_, _K Nearest Neighbor(KNN)_, dan _Decision Tree_.
 
# Data Understanding
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi dataset terkait diabetes. Dataset yang digunakan dapat didownload pada [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/code) berikut ini
 
File ini berisi data medis dan demografis 100.000 pasien bersama dengan status diabetes mereka, baik positif maupun negatif. Data ini terdiri dari berbagai fitur seperti _age_, _gender_, _body mass index (BMI)_, _hypertension_, _heart disease_, _smoking history_, _HbA1c level_, dan _blood glucose level_. 
 
### Variabel-variabel yang terdapat dalam dataset Diabetes:
- gender = Mengacu pada jenis kelamin biologis seseoraBerng, yang dapat berdampak pada kerentanan mereka terhadap diabetes. Ada tiga kategori di dalamnya, yaitu _ male_, _female_. dan _other_.
 
- age = Mengacu pada usia,merupakan salah satu  faktor penting karena diabetes lebih sering didiagnosis pada orang dewasa yang lebih tua, usia berkisar antara 0-80 pada dataset ini.
 
- hypertension = Kondisi medis dimana tekanan darah di arteri terus meningkat. Ini memiliki nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki hipertensi dan untuk 1 berarti mereka memiliki hipertensi.
 
- heart_disease = kondisi medis lain(penyakit jantung) yang dikaitkan dengan peningkatan risiko diabetes. Ini memiliki nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki penyakit jantung dan untuk 1 berarti mereka memiliki penyakit jantung.
 
- smoking_history = Riwayat merokok juga dianggap sebagai faktor risiko diabetes dan dapat memperburuk komplikasi yang terkait dengan - diabetes, pada dataset ini memiliki 5 kategori yaitu _not current_, _former_,_ No Info_, _current_ dan _never_.
 
- bmi = BMI (Indeks Massa Tubuh) adalah ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang lebih tinggi dikaitkan dengan risiko diabetes yang lebih tinggi. Kisaran BMI dalam dataset ini adalah dari 10,16 hingga 71,55. BMI kurang dari 18,5 berarti kekurangan berat badan, 18,5-24,9 berarti normal, 25-29,9 berarti kelebihan berat badan, dan 30 atau lebih berarti obesitas.
 
- hbA1c_level =  Hemoglobin A1c adalah ukuran kadar gula darah rata-rata seseorang selama 2-3 bulan terakhir. Kadar yang lebih tinggi menunjukkan risiko yang lebih besar terkena diabetes. Sebagian besar kadar HbA1c lebih dari 6,5% mengindikasikan diabetes.
 
- blood_glucose_level = Mengacu pada jumlah glukosa dalam aliran darah pada waktu tertentu. Kadar glukosa darah yang tinggi adalah atau lebih dari 140 bisa menjadi indikator utama diabetes.

- diabetes = Variabel target yang diprediksi, dengan nilai 1 menunjukkan adanya diabetes(positif) dan 0 menunjukkan tidak adanya diabetes(negatif).
 
Membuat _exploratory data analysis_ yang digunakan untuk memahami variabel-variabel yang terdapat dalam dataset. EDA disini terdiri dari_univariate analysis_ dan _bivariate analysis_. Dalam analisis univariat, variabel tersebut dipelajari secara terpisah tanpa mempertimbangkan hubungannya dengan variabel lain.
  
![unigender](https://github.com/zashnf/Predictive-Analytic/assets/89719711/f64f195c-d641-41c9-872b-9ce3c50becda)

Gambar 1. Fitur Gender
 
Tabel 1.  Hasil Analisis Fitur Gender
        
|            | **count**| **percentage**| 
| -----------| ---------| --------------|
| **Female** |   58522  |   58.552      | 
| **Male** 		|   41430  |   41.430      | 
| **Other**  |    18    |   0.018       |   


![unsmok](https://github.com/zashnf/Predictive-Analytic/assets/89719711/5fb59602-baf5-4653-84ca-d02cbbf4c320)

Gambar 2. Fitur Smoking history
 
Tabel 2. Hasil Analisis Fitur Smoking History

|                |  **count**  |**percentage**|
| ------------   | ------------| ------------ |
| **No Info**    |    35816    |    35.816    |
| **never**      |    35095    |    35.095    |
| **former**     |    9352     |     9.352    |
| **current**    |    9286     |     9.286    |
| **not current**|    6447     |     6.447    |
| **ever**       |    4004     |     4.004    |

Sedangkan analisis bivariat adalah metode analisis data yang digunakan untuk menganalisis hubungan antara dua variabel atau lebih dalam satu waktu.

![bivasmokgen](https://github.com/zashnf/Predictive-Analytic/assets/89719711/28d1db80-4778-4d32-a4d6-52e1d6131194)

Gambar 3. Fitur Smoking history dan gender
 
![bigendia](https://github.com/zashnf/Predictive-Analytic/assets/89719711/1e3c60d7-f4c5-4c64-a609-765fe477969e)

Gambar 4. Fitur gender dan diabetes

## Data Preparation
Teknik _data preparation_ yang dilakukan, yaitu sebagai berikut:
- Mengubah dataset diabetes menjadi _dataframe_ dengan menggunakan pandas.
- Membagi data menjadi data latih dan data uji. Pada proyek ini saya membagi data latih menjadi 70% dan data uji sebesar 30%.
- Melakukan proses _encoding_, dimana _encoding_ sendiri merupakan proses yang mengubah data kategorik menjadi data numerik.Dalam hal ini meng_encode_ _gender_ dan _smoking_history_ karena nilai pada data tersebut masih kategorikal sehingga harus diubah menjadi numerik.
- Melakukan _feature selection_ untuk menyaring fitur-fitur penting karena semua fitur yang ada dalam dataset tidak sama pentingnya. Menggunakan _variance Threshold_ pada data latih untuk mengecheck apakah ada fitur yang memiliki varians rendah.

![corr](https://github.com/zashnf/Predictive-Analytic/assets/89719711/20bfad2c-d489-47fd-9628-6f0c99591d92)

Gambar 5. Correlation Matrix

Koefisien korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah. Pada gambar 5 dapat dilihat bahwa tidak ada fitur yang berkorelasi tinggi satu sama lain. 

- Menangani data set yang tidak seimbang menggunakan SMOTE
 
![download (2)](https://github.com/zashnf/Predictive-Analytic/assets/89719711/895302bb-ab46-4d9b-9a2b-2fb0eb002247)

Gambar 6. Imbalance data
 

Analisa


Menurut Gambar 6. dataset yang digunakan tidak seimbang (pada 1 kelas/kategori memiliki jumlah contoh yang jauh lebih besar daripada kelas lainnya). 

![before](https://github.com/zashnf/Predictive-Analytic/assets/89719711/dbdf2477-5c38-4dc5-a04d-237dacc43204)
 
Gambar 7. Sebelum menggunakan SMOTE
 
Pada Gambar 7 dapat dilihat bahwa pada kelas 0 terdapat 91.5% dan pada kelas 1 hanya sekitar 8.5%,sehingga terjadi ketidakseimbangan antar kelas 0 dan 1. Oleh karena itu kita dapat menggunakan SMOTE (_Synthetic Minority Over-sampling Technique_) untuk mengatasi masalah ketidakseimbangan kelas dengan meningkatkan jumlah instance dalam kelas minoritas dalam hal ini kelas 1.
 ![after](https://github.com/zashnf/Predictive-Analytic/assets/89719711/a3253f0d-51e7-4aa7-83cf-d7b6b8f74f8a)

Gambar 8. Setelah menggunakan SMOTE
 
Setelah menggunakan SMOTE  (_Synthetic Minority Over-sampling Technique_) dapat dilihat pada gambar 8 bahwa nilai antar kelas 0 dan 1 sudah sama yaitu 50% dan 50%, sehingga sudah tidak ada ketidakseimbangan antar kelas.

- Melakukan pelatihan model ( _Logistic Regression, K Nearest Neighbor,_ dan _Decision Tree_).
 
 
## Modeling
Pada tahap ini model _machine learning_ yang digunakan antara lainnya adalah _Logistic Regression_,_K Nearest Neighbor_,dan _Decision Tree_.
 
### - _Logistic Regression_ 
 _Logistic Regression_ adalah salah satu algoritma Machine Learning yang paling populer, yang termasuk dalam teknik _Supervised Learning_. Algoritma ini digunakan untuk memprediksi variabel dependen kategorik dengan menggunakan sekumpulan variabel independen.
Cara Kerja _Logistic   Regression_   menghitung   nilai   probabilitas   suatu _instance_ data masuk ke dalam kelas tertentu. Hal ini dilakukan dengan memperhitungkan bobot setiap variabel input pada suatu fungsi logistik. 
Fungsi logistik digunakan untuk mengubah hasil perhitungan bobot variabel input menjadi nilai probabilitas yang berkisar antara 0 dan 1. Dalam proses training,_ Logistic Regression_ meminimalkan _error_   prediksi   dengan   memperbarui   bobot   variabel   input menggunakan   teknik   optimasi   seperti   _Gradient   Descent_. Tujuannya   adalah   untuk   menemukan   bobot   yang   terbaik sehingga model dapat memprediksi kelas dari instance data yang belum dilihat dengan akurasi yang tinggi
Kelebihan dari _Logistic Regression_ antara lain sangat efektif untuk klasifikasi data biner, sangat mudah diimplementasikan dan diinterpretasikan, dapat digunakan pada dataset yang memiliki banyak variabel input.
Selain kelebihan yang dimiliki ia juga memiliki kekurangan seperti tidak   efektif   untuk   klasifikasi   data   yang   kompleks   dengan banyak variabel input, rentan terhadap overfitting jika digunakan  pada dataset  yang tidak seimbang, tidak   dapat   mengatasi   masalah   _multicollinearity _  di   antara variabel input.
Parameter yang digunakan pada model ini :
max_iter = Jumlah iterasi maksimum yang dibutuhkan agar pemecah masalah mencapai konvergen. Misal max_iter=1000 berarti algoritma akan mengambil kesempatan hingga 1000 iterasi untuk mendapatkan nilai konvergensi yang diberikan.
 
### _K Nearest Neighbor (KNN)_ 
_K Nearest Neighbor (KNN)_ adalah teknik pembelajaran mesin yang populer digunakan untuk tugas klasifikasi dan regresi. Algoritma ini bergantung pada gagasan bahwa titik data yang serupa cenderung memiliki label atau nilai yang serupa.
Selama fase pelatihan, algoritma KNN menyimpan seluruh kumpulan data pelatihan sebagai referensi. Saat membuat prediksi, algoritma ini menghitung jarak antara titik data input dan semua contoh pelatihan, menggunakan metrik jarak yang dipilih seperti jarak Euclidean.
Selanjutnya, algoritma mengidentifikasi K tetangga terdekat dari titik data input berdasarkan jaraknya. Dalam kasus klasifikasi, algoritma memberikan label kelas yang paling umum di antara K tetangga sebagai label yang diprediksi untuk titik data input. Untuk regresi, algoritma ini menghitung rata-rata atau rata-rata tertimbang dari nilai target dari K tetangga untuk memprediksi nilai untuk titik data input.
Mendefinisikan k dapat menjadi tindakan penyeimbang karena nilai yang berbeda dapat menyebabkan _overfitting_ atau _underfitting_. Nilai k yang lebih rendah dapat memiliki varians yang tinggi, tetapi bias yang rendah. Sedangkan nilai k yang lebih besar dapat menyebabkan bias yang tinggi dan varians yang lebih rendah. 
Pilihan k akan sangat bergantung pada data input karena data dengan lebih banyak _outlier_ atau _noise _ kemungkinan akan berkinerja lebih baik dengan nilai k yang lebih tinggi. Secara keseluruhan, disarankan untuk memilih nilai k berupa angka ganjil untuk menghindari ikatan dalam klasifikasi. Nilai k pada algoritma KNN mendefinisikan berapa banyak tetangga yang akan diperiksa untuk menentukan klasifikasi titik kueri tertentu. Misalnya, jika k=1, _instance_ akan ditugaskan ke kelas yang sama dengan tetangga terdekatnya. 
Kelebihan dari _K Nearest Neighbor_ adalah mudah dipahami dan diimplementasikan.  Namun kekurangannya adalah ia masih perlu menunjukkan parameter k (jumlah tetangga terdekat), tidak menangani nilai hilang (_missing value_) secara implisit.
Parameter yang digunakan pada model ini:
n_neighbors: Parameter ini menentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Nilai yang umumnya digunakan adalah bilangan ganjil untuk menghindari situasi kesetimbangan kelas. Jumlah tetangga yang terlalu rendah dapat menghasilkan model yang sensitif terhadap noise, sementara jumlah tetangga yang terlalu tinggi dapat menghasilkan model yang terlalu umum. Disini saya menggunakan n_neighbors sebesar 3.
 
 
###  _Decision tree_
 _Decision tree _ adalah jenis algoritma klasifikasi yang strukturnya mirip seperti sebuah pohon yang memiliki akar, ranting, dan daun. Simpul akar (_internal node_) mewakili fitur pada dataset, simpul ranting (_branch node_) mewakili aturan keputusan (_decision rule_), dan tiap-tiap simpul daun (_leaf node_) mewakili hasil keluaran.
Kelebihan dari algoritma ini antara lain memiliki akurasi yang baik, dapat menemukan kombinasi data yang tidak terduga, dapat menghilangkan perhitungan yang tidak diperlukan, karena dengan metode ini sampel hanya diuji berdasarkan kriteria atau kelas tertentu.
Kekurangan dari algoritma ini diantaranya tumpang tindih dapat terjadi jika banyak kelas dan kriteria yang digunakan, sehingga dapat menyebabkan waktu keputusan menjadi lebih lama dan memori yang dibutuhkan juga lebih besar. 
Parameter yang digunakan pada model ini :
max_depth: Parameter ini menentukan kedalaman maksimum dari setiap pohon dalam model. Kedalaman yang lebih dalam dapat menghasilkan model yang lebih kompleks, tetapi juga dapat menyebabkan _overfitting_.
 
## Evaluation
Metrik evaluasi yang digunakan antara lain adalah metrik _accuracy_, _precision_, _recall_, dan _F1 score_. Di mana:
- _Accuracy_ = Menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar.  Dengan kata lain, _accuracy_ merupakan tingkat kedekatan nilai prediksi dengan nilai sebenarnya.
Accuracy = (TP + TN )/ (TP+FP+FN+TN)
 
- Presisi = Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positif.
Precision = (TP) / (TP+FP)
 
- _Recall_ = Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
Recall = (TP) / (TP + FN)
 
- _F1 Score_ = Merupakan perbandingan rata-rata presisi dan recall yang dibobotkan
F1 Score = 2 * (Recall*Precision) / (Recall + Precision)
 
 
Tabel 1. Metrik Evaluasi

 
|                	  	     |   **accuracy**  |   **precision**  |  **recall** | **f1-score** |
| --------------------------------| --------------------| --------------------| --------------| ----------------| 
| **Logistic Regression** | 0.88                 |  0.99	         | 0.89          | 0.93            |
| **KNN** 		     | 0.93                 |  0.97	         | 0.95          | 0.96            |
| **Decision Tree**     	     | 0.91                 |  0.91	         | 1.00          | 0.95             |
 
Berdasarkan Tabel 1. Dapat disimpulkan bahwa nilai akurasi terbaik ada pada model _KNN_ dengan nilai 0.93 kemudian diikuti oleh model _Decision Tree_ dengan nilai akurasi sebesar 0.91 dan diperingkat terakhir ditempati oleh model _Logistic Regression_ yang memiliki nilai akurasi sebesar 0.88.
Untuk visualisasi perbandingan akurasi masing masing model dapat dilihat pada Gambar 9 dibawah ini.
![modelakurasi](https://github.com/zashnf/Predictive-Analytic/assets/89719711/e8838658-4712-4064-9908-2b2103eb642b)

Gambar 9. Perbandingan Hasil Akurasi Model


Referensi:   

[1] (2020). Machine Learning-based Web Application for Early Diagnosis of Diabetes. Journal of Applied and Emerging Sciences 133–142.

[2] (2021). International Diabetes Federation.
IDF Diabetes Atlas, 10th edn. Brussels,
Belgium: International Diabetes Federation.
