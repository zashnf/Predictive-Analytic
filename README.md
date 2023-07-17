
# Domain proyek
Diabetes merupakan penyakit kronis yang disebabkan oleh gagalnya organ pankreas memproduksi jumlah hormon insulin secara memadai sehingga menyebabkan peningkatan kadar glukosa dalam darah. Pemeriksaan pada penyakit Diabetes dalam dunia medis dapat dilakukan dengan cara pendiagnosaan penyakit berdasarkan gejala-gejala yang diderita oleh penderita yang dapat menghasilkan data hasil uji laboratorium dan rekam medis gejala sakit.

Data dari WHO mengatakan bahwa tahun 2015 ada sekitar 415 juta penderita Diabetes di Indonesia. Prediksi berbicara bahwa di tahun 2040, penderita akan bertambah menjadi 2040. Peningkatan penderita Diabetes meningkat dari 5,7 % menjadi 6,9 %. jumlah pastinya sekitar 9,1 juta di tahun 2013.

Untuk meminimalisir angka kematian dari penyakit Diabetes ini, para pakar kesehatan harus melakukan pendiagnosaan penyakit sedini mungkin. Klasifikasi dapat dijadikan salah satu penanganan dini dari penyakit ini. Dataset pada proyek ini digunakan untuk membangun model pembelajaran mesin yang dapat memprediksi kemungkinan diabetes pada pasien berdasarkan riwayat medis dan detail demografis mereka.

Referensi:   https://p2ptm.kemkes.go.id/kegiatan-p2ptm/subdit-penyakit-diabetes-melitus-dan-gangguan-metabolik/perlunya-deteksi-dini-untuk-cegah-dan-kurangi-risiko-diabetes

# Business Understanding
- Problem Statement
Berdasarkan penjelasan yang telah disampaikan sebelumnya, maka problem statements (rumusan masalah), yaitu sebagai berikut: mengidentifikasi faktor-faktor utama yang mempengaruhi kemungkinan penyakit diabetes.
- Goals
Tujuan yang ingin dicapai dari prediksi diabetes adalah membandingkan akurasi model untuk memilih yang terbaik. Membuat aplikasi yang dapat memprediksi kemungkinan diabetes pada pasien berdasarkan riwayat medis dan detail demografis.
- Solution statements
Solusi yang dapat dilakukan untuk menangani permasalahan sebagaimana terdapat dalam _problem statements_, yaitu dengan membuat aplikasi prediksi diabetes. Adapun aplikasi tersebut dibuat dengan menerapkan teknologi _machine learning_ serta bahasa pemrograman python. Algoritma machine learning yang akan digunakan, yaitu _Logistic Regression_, _K Nearest Neighbor(KNN)_, dan _Decision Tree_.

# Data Understanding
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi dataset terkait diabetes. Dataset yang digunakan dapat didownload pada link berikut ini:   https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/code

Jumlah data yang terdapat di dalam file tersebut sebanyak 100000 data. Variabel-variabel yang terdapat dalam dataset Diabetes:
- gender = Mengacu pada jenis kelamin biologis seseorang, yang dapat berdampak pada kerentanan mereka terhadap diabetes. Ada tiga kategori di dalamnya, yaitu _ male_, _female_. dan _other_.
- age = Mengacu pada usia,merupakan salah satu  faktor penting karena diabetes lebih sering didiagnosis pada orang dewasa yang lebih tua, usia berkisar antara 0-80 pada dataset ini.
- hypertension = Kondisi medis dimana tekanan darah di arteri terus meningkat. Ini memiliki nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki hipertensi dan untuk 1 berarti mereka memiliki hipertensi.
- heart_disease = kondisi medis lain(penyakit jantung) yang dikaitkan dengan peningkatan risiko diabetes. Ini memiliki nilai 0 atau 1 di mana 0 menunjukkan mereka tidak memiliki penyakit jantung dan untuk 1 berarti mereka memiliki penyakit jantung.
- smoking_history = Riwayat merokok juga dianggap sebagai faktor risiko diabetes dan dapat memperburuk komplikasi yang terkait dengan - diabetes, pada dataset ini memiliki 5 kategori yaitu _not current_, _former_,_ No Info_, _current_ dan _never_.
- bmi = BMI (Indeks Massa Tubuh) adalah ukuran lemak tubuh berdasarkan berat dan tinggi badan. Nilai BMI yang lebih tinggi dikaitkan dengan risiko diabetes yang lebih tinggi. Kisaran BMI dalam dataset ini adalah dari 10,16 hingga 71,55. BMI kurang dari 18,5 berarti kekurangan berat badan, 18,5-24,9 berarti normal, 25-29,9 berarti kelebihan berat badan, dan 30 atau lebih berarti obesitas.
- hbA1c_level =  Hemoglobin A1c adalah ukuran kadar gula darah rata-rata seseorang selama 2-3 bulan terakhir. Kadar yang lebih tinggi menunjukkan risiko yang lebih besar terkena diabetes. Sebagian besar kadar HbA1c lebih dari 6,5% mengindikasikan diabetes.
- blood_glucose_level = Mengacu pada jumlah glukosa dalam aliran darah pada waktu tertentu. Kadar glukosa darah yang tinggi adalah atau lebih dari 140 bisa menjadi indikator utama diabetes.
- diabetes = Variabel target yang diprediksi, dengan nilai 1 menunjukkan adanya diabetes dan 0 menunjukkan tidak adanya diabetes.

# Data Preparation
Teknik _data preparation_ yang dilakukan, yaitu sebagai berikut:
- Mengubah dataset diabetes menjadi _dataframe_ dengan menggunakan pandas.
- Melakukan _exploratory data analysis_ untuk memahami variabel-variabel yang terdapat dalam dataset.
- Memvisualisasikan data dengan menggunakan plot.
- Memvisualisasikan data persentase dengan diagram pie.
- Membagi data menjadi data latih dan data uji. Pada proyek ini saya membagi data latih menjadi 70% dan data uji sebesar 30%.
- Melakukan proses _encoding_, dimana _encoding_ sendiri merupakan proses yang mengubah data kategorikal menjadi data numerikal.Dalam hal ini meng_encode_ _gender_ dan _smoking_history_ karena nilai pada data tersebut masih kategorikal sehingga harus diubah menjari numerik.
- Melakukan _feature selection_ untuk menyaring fitur-fitur penting karena semua fitur yang ada dalam dataset tidak sama pentingnya. Menggunakan _variance Threshold_ pada data latih untuk mengecheck apakah ada fitur yang memiliki varians rendah.
![download (5)](https://github.com/zashnf/Predictive_Systems/assets/89719711/3eb71f93-fc8c-4d66-9766-5ae6de6c49cd)
Gambar 1.
- Menangani data set yang tidak seimbang menggunakan SMOTE

![download (2)](https://github.com/zashnf/Predictive_Systems/assets/89719711/44e3168f-2ff4-4a03-b626-17c59e623511)

Gambar 2.

Menurut countplot diatas , dataset yang digunakan tidak seimbang( pada 1 kelas/kategori memiliki jumlah contoh yang jauh lebih besar daripada kelas lainnya). Sehingga disini dapat menggunakan SMOTE (_Synthetic Minority Over-sampling Technique_) untuk mengatasi masalah ketidakseimbangan kelas dengan meningkatkan jumlah instance dalam kelas minoritas.
Metode SMOTE digunakan karena adanya ketidakseimbangan pada kategori diabetes

![Capture](https://github.com/zashnf/Predictive_Systems/assets/89719711/70be4de1-4d15-4859-beb2-e49ca4b92d92)
Gambar 3.


- Melakukan pelatihan model ( _Logistic Regression, K Nearest Neighbor,_ dan _Decision Tree_).


# Modeling
Model _machine learning_ yang digunakan adalah_ Logistic Regression,K Nearest Neighbor,_ dan _Decision Tree_.
_Logistic Regression_ adalah salah satu algoritma Machine Learning yang paling populer, yang termasuk dalam teknik _Supervised Learning_. Algoritma ini digunakan untuk memprediksi variabel dependen kategorik dengan menggunakan sekumpulan variabel independen.
Cara Kerja _Logistic   Regression_   menghitung   nilai   probabilitas   suatu _instance_ data masuk ke dalam kelas tertentu. Hal ini dilakukan dengan memperhitungkan bobot setiap variabel input pada suatu fungsi logistik. 
Fungsi logistik digunakan untuk mengubah hasil perhitungan bobot variabel input menjadi nilai probabilitas yang berkisar antara 0 dan 1. Dalam proses training,_ Logistic Regression_ meminimalkan _error_   prediksi   dengan   memperbarui   bobot   variabel   input menggunakan   teknik   optimasi   seperti   _Gradient   Descent_. Tujuannya   adalah   untuk   menemukan   bobot   yang   terbaik sehingga model dapat memprediksi kelas dari instance data yang belum dilihat dengan akurasi yang tinggi
Kelebihan dari _Logistic Regression_ antara lain sangat efektif untuk klasifikasi data biner, sangat mudah diimplementasikan dan diinterpretasikan, dapat digunakan pada dataset yang memiliki banyak variabel input.
Selain kelebihan yang dimiliki ia juga memiliki kekurangan seperti tidak   efektif   untuk   klasifikasi   data   yang   kompleks   dengan banyak variabel input, rentan terhadap overfitting jika digunakan  pada dataset  yang tidak seimbang, tidak   dapat   mengatasi   masalah   _multicollinearity _  di   antara variabel input.

_K Nearest Neighbor (KNN)_ adalah teknik pembelajaran mesin yang populer digunakan untuk tugas klasifikasi dan regresi. Algoritma ini bergantung pada gagasan bahwa titik data yang serupa cenderung memiliki label atau nilai yang serupa.
Selama fase pelatihan, algoritma KNN menyimpan seluruh kumpulan data pelatihan sebagai referensi. Saat membuat prediksi, algoritma ini menghitung jarak antara titik data input dan semua contoh pelatihan, menggunakan metrik jarak yang dipilih seperti jarak Euclidean.
Selanjutnya, algoritma mengidentifikasi K tetangga terdekat dari titik data input berdasarkan jaraknya. Dalam kasus klasifikasi, algoritma memberikan label kelas yang paling umum di antara K tetangga sebagai label yang diprediksi untuk titik data input. Untuk regresi, algoritma ini menghitung rata-rata atau rata-rata tertimbang dari nilai target dari K tetangga untuk memprediksi nilai untuk titik data input.
Mendefinisikan k dapat menjadi tindakan penyeimbang karena nilai yang berbeda dapat menyebabkan _overfitting_ atau _underfitting_. Nilai k yang lebih rendah dapat memiliki varians yang tinggi, tetapi bias yang rendah. Sedangkan nilai k yang lebih besar dapat menyebabkan bias yang tinggi dan varians yang lebih rendah. 
Pilihan k akan sangat bergantung pada data input karena data dengan lebih banyak _outlier_ atau _noise _kemungkinan akan berkinerja lebih baik dengan nilai k yang lebih tinggi. Secara keseluruhan, disarankan untuk memilih nilai k berupa angka ganjil untuk menghindari ikatan dalam klasifikasi. Nilai k pada algoritma KNN mendefinisikan berapa banyak tetangga yang akan diperiksa untuk menentukan klasifikasi titik kueri tertentu. Misalnya, jika k=1, _instance_ akan ditugaskan ke kelas yang sama dengan tetangga terdekatnya. 
Kelebihan dari _K Nearest Neighbor _adalah mudah dipahami dan diimplementasikan.  Namun kekurangannya adalah ia masih perlu menunjukkan parameter k (jumlah tetangga terdekat), tidak menangani nilai hilang (_missing value_) secara implisit.
_Decision tree _adalah jenis algoritma klasifikasi yang strukturnya mirip seperti sebuah pohon yang memiliki akar, ranting, dan daun. Simpul akar (_internal node_) mewakili fitur pada dataset, simpul ranting (_branch node_) mewakili aturan keputusan (_decision rule_), dan tiap-tiap simpul daun (_leaf node_) mewakili hasil keluaran.
Kelebihan dari algoritma ini antara lain memiliki akurasi yang baik, dapat menemukan kombinasi data yang tidak terduga, dapat menghilangkan perhitungan yang tidak diperlukan, karena dengan metode ini sampel hanya diuji berdasarkan kriteria atau kelas tertentu.
Kekurangan dari algoritma ini diantaranya tumpang tindih dapat terjadi jika banyak kelas dan kriteria yang digunakan, sehingga dapat menyebabkan waktu keputusan menjadi lebih lama dan memori yang dibutuhkan juga lebih besar. 

# Evaluation
Metrik evaluasi yang digunakan antara lain adalah metrik _accuracy_, _precision_, _recall_, dan _F1 score_. Di mana:
- _Accuracy_ = Menggambarkan seberapa akurat model dapat mengklasifikasikan dengan benar.  Dengan kata lain, _accuracy _merupakan tingkat kedekatan nilai prediksi dengan nilai sebenarnya.
Accuracy = (TP + TN )/ (TP+FP+FN+TN)
- Presisi = Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positif.
Precision = (TP) / (TP+FP)
- _Recall_ = Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
Recall = (TP) / (TP + FN)
- _F1 Score_ = Merupakan perbandingan rata-rata presisi dan recall yang dibobotkan
F1 Score = 2 * (Recall*Precission) / (Recall + Precission)

# Kesimpulan
Kesimpulan dari proyek ini didapati bahwa faktor-faktor yang dapat mempengaruhi kemungkinan penyakit diabetes lebih banyak terjadi pada seseorang dengan riwayat merokok aktif dan lebih banyak menyerang ke wanita (_female_), untuk visualisasi perbandingannya dapat dilihat pada gambar dibawah ini.
![download (4)](https://github.com/zashnf/Predictive_Systems/assets/89719711/2a68457f-1869-496c-8ebc-962e3b524df0)

Gambar 4.

![download (3)](https://github.com/zashnf/Predictive_Systems/assets/89719711/57a75876-45d9-4b5f-a742-a4b64aa32116)

Gambar 5.

Dengan perhitungan metrik evaluasi saya mengambil nilai _accuracy_ dari tiap tiap model algoritma untuk dibandingkan di dalam satu bar plot untuk melihat dari model manakah yang _accuracy_nya lebih baik.

![download (1)](https://github.com/zashnf/Predictive_Systems/assets/89719711/d9df4e07-d67b-4a93-ba5c-df8d6ea5675e)

Gambar 6.
Pada gambar 6  dapat dilihat bahwa model KNN memiliki nilai _accuraccy_ tertinggi dibanding dengan _Logistic Regression_ dan _Decision Tree_.

