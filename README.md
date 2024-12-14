# Tugas Besar 2 Natural Language Processing

**Nama  :** Muhammad Aryaka Zamzami
**NIM   :** 41522010100

## News Classification
Proyek News Classification ini bertujuan untuk mengembangkan sistem klasifikasi berita otomatis yang mampu mengelompokkan berita ke dalam kategori tertentu menggunakan pendekatan Natural Language Processing (NLP) dan Machine Learning.

Dataset yang digunakan berasal dari News Category Dataset di Kaggle, yang terdiri dari sekitar 200.000 data berita dalam berbagai kategori. Proses awal pengolahan data dilakukan dengan preprocessing, meliputi pembersihan teks, penghapusan kata-kata umum (stopwords). Data teks ini kemudian dikonversi menjadi representasi numerik menggunakan teknik TF-IDF, yang menilai bobot setiap kata berdasarkan frekuensi dan kepentingannya dalam dokumen.

Model yang digunakan untuk klasifikasi adalah Logistic Regression. Model ini dilatih menggunakan data yang telah diproses, dan kinerjanya dievaluasi dengan menggunakan metrik seperti precision, recall, dan F1-score. 

[Link Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)