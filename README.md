# Workflow-CI — Heart Disease Classification

Repository ini berisi MLflow Project dan GitHub Actions CI
untuk pelatihan model Heart Disease Classification secara otomatis.

**Author**: Yusfitasari

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   ├── modelling.py
│   └── heart_preprocessing/
│       ├── heart_train.csv
│       └── heart_test.csv
├── automate_Yusfitasari.py
├── heart.csv
└── README.md
```

## Cara Menjalankan Secara Lokal

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn mlflow matplotlib seaborn

# 2. Jalankan preprocessing
python automate_Yusfitasari.py --input heart.csv

# 3. Jalankan MLflow Project
mlflow run MLProject/ --env-manager=local

# 4. Lihat hasil di MLflow UI
mlflow ui
```

## CI/CD

Workflow otomatis berjalan setiap kali ada push ke branch `main`.
Artefak hasil training tersimpan di GitHub Actions Artifacts.
