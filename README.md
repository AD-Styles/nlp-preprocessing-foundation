# 📝 NLP Preprocessing & Sequence Modeling Foundation
## NLP 전처리 및 시퀀스 모델링 기초

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

본 프로젝트는 텍스트 데이터의 수치 표상(Representation)과 초기 언어 모델링의 핵심 원리를 체계적으로 구현한 결과물입니다. BERT와 같은 고성능 트랜스포머 아키텍처를 이해하기 위한 필수 단계로서, 자연어의 특성을 보존하며 기계 학습 모델에 최적화된 형태로 데이터를 가공하는 엔지니어링 역량을 담고 있습니다.

## 📌 프로젝트 개요 (Project Overview)
비정형 텍스트 데이터를 정형 데이터로 변환하는 핵심 기법들을 탐구하고, 언어의 순차적(Sequential) 특성을 학습하기 위한 기초 딥러닝 아키텍처를 구축합니다. 단순한 모델 호출을 넘어, 데이터 전처리와 벡터화 방식이 전체 파이프라인 성능에 미치는 영향을 코드 수준에서 검증하는 것을 목표로 합니다.

## 📂 프로젝트 구조 (Project Structure)
```text
📂 src/                                     
    └── 1_preprocessing_pipeline.py     # OOP 기반 텍스트 정제 및 TF-IDF 벡터화 파이프라인 구현
├── .gitignore                          # 프로젝트 관리 제외 설정
├── LICENSE                             # MIT License (AD-Styles)
├── README.md                           # 프로젝트 요약 및 가이드
└── requirements.txt                    # 필수 라이브러리 목록
```

---

## 🛠️ 기술 스택 (Tech Stack)
| 구분 | 상세 항목 |
| :--- | :--- |
| **Language** | Python |
| **Libraries** | NLTK, Spacy, Scikit-learn (전처리 및 벡터화) |
| **Frameworks** | PyTorch / TensorFlow (순환 신경망 구현) |
| **Visualization** | Matplotlib, Seaborn |

## 🚀 주요 기능 (Key Features)

| 단계 | 주요 기능 (Features) | 핵심 기술 (Key Tech) |
| :--- | :--- | :--- |
| **1. 정제 및 정규화** | 텍스트 노이즈 제거 및 일관성 확보 | Cleaning, Stemming, Lemmatization |
| **2. 토큰화 전략** | 의미 단위 분절 및 효율적 인덱싱 | Word/Char/Subword Tokenization |
| **3. 데이터 수치화** | 텍스트의 벡터 공간 투영 및 수치 변환 | TF-IDF, Word2Vec, GloVe Embedding |
| **4. 순환 모델링** | 시계열 데이터의 맥락 파악 및 학습 | RNN, LSTM (Gate 구조 분석) |
