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

---

## 🚀 주요 기능 (Key Features)
| 단계 | 주요 기능 (Features) | 핵심 기술 (Key Tech) |
| :--- | :--- | :--- |
| **1. 정제 및 정규화** | 텍스트 노이즈 제거 및 일관성 확보 | Cleaning, Stemming, Lemmatization |
| **2. 토큰화 전략** | 의미 단위 분절 및 효율적 인덱싱 | Word/Char/Subword Tokenization |
| **3. 데이터 수치화** | 텍스트의 벡터 공간 투영 및 수치 변환 | TF-IDF, Word2Vec, GloVe Embedding |
| **4. 순환 모델링** | 시계열 데이터의 맥락 파악 및 학습 | RNN, LSTM (Gate 구조 분석) |

---

## 💡 회고록 (Retrospective)
&emsp;&emsp;처음 AI와 자연어 처리를 접했을 때는 근사한 모델을 돌려보는 것에만 관심이 있었는데, 막상 프로젝트를 시작해 보니 '데이터를 기계가 알아먹게 만드는 과정'이 얼마나 고된지 뼈저리게 느꼈습니다. 아무리 좋은 모델을 가져와도 입력되는 텍스트가 지저분하면 결과도 엉망이라는 'Garbage In, Garbage Out'의 교훈을 직접 체감할 수 있었습니다.
<br>&emsp;&emsp;이번 실습에서 가장 공을 들인 부분은 전처리 파이프라인의 설계였습니다. 단순히 정규표현식으로 특수문자를 지우는 것뿐만 아니라, 문맥상 큰 의미가 없는 불용어를 걸러내고 단어들을 토큰 단위로 쪼개는 과정이 모델의 학습 효율에 얼마나 직접적인 영향을 주는지 알게 되었습니다. 특히 TF-IDF 기법을 통해 텍스트라는 비정형 데이터를 고차원 벡터로 변환해 보면서, 기계가 단어의 중요도를 숫자로 이해하는 수치 표상(Representation)의 원리를 명확히 이해할 수 있었습니다.
<br>&emsp;&emsp;딥러닝 아키텍처 측면에서는 순차적인 데이터를 다루기 위한 RNN의 구조를 깊게 파고들었습니다. 문장이 길어질수록 앞쪽의 정보를 잊어버리는 RNN의 치명적인 단점을 보완하기 위해 탄생한 LSTM의 게이트(Gate) 메커니즘을 공부하며, '정보를 잊을지 말지'를 스스로 학습한다는 개념이 매우 흥미로웠습니다. 수식으로만 볼 때는 막막했는데, 실제 데이터가 흐르는 과정을 고민하며 코드를 구현해 보니 비로소 장기 의존성(Long-term Dependency) 문제 해결의 실마리가 보이기 시작했습니다.
<br>&emsp;&emsp;물론 아직 넘어야 할 산이 많습니다. 현재의 전처리 방식은 정적인 규칙에 의존하는 경향이 있어, 실제 복잡한 한국어 구어체나 신조어 대응에는 한계가 있음을 느꼈습니다. 앞으로는 단순한 빈도수 기반의 벡터화를 넘어, 문맥에 따라 단어의 의미를 다르게 해석하는 BERT와 같은 고도화된 언어 모델로 파이프라인을 확장해 나가는 것이 다음 목표입니다. 기초를 탄탄히 다진 만큼, 이제는 더 거대한 파라미터를 가진 모델을 어떻게 효율적으로 다루고 배포할 수 있을지에 대해 더 치열하게 고민해 보려 합니다.
