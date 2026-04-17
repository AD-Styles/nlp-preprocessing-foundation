import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

class NLPPreprocessor:
    """텍스트 데이터 정제 및 토큰화를 위한 전처리 클래스"""
    
    def __init__(self):
        self.stopwords = {"is", "the", "in", "and", "to", "a", "of", "about"}

    def clean_text(self, text: str) -> str:
        """소문자 변환 및 특수문자 제거"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def tokenize_and_remove_stopwords(self, text: str) -> List[str]:
        """단어 토큰화 및 불용어 제거"""
        tokens = text.split()
        cleaned_tokens = [word for word in tokens if word not in self.stopwords]
        return cleaned_tokens

    def process_corpus(self, corpus: List[str]) -> List[str]:
        """전체 코퍼스에 대한 전처리 파이프라인 실행"""
        processed_corpus = []
        for doc in corpus:
            cleaned = self.clean_text(doc)
            tokens = self.tokenize_and_remove_stopwords(cleaned)
            processed_corpus.append(" ".join(tokens))
        return processed_corpus

class TextVectorizer:
    """전처리된 텍스트를 수치 벡터로 변환하는 클래스"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, corpus: List[str]):
        return self.vectorizer.fit_transform(corpus)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

if __name__ == "__main__":
    sample_data = [
        "Natural Language Processing is highly fascinating!",
        "I love learning about machine learning and NLP.",
        "Text preprocessing is the first step in the NLP pipeline."
    ]

    preprocessor = NLPPreprocessor()
    processed_data = preprocessor.process_corpus(sample_data)

    vectorizer = TextVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_data)

    print("--- NLP Preprocessing Pipeline Result ---")
    print(f"Vocabulary: {vectorizer.get_feature_names()}")
    print(f"Matrix Shape: {tfidf_matrix.shape}")
