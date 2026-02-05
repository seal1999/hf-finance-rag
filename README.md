# hf-finance-rag

허깅 페이스 금융 데이터셋을 이용한 RAG 구축 및 평가 실습

## 목차

1. [데이터셋 소개](#데이터셋-소개)
2. [사전 요구사항](#사전-요구사항)
3. [환경 설정](#환경-설정)
4. [ChromaDB 구축](#chromadb-구축)
5. [Hybrid RAG 시스템](#hybrid-rag-시스템)
6. [Chat App 실행](#chat-app-실행)
7. [테스트 데이터 생성 및 평가](#테스트-데이터-생성-및-평가)

---

## 데이터셋 소개

### 허깅 페이스 금융 데이터셋

| 데이터셋 | 행 수 | 주요 카테고리 | 컬럼 | 출처 |
|----------|-------|---------------|------|------|
| opensource_korean_finance | 503K | 뉴스 | source, text, category, token_count | [링크](https://huggingface.co/datasets/nmixx-fin/opensource_korean_finance_datasets) |
| synthetic_financial_report | 20.8K | 시황 등 7종 | text, category, source, token_count | [링크](https://huggingface.co/datasets/nmixx-fin/synthetic_financial_report_korean) |
| synthetic_dart_report | 5.08K | 공시 | text, category, source, token_count | [링크](https://huggingface.co/datasets/nmixx-fin/synthetic_dart_report_korean) |

**총 문서 수**: 약 528,747개 (청킹 후 818,779개 청크)

---

## 사전 요구사항

- **Python 3.10+**
- **uv** — [설치 가이드](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** — 로컬 LLM 실행용

```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ollama 설치 (macOS)
brew install ollama

# Ollama 모델 다운로드
ollama pull llama3.2
```

---

## 환경 설정

```bash
# 저장소 클론 후 프로젝트 디렉터리로 이동
cd hf-finance-rag

# 가상환경 생성
uv venv

# 의존성 설치
uv sync

# 가상환경 활성화
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

---

## ChromaDB 구축

### 1. 데이터 수집 및 저장

`ingest_to_chroma.py`를 실행하여 허깅 페이스 데이터셋을 로드하고 ChromaDB에 저장합니다.

```bash
uv run python ingest_to_chroma.py
```

### 처리 과정

1. **데이터 로드**: 3개의 허깅 페이스 금융 데이터셋 로드
2. **텍스트 청킹**: RecursiveCharacterTextSplitter 사용
   - chunk_size: 500자
   - chunk_overlap: 50자
3. **임베딩**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
4. **저장**: ChromaDB에 벡터 저장 (`./chroma_db` 디렉터리)

### 실행 결과

```
총 문서 수: 528,747
총 청크 수: 818,779
저장 위치: ./chroma_db
```

---

## Hybrid RAG 시스템

### 주요 파일: `hybrid_rag_query.py`

### 핵심 기능

#### 1. Hybrid Search (하이브리드 검색)

- **BM25 키워드 검색**: Kiwi 한국어 토크나이저 사용
- **Vector 검색**: ChromaDB 유사도 검색
- **RRF (Reciprocal Rank Fusion)**: 두 검색 결과 통합

```python
# 가중치 설정
bm25_weight=0.3
vector_weight=0.7
```

#### 2. Query Transformation (쿼리 변환)

| 기능 | 설명 |
|------|------|
| **Rewrite** | 질문을 검색에 최적화된 형태로 재작성 |
| **Decomposition** | 복잡한 질문을 하위 질문으로 분해 |
| **HyDE** | 가상의 답변 문서 생성 후 검색 |

#### 3. 2-Stage Retrieval (2단계 검색)

1. **1단계 (Bi-Encoder)**: 빠른 후보 검색 (top_k=15)
2. **2단계 (Cross-Encoder Reranker)**: 정밀 재순위화 (rerank_top_n=5)
   - 모델: `BAAI/bge-reranker-v2-m3`

### 사용 예시

```python
from hybrid_rag_query import AdvancedHybridRAG

# RAG 시스템 초기화
rag = AdvancedHybridRAG(
    top_k=15,
    rerank_top_n=5,
    bm25_weight=0.3,
    vector_weight=0.7,
    use_rewrite=True,
    use_decomposition=False,
    use_hyde=False,
    use_reranker=True,
)

# 질문에 대한 답변 생성
question = "삼성전자 반도체 실적은?"
transformed = rag.transform_query(question)
documents = rag.hybrid_search(question, transformed)
answer = rag.generate_answer(question, documents)
print(answer)
```

---

## Chat App 실행

### 주요 파일: `chat_app.py`

Gradio 기반의 웹 채팅 인터페이스입니다.

### 실행 방법

```bash
# Ollama 서버 실행 (별도 터미널)
ollama serve

# Chat App 실행
uv run python chat_app.py
```

### 접속

- **URL**: http://localhost:7860
- 첫 질문 시 RAG 시스템 초기화 (Lazy Loading)

### 주요 기능

- 금융 관련 질문에 대한 AI 답변
- 참고 문서(출처) 표시
- 예시 질문 제공

---

## 테스트 데이터 생성 및 평가

### 주요 파일: `evaluate_rag.py`

LLM-as-Judge 방식으로 RAG 시스템을 평가합니다.

### 실행 방법

```bash
# Ollama 서버 실행 (별도 터미널)
ollama serve

# 평가 실행
uv run python evaluate_rag.py
```

### 평가 과정

1. **테스트셋 생성**: ChromaDB에서 문서 샘플링 후 LLM으로 질문/답변 쌍 생성 (10개)
2. **RAG 답변 생성**: 생성된 질문에 대해 RAG 시스템으로 답변
3. **LLM-as-Judge 평가**: 3가지 메트릭으로 점수 산정

### 평가 메트릭

| 메트릭 | 설명 | 점수 범위 |
|--------|------|-----------|
| **Faithfulness** | RAG 답변이 검색된 문서에 충실한가 (환각 여부) | 1-5 |
| **Relevancy** | 답변이 질문의 의도에 적절히 부합하는가 | 1-5 |
| **Correctness** | 답변이 참조 정답과 일치하는가 | 1-5 |

### 점수 기준

- **4.0 이상**: 우수 (Production Ready)
- **3.0-4.0**: 양호 (개선 여지 있음)
- **3.0 미만**: 개선 필요

### 출력 파일

실행 시 타임스탬프가 포함된 파일명으로 저장됩니다:

- `evaluation_report_YYYYMMDD_HHMMSS.md` — 마크다운 형식 보고서
- `evaluation_results_YYYYMMDD_HHMMSS.csv` — 상세 결과 CSV

---

## 프로젝트 구조

```
hf-finance-rag/
├── ingest_to_chroma.py      # 데이터 수집 및 ChromaDB 저장
├── hybrid_rag_query.py      # Hybrid RAG 시스템 (핵심)
├── chat_app.py              # Gradio 채팅 UI
├── evaluate_rag.py          # RAG 평가 프로그램
├── advanced_rag_practice.ipynb  # 실습 노트북
├── chroma_db/               # ChromaDB 저장소
├── .env                     # 환경 변수 (API 키)
└── README.md
```

---

## 의존성 추가/업데이트

```bash
# 패키지 추가 시 pyproject.toml의 dependencies에 추가 후
uv sync

# lock 파일만 갱신 (설치 없이)
uv lock
```
