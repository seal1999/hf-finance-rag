"""
Advanced Hybrid RAG 시스템
- Query Transformation: Rewrite, Decomposition, HyDE
- Vector Search + BM25 Rerank
- Cross-Encoder Reranker (2-Stage Retrieval)
- Ollama (llama3.2)로 답변 생성 (무료, 로컬)
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from sentence_transformers import CrossEncoder
import re

# 설정
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 다국어 지원 Reranker
COLLECTION_NAME = "finance_docs"


class CrossEncoderReranker:
    """Cross-Encoder 기반 Reranker (2-Stage Retrieval의 2단계)"""

    def __init__(self, model_name: str = RERANKER_MODEL, top_n: int = 5):
        """
        Args:
            model_name: Cross-Encoder 모델 이름
            top_n: 최종 반환할 문서 수
        """
        self.top_n = top_n
        print(f"[Reranker 로드 중... ({model_name})]")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """문서를 쿼리와의 관련성 기준으로 재순위화"""
        if not documents:
            return []

        # (query, document) 쌍 생성
        pairs = [(query, doc.page_content) for doc in documents]

        # Cross-Encoder로 점수 계산
        scores = self.model.predict(pairs)

        # 점수와 문서를 쌍으로 묶어서 정렬
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 상위 top_n개 반환
        return [doc for _, doc in scored_docs[:self.top_n]]


class QueryTransformer:
    """쿼리 변환 클래스: Rewrite, Decomposition, HyDE"""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        """쿼리 재작성: 더 명확하고 검색에 적합한 형태로 변환"""
        prompt = f"""당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 검색에 더 적합한 형태로 재작성해 주세요.

원칙:
- 고유명사(회사명, 인물명, 제품명)는 반드시 유지
- 핵심 키워드를 명확히 포함
- 불필요한 조사나 어미 제거
- 한 문장으로 간결하게

원본 질문: {query}

재작성된 쿼리 (한 문장만 출력, 고유명사 유지):"""

        response = self.llm.invoke(prompt)
        rewritten = response.strip().split('\n')[0].strip()
        rewritten = rewritten.strip('"\'')
        return rewritten if rewritten else query

    def decompose(self, query: str) -> list[str]:
        """쿼리 분해: 복잡한 질문을 여러 하위 질문으로 분해"""
        prompt = f"""당신은 질문 분석 전문가입니다.
복잡한 질문을 검색하기 쉬운 2-3개의 하위 질문으로 분해해 주세요.

원칙:
- 각 하위 질문은 독립적으로 검색 가능해야 함
- 원본 질문의 모든 측면을 커버해야 함
- 간결하고 명확하게 작성

원본 질문: {query}

하위 질문들 (각 줄에 하나씩, 번호 없이):"""

        response = self.llm.invoke(prompt)

        sub_queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^[\d\.\-\*\•]+\s*', '', line)
            if line and len(line) > 5:
                sub_queries.append(line)

        if not sub_queries:
            return [query]
        return sub_queries[:3]

    def generate_hyde(self, query: str) -> str:
        """HyDE (Hypothetical Document Embeddings): 가상의 답변 문서 생성"""
        prompt = f"""당신은 금융 분야 전문가입니다.
다음 질문에 대한 답변이 포함된 가상의 금융 뉴스/보고서 문단을 작성해 주세요.

원칙:
- 실제 금융 문서처럼 작성
- 구체적인 정보와 수치 포함 (가상이어도 됨)
- 2-3문장으로 간결하게
- 검색에 유용한 키워드 포함

질문: {query}

가상 문서 (2-3문장):"""

        response = self.llm.invoke(prompt)
        hyde_doc = response.strip()
        return hyde_doc if hyde_doc else query


class AdvancedHybridRAG:
    """Advanced Hybrid RAG: Query Transformation + BM25 + Vector Search + Reranker"""

    def __init__(
        self,
        top_k: int = 10,
        rerank_top_n: int = 5,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        use_rewrite: bool = True,
        use_decomposition: bool = True,
        use_hyde: bool = True,
        use_reranker: bool = True,
    ):
        """
        Args:
            top_k: 1차 검색에서 가져올 문서 수
            rerank_top_n: Reranker 후 최종 반환할 문서 수
            bm25_weight: BM25 점수 가중치
            vector_weight: Vector 검색 점수 가중치
            use_rewrite: 쿼리 재작성 사용 여부
            use_decomposition: 쿼리 분해 사용 여부
            use_hyde: HyDE 사용 여부
            use_reranker: Cross-Encoder Reranker 사용 여부
        """
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.use_rewrite = use_rewrite
        self.use_decomposition = use_decomposition
        self.use_hyde = use_hyde
        self.use_reranker = use_reranker

        # 형태소 분석기 초기화 (한국어)
        self.kiwi = Kiwi()

        # 임베딩 모델 로드
        print("[임베딩 모델 로드 중...]")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ChromaDB 로드
        print("[ChromaDB 로드 중...]")
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        # LLM 초기화 (Ollama - 무료, 로컬)
        print("[LLM 초기화 중... (Ollama llama3.2)]")
        self.llm = Ollama(
            model="llama3.2",
            temperature=0,
        )

        # Query Transformer 초기화
        self.query_transformer = QueryTransformer(self.llm)

        # Cross-Encoder Reranker 초기화
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(top_n=rerank_top_n)
        else:
            self.reranker = None

        print("[초기화 완료!]")

    def _tokenize(self, text: str) -> list[str]:
        """한국어 형태소 분석"""
        tokens = []
        for token in self.kiwi.tokenize(text):
            if token.tag.startswith(("NN", "VV", "VA", "XR")):
                tokens.append(token.form)
        return tokens

    def _vector_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Vector 검색 (ChromaDB)"""
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
        return results

    def _bm25_rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """BM25로 후보 문서 재순위화"""
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]
        tokenized_docs = [self._tokenize(text) for text in doc_texts]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        return [(doc, score) for doc, score in zip(documents, scores)]

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """점수 정규화 (0~1)"""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _merge_documents(self, doc_lists: list[list[tuple[Document, float]]]) -> list[Document]:
        """여러 검색 결과를 융합 (Reciprocal Rank Fusion)"""
        doc_scores = {}
        k = 60  # RRF 상수

        for doc_list in doc_lists:
            for rank, (doc, _) in enumerate(doc_list):
                doc_id = doc.page_content[:100]
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0}
                doc_scores[doc_id]["score"] += 1 / (k + rank + 1)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:self.top_k * 2]]  # Reranker용으로 더 많이 가져옴

    def transform_query(self, query: str, verbose: bool = True) -> dict:
        """쿼리 변환 수행"""
        result = {
            "original": query,
            "rewritten": None,
            "sub_queries": None,
            "hyde_doc": None,
        }

        if self.use_rewrite:
            if verbose:
                print("  - Rewrite 수행 중...")
            result["rewritten"] = self.query_transformer.rewrite(query)
            if verbose:
                print(f"    → {result['rewritten']}")

        if self.use_decomposition:
            if verbose:
                print("  - Decomposition 수행 중...")
            result["sub_queries"] = self.query_transformer.decompose(query)
            if verbose:
                for i, sq in enumerate(result["sub_queries"], 1):
                    print(f"    → 하위질문 {i}: {sq}")

        if self.use_hyde:
            if verbose:
                print("  - HyDE 문서 생성 중...")
            result["hyde_doc"] = self.query_transformer.generate_hyde(query)
            if verbose:
                print(f"    → {result['hyde_doc'][:100]}...")

        return result

    def hybrid_search(self, query: str, transformed: dict, verbose: bool = True) -> list[Document]:
        """변환된 쿼리로 하이브리드 검색 + Reranking"""
        all_results = []

        # 1. 원본 쿼리로 검색
        original_results = self._vector_search(query, self.top_k * 2)
        all_results.append(original_results)

        # 2. Rewritten 쿼리로 검색
        if transformed.get("rewritten"):
            rewritten_results = self._vector_search(transformed["rewritten"], self.top_k * 2)
            all_results.append(rewritten_results)

        # 3. Sub-queries로 검색
        if transformed.get("sub_queries"):
            for sub_query in transformed["sub_queries"]:
                sub_results = self._vector_search(sub_query, self.top_k)
                all_results.append(sub_results)

        # 4. HyDE 문서로 검색
        if transformed.get("hyde_doc"):
            hyde_results = self._vector_search(transformed["hyde_doc"], self.top_k * 2)
            all_results.append(hyde_results)

        # 결과 융합 (RRF)
        merged_docs = self._merge_documents(all_results)

        # BM25 재순위화
        bm25_results = self._bm25_rerank(query, merged_docs)

        # Vector + BM25 점수 융합
        doc_scores = {}

        for i, doc in enumerate(merged_docs):
            doc_id = doc.page_content[:100]
            doc_scores[doc_id] = {"doc": doc, "vector": 1.0 - (i / len(merged_docs)), "bm25": 0}

        bm25_scores = [score for _, score in bm25_results]
        bm25_normalized = self._normalize_scores(bm25_scores)
        for (doc, _), norm_score in zip(bm25_results, bm25_normalized):
            doc_id = doc.page_content[:100]
            if doc_id in doc_scores:
                doc_scores[doc_id]["bm25"] = norm_score

        final_results = []
        for data in doc_scores.values():
            final_score = (self.bm25_weight * data["bm25"]) + (self.vector_weight * data["vector"])
            final_results.append((data["doc"], final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)
        candidate_docs = [doc for doc, _ in final_results[:self.top_k]]

        # Cross-Encoder Reranker 적용 (2-Stage Retrieval의 2단계)
        if self.use_reranker and self.reranker:
            if verbose:
                print(f"  - Cross-Encoder Reranking 중... ({len(candidate_docs)}개 → {self.rerank_top_n}개)")
            final_docs = self.reranker.rerank(query, candidate_docs)
        else:
            final_docs = candidate_docs[:self.rerank_top_n]

        return final_docs

    def generate_answer(self, query: str, documents: list[Document]) -> str:
        """검색된 문서를 기반으로 답변 생성 (환각 방지 강화)"""
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])

        prompt = f"""당신은 금융 분야 전문가입니다. 아래 제공된 참고 문서를 기반으로 질문에 답변하세요.

## 참고 문서
{context}

## 질문
{query}

## 답변 규칙
1. 참고 문서에 있는 내용만 사용하여 답변하세요.
2. 문서에서 직접 관련된 내용을 찾아 인용하세요.
3. 문서의 표현을 최대한 그대로 사용하세요.
4. 2-4문장으로 간결하게 답변하세요.

## 답변
"""

        response = self.llm.invoke(prompt)
        return response.strip()

    def query(self, question: str, verbose: bool = True) -> str:
        """질문에 대한 답변 생성 (Query Transformation + Reranking 포함)"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"[질문] {question}")
            print("="*60)

        # 1. 쿼리 변환
        if verbose:
            print("\n[1. 쿼리 변환 중...]")
        transformed = self.transform_query(question, verbose)

        # 2. 하이브리드 검색 + Reranking
        if verbose:
            print("\n[2. 하이브리드 검색 + Reranking 중...]")
        documents = self.hybrid_search(question, transformed, verbose)

        if verbose:
            print(f"  - 최종 문서 수: {len(documents)}")
            print("\n[검색된 문서 미리보기]")
            for i, doc in enumerate(documents[:3], 1):
                print(f"\n--- 문서 {i} ---")
                print(f"{doc.page_content[:200]}...")

        # 3. 답변 생성
        if verbose:
            print("\n[3. 답변 생성 중...]")
        answer = self.generate_answer(question, documents)

        if verbose:
            print("\n" + "="*60)
            print("[답변]")
            print("="*60)
            print(answer)

        return answer


def main():
    # Advanced RAG 시스템 초기화
    # 2-Stage Retrieval: 1차(top_k=20) → 2차 Rerank(top_n=7)
    rag = AdvancedHybridRAG(
        top_k=20,           # 1차 검색에서 가져올 후보 문서 수
        rerank_top_n=7,     # Reranker 후 최종 문서 수 (증가)
        bm25_weight=0.4,    # BM25 가중치 증가 (키워드 매칭 강화)
        vector_weight=0.6,
        use_rewrite=True,
        use_decomposition=False,  # 단순 질문에서는 비활성화
        use_hyde=False,           # 환각 방지를 위해 비활성화
        use_reranker=True,  # Cross-Encoder Reranker 활성화
    )

    # 테스트 질문
    question = "삼성전자와 하이닉스 가격 상승 요인을 비교해 봐"

    # 질문 실행
    rag.query(question)


if __name__ == "__main__":
    main()
