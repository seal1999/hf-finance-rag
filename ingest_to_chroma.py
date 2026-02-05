"""
HuggingFace 금융 데이터셋을 로드하고, 청킹하여 ChromaDB에 저장하는 스크립트

데이터셋:
1. nmixx-fin/opensource_korean_finance_datasets (503K rows)
2. nmixx-fin/synthetic_financial_report_korean (20.8K rows)
3. nmixx-fin/synthetic_dart_report_korean (5.08K rows)
"""

from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# .env 파일 로드
load_dotenv()

# 설정
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 1000  # ChromaDB 배치 저장 크기

# 데이터셋 정보
DATASETS = [
    {
        "name": "nmixx-fin/opensource_korean_finance_datasets",
        "text_column": "text",
        "metadata_columns": ["source", "category", "token_count"],
    },
    {
        "name": "nmixx-fin/synthetic_financial_report_korean",
        "text_column": "text",
        "metadata_columns": ["category", "source", "token_count"],
    },
    {
        "name": "nmixx-fin/synthetic_dart_report_korean",
        "text_column": "text",
        "metadata_columns": ["category", "source", "token_count"],
    },
]


def load_hf_dataset(dataset_name: str, split: str = "train", max_rows: Optional[int] = None):
    """HuggingFace 데이터셋 로드"""
    print(f"\n[로드] {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    print(f"  - 로드된 행 수: {len(dataset):,}")
    return dataset


def create_documents(dataset, text_column: str, metadata_columns: list, dataset_name: str) -> list[Document]:
    """데이터셋을 LangChain Document 객체로 변환"""
    documents = []

    for row in tqdm(dataset, desc=f"문서 변환 중"):
        text = row.get(text_column, "")
        if not text or not isinstance(text, str):
            continue

        metadata = {
            "dataset": dataset_name,
        }
        for col in metadata_columns:
            if col in row:
                value = row[col]
                # ChromaDB는 None 값을 허용하지 않음
                if value is not None:
                    metadata[col] = value

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """문서를 청크로 분할"""
    print(f"\n[청킹] chunk_size={chunk_size}, overlap={chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"  - 원본 문서 수: {len(documents):,}")
    print(f"  - 청크 수: {len(chunks):,}")

    return chunks


def create_embeddings():
    """임베딩 모델 생성"""
    print(f"\n[임베딩 모델] {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def store_to_chroma(chunks: list[Document], embeddings, collection_name: str = "finance_docs"):
    """ChromaDB에 저장"""
    print(f"\n[ChromaDB 저장] 컬렉션: {collection_name}, 저장 경로: {CHROMA_PERSIST_DIR}")
    print(f"  - 총 청크 수: {len(chunks):,}")

    # 배치로 나눠서 저장 (메모리 효율)
    vectorstore = None

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="배치 저장 중"):
        batch = chunks[i:i + BATCH_SIZE]

        if vectorstore is None:
            # 첫 배치: 새 컬렉션 생성
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        else:
            # 이후 배치: 기존 컬렉션에 추가
            vectorstore.add_documents(batch)

    print(f"  - 저장 완료!")
    return vectorstore


def main(max_rows_per_dataset: Optional[int] = None):
    """메인 실행 함수"""
    print("=" * 60)
    print("HuggingFace 금융 데이터셋 -> ChromaDB 저장 파이프라인")
    print("=" * 60)

    all_documents = []

    # 1. 데이터셋 로드 및 문서 변환
    for ds_info in DATASETS:
        dataset = load_hf_dataset(ds_info["name"], max_rows=max_rows_per_dataset)
        documents = create_documents(
            dataset,
            ds_info["text_column"],
            ds_info["metadata_columns"],
            ds_info["name"],
        )
        all_documents.extend(documents)
        print(f"  - 변환된 문서 수: {len(documents):,}")

    print(f"\n[총 문서 수] {len(all_documents):,}")

    # 2. 청킹
    chunks = chunk_documents(all_documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. 임베딩 모델 로드
    embeddings = create_embeddings()

    # 4. ChromaDB에 저장
    vectorstore = store_to_chroma(chunks, embeddings)

    # 5. 테스트 쿼리
    print("\n" + "=" * 60)
    print("[테스트 쿼리]")
    test_query = "삼성전자 주가 전망"
    results = vectorstore.similarity_search(test_query, k=3)
    print(f"쿼리: '{test_query}'")
    print(f"결과 수: {len(results)}")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 결과 {i} ---")
        print(f"내용: {doc.page_content[:200]}...")
        print(f"메타데이터: {doc.metadata}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HuggingFace 금융 데이터셋을 ChromaDB에 저장")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="데이터셋당 최대 행 수 (테스트용, 기본값: 전체)",
    )
    args = parser.parse_args()

    main(max_rows_per_dataset=args.max_rows)
