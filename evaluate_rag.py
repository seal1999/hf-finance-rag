"""
RAG 시스템 평가 프로그램
1. ChromaDB에서 문서 샘플링하여 합성 테스트셋 생성 (LLM 기반)
2. RAG 시스템으로 답변 생성
3. LLM-as-Judge로 평가 (Faithfulness, Relevancy, Correctness)
"""

import json
import pandas as pd
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from hybrid_rag_query import AdvancedHybridRAG

# 설정
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "finance_docs"
NUM_TEST_QUESTIONS = 10


def load_sample_documents(num_docs: int = 30):
    """ChromaDB에서 샘플 문서 로드"""
    print(f"[1/4] ChromaDB에서 샘플 문서 로드 중...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # 다양한 쿼리로 문서 샘플링
    sample_queries = [
        "삼성전자 주가",
        "반도체 시장",
        "코스피 지수",
        "외국인 투자",
        "환율 영향",
        "금리 인상",
        "실적 발표",
        "배당금",
        "IPO 상장",
        "기업 인수",
    ]

    all_docs = []
    seen_contents = set()

    for query in sample_queries:
        docs = vectorstore.similarity_search(query, k=5)
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)
                if len(all_docs) >= num_docs:
                    break
        if len(all_docs) >= num_docs:
            break

    print(f"  - 로드된 문서 수: {len(all_docs)}")
    return all_docs


def generate_test_questions(docs, llm, num_questions: int = 10):
    """LLM을 사용하여 문서 기반 테스트 질문/답변 생성 (개별 생성 방식)"""
    print(f"\n[2/4] 테스트 질문 {num_questions}개 생성 중...")

    qa_pairs = []

    # 문서별로 질문 생성
    for i, doc in enumerate(docs[:num_questions]):
        doc_content = doc.page_content[:1000]

        prompt = f"""다음 금융 문서를 읽고 질문 1개와 정답 1개를 만들어주세요.

문서:
{doc_content}

지침:
- 문서 내용을 기반으로 답변할 수 있는 질문
- 질문은 구체적이고 명확하게
- 정답은 1-2문장으로 간결하게

다음 형식으로만 출력하세요:
질문: [질문 내용]
정답: [정답 내용]"""

        try:
            response = llm.invoke(prompt)

            # 파싱
            lines = response.strip().split("\n")
            question = ""
            answer = ""

            for line in lines:
                line = line.strip()
                if line.startswith("질문:") or line.startswith("Question:"):
                    question = line.split(":", 1)[1].strip()
                elif line.startswith("정답:") or line.startswith("Answer:"):
                    answer = line.split(":", 1)[1].strip()

            if question and answer:
                qa_pairs.append({"question": question, "answer": answer})
                print(f"  [{i+1}/{num_questions}] ✓ {question[:40]}...")
            else:
                print(f"  [{i+1}/{num_questions}] ✗ 파싱 실패")

        except Exception as e:
            print(f"  [{i+1}/{num_questions}] ✗ 오류: {e}")

        if len(qa_pairs) >= num_questions:
            break

    print(f"  - 생성된 질문 수: {len(qa_pairs)}")
    return qa_pairs


def evaluate_answer(question, answer, reference, context, llm):
    """LLM-as-Judge로 답변 평가"""
    prompt = f"""다음 RAG 시스템의 답변을 평가해주세요.

## 질문
{question}

## 정답 (참조)
{reference}

## RAG 시스템 답변
{answer}

## 검색된 문서 (Context)
{context[:1500]}...

## 평가 기준 (각 1-5점)
1. **faithfulness**: 답변이 검색된 문서(Context)에 충실한가? (할루시네이션 없는가?)
2. **relevancy**: 답변이 질문의 의도에 적절히 부합하는가?
3. **correctness**: 답변이 정답(참조)과 일치하는가?

## 출력 형식 (JSON만 출력)
{{"faithfulness": 점수, "relevancy": 점수, "correctness": 점수, "comment": "한줄평"}}

JSON:"""

    response = llm.invoke(prompt)

    try:
        result = response.strip()
        # JSON 객체 찾기
        start_idx = result.find("{")
        end_idx = result.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            result = result[start_idx:end_idx]
        return json.loads(result)
    except:
        return {"faithfulness": 0, "relevancy": 0, "correctness": 0, "comment": "파싱 실패"}


def generate_markdown_report(results, avg_scores, timestamp):
    """마크다운 형식의 평가 보고서 생성"""
    overall_avg = sum(avg_scores.values()) / 3

    report = f"""# RAG 평가 결과 보고서

## 📊 평가 결과 요약

| 메트릭 | 평균 | 해석 |
|--------|------|------|
| Faithfulness | {avg_scores['faithfulness']:.2f} | 문서 충실도 |
| Relevancy | {avg_scores['relevancy']:.2f} | 질문 관련성 |
| Correctness | {avg_scores['correctness']:.2f} | 정답 일치도 |
| **전체 평균** | **{overall_avg:.2f}** | |

### 점수 가이드

- **4.0 이상**: 우수 (Production Ready)
- **3.0–4.0**: 양호 (개선 여지 있음)
- **3.0 미만**: 개선 필요

---

## 📋 개별 평가 결과

"""

    for i, r in enumerate(results, 1):
        report += f"""### [{i}] {r['question']}

- **정답**: {r['reference']}
- **RAG 답변**: {r['rag_answer']}
- **점수**: F={r['faithfulness']}, R={r['relevancy']}, C={r['correctness']}
- **평가**: {r['comment'] if r['comment'] else '(평가 코멘트 없음)'}

---

"""

    report += """## 메트릭 설명

- **F (Faithfulness)**: RAG 답변이 검색된 문서에 얼마나 충실한지 (환각 여부)
- **R (Relevancy)**: 답변이 질문과 얼마나 관련 있는지
- **C (Correctness)**: 답변이 참조 정답과 얼마나 일치하는지

*상세 수치 결과는 `evaluation_results_{timestamp}.csv`에 저장됨.*
"""

    return report.replace("{timestamp}", timestamp)


def run_evaluation():
    """전체 평가 파이프라인 실행"""
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("🧪 RAG 시스템 평가 시작")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. LLM 초기화
    print("\n[0/4] LLM 초기화 중...")
    llm = Ollama(model="llama3.2", temperature=0.3)

    # 2. 샘플 문서 로드
    docs = load_sample_documents(num_docs=30)

    # 3. 테스트 질문 생성
    qa_pairs = generate_test_questions(docs, llm, num_questions=NUM_TEST_QUESTIONS)
    if not qa_pairs:
        print("❌ 테스트 질문 생성 실패")
        return

    # DataFrame으로 변환
    test_df = pd.DataFrame(qa_pairs)
    test_df.columns = ["question", "reference"]

    print("\n📝 생성된 테스트 질문:")
    print("-" * 40)
    for i, row in test_df.iterrows():
        print(f"  {i+1}. {row['question'][:60]}...")
    print("-" * 40)

    # 4. RAG 시스템 초기화
    print("\n[3/4] RAG 시스템 초기화 중...")
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

    # 5. 평가 실행
    print(f"\n[4/4] RAG 평가 진행 중... ({len(test_df)}개 질문)")
    print("-" * 40)

    results = []
    scores = {"faithfulness": [], "relevancy": [], "correctness": []}

    for i, row in test_df.iterrows():
        question = row["question"]
        reference = row["reference"]

        print(f"\n  [{i+1}/{len(test_df)}] 평가 중: {question[:40]}...")

        # RAG로 답변 생성
        transformed = rag.transform_query(question, verbose=False)
        documents = rag.hybrid_search(question, transformed, verbose=False)
        answer = rag.generate_answer(question, documents)
        context = " ".join([doc.page_content for doc in documents])

        # LLM-as-Judge 평가
        eval_result = evaluate_answer(question, answer, reference, context, llm)

        results.append({
            "question": question,
            "reference": reference,
            "rag_answer": answer[:300] + "..." if len(answer) > 300 else answer,
            "faithfulness": eval_result.get("faithfulness", 0),
            "relevancy": eval_result.get("relevancy", 0),
            "correctness": eval_result.get("correctness", 0),
            "comment": eval_result.get("comment", ""),
        })

        # 점수 수집
        for key in scores:
            scores[key].append(eval_result.get(key, 0))

        print(f"       → F:{eval_result.get('faithfulness', 0)} R:{eval_result.get('relevancy', 0)} C:{eval_result.get('correctness', 0)}")
        print(f"       → {eval_result.get('comment', '')[:50]}")

    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("📊 평가 결과 요약")
    print("=" * 60)

    avg_scores = {key: sum(vals) / len(vals) if vals else 0 for key, vals in scores.items()}

    print(f"""
┌────────────────────┬─────────┬─────────────────────┐
│ 메트릭             │ 평균    │ 해석                │
├────────────────────┼─────────┼─────────────────────┤
│ Faithfulness       │ {avg_scores['faithfulness']:.2f}    │ 문서 충실도         │
│ Relevancy          │ {avg_scores['relevancy']:.2f}    │ 질문 관련성         │
│ Correctness        │ {avg_scores['correctness']:.2f}    │ 정답 일치도         │
├────────────────────┼─────────┼─────────────────────┤
│ 전체 평균          │ {sum(avg_scores.values())/3:.2f}    │                     │
└────────────────────┴─────────┴─────────────────────┘

📈 점수 가이드:
  - 4.0 이상: 우수 (Production Ready)
  - 3.0-4.0: 양호 (개선 여지 있음)
  - 3.0 미만: 개선 필요
""")

    # 7. 상세 결과 저장 (타임스탬프 포함)
    csv_filename = f"evaluation_results_{timestamp}.csv"
    md_filename = f"evaluation_report_{timestamp}.md"

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"💾 상세 결과가 '{csv_filename}'에 저장되었습니다.")

    # 마크다운 보고서 저장
    markdown_report = generate_markdown_report(results, avg_scores, timestamp)
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"📝 평가 보고서가 '{md_filename}'에 저장되었습니다.")

    # 8. 개별 결과 출력
    print("\n" + "=" * 60)
    print("📋 개별 평가 결과")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"""
[{i}] Q: {r['question'][:50]}...
    정답: {r['reference'][:50]}...
    RAG: {r['rag_answer'][:50]}...
    점수: F={r['faithfulness']} R={r['relevancy']} C={r['correctness']}
    평가: {r['comment']}
""")

    print("=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    run_evaluation()
