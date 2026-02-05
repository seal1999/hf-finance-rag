"""
RAG 시스템 평가 프로그램
1. 고정된 테스트셋 사용 (일관된 비교를 위해)
2. RAG 시스템으로 답변 생성
3. LLM-as-Judge로 평가 (Faithfulness, Relevancy, Correctness)
"""

import json
import pandas as pd
from datetime import datetime
from langchain_community.llms import Ollama
from hybrid_rag_query import AdvancedHybridRAG

# 고정된 테스트셋 (일관된 평가를 위해)
FIXED_TEST_SET = [
    {
        "question": "지능형 반도체 팹리스 기업을 찾은 최기영 장관이 어떤 지침을 받았는지 알려주세요.",
        "reference": "최기영 장관은 문서 내용을 기반으로 지침을 받았다고 알려져 있지 않습니다."
    },
    {
        "question": "엑손 모빌 코퍼레이션의 자회사로 사용되는 이름이 무엇인가요?",
        "reference": "엑슨 모빌 코퍼레이션."
    },
    {
        "question": "윌리엄스 컴퍼니즈 주식회사가 최근 확장한 목적은 무엇인가?",
        "reference": "윌리엄스 컴퍼니즈 주식회사는 최근 확장을 통해 비즈니스 활동을 확대하고, 새로운 시장에 진출하여 성장할 수 있는 기회를 찾고 있습니다."
    },
    {
        "question": "경영진이 확인서를 발급하였을 때, 이란에 대한 정보가 포함되어 있는지 여부는 무엇일까요?",
        "reference": "확인서에는 대표이사 확인의 내용만 포함되어 있을 수 있습니다."
    },
    {
        "question": "반도체 시장의 성장을 어떻게 설명할 수 있나요?",
        "reference": "반도체 시장은 2020년 이후 전 세계적으로 빠르게 성장하고 있습니다. 이는 반도체 기술의 발전과 모바일, 컴퓨터, 자동차 등 다양한 산업에서 반도체의 중요성에 대한 인식 증가로 인해 발생했습니다."
    },
    {
        "question": "주식 시장의 상승폭이 어떤 방향으로 이동했는지 설명해주세요.",
        "reference": "주식 시장의 상승폭은 줄어들었습니다."
    },
    {
        "question": "반도체 시장의 두 가지 주요 세그먼트는 무엇입니까?",
        "reference": "로직 반도체와 메모리 반도체로, 로직 반도체는 마이크로프로세서, 센서 및 기타 컴퓨팅 장치에 사용되며, 메모리 반도체는 DRAM, SRAM 및 플래시 메모리와 같은 다양한 유형의 메모리 칩을 포함합니다."
    },
    {
        "question": "반도체 패키징 소재 시장이 성장하는 이유는 무엇인가?",
        "reference": "반도체의 집적화 및 소형화가 진행됨에 따라 반도체 패키징 소재 시장이 성장하고 있습니다."
    },
    {
        "question": "월스트리트의 개념은 무엇입니까?",
        "reference": "월스트리트는 세계에서 가장 큰 금융 시장입니다."
    },
    {
        "question": "AbbVie의 주가 강세는 어떤 요인에 의해 인상되나요?",
        "reference": "AbbVie의 주가 강세는 AbbVie의 새로운 약물 개발과 의료 분야의 성장 potential에 의해 인상되며, 이로 인해 투자자들의 신뢰를 얻고 있습니다."
    },
]


def evaluate_answer(question, answer, reference, context, llm):
    """LLM-as-Judge로 답변 평가"""
    prompt = f"""RAG 시스템의 답변을 평가하세요.

질문: {question}
정답: {reference}
RAG답변: {answer[:500]}
문서: {context[:1000]}

평가 기준 (1-5점):
- faithfulness: 답변이 문서에 충실한가?
- relevancy: 답변이 질문에 적절한가?
- correctness: 답변이 정답과 일치하는가?

JSON 형식으로만 출력:
{{"faithfulness": 숫자, "relevancy": 숫자, "correctness": 숫자, "comment": "평가"}}"""

    response = llm.invoke(prompt)

    try:
        result = response.strip()
        # JSON 객체 찾기
        start_idx = result.find("{")
        end_idx = result.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            result = result[start_idx:end_idx]
        parsed = json.loads(result)
        # 점수 검증 및 기본값
        for key in ["faithfulness", "relevancy", "correctness"]:
            if key not in parsed or not isinstance(parsed[key], (int, float)):
                parsed[key] = 3  # 파싱 실패 시 중간값
        return parsed
    except:
        return {"faithfulness": 3, "relevancy": 3, "correctness": 3, "comment": "파싱 실패 - 중간값 적용"}


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

    report += f"""## 메트릭 설명

- **F (Faithfulness)**: RAG 답변이 검색된 문서에 얼마나 충실한지 (환각 여부)
- **R (Relevancy)**: 답변이 질문과 얼마나 관련 있는지
- **C (Correctness)**: 답변이 참조 정답과 얼마나 일치하는지

*상세 수치 결과는 `evaluation_results_{timestamp}.csv`에 저장됨.*
"""

    return report


def run_evaluation():
    """전체 평가 파이프라인 실행"""
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("🧪 RAG 시스템 평가 시작 (고정 테스트셋)")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. LLM 초기화
    print("\n[0/3] LLM 초기화 중...")
    llm = Ollama(model="llama3.2", temperature=0.3)

    # 2. 고정 테스트셋 사용
    test_df = pd.DataFrame(FIXED_TEST_SET)

    print("\n📝 테스트 질문 (고정):")
    print("-" * 40)
    for i, row in test_df.iterrows():
        print(f"  {i+1}. {row['question'][:60]}...")
    print("-" * 40)

    # 3. RAG 시스템 초기화
    print("\n[1/3] RAG 시스템 초기화 중...")
    rag = AdvancedHybridRAG(
        top_k=25,            # 더 많은 후보 검색
        rerank_top_n=7,      # 더 많은 문서 사용
        bm25_weight=0.5,     # BM25 가중치 증가 (키워드 매칭 강화)
        vector_weight=0.5,
        use_rewrite=True,
        use_decomposition=False,
        use_hyde=False,
        use_reranker=True,
    )

    # 4. 평가 실행
    print(f"\n[2/3] RAG 평가 진행 중... ({len(test_df)}개 질문)")
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
            "rag_answer": answer[:500] + "..." if len(answer) > 500 else answer,
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

    # 5. 결과 요약
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

    # 6. 상세 결과 저장 (타임스탬프 포함)
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

    # 7. 개별 결과 출력
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
