"""
금융 RAG 챗봇 - Gradio UI (간소화 버전)
"""

import gradio as gr
from hybrid_rag_query import AdvancedHybridRAG

# 전역 RAG 객체 (lazy loading)
rag = None

def init_rag():
    """RAG 시스템 초기화 (최초 질문 시 1회만)"""
    global rag
    if rag is None:
        print("🚀 RAG 시스템 초기화 중...")
        rag = AdvancedHybridRAG(
            top_k=15,
            rerank_top_n=5,
            bm25_weight=0.3,
            vector_weight=0.7,
            use_rewrite=True,
            use_decomposition=False,  # 속도를 위해 비활성화
            use_hyde=False,           # 속도를 위해 비활성화
            use_reranker=True,
        )
        print("✅ 초기화 완료!")
    return rag


def chat(message, history):
    """채팅 응답 생성"""
    if not message.strip():
        return "질문을 입력해주세요."

    # RAG 초기화 (lazy loading)
    rag_system = init_rag()

    # 쿼리 변환
    transformed = rag_system.transform_query(message, verbose=False)

    # 하이브리드 검색 + Reranking
    documents = rag_system.hybrid_search(message, transformed, verbose=False)

    # 답변 생성
    answer = rag_system.generate_answer(message, documents)

    # 출처 추가
    sources = "\n\n---\n\n**📚 참고 문서:**\n"
    for i, doc in enumerate(documents[:3], 1):
        meta = doc.metadata
        source_text = f"- **출처 {i}**: "
        if "category" in meta:
            source_text += f"[{meta['category']}] "
        source_text += f"{doc.page_content[:150]}...\n"
        sources += source_text

    return answer + sources


# Gradio 인터페이스
demo = gr.ChatInterface(
    fn=chat,
    title="💰 금융 RAG 챗봇",
    description="한국 금융 데이터 기반 질의응답 시스템 (첫 질문 시 초기화에 시간이 걸립니다)",
    examples=[
        "삼성전자와 하이닉스 가격 상승 요인을 비교해줘",
        "반도체 시장 전망은 어때?",
        "외국인 투자자들의 최근 매매 동향은?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
