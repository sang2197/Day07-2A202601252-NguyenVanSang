"""
bench.py — chạy 5 câu hỏi benchmark của nhóm qua chiến lược chunking của MÌNH.

Chỉ đổi dòng chọn `chunker` bên dưới so với các thành viên khác trong nhóm;
mọi thứ còn lại (corpus, 5 câu hỏi, embedder) phải giữ nguyên để so sánh công bằng.
"""
from __future__ import annotations

from ingest import build_knowledge_base
from main import _select_embedder, demo_llm
from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker

DATA_DIR = "data/k4_ecommerce"

# 5 câu hỏi chốt của nhóm (xem report/REPORT_NHOM.md mục 3).
DEFAULT_TOP_K = 3

QUERIES = [
    {
        "id": "Q1",
        "category": "số liệu",
        "query": "Người mua được gửi yêu cầu trả hàng hoặc hoàn tiền trong bao lâu sau khi đơn được giao?",
        "gold_answer": "Người mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng mười lăm (15) ngày dương lịch sau khi trạng thái đơn hàng được cập nhật thành Đã giao hàng.",
        "gold_quote": "Người mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng mười lăm (15) ngày dương lịch sau khi trạng thái đơn hàng được cập nhật thành Đã giao hàng.",
        "expected_doc_ids": ["tiktok-buyer-return-eligibility"],
        "expected_chunk": {
            "heading": "Khi nào người mua có thể yêu cầu trả hàng hoàn tiền",
            "reference_chunk_index": 0,
        },
        "required_terms": ["mười lăm (15) ngày dương lịch", "đã giao hàng"],
        "metadata_filter": {"customer_role": "buyer"},
    },
    {
        "id": "Q2",
        "category": "điều kiện",
        "query": "Điều gì xảy ra nếu khách hàng không nhận đơn COD ba lần trong khoảng thời gian 60 ngày?",
        "gold_answer": "TikTok Shop có quyền tạm thời tắt phương thức thanh toán COD của khách hàng trong sáu mươi (60) ngày nếu họ không nhận đơn hàng COD ba lần trong khoảng thời gian 60 ngày.",
        "gold_quote": "TikTok Shop có quyền tạm thời tắt phương thức thanh toán COD của khách hàng trong sáu mươi (60) ngày nếu họ không nhận đơn hàng COD ba lần trong khoảng thời gian 60 ngày.",
        "expected_doc_ids": ["tiktok-cod-policy"],
        "expected_chunk": {
            "heading": "Tính đủ điều kiện > 3.2 Khách hàng",
            "reference_chunk_index": 7,
        },
        "required_terms": ["sáu mươi (60) ngày", "không nhận đơn hàng COD ba lần"],
    },
    {
        "id": "Q3",
        "category": "quy trình",
        "query": "Quy trình và thời hạn để người bán kháng nghị một hành động thực thi như thế nào?",
        "gold_answer": "Người bán có thể gửi tối đa hai kháng nghị. Kháng nghị đầu tiên phải nộp trong vòng 30 ngày từ thông báo hành động thực thi; kháng nghị thứ hai phải nộp trong vòng mười lăm (15) ngày từ ngày kháng nghị đầu tiên bị từ chối.",
        "gold_quote": "Người bán có thể gửi tối đa hai kháng nghị đối với một hành động thực thi. Trong vòng 30 ngày kể từ ngày nhận được thông báo về hành động thực thi, Người bán phải nộp kháng nghị đầu tiên. Phải nộp kháng nghị thứ hai trong vòng mười lăm (15) ngày kể từ ngày từ chối kháng nghị đầu tiên.",
        "expected_doc_ids": ["tiktok-seller-performance-policy"],
        "expected_chunk": {
            "heading": "8. Kháng nghị > 8.1 Quy trình kháng nghị",
            "reference_chunk_index": 27,
        },
        "required_terms": ["tối đa hai kháng nghị", "trong vòng 30 ngày", "mười lăm (15) ngày"],
        "metadata_filter": {"customer_role": "seller"},
    },
    {
        "id": "Q4",
        "category": "liệt kê",
        "query": "Muốn đăng bán mỹ phẩm thuộc nhóm hạn chế thì người bán cần chuẩn bị và thể hiện những thông tin, tài liệu gì?",
        "gold_answer": "Cần thông báo về sản phẩm mỹ phẩm và hình ảnh nhãn bao bì có tên sản phẩm, chức năng, hướng dẫn sử dụng, thành phần, tên và quốc gia xuất xứ, kích thước/thành phần/khối lượng tịnh, ngày hết hạn và cảnh báo/thận trọng.",
        "gold_quote": "Thông báo về sản phẩm mỹ phẩm; (Các) hình ảnh nhãn bao bì có chứa tất cả thông tin bắt buộc: Tên sản phẩm mỹ phẩm; Chức năng; Hướng dẫn sử dụng; Thành phần; Tên và quốc gia xuất xứ của sản phẩm; Kích thước, thành phần hoặc khối lượng tịnh; Ngày hết hạn; Tuyên bố cảnh báo/thận trọng.",
        "expected_doc_ids": ["tiktok-restricted-products"],
        "expected_chunk": {
            "heading": "Sản phẩm làm đẹp và chăm sóc cá nhân > Tài liệu được yêu cầu",
            "reference_chunk_index": 3,
        },
        "required_terms": ["thông báo về sản phẩm mỹ phẩm", "hình ảnh nhãn bao bì"],
        "metadata_filter": {"customer_role": "seller"},
    },
    {
        "id": "Q5",
        "category": "ngoại lệ",
        "query": "Điểm AHR bị trừ thường được tự động xóa sau bao lâu, và ngoại lệ nào khiến quy tắc này không áp dụng?",
        "gold_answer": "Điểm tình trạng tài khoản (AHR) bị trừ do vi phạm sẽ được tự động xóa sau mỗi 90 ngày trừ khi cửa hàng của Người bán đã bị nền tảng đóng, tức là Điểm tình trạng tài khoản (AHR) = 0.",
        "gold_quote": "Điểm tình trạng tài khoản (AHR) bị trừ do vi phạm sẽ được tự động xóa sau mỗi 90 ngày trừ khi cửa hàng của Người bán đã bị nền tảng đóng, tức là Điểm tình trạng tài khoản (AHR) = 0.",
        "expected_doc_ids": ["tiktok-seller-performance-policy"],
        "expected_chunk": {
            "heading": "2. Trừ điểm tình trạng tài khoản AHR",
            "reference_chunk_index": 4,
        },
        "required_terms": ["tự động xóa sau mỗi 90 ngày", "cửa hàng của người bán đã bị nền tảng đóng"],
        "metadata_filter": {"customer_role": "seller"},
    },
]


def main() -> int:
    # 1. Chọn chunker của riêng bạn — DÒNG DUY NHẤT khác với bạn cùng nhóm.
    chunker = FixedSizeChunker(chunk_size=700)

    embedding_fn = _select_embedder()
    backend = getattr(embedding_fn, "_backend_name", embedding_fn.__class__.__name__)

    print(f"Strategy: {chunker.__class__.__name__} params={vars(chunker)}")
    print(f"Embedding backend: {backend}")
    if backend == "mock embeddings fallback":
        print(
            "Cảnh báo: mock embedding gần-ngẫu-nhiên, không phản ánh chất lượng ngữ nghĩa. "
            "Đặt EMBEDDING_PROVIDER=local để so sánh chunking có ý nghĩa."
        )

    # 2. Nạp cả thư mục corpus.
    store = build_knowledge_base(DATA_DIR, embedding_fn=embedding_fn, chunker=chunker)
    print(f"Đã nạp {store.get_collection_size()} chunk vào EmbeddingStore\n")

    # 3. Chạy 5 câu hỏi, in top-3 và câu trả lời của agent.
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    for item in QUERIES:
        question = item["query"]
        top_k = item.get("top_k", DEFAULT_TOP_K)
        metadata_filter = item.get("metadata_filter")

        print(f"=== {item['id']} ({item['category']}): {question} ===")
        if metadata_filter:
            print(f"(metadata_filter={metadata_filter})")
            results = store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = store.search(question, top_k=top_k)

        for rank, r in enumerate(results, start=1):
            preview = r["content"][:100].replace("\n", " ")
            print(f"  {rank}. score={r['score']:.3f} doc_id={r['metadata'].get('doc_id')} | {preview}...")

        answer = agent.answer(question, top_k=top_k, metadata_filter=metadata_filter)
        print(f"Answer: {answer}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
