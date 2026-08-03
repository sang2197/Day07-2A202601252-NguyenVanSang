# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Văn Sáng
**Nhóm:** Mr.Peak VN
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao cho thấy hai văn bản có ý nghĩa hoặc ngữ cảnh gần giống nhau, dù cách diễn đạt có thể khác nhau. Giá trị cosine càng gần 1 thì mức độ tương đồng càng lớn.

**Ví dụ có độ tương tự CAO:**
- Câu A: Tôi muốn đổi trả sản phẩm không đúng mô tả.
- Câu B: Làm thế nào để trả lại hàng khi sản phẩm bị thiếu chi tiết?
- Tại sao tương đồng: Cả hai đều nói về việc trả lại sản phẩm do không đúng mô tả, chỉ khác cách diễn đạt.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Chính sách hoàn tiền khi trả hàng là gì?
- Câu B: Cách đăng ký trở thành người bán?
- Tại sao khác: Hai câu hỏi thuộc hai chủ đề khác nhau, nên ngữ nghĩa không liên quan.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity so sánh hướng của các vector nên phản ánh mức độ giống nhau về ngữ nghĩa, ít bị ảnh hưởng bởi độ lớn của vector. Trong khi đó, khoảng cách Euclid phụ thuộc vào độ lớn của vector nên thường không phản ánh tốt mức độ tương đồng giữa các văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước nhảy (stride) = 500 − 50 = 450. Số chunk = ceil((10000 - 500) / 450) + 1
> Đáp án: 23 chunks

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi tăng overlap từ 50 lên 100, số lượng chunk tăng từ 23 lên 25 vì mỗi chunk mới dịch ít hơn nên cần nhiều chunk hơn để bao phủ toàn bộ tài liệu. Overlap lớn giúp giữ được ngữ cảnh giữa các chunk, giảm nguy cơ mất thông tin ở ranh giới chunk và cải thiện chất lượng truy xuất (retrieval).

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng `re.split(r"(?<=[.!?])\s+|(?<=\.)\n", text)` để tách câu tại khoảng trắng/newline ngay sau `.`, `!`, `?`, dùng lookbehind `(?<=...)` để dấu câu không bị regex nuốt mất. Sau khi `strip()` và loại bỏ chuỗi rỗng, các câu được gộp theo từng nhóm `max_sentences_per_chunk` bằng slicing `sentences[i:i+limit]`. Edge case xử lý: text rỗng trả `[]`, và regex không tách nhầm số thập phân/viết tắt vì chỉ nhận diện `.`/`!`/`?` theo sau bởi khoảng trắng.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> `_split` thử lần lượt các separator theo độ ưu tiên `["\n\n", "\n", ". ", " ", ""]`: nếu separator không xuất hiện trong text thì đệ quy tiếp với `remaining_separators[1:]` (tiến gần điều kiện dừng vì danh sách separator ngắn dần). Base case 1 là `len(text) <= chunk_size` (trả nguyên text), base case 2 là hết separator hoặc separator rỗng (cắt cố định theo `chunk_size`). Khi split được, các phần liền kề được gộp lại tới khi gần chạm `chunk_size`; phần đơn lẻ vẫn còn quá dài thì gọi đệ quy `_split(part, rest)` với separator ưu tiên thấp hơn — đây là điểm mấu chốt để tránh vòng lặp vô hạn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` embed nội dung từng `Document` qua `_make_record` (copy metadata, `setdefault("doc_id", doc.id)`, id record dạng `f"{doc.id}::{index}"` để không trùng), rồi append vào `self._store` (list dict trong RAM). `search` embed câu hỏi **một lần**, tính `_dot(query_vector, record["embedding"])` cho từng record (cosine similarity vì embedding đã chuẩn hóa), sort giảm dần theo score và cắt `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc **trước**, rank **sau**: `search_with_filter` duyệt `self._store`, chỉ giữ record khớp **mọi** cặp key/value trong `metadata_filter`, rồi mới đưa tập đã lọc vào cùng hàm `_search_records` mà `search()` dùng — nhờ vậy khi `metadata_filter=None` hai hàm luôn cho kết quả giống hệt nhau. Nếu lọc sau khi lấy top-k thì có thể còn 0 kết quả dù store vẫn còn tài liệu hợp lệ. `delete_document` so `len(self._store)` trước/sau khi lọc bỏ record có `metadata["doc_id"] == doc_id`, trả `True` nếu size giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `store.search(question, top_k)` (hoặc `search_with_filter` nếu có `metadata_filter`), sau đó ghép các chunk lấy được thành context có đánh số `[1]`, `[2]`... kèm `doc_id` để truy vết nguồn khi debug. Prompt gồm 4 phần rõ ràng: Instruction (chỉ dùng context, nói rõ khi thiếu thông tin) → Context → Question → nhãn `Answer:`. Nếu store rỗng/không có kết quả thì trả thông báo cố định, không gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: F:\Vin\Source\Day07-2A202601252-NguyenVanSang
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 1.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | COD là gì? | Thanh toán khi nhận hàng là gì? | cao (đồng nghĩa: COD = thanh toán khi nhận hàng) | 0.217 | Sai |
| 2 | Người mua có bao nhiêu ngày để yêu cầu hoàn tiền? | Người bán có bao nhiêu ngày để nộp kháng nghị? | thấp (khác đối tượng, khác mục đích dù đều hỏi "bao nhiêu ngày") | 0.830 | Sai |
| 3 | Chính sách hoàn tiền là gì? | Làm sao để đăng ký trở thành người bán? | thấp (hai chủ đề không liên quan) | 0.095 | Đúng |
| 4 | Sản phẩm bị hạn chế là gì? | Những mặt hàng nào cần TikTok Shop phê duyệt trước khi bán? | cao (paraphrase cùng nghĩa) | 0.409 | Một phần (trung bình, không hẳn "cao") |
| 5 | Phí giao dịch của người bán là bao nhiêu? | Phí vận chuyển do khách hàng thanh toán là gì? | thấp (cùng chữ "phí" nhưng khác loại phí, khác đối tượng trả) | 0.721 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 1 và cặp 2 cho kết quả ngược hẳn dự đoán. Cặp 1 ("COD" vs "thanh toán khi nhận hàng") tuy đồng nghĩa 100% trong domain nhưng embedder cho điểm rất thấp (0.217) — mô hình đa ngữ có vẻ không nối được ký hiệu viết tắt tiếng Anh "COD" với cụm giải nghĩa tiếng Việt tương ứng. Ngược lại, cặp 2 (quyền lợi của người mua vs nghĩa vụ của người bán — hai chủ thể và mục đích hoàn toàn khác nhau) lại được chấm rất cao (0.830), chỉ vì hai câu có cấu trúc cú pháp và từ vựng bề mặt giống nhau ("bao nhiêu ngày để..."). Điều này cho thấy embedding hiện tại thiên về **tương đồng từ vựng/cấu trúc câu** hơn là hiểu sâu ngữ nghĩa/thực thể — đây chính xác là lý do câu hỏi Q1 của nhóm (đề bài K4) cố tình dùng chung số "15 ngày" cho cả buyer và seller để buộc phải lọc bằng `metadata_filter` thay vì tin vào retrieval thuần túy.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

> Chiến lược cá nhân (đã thống nhất với nhóm, xem `REPORT_NHOM.md` mục 2): **`FixedSizeChunker(chunk_size=700, overlap=50)`**, chạy qua `bench.py`. Backend nhúng: `EMBEDDING_PROVIDER=local` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), nạp 125 chunk. Đã verify từng chunk top-1 chứa nguyên văn `gold_quote` bằng substring match (không chỉ đọc bằng mắt).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua được gửi yêu cầu trả hàng/hoàn tiền trong bao lâu? | `tiktok-buyer-return-eligibility`, chunk đầu — chứa nguyên văn "mười lăm (15) ngày dương lịch sau khi trạng thái đơn hàng được cập nhật thành Đã giao hàng" | 0.606 | Có (top-1, khớp nguyên văn gold) | Trích đúng "15 ngày dương lịch sau khi Đã giao hàng" |
| 2 | Điều gì xảy ra nếu khách hàng không nhận đơn COD 3 lần/60 ngày? | `tiktok-cod-policy` — chứa nguyên văn "TikTok Shop có quyền tạm thời tắt phương thức thanh toán COD của khách hàng trong sáu mươi (60) ngày nếu họ không nhận đơn hàng COD ba lần..." | 0.683 | Có (top-1, khớp nguyên văn gold) | Trích đúng "tạm tắt COD 60 ngày nếu không nhận 3 lần" |
| 3 | Quy trình và thời hạn kháng nghị của người bán? | `tiktok-seller-performance-policy`, mục "8.1 Quy trình kháng nghị" — chứa nguyên văn "tối đa hai kháng nghị... 30 ngày... mười lăm (15) ngày" | 0.755 | Có (top-1, khớp nguyên văn gold) | Trích đúng quy trình 2 lần kháng nghị + mốc 30/15 ngày |
| 4 | Người bán cần chuẩn bị tài liệu gì khi bán mỹ phẩm hạn chế? | `tiktok-restricted-products`, mục "❗ Tài liệu được yêu cầu" — chunk chứa đủ toàn bộ danh sách (thông báo mỹ phẩm, hình ảnh nhãn bao bì, tên, chức năng, hướng dẫn sử dụng, thành phần, xuất xứ, kích thước, ngày hết hạn, cảnh báo) | 0.657 | Có (top-1, đủ toàn bộ danh sách gold) | Liệt kê đủ các tài liệu/thông tin yêu cầu |
| 5 | Điểm AHR tự động xóa sau bao lâu, ngoại lệ nào? | `tiktok-seller-performance-policy` — chứa nguyên văn "tự động xóa sau mỗi 90 ngày trừ khi cửa hàng của Người bán đã bị nền tảng đóng" | 0.736 | Có (top-1, khớp nguyên văn gold) | Trích đúng "90 ngày, trừ khi cửa hàng bị đóng" |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (cả 5 câu top-1 đều chứa nguyên văn/đầy đủ gold answer — đã verify bằng substring match)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> So sánh `FixedSizeChunker(700)` của mình với `HeadingAwareChunker(700)` của Công: chiến lược của Công giữ breadcrumb heading nên chunk có nhãn mục rõ hơn, nhưng tạo nhiều chunk hơn (187 so với 125) và tốn embedding hơn; chiến lược của mình đơn giản hơn nhưng vẫn đạt top-1 đúng cả 5/5 câu vì corpus TikTok Shop có câu trả lời nằm liền trong 1-2 câu văn, không cần ngữ cảnh xa như heading cha. Bài học: chiến lược phức tạp hơn không nhất thiết cho kết quả tốt hơn nếu tài liệu đã đủ ngắn/rõ ràng.*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 (42/42 test pass) |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 (đã dự đoán, đo thực tế, phản ánh — dù 3/5 dự đoán sai, đó chính là insight của bài) |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 (cả 5/5 câu top-1 khớp nguyên văn gold answer theo rubric 2đ/câu) |
| **Tổng phần cá nhân** | **60 / 60** |
