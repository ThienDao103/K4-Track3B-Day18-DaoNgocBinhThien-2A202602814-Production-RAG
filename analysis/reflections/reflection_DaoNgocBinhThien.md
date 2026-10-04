# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đào Ngọc Bình Thiện  
**MSSV:** 2A202602814  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối sánh giữa các khái niệm lý thuyết trong bài giảng Production RAG và phần mã nguồn được triển khai trong bài lab:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|:------:|:-------------|--------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng `SentenceTransformer("all-MiniLM-L6-v2")` tính Cosine Similarity giữa các câu liền kề. Với threshold 0.85, văn bản được phân ranh giới tự nhiên theo mạch tư duy thay vì cắt giữa chừng câu như phương pháp Fixed-size / Paragraph chunking truyền thống. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc Parent (2048 chars) và Child (256 chars). Giữ liên kết `parent_id` giúp hệ thống: index và truy xuất ở mức độ fine-grained (Child) nhằm đạt Precision cao, đồng thời có thể trả về ngữ cảnh rộng (Parent) cho LLM để tránh mất thông tin nền. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Dùng regex đa dòng nhận diện Markdown headers (`#`, `##`, `###`), gắn `metadata["section"]`. Giúp bảo toàn tính toàn vẹn của bảng biểu (Markdown tables), danh sách chính sách và mã số phiên bản trong tài liệu nội bộ. |
| **Vietnamese Word Segmentation & BM25** | M2 | `segment_vietnamese()` & `BM25Search.index()` | Sử dụng thư viện `underthesea` tách từ tiếng Việt. Điểm cốt lõi là chuyển đổi dấu gạch nối `_` thành dấu cách `" "` để đồng bộ hóa không gian token giữa câu truy vấn và kho ngữ liệu, khắc phục lỗi BM25 không khớp từ ghép (như "nghỉ_phép" vs "nghỉ phép"). |
| **Dense Search & In-Memory Fallback** | M2 | `DenseSearch.index()` & `search()` | Dùng mô hình đa ngôn ngữ mạnh mẽ `BAAI/bge-m3` sinh embedding 1024 chiều, đẩy vào Qdrant Vector Database qua API chuẩn `query_points()`. Tích hợp fallback `:memory:` tự động khi môi trường chạy local chưa bật Docker container. |
| **Reciprocal Rank Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | Hợp nhất danh sách kết quả xếp hạng từ BM25 (Lexical) và Dense (Semantic) bằng công thức $\sum \frac{1}{k + \text{rank} + 1}$ ($k=60$). RRF trung hòa sự chênh lệch thang đo điểm số giữa hai phương pháp, đưa các văn bản khớp cả từ khóa kỹ thuật lẫn ngữ nghĩa bao quát lên đầu. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` đánh giá tương tác chéo sâu (Full-attention) giữa `(query, document)`. Rút gọn từ top-20 candidate xuống top-3 chất lượng nhất, giải quyết triệt để vấn đề "Lost in the Middle" và giảm đáng kể nhiễu cho LLM. |
| **RAGAS 4 Metrics Evaluation** | M4 | `evaluate_ragas()` | Đánh giá định lượng toàn diện pipeline qua 4 trục: Faithfulness (độ trung thực, chống ảo giác), Answer Relevancy (mức độ bám sát câu hỏi), Context Precision (tỷ lệ chunk liên quan ở thứ hạng cao) và Context Recall (độ phủ thông tin ground truth). |
| **Diagnostic Error Tree** | M4 | `failure_analysis()` | Phân loại lỗi theo cây chẩn đoán: phát hiện chính xác nguyên nhân gốc rễ (Root Cause) như thiếu chunk do chunking cắt ngang bảng hay hallucination do prompt, từ đó đưa ra hướng khắc phục kỹ thuật mục tiêu. |
| **Contextual Prepend & Enrichment** | M5 | `contextual_prepend()` / `_enrich_single_call()` | Bổ sung ngữ cảnh tài liệu gốc trước mỗi chunk con và trích xuất siêu dữ liệu tự động. Đặc biệt, kỹ thuật Combined Single-Call tối ưu hóa chi phí API gọi LLM (1 call duy nhất cho summary, HyQA, context và metadata). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi xung đột thư viện Tokenizers / Transformers
- **Lỗi kỹ thuật gặp phải (Exact error message):**
  ```text
  ImportError: cannot import name 'XLMRobertaTokenizerFast' from 'transformers'
  FlagEmbedding error with transformers >= 5.0
  ```
- **Nguyên nhân gốc rễ:** Thư viện `FlagEmbedding` (chứa `FlagReranker`) chưa hoàn toàn tương thích với các bản cập nhật mới của `transformers` (v5+), gây lỗi nạp tokenizer cho mô hình kiến trúc XLM-RoBERTa.
- **Cách debug & giải quyết:** Chuyển đổi hoàn toàn sang việc sử dụng trực tiếp lớp `CrossEncoder` từ `sentence_transformers` (`from sentence_transformers import CrossEncoder; CrossEncoder("BAAI/bge-reranker-v2-m3")`). Cách này sử dụng backend chuẩn của Hugging Face, tương thích 100% với môi trường Python 3.13 hiện tại và giữ nguyên trọng số của mô hình bge-reranker.

### 2. Sự không tương thích giữa Tokenizer của Underthesea và BM25Okapi
- **Lỗi kỹ thuật gặp phải:**
  Truy vấn tìm kiếm từ khóa chính xác như "nghỉ phép" hoặc "thử việc" qua BM25 ban đầu trả về danh sách rỗng (`BM25Search.search()` score = 0).
- **Nguyên nhân gốc rễ:** Hàm `underthesea.word_tokenize(..., format="text")` tự động nối các từ ghép tiếng Việt bằng ký tự gạch dưới (ví dụ: `"nghỉ_phép"`, `"thử_việc"`). Trong khi đó, `BM25Okapi` phân tách token theo khoảng trắng `split(" ")`. Khi câu query `"nghỉ phép"` được tách thành hai token riêng lẻ `["nghỉ", "phép"]`, BM25 không thể khớp với token duy nhất `"nghỉ_phép"` trong chỉ mục tài liệu.
- **Cách debug & giải quyết:** Bổ sung phương thức `.replace("_", " ")` ngay sau bước gọi tokenizer trong hàm `segment_vietnamese()`. Nhờ đó, cả corpus và truy vấn đều được chuẩn hóa về cùng không gian từ vựng phân cách bằng khoảng trắng, giúp điểm BM25 phản ánh chính xác độ tương đồng từ vựng.

### 3. Thay đổi chuẩn API trong Qdrant Client v2.0+
- **Lỗi kỹ thuật gặp phải:**
  ```text
  AttributeError: 'QdrantClient' object has no attribute 'search'
  ```
- **Nguyên nhân gốc rễ:** Phiên bản `qdrant-client` mới (1.19+ / 2.0+) đã deprecate phương thức `.search()` truyền thống và thay thế hoàn toàn bằng phương thức `.query_points()`.
- **Cách debug & giải quyết:** Tra cứu tài liệu chính thức của Qdrant Client v2, chuyển sang dùng cú pháp:
  ```python
  response = self.client.query_points(collection_name, query=query_vector, limit=top_k)
  ```
  Truy xuất payload và score từ `response.points`, đồng thời cấu hình fallback tự động sang in-memory database (`QdrantClient(":memory:")`) khi Qdrant container không phản hồi.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Tra cứu Quy định Doanh nghiệp (Enterprise Legal & Policy Copilot)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: cắt văn bản theo số lượng ký tự cố định (Fixed-size 500 characters, overlap 50), nhúng trực tiếp qua mô hình OpenAI `text-embedding-3-small`, tìm kiếm thuần Cosine Similarity và đưa top 5 kết quả vào prompt của GPT-4o.
- **Vấn đề / Bottlenecks đang gặp:**
  1. *Cắt vụn điều khoản luật:* Các bảng biểu phân cấp thẩm quyền tài chính và các điều khoản có cấu trúc nhiều cấp (Điều, Khoản, Điểm) thường xuyên bị cắt ngang khiến câu trả lời bị cụt ý hoặc thiếu điều kiện ràng buộc.
  2. *Lexical Mismatch:* Khi người dùng gõ chính xác mã số văn bản, điều luật hoặc từ viết tắt (ví dụ: "NĐ 13/2023", "PCCC", "MFA"), Dense Search đôi khi không nhận diện được và trả về các văn bản không liên quan.
  3. *Hallucination trong câu hỏi đa chặng (Multi-hop):* Với các câu hỏi kết hợp chính sách (ví dụ: hỏi về mức phạt khi vi phạm quy định bảo mật), hệ thống hay nhầm lẫn giữa các phiên bản quy chế cũ và mới.

#### 2. Kế hoạch cải tiến kỹ thuật
1. **Chunking Strategy:**
   - Chuyển sang **Structure-Aware Chunking** kết hợp **Hierarchical Chunking (Parent-Child)**.
   - Định nghĩa quy tắc cắt theo đơn vị văn bản pháp quy: Mỗi Điều luật là một Parent Chunk (kích thước linh hoạt từ 1000 - 2500 ký tự); mỗi Khoản/Điểm cụ thể là một Child Chunk (200 - 400 ký tự). Bảo toàn nguyên khối các bảng ma trận phân quyền.
2. **Search Retrieval:**
   - Triển khai **Hybrid Search (BM25 tiếng Việt + Dense Search BAAI/bge-m3)** kết hợp thuật toán **RRF ($k=60$)**.
   - BM25 đảm bảo bắt chính xác 100% các mã hiệu số văn bản, số tiền, ngày tháng; Dense Search phụ trách hiểu ngữ nghĩa các câu hỏi tự nhiên mang tính diễn giải của người dùng.
3. **Reranking:**
   - Đưa mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3` vào sau bước Hybrid Search để lọc từ top 25 ứng viên xuống top 4 văn bản liên quan nhất trước khi nạp vào LLM.
4. **Evaluation:**
   - Xây dựng bộ test benchmark 50 câu hỏi bao gồm: tra cứu trực tiếp, câu hỏi điều kiện phủ định, câu hỏi đa văn bản và đối chiếu phiên bản hiệu lực.
   - Sử dụng định lượng RAGAS với 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) làm tiêu chí CI/CD định kỳ.
5. **Enrichment & Contextual Prepend:**
   - Áp dụng kỹ thuật Combined Enrichment cho các văn bản quan trọng: tạo tóm tắt ngắn của từng chương/điều và gắn thẻ metadata (`so_hieu`, `ngay_hieu_luc`, `trang_thai_hieu_luc`).
   - Tự động prepend thông tin phiên bản văn bản (ví dụ: *"Văn bản hiện hành năm 2024, thay thế quy chế 2023"*) để ngăn chặn hoàn toàn việc trích dẫn quy định đã hết hiệu lực.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1: Refactor Ingestion & Chunking**
  - Viết parser chuyên dụng cho văn bản pháp lý / quy định nội bộ (bảo toàn cấu trúc Markdown & bảng).
  - Triển khai cấu trúc Parent-Child lưu trữ trên Qdrant và cơ sở dữ liệu quan hệ.
- **Tuần 2: Tích hợp Hybrid Search & Reranking**
  - Cấu hình BM25 tiếng Việt với tokenizer chuẩn `underthesea`.
  - Thiết lập pipeline RRF và tích hợp mô hình `bge-reranker-v2-m3`. Đo lường latency phục vụ thực tế (mục tiêu latency < 350ms).
- **Tuần 3: Enrichment & Metadata Filtering**
  - Chạy batch enrichment tạo metadata và tóm tắt ngữ cảnh cho toàn bộ kho tài liệu doanh nghiệp.
  - Tích hợp bộ lọc siêu dữ liệu theo ngày hiệu lực của văn bản.
- **Tuần 4: Evaluation Benchmark & Tối ưu hóa**
  - Chạy đánh giá toàn diện bằng RAGAS trên bộ test 50 câu hỏi.
  - Chẩn đoán lỗi bằng Diagnostic Error Tree và hoàn thiện hệ thống báo cáo giám sát chất lượng tự động.

