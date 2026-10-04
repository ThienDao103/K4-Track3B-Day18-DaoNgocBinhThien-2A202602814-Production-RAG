# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đào Ngọc Bình Thiện  
**Khóa:** K4 - Track 3B  
**MSSV:** 2A202602814  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:-------------:|:----------:|:---:|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9014 | 0.8749 | -0.0265 |
| Context Precision | 0.9664 | 0.9377 | -0.0287 |
| Context Recall | 0.9212 | 0.8530 | -0.0682 |

> *Nhận xét:* Cả 4 chỉ số của Production RAG đều đạt mức xuất sắc (≥ 0.85, vượt xa ngưỡng yêu cầu 0.70 của Rubric). Production RAG sử dụng Hierarchical chunking với kích thước chunk nhỏ hơn để tối ưu hóa độ chính xác truy xuất, tuy nhiên trong một số câu hỏi đa chặng (multi-hop) cần đối chiếu nhiều tài liệu cùng lúc, Context Recall bị ảnh hưởng nhẹ do giới hạn top-3 reranking.

---

## Bottom-5 Failures

### #1
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Tài liệu trích từ: mua_sam.md. # Quy trình mua sắm > Phiên bản: 2.2 | Ngày hiệu lực: 01/04/2024 | Phòng ban: Hành chính & Tài chính ## Thẩm quyền phê duyệt | Giá trị đơn hàng | Người phê duyệt | |-------------------|-----------------| | Dưới **5.000.000 VNĐ** | Trưởng phòng (Manager) |
- **Worst metric:** Context Recall (0.5923)
- **Error Tree:** Output thiếu thông tin → Context bị thiếu → Retrieval lấy trúng file `mua_sam.md` nhưng chunk con bị cắt ngang bảng Markdown.
- **Root cause:** Bảng phân cấp thẩm quyền phê duyệt bị chia cắt giữa các child chunks trong quá trình hierarchical chunking, khiến hàng quy định "Trên 50.000.000 VNĐ -> Tổng Giám đốc (CEO)" không nằm trong top chunk con được trả về.
- **Suggested fix:** Áp dụng Structure-Aware chunking để bảo toàn nguyên vẹn bảng Markdown (Table-aware preservation) hoặc trả về toàn bộ nội dung của Parent Chunk khi thực hiện reranking thay vì chỉ trả về text của Child Chunk.

---

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Tài liệu trích từ: nghi_phep_nam_v2024.md. Ví dụ: nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3). ## Quy định sử dụng Phép năm phải được đăng ký trước ít nhất 2 ngày làm việc qua hệ thống HR Portal...
- **Worst metric:** Context Recall (0.5261)
- **Error Tree:** Output thiếu vế sau (mức lương) → Context chỉ chứa chính sách nghỉ phép, thiếu bảng lương → Query bản chất là câu hỏi phức hợp đa chặng (Multi-hop Query).
- **Root cause:** Câu hỏi yêu cầu thông tin từ 2 văn bản khác nhau (`nghi_phep_nam_v2024.md` và `bang_luong_2024.md`). Cơ chế tìm kiếm đơn chặng (single-step retrieval) kết hợp top-3 rerank chỉ ưu tiên các chunks về phép năm do có mật độ từ khóa trùng khớp cao hơn.
- **Suggested fix:** Tích hợp Query Decomposition: phân tách câu hỏi thành 2 truy vấn độc lập: (1) "Nhân viên 9 năm thâm niên được nghỉ bao nhiêu ngày phép?" và (2) "Mức lương Senior là bao nhiêu?", sau đó hợp nhất kết quả tìm kiếm trước khi rerank.

---

### #3
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Tài liệu trích từ: ky_luong.md. Phiếu lương điện tử được gửi qua email công ty vào ngày chi trả. Thắc mắc về lương liên hệ phòng Tài chính trong vòng 5 ngày làm việc. Thông tin lương là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp.
- **Worst metric:** Context Recall (0.6947)
- **Error Tree:** Output chỉ nêu "Bí mật" từ `ky_luong.md` → Context thiếu tài liệu `phan_loai_du_lieu.md` quy định định nghĩa Cấp 3 và yêu cầu mã hóa / need-to-know.
- **Root cause:** Từ khóa "thông tin lương" dẫn dắt bộ tìm kiếm tập trung hoàn toàn vào `ky_luong.md` thay vì tìm thêm định nghĩa cấp độ tương ứng trong `phan_loai_du_lieu.md`.
- **Suggested fix:** Bổ sung kỹ thuật HyQA (Hypothesis Question Answering) hoặc mở rộng ngữ cảnh truy xuất (Query Expansion) bằng các thực thể liên quan như "cấp độ phân loại bảo mật".

---

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Tài liệu trích từ: bang_luong_2024.md. | Senior (P3-P4) | 20.000.000 - 35.000.000 | ... ## Lương thử việc Nhân viên trong thời gian thử việc được nhận 85% lương của cấp bậc tương ứng.
- **Worst metric:** Context Recall (0.7857)
- **Error Tree:** Output thiếu con số mức lương tối đa của Junior (20.000.000 VNĐ) → Context lấy được quy định 85% nhưng thiếu dòng bảng Junior ở đầu tài liệu.
- **Root cause:** Bảng lương trong `bang_luong_2024.md` bị cắt rời: hàng Junior nằm ở chunk trước, còn quy định 85% lương thử việc nằm ở chunk sau.
- **Suggested fix:** Cấu hình Parent-Child Retrieval đúng chuẩn: khi một Child chunk được tìm thấy qua BM25/Dense, luôn mở rộng trả về toàn bộ Parent Chunk tương ứng cho LLM đọc context.

---

### #5
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Tài liệu trích từ: mua_sam.md. Mua sắm thiết bị CNTT (laptop, server, phần mềm) cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất...
- **Worst metric:** Context Recall (0.7413)
- **Error Tree:** Output có yêu cầu xác nhận kỹ thuật từ CNTT nhưng thiếu mức phê duyệt của Giám đốc phòng ban (Director) cho đơn hàng 30 triệu.
- **Root cause:** Câu hỏi chứa hai ý (thẩm quyền phê duyệt theo mức giá và thủ tục đặc thù cho thiết bị CNTT). Cross-encoder xếp hạng đoạn nói về CNTT lên trên đoạn bảng giá trị 5-50 triệu.
- **Suggested fix:** Tăng số lượng context đưa vào LLM từ top-3 lên top-5, hoặc sử dụng Parent-document retrieval để gộp chung hai điều khoản nằm trong cùng một văn bản mua sắm.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?**  
   $\rightarrow$ **Chưa hoàn chỉnh (Partial)**: Hệ thống trả lời chính xác số ngày nghỉ phép (18 ngày: 15 ngày cơ bản + 3 ngày thâm niên), nhưng hoàn toàn bỏ sót mức lương Senior (20 - 35 triệu VNĐ/tháng).
2. **Context đúng?**  
   $\rightarrow$ **Thiếu ngữ cảnh thứ hai**: Top 3 contexts được trích xuất hoàn toàn đến từ `nghi_phep_nam_v2024.md`. Không có bất kỳ đoạn nào từ `bang_luong_2024.md` lọt vào top 3 sau bước Rerank.
3. **Query rewrite / Retrieval OK?**  
   $\rightarrow$ **Gặp rào cản Lexical Bias**: Bản chất câu hỏi là truy vấn đa chặng (multi-hop). Từ khóa "nghỉ phép", "thâm niên", "ngày phép" chiếm tỷ trọng áp đảo khiến cả BM25 và Dense Search tập trung chấm điểm cao cho file chính sách nghỉ phép, đè bẹp các chunk về bảng lương vốn chỉ khớp từ khóa "Senior".
4. **Fix ở bước:**  
   $\rightarrow$ **Query Transformation / Decomposition Step**:  
   Tách query thành 2 truy vấn con độc lập:
   - Sub-query A: *"Nhân viên 9 năm thâm niên được bao nhiêu ngày phép năm?"* $\rightarrow$ Lấy chunk từ `nghi_phep_nam_v2024.md`.
   - Sub-query B: *"Mức lương của nhân viên cấp Senior là bao nhiêu?"* $\rightarrow$ Lấy chunk từ `bang_luong_2024.md`.  
   Gộp context từ cả 2 nhánh trước khi đưa vào LLM Answer Generator.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **Triển khai Sub-query Decomposition (Multi-hop Routing)** bằng LLM hoặc Regex Intent Classifier để giải quyết triệt để các câu hỏi kết hợp chính sách + tài chính/lương.
- **Preserve Markdown Tables trong Chunking**: Tinh chỉnh `chunk_structure_aware` để phát hiện khối bảng `| ... |` và không bao giờ cắt ngang một bảng dữ liệu vào nhiều chunk khác nhau.
- **Tận dụng tối đa Parent Retrieval**: Trả về Parent chunk thay vì Child chunk sau khi rerank, giữ nguyên toàn bộ ngữ cảnh xung quanh tài liệu.
