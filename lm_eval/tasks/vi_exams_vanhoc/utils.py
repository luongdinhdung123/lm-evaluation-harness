import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      # modifies the contents of a single
      # document in our dataset.
      doc["question"] = "Question: " + doc["question"]
      doc["choices"] = doc["choices"]["text"]
      doc["gold"] = ["A", "B", "C", "D"].index(doc["answerKey"].strip())
      if getattr(doc, "few_shot", None) is not None:
          doc["few_shot"] = True
      return doc

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN VĂN") # returns back a datasets.Dataset object

def _doc_to_text(doc):
    return doc["question"]

def _doc_to_choice(doc):
    return doc["choices"]
    
def _doc_to_target(doc):
    return doc["gold"]

def _should_or_not_decontamination():
    return True

def _doc_to_decontamination_query(doc):
    return doc["question"]

# cau hoi dau tien trang 157, 158, 159, 160, 161
# tu trang 157 den 192
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": """Đọc câu chuyện sau và trả lời câu hỏi: NEWTON ĐÃ LÀM GÌ KHI NGHỈ HỌC VÌ ĐẠI DỊCH?""",
        "gold": "Tự sự",
        "few_shot": "1",
        },
        {
        "question": """Đọc văn bản sau và trả lời câu hỏi: "Hỡi sông Hồng tiếng hát bốn nghìn năm Tổ quốc bao giờ đẹp thế này chăng? - Chưa đâu! Và ngay cả trong những ngày đẹp nhất Khi Nguyễn Trãi làm thơ và đánh giặc Nguyễn Du viết Kiều, đất nước hoá thành văn Khi Nguyễn Huệ cưỡi voi vào cửa Bắc Hưng Đạo diệt quân Nguyên trên sóng Bạch Đằng... Những ngày tôi sống đây là ngày đẹp hơn tất cả Dù mai sau đời muôn vạn lần hơn! Trái cây rơi vào áo người ngắm quả Đường nhân loại đi qua bóng lá xanh rờn Mặt trời đến mỗi ngày như khách lạ Gặp mỗi mặt người đều muốn ghé môi hôn... ...Không ai có thể ngủ yên trong đời chật Buổi thuỷ triều vẫy gọi những vầng trăng Mỗi gié lúa đều muốn thêm nhiều hạt Gỗ trăm cây đều muốn hoá nên trầm Mỗi chú bé đều nằm mơ ngựa sắt Mỗi con sông đều muốn hoá Bạch Đằng...." -1965- (Trích Tổ Quốc bao giờ đẹp thế này chăng? - Chế Lan Viên - NXB Văn học, 2002) Biện pháp nghệ thuật được sử dụng trong câu thơ: Tổ quốc bao giờ đẹp thế này chăng?""",
        "gold": "Câu hỏi tu từ",
        "few_shot": "1",
        },
        {
        "question": "Lưu biệt khi xuất dương của tác giả nào?",
        "gold": "Phan Bội Châu",
        "few_shot": "1",
        },
        {
        "question": "Văn học Việt Nam từ đầu thế kỉ XX đến Cách mạng tháng Tám 1945 phát triển dưới chế độ xã hội nào ?",
        "gold": "Thực dân, nửa phong kiến",
        "few_shot": "1",
        },
        {
        "question": "Tác phẩm Lửa thiêng của Huy Cận thuộc thể loại nào?",
        "gold": "Thơ",
        "few_shot": "1",
        },
    ]
    return five_examples