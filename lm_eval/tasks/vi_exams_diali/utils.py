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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN ĐỊA") # returns back a datasets.Dataset object

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

#tu trang 1 - 23: Dia ly
#cau hoi dau tien trang 1, 2, 3, 4, 5
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": "Biển Đông mang lại nguồn nhiệt ẩm dồi dào, lượng mưa lớn cho nước ta chủ yếu do biển Đông có đặc điểm gì?",
        "gold": "Nóng, ẩm và chịu ảnh hưởng của gió mùa",
        "few_shot": "1",
        },
        {
        "question": "Thuận lợi chủ yếu để phát triển công nghiệp ở Đồng bằng sông Hồng là gì?",
        "gold": "có nhiều lao động kỹ thuật, cơ sở hạ tầng khá tốt",
        "few_shot": "1",
        },
        {
        "question": "Duyên hải Nam Trung Bộ có nhiều điều kiện thuận lợi để xây dựng các cảng nước sâu, trong đó chủ yếu là do ?",
        "gold": "bờ biển có nhiều vũng vịnh, thềm lục địa sâu, ít bị sa bồi",
        "few_shot": "1",
        },
        {
        "question": "Ý nghĩa chủ yếu của phát triển du lịch ở Trung du và miền núi Bắc Bộ là gì?",
        "gold": "phát huy tiềm năng, tăng thu nhập vùng, nâng cao đời sống nhân dân,",
        "few_shot": "1",
        },
        {
        "question": "Năng suất lúa nước ta tăng nhanh là do nguyên nhân chính nào?",
        "gold": "Áp dụng các biện pháp thâm canh",
        "few_shot": "1",
        },
    ]
    return five_examples