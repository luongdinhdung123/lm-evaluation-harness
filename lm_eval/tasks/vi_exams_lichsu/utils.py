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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN SỬ") # returns back a datasets.Dataset object

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

#tu trang 24 - giua trang 77: Lich su
#cau hoi dau tien trang 23, 24, 25, 26, 27
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": "Chiến thắng nào của quân dân miền Nam trong cuộc chiến đấu chống chiến lược “chiến tranh cục bộ” (1965-1968) được coi là Ấp Bắc đối với quân Mĩ?",
        "gold": "Chiến thắng Vạn Tường (1965)",
        "few_shot": "1",
        },
        {
        "question": "Nhiệm vụ của miền Bắc Việt Nam sau hiệp định Giơnevơ năm 1954 về Đông Dương là gì?",
        "gold": "Khôi phục kinh tế, hàn gắn vết thương chiến tranh và tiến hành cách mạng xã hội chủ nghĩa",
        "few_shot": "1",
        },
        {
        "question": "Đại hội đại biểu nào của Đảng được coi là “Đại hội Kháng chiến thắng lợi”?",
        "gold": "Đại hội đại biểu lần thứ II (1951)",
        "few_shot": "1",
        },
        {
        "question": "Trong cuộc kháng chiến chống thực dân Pháp (1945 – 1954), thắng lợi quân sự nào của quân dân Việt Nam đã làm xoay chuyển cục diện chiến tranh, giáng đòn quyết định vào ý chí xâm lược của thực dân Pháp?",
        "gold": "Chiến thắng lịch sử Điện Biên Phủ",
        "few_shot": "1",
        },
        {
        "question": "Nhiệm vụ hàng đầu của cách mạng Việt Nam sau khi cách mạng tháng Tám thành công là gì?",
        "gold": "Xây dựng và bảo vệ chính quyền cách mạng",
        "few_shot": "1",
        },
    ]
    return five_examples