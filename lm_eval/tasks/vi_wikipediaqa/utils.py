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

    return dataset.map(_helper) # returns back a datasets.Dataset object

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
    
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": "Thủy điện Vĩnh Sơn ở tỉnh nào?",
        "gold": "Bình Định",
        "few_shot": "1",
        },
        {
        "question": "Nghị định thư Kyoto là nghị định về vấn đề gì?",
        "gold": "Biến đổi khí hậu",
        "few_shot": "1",
        },
        {
        "question": "Đầu tháng 9/1858, sự kiện nào đánh dấu sự bắt đầu của cuộc chiến tranh xâm lược của thực dân Pháp vào Việt Nam?",
        "gold": "Nổ súng tiến công Đà Nẵng",
        "few_shot": "1",
        },
        {
        "question": "Đầu tháng 9/1858, sự kiện nào đánh dấu sự bắt đầu của cuộc chiến tranh xâm lược của thực dân Pháp vào Việt Nam?",
        "gold": "Pháp",
        "few_shot": "1",
        },
        {
        "question": "Lịch Công nguyên được phát triển từ năm nào trước công nguyên?",
        "gold": "46",
        "few_shot": "1",
        },
    ]
    return five_examples