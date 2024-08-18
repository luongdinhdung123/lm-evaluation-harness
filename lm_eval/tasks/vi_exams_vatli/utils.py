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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN LÍ") # returns back a datasets.Dataset object

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

# cau hoi dau tien trang 120, 121, 122, 123, 124
# tu trang 120 den giua trang 128
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": """Một con lắc đơn có chiều dài \(l\). Kéo con lắc lệch khỏi vị trí cân bằng một góc \({\alpha _0} = {45^0}\) rồi thả nhẹ cho dao động. Khi đi qua vị trí cân bằng dây treo bị vướng vào một chiếc đinh nằm trên đường thẳng đứng cách điểm treo con lắc một đoạn \(\dfrac{{2l}}{5}\). Tính biên độ góc α<sub>0</sub><sup>’</sup> mà con lắc đạt được sau khi vướng đinh?""",
        "gold": "59,2<sup>0</sup>",
        "few_shot": "1",
        },
        {
        "question": "Một sợi dây đàn hồi dài 1,2m được treo lơ lửng trên một cần rung. Cần rung có thể dao động theo phương ngang với tần số thay đổi được từ 50Hz đến 75Hz. Tốc độ truyền sóng trên dây là 6m/s. Xem đầu nối với cần rung là nút sóng khi có sóng dừng trên dây. Trong quá trình thay đổi tần số rung, số lần tạo ra sóng dừng trên dây là gì?",
        "gold": "10",
        "few_shot": "1",
        },
        {
        "question": "Cho 3 pin giống nhau, mỗi pin có suất điện động 3V. Ghép 3 pin nối tiếp với nhau thì suất điện động của bộ pin là gì?",
        "gold": "9 V",
        "few_shot": "1",
        },
        {
        "question": "Một vật có khối lượng 0,2 kg được ném thẳng đứng từ mặt đất với vận tốc 10 m/s. Lấy g = 10 m/s<sup>2</sup>. Bỏ qua sức cản. Khi vật đi được quãng đường 8 m thì động năng của vật có giá trị bằng gì?",
        "gold": "6 J",
        "few_shot": "1",
        },
        {
        "question": "Bóng đèn có điện trở \(9\Omega \) và hiệu điện thế qua nó là \(24V\) thì nó sáng bình thường. Tính công suất định mức của bóng đèn?",
        "gold": "64W",
        "few_shot": "1",
        },
    ]
    return five_examples