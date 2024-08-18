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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN TOÁN") # returns back a datasets.Dataset object

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

# cau hoi dau tien trang 129, 130, 131, 132, 133
# tu giua trang 128 den trang 156
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": """Tiệm cận đứng và tiệm cận ngang của đồ thị hàm số $y = \dfrac{{2{\rm{x}} - 1}}{{x + 1}}$ lần lượt là gì?""",
        "gold": "$x = {\\rm{\\;}} - 1;y = 2$",
        "few_shot": "1",
        },
        {
        "question": "Trong không gian với hệ tọa độ $Oxyz$, cho mặt phẳng \((P):2x - y - 2z + 1 = 0\) và ba điểm\(A(1; - 2;0)\), \(B(1;0; - 1)\) và \(C(0;0; - 2)\). Hỏi có tất cả bao nhiêu mặt cầu có tâm thuộc mặt phẳng $(P)$ và tiếp xúc với ba đường thẳng $AB, AC, BC$?",
        "gold": "$4$ mặt cầu",
        "few_shot": "1",
        },
        {
        "question": "Đồ thị hàm số \(y = 13{x^4} - 3{x^2} - 2020\) cắt trục hoành tại bao nhiêu điểm?",
        "gold": "2 điểm",
        "few_shot": "1",
        },
        {
        "question": """Giao điểm của hai đường thẳng \(d:\left\{ \begin{array}{l}x = - 3 + 2t\\y = - 2 + 3t\\z = 6 + 4t\end{array} \right.\) và \(d':\left\{ \begin{array}{l}x = 5 + t'\\y = - 1 - 4t'\\z = 20 + t'\end{array} \right.\) có tọa độ là gì?""",
        "gold": """\\((3;7;18)\\)""",
        "few_shot": "1",
        },
        {
        "question": """Trong các số phức \({z_1} = - 2i,\,\,{z_2} = 2 - i,\,\,{z_3} = 5i,\,\,{z_4} = 4\) có bao nhiêu số thuần ảo?""",
        "gold": "$2$",
        "few_shot": "1",
        },
    ]
    return five_examples