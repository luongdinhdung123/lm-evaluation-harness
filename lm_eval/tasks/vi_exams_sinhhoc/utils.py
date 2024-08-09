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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN SINH") # returns back a datasets.Dataset object

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

#tu giua trang 77 den giua trang 108
#cau hoi dau tien trang 78, 79, 80, 81, 82
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": """Một quần thể có tỉ lệ kiểu gen <a id="_Hlk75512231"></a>0,2AA: 0,5Aa: 0,3aa. Tần số alen A của quần thể là gì?""",
        "gold": "0,45",
        "few_shot": "1",
        },
        {
        "question": "Nguyên nhân nào khiến cách ly địa lý trở thành một nhân tố vô cùng quan trọng trong quá trình tiến hóa của sinh vật?",
        "gold": "Vì cách li địa lí duy trì sự khác biệt về vốn gen giữa các quần thể",
        "few_shot": "1",
        },
        {
        "question": "Một gen của vi khuẩn tiến hành phiên mã đã cần môi trường nội bào cung cấp 900U; 1200G; 1500A; 900X. Biết phân tử mARN này có tổng số 1500 nucleotit. Số phân tử mARN tạo ra là gì?",
        "gold": "3",
        "few_shot": "1",
        },
        {
        "question": "Theo Đacuyn, biến dị cá thể muốn di truyền lại cho các thế hệ sau thì cần trải qua?",
        "gold": "Sự sinh sản",
        "few_shot": "1",
        },
        {
        "question": "Một gen của Vi khuẩn dài 510 (nm), mạch 1 có A1: T1: G1: X1= 1:2:3:4. Gen phiên mã tạo ra một mARN có nucleotit loại A là 300. Số nucleotit loại G môi trường cung cấp cho quá trình phiên mã là gì?",
        "gold": "600",
        "few_shot": "1",
        },
    ]
    return five_examples