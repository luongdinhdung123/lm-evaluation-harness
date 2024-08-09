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

    return dataset.map(_helper).filter(lambda x: x["metadata"]["subject"] == "MÔN HOÁ") # returns back a datasets.Dataset object

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

#tu giua trang 108 den giua trang 108
#cau hoi dau tien trang 109, 110, 111, 112, 113
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "question": """Điện phân 200 ml dung dịch CuCl<sub>2</sub> sau một thời gian người ta thu được 1,12 lít khí (đktc) ở anot. Ngâm đinh sắt sạch trong dung dịch còn lại sau khi điện phân, phản ứng xong thấy khối lượng đinh sắt tăng 1,2 gam. Nồng độ mol ban đầu của dung dịch CuCl<sub>2</sub> là gì?""",
        "gold": "1,0M",
        "few_shot": "1",
        },
        {
        "question": "Đốt cháy hoàn toàn 0,15 mol etyl axeta rồi dẫn sản phẩm cháy vào bình đựng dung dịch Ca(OH)<sub>2</sub> dư thu được m gam kết tủa. Giá trị của m là gì?",
        "gold": "45",
        "few_shot": "1",
        },
        {
        "question": "Cho 35,6 gam alanin tác dụng với lượng dư dung dịch HCl thu được m gam muối khan. Giá trị của m là gì?",
        "gold": "50,20",
        "few_shot": "1",
        },
        {
        "question": "Cho 100 ml dung dịch NaOH x mol/l (x &gt; 0,4M) vào dung dịch chứa 0,02 mol MgCl<sub>2</sub> và 0,02 mol AlCl<sub>3</sub>. Lọc lấy kết tủa và nung đến khối lượng không đổi được m gam chất rắn. Để m nhỏ nhất thì giá trị của x tối thiểu là gì?",
        "gold": "1,2",
        "few_shot": "1",
        },
        {
        "question": "Cho các chất: Al, Al<sub>2</sub>O<sub>3</sub>, Al<sub>2</sub>(SO<sub>4</sub>)<sub>3</sub>, Zn(OH)<sub>2</sub>, NaHS, KHSO<sub>3</sub>, (NH<sub>4</sub>)<sub>2</sub>CO<sub>3</sub>. Số chất phản ứng được với cả dung dịch HCl và dung dịch NaOH là gì?",
        "gold": "6",
        "few_shot": "1",
        },
    ]
    return five_examples