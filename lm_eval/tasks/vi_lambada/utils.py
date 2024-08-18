import datasets

def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
      if getattr(doc, "few_shot", None) is not None:
          doc["few_shot"] = True
      return doc

    return dataset.map(_helper) # returns back a datasets.Dataset object

def _doc_to_text(doc):
    return doc["context"]
    
def _doc_to_target(doc):
    return " " + doc["target_word"]

def _should_or_not_decontamination():
    return True

def _doc_to_decontamination_query(doc):
    return doc["text"]

#trong tap valid: cot1: 1, 2; cot2: 1, 2; cot3: 1
def list_fewshot_samples() -> list[dict]:
    five_examples = [
        {
        "context": "Tại UEFA Nations League 2020-21, do UEFA sửa đổi thể thức nên Đức không bị xuống hạng, Đức nằm cùng bảng A4 với Tây Ban Nha, Thuỵ Sĩ & Ukraina, Đức hoà Tây Ban Nha 1-1 tại sân nhà, hoà Thuỵ Sĩ cùng tỷ số tại Basel, thắng Ukraina 2-1 tại Kiev, họ để Thuỵ Sĩ cầm hoà 3-3, thắng Ukraina 3-1 tại sân nhà sau đó thua sốc Tây Ban Nha 0-6 tại Seville. Theo UEFA, đây là trận thua thảm hoạ lịch sử đối với bóng đá Đức. Lần gần nhất họ thua cách biệt tới 5 bàn trong một trận đấu là ở World Cup 1954 (thua Hungary 3-8 ở vòng bảng). Lùi xa hơn thì vào năm 1931, Đức thua Áo 0-6 trong một trận giao hữu. Thế nên trận thua 0-6 là trận thua đậm nhất lịch sử của đội tuyển Đức trong một trận đấu quốc tế thuộc FIFA. Thất bại được xem là đậm nhất lịch sử của Đức đến nay được ghi nhận là trận thua 0-9 trước Anh vào năm 1909, ở giải quốc tế Đức lần thứ 4. Nhưng khi đó bóng đá vẫn là nghiệp dư. Đây là lần đầu tiên trong lịch sử đội tuyển Đức thua tới 6 bàn không gỡ. Cả trận, họ hứng chịu 23 cú dứt điểm và 10 trong số đó hướng trúng khung thành. Trận thua này khiến Đức dừng bước tại",
        "target_word": "Nations League",
        "few_shot": "1",
        },
        {
        "context": "Ngày 17 tháng 9 năm 1939, sức kháng cự của Ba Lan bị quân Đức bẻ gãy, hy vọng cuối cùng của Ba Lan là rút lui và tái tập hợp dọc theo đầu cầu Romania. Tuy nhiên, kế hoạch này trở nên lỗi thời chỉ trong một đêm, khi hơn 800 ngàn quân Liên Xô tiến công với hai Phương diện quân Belarussia và Ukrainia, đánh vào khu vực Kresy ở phía đông Ba Lan. Về mặt ngoại giao, Liên Xô tuyên bố họ hành động là để bảo vệ người Ukraina và Belarusia sống ở miền đông Ba Lan trước bối cảnh thất bại của Ba Lan đã rõ ràng. Một lý do thực dụng hơn là những vùng đất phía Đông này vốn do Ba Lan chiếm của Nga trong Chiến tranh Nga-Ba Lan (1919-1921), Liên Xô muốn nhân cơ hội Ba Lan sắp bị Đức đánh bại để thu hồi lại những vùng đất này mà không cần phải đổ nhiều",
        "target_word": "máu",
        "few_shot": "1",
        },
        {
        "context": "Ngày 31 tháng 10 năm 1920: lập lại tỉnh Đồng Nai Thượng, tỉnh lỵ đặt tại Di Linh, cử O'Connell tiếp tục cai trị. Năm 1928, thời công sứ Cunhac, Pháp chuyển tỉnh lỵ tỉnh Đồng Nai Thượng về Đà Lạt. Năm 1941, khi lập lại tỉnh Lâm Viên, tỉnh lỵ đặt tại Đà Lạt thì tỉnh lỵ tỉnh Đồng Nai Thượng chuyển về",
        "target_word": "Di Linh",
        "few_shot": "1",
        },
        {
        "context": """Ushkowitz sinh ra ở Hàn Quốc; cô được nhận làm con nuôi và lớn lên ở East Meadow, New York. Ushkowitz theo Công giáo, cô đi học tại Parkway Elementary School và Holy Trinity Diocesan High School, một trường công giáo ở Hicksville, Long Island nổi tiếng với khoa sân khấu. Khi còn học trung học, Ushkowitz tham gia một số vở nhạc kịch, trong số đó có "Những người khốn khổ". Các vai diễn khác của cô bao gồm Penny trong "Honk!", Inez trong "The Baker's Wife", Cô bé quàng khăn đỏ trong "Into the Woods" và Romaine Patterson trong "The Laramie Project". Cô tốt nghiệp trung học vào năm 2004 và theo học trường đại học Marymount Manhattan, nơi cô một lần nữa đóng vai Cô bé quàng khăn đỏ trong "Into the Woods". Cô tốt nghiệp đại học vào năm 2007 với bằng cử nhân nghệ thuật sân khấu chuyên ngành biểu diễn và""",
        "target_word": "nhạc kịch",
        "few_shot": "1",
        },
        {
        "context": "Philippe là con trai thứ hai của Vua Philippe IV của Pháp và Nữ hoàng Joan I của Navarre. Ông được ban cho một thái ấp, Quận Poitiers, trong khi anh trai của ông, Louis X, thừa kế ngai vàng của Pháp và Navarrese. Khi Louis qua đời vào năm 1316, ông để lại một cô con gái và một người vợ đang mang thai, Clementia của Hungary. Philippe Cao kều đã giành được quyền nhiếp chính thành công. Nữ hoàng Clementia hạ sinh một bé trai, người được xưng vương là John I, nhưng vị vua sơ sinh chỉ sống được năm",
        "target_word": "ngày",
        "few_shot": "1",
        },
    ]
    return five_examples