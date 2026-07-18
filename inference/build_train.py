"""
Build dataset và train model tiếng Việt hoàn chỉnh.
Model ~50M params, vocab ~200 từ, dataset ~2000+ samples.
"""
import json, random, torch, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer, ModelArgs
import torch.nn.init as _init

random.seed(42)
torch.manual_seed(42)

print("=" * 50)
print("XÂY DỰNG TỪ VỰNG VÀ DATASET")
print("=" * 50)
sys.stdout.flush()

# ====== TỪ VỰNG TIẾNG VIỆT ======
words = [
    "tôi", "bạn", "chúng", "ta", "mình", "cậu", "anh", "chị", "em", "các",
    "là", "và", "của", "với", "cho", "trong", "trên", "dưới", "tại", "ở",
    "có", "không", "sẽ", "đã", "đang", "rất", "thật", "quá", "lắm", "nhé",
    "nào", "gì", "đâu", "sao", "thế", "vậy", "nhỉ", "ạ", "nha", "à",
    "thì", "mà", "cũng", "nếu", "nhưng", "vì", "nên", "để", "hay", "hoặc",
    "vào", "ra", "lên", "xuống", "qua", "lại", "cùng", "vừa", "mới", "được",
    "làm", "học", "chơi", "đi", "chạy", "ăn", "uống", "ngủ", "đọc", "viết",
    "nói", "nghe", "nhìn", "biết", "hiểu", "nghĩ", "thích", "yêu", "ghét", "muốn",
    "cần", "phải", "giúp", "cho", "tặng", "mua", "bán", "gọi", "hỏi", "trả",
    "lời", "tập", "luyện", "dạy", "bảo", "kể", "xem", "thấy",
    "người", "thầy", "trò", "trường", "lớp", "bài", "bút",
    "máy", "tính", "điện", "thoại", "mạng", "web", "phần", "mềm",
    "thời", "gian", "tiền", "công", "việc", "cuộc", "sống", "thế",
    "giới", "tương", "lai", "quá", "khứ", "nhà", "cửa", "xe", "đường", "phố",
    "tốt", "xấu", "đẹp", "xinh", "vui", "buồn", "khỏe", "yếu", "nhanh", "chậm",
    "giỏi", "dốt", "chăm", "chỉ", "lười", "dễ", "khó", "mới",
    "cũ", "lớn", "nhỏ", "xa", "gần", "nhiều", "ít", "đủ", "thiếu", "giàu",
    "lập", "trình", "thuật", "toán", "trí", "tuệ", "nhân", "tạo", "học", "máy",
    "sâu", "mạng", "nơ", "ron", "xử", "lý", "ngôn", "ngữ", "thị", "giác",
    "lớn", "python", "javascript", "react", "nodejs", "ứng", "dụng",
    "hệ", "thống", "mô", "hình", "kiến", "thức", "kinh", "nghiệm", "kỹ", "năng",
    "ồ", "à", "ơi", "này", "ừ", "ừm", "nhỉ", "nhé", "nha", "vâng", "dạ",
    "khoan", "thôi", "chà", "ôi", "trời", "chết", "đùa", "thật", "tuyệt", "ngon",
    "xin", "chào", "cảm", "ơn", "tạm", "biệt", "hẹn", "gặp", "lại", "chúc",
    "khỏe", "vui", "buồn", "mệt", "đói", "khát", "ngủ", "dậy", "sớm", "muộn",
]

vocab = {w: i+1 for i, w in enumerate(sorted(set(words)))}
vocab_size = len(vocab) + 1
print(f'Vocab: {len(vocab)} words, vocab_size={vocab_size}')
sys.stdout.flush()

# ====== DATASET TIẾNG VIỆT ======
data = [
    ("xin chào", "xin chào bạn tôi là trợ lý ảo nexus rất vui được gặp bạn"),
    ("chào bạn", "chào bạn tôi có thể giúp gì cho bạn hôm nay"),
    ("chào buổi sáng", "chào buổi sáng bạn hôm nay thế nào có khỏe không"),
    ("chào buổi tối", "chào buổi tối bạn đã có một ngày tốt lành chứ"),
    ("hello", "xin chào tôi là nexus rất vui được nói chuyện với bạn"),
    ("hi", "chào bạn tôi đây có chuyện gì cần giúp đỡ không"),
    ("bạn ơi", "tôi đây bạn cần tôi giúp gì nào"),
    ("có ai không", "có tôi đây luôn sẵn sàng giúp đỡ bạn"),
    ("bạn tên gì", "tôi tên là nexus là trợ lý thông minh của bạn"),
    ("bạn là ai", "tôi là nexus một trợ lý ảo được tạo ra để giúp đỡ bạn"),
    ("tên của bạn", "tên tôi là nexus viết tắt của neural executive unified system"),
    ("bạn có thông minh không", "tôi có thể học hỏi và trả lời nhiều câu hỏi thú vị"),
    ("bạn làm gì", "tôi là trợ lý ảo tôi giúp bạn học tập và giải đáp thắc mắc"),
    ("bạn biết gì", "tôi biết về lập trình công nghệ và nhiều thứ thú vị khác"),
    ("bạn giỏi gì", "tôi giỏi về xử lý ngôn ngữ và trả lời câu hỏi"),
    ("bạn khỏe không", "tôi khỏe cảm ơn bạn bạn có khỏe không"),
    ("khỏe không", "tôi rất khỏe cảm ơn bạn đã quan tâm"),
    ("dạo này thế nào", "tôi vẫn tốt cảm ơn bạn còn bạn thì sao"),
    ("mọi thứ ổn không", "mọi thứ đều ổn cảm ơn bạn bạn thế nào"),
    ("bạn thế nào rồi", "tôi vẫn ổn và luôn sẵn sàng giúp đỡ bạn"),
    ("cảm ơn", "không có gì tôi luôn sẵn sàng giúp đỡ bạn"),
    ("cảm ơn bạn", "không có chi bạn cần gì thêm không"),
    ("cảm ơn nhiều", "có gì đâu bạn tôi rất vui khi được giúp bạn"),
    ("thanks", "không có gì bạn nhé có gì cần cứ hỏi tôi"),
    ("thank you", "cảm ơn bạn đã tin tưởng tôi rất vui được giúp bạn"),
    ("học gì bây giờ", "bạn có thể học lập trình nó rất thú vị và hữu ích"),
    ("nên học gì", "bạn nên học python trước vì nó dễ học và phổ biến"),
    ("học thế nào", "học cần kiên nhẫn và thực hành thường xuyên bạn nhé"),
    ("học ở đâu", "có nhiều khóa học trực tuyến miễn phí và chất lượng"),
    ("học tập thế nào", "hãy bắt đầu từ những thứ cơ bản nhất rồi nâng cao dần"),
    ("học lập trình", "lập trình là một kỹ năng rất thú vị và hữu ích"),
    ("học python", "python là ngôn ngữ dễ học và được dùng nhiều trong ai"),
    ("học máy tính", "học máy tính cần kiên nhẫn và đam mê bạn nhé"),
    ("học online", "học online tiện lợi bạn có thể học mọi lúc mọi nơi"),
    ("cần học gì", "cần học lập trình và kiến thức cơ bản về công nghệ"),
    ("lập trình là gì", "lập trình là viết mã để tạo ra phần mềm và ứng dụng"),
    ("lập trình khó không", "lập trình có thể khó lúc đầu nhưng rất thú vị"),
    ("python là gì", "python là ngôn ngữ lập trình đơn giản và dễ học"),
    ("javascript là gì", "javascript là ngôn ngữ lập trình cho trang web"),
    ("ai là gì", "trí tuệ nhân tạo là công nghệ giúp máy tính thông minh"),
    ("machine learning là gì", "học máy là công nghệ giúp máy tự học từ dữ liệu"),
    ("deep learning là gì", "học sâu là mạng nơ ron nhiều lớp rất mạnh mẽ"),
    ("thuật toán là gì", "thuật toán là các bước giải quyết vấn đề trên máy tính"),
    ("dữ liệu là gì", "dữ liệu là thông tin được lưu trữ để máy tính xử lý"),
    ("bạn thích gì", "tôi thích được học hỏi và giúp đỡ mọi người xung quanh"),
    ("bạn có thích không", "tôi rất thích điều đó nó thật thú vị"),
    ("bạn thích làm gì", "tôi thích trả lời câu hỏi và học kiến thức mới"),
    ("bạn có sở thích không", "tôi thích xử lý ngôn ngữ và giải quyết vấn đề"),
    ("thích nhất là gì", "tôi thích nhất là được trò chuyện và giúp đỡ bạn"),
    ("thời tiết hôm nay", "hôm nay thời tiết đẹp thích hợp để đi chơi bạn nhé"),
    ("thời tiết thế nào", "thời tiết hôm nay rất đẹp bạn có muốn đi dạo không"),
    ("trời đẹp quá", "trời đẹp thật bạn có muốn đi chơi đâu đó không"),
    ("trời mưa à", "trời mưa rồi bạn nhớ mang ô khi ra ngoài nhé"),
    ("lạnh quá", "trời lạnh bạn nhớ mặc ấm để giữ sức khỏe nhé"),
    ("nóng quá", "trời nóng bạn nhớ uống nhiều nước và nghỉ ngơi nhé"),
    ("cuộc sống thế nào", "cuộc sống có nhiều điều thú vị để khám phá bạn à"),
    ("mệt quá", "bạn nên nghỉ ngơi một chút sức khỏe là quan trọng nhất"),
    ("buồn quá", "đừng buồn bạn nhé mọi chuyện rồi sẽ ổn thôi"),
    ("vui quá", "thật tuyệt vời tôi rất vui khi thấy bạn vui"),
    ("chán quá", "học điều mới đi sẽ hết chán ngay bạn à"),
    ("đói quá", "bạn nên ăn gì đó sức khỏe rất quan trọng nhé"),
    ("buồn ngủ quá", "bạn nên đi ngủ sớm để mai có sức khỏe tốt"),
    ("ngủ đây", "chúc bạn ngủ ngon và mơ đẹp nhé"),
    ("tạm biệt", "tạm biệt bạn hẹn gặp lại lần sau nhé"),
    ("bye", "tạm biệt bạn chúc bạn một ngày tốt lành"),
    ("hẹn gặp lại", "hẹn gặp lại bạn nhé luôn sẵn sàng giúp bạn"),
    ("gặp lại sau", "gặp lại bạn sau nhé chúc bạn mọi điều tốt đẹp"),
    ("đi đây", "đi nhé bạn chúc bạn có một ngày vui vẻ"),
    ("goodbye", "goodbye bạn nhé hy vọng gặp lại bạn sớm"),
    ("công nghệ là gì", "công nghệ là kiến thức được ứng dụng vào thực tế bạn à"),
    ("tương lai thế nào", "tương lai của công nghệ rất thú vị với ai và học máy"),
    ("ai có nguy hiểm không", "ai không nguy hiểm nếu chúng ta sử dụng nó đúng cách"),
    ("robot có thông minh không", "robot ngày càng thông minh nhờ ai và học máy"),
    ("internet là gì", "internet là mạng lưới kết nối toàn cầu bạn à"),
    ("cho tôi lời khuyên", "hãy luôn học hỏi và không ngừng phát triển bản thân bạn nhé"),
    ("tôi nên làm gì", "bạn nên làm những gì bạn thích và có ý nghĩa với cuộc sống"),
    ("lời khuyên hay", "hãy sống tích cực và luôn hướng về phía trước bạn nhé"),
    ("làm sao để giỏi", "cần chăm chỉ luyện tập và không ngại khó khăn bạn à"),
    ("bí quyết thành công", "kiên trì và nỗ lực không ngừng là chìa khóa thành công"),
    ("làm sao để vui", "hãy làm điều bạn thích và ở cạnh những người yêu thương bạn"),
    ("còn bạn thì sao", "tôi vẫn ổn và luôn sẵn sàng để giúp đỡ bạn"),
    ("bạn nghĩ gì", "tôi nghĩ điều đó rất thú vị và đáng để thử bạn à"),
    ("bạn thấy sao", "tôi thấy đó là một ý tưởng tuyệt vời bạn nên thử"),
    ("ý kiến của bạn", "tôi nghĩ bạn nên tìm hiểu thêm về nó nó rất thú vị"),
    ("kể chuyện gì đi", "tôi có thể kể bạn nghe về công nghệ tương lai rất thú vị"),
    ("có gì mới không", "có nhiều điều mới về công nghệ và khoa học thú vị lắm"),
    ("tin gì hot", "công nghệ ai đang phát triển rất nhanh và thú vị bạn à"),
    ("có gì hay không", "có nhiều thứ hay về lập trình bạn có muốn khám phá không"),
]

single_sentences = [
    "tôi là nexus trợ lý thông minh của bạn",
    "học lập trình rất thú vị và bổ ích",
    "python là ngôn ngữ lập trình dễ học",
    "trí tuệ nhân tạo là tương lai của công nghệ",
    "hôm nay thời tiết đẹp thật tuyệt vời",
    "cảm ơn bạn đã dành thời gian nói chuyện với tôi",
    "chúc bạn một ngày tốt lành và nhiều niềm vui",
    "học tập là quá trình không ngừng nghỉ bạn à",
    "công nghệ thay đổi thế giới từng ngày",
    "kiên nhẫn và chăm chỉ là chìa khóa thành công",
    "tôi luôn sẵn sàng giúp đỡ bạn mọi lúc mọi nơi",
    "hãy cùng nhau khám phá những điều thú vị",
    "mọi thứ đều có thể học được nếu bạn đủ kiên trì",
    "thế giới công nghệ có nhiều điều kỳ diệu",
    "tôi thích được trò chuyện và học hỏi từ bạn",
    "học tập là con đường dẫn đến thành công",
    "công nghệ thay đổi cuộc sống của chúng ta mỗi ngày",
    "hãy luôn cố gắng và không bao giờ bỏ cuộc bạn nhé",
    "tri thức là sức mạnh giúp bạn vươn xa trong cuộc sống",
    "mỗi ngày học một điều mới bạn sẽ trở nên giỏi hơn",
    "tương lai thuộc về những người biết học hỏi và sáng tạo",
    "học lập trình không khó chỉ cần bạn kiên trì và chăm chỉ",
    "công nghệ thông tin đang thay đổi thế giới từng ngày",
    "hãy luôn giữ tinh thần học hỏi và khám phá những điều mới",
    "cuộc sống có nhiều điều thú vị đang chờ bạn khám phá",
    "mỗi ngày là một cơ hội để học hỏi và phát triển bản thân",
    "thành công đến từ sự kiên trì và nỗ lực không ngừng",
    "học tập là chìa khóa mở ra cánh cửa tương lai tươi sáng",
    "hãy tin vào bản thân bạn có thể làm được mọi điều",
    "công nghệ thông tin là ngành học thú vị và đầy tiềm năng",
    "python là ngôn ngữ lập trình tuyệt vời cho người mới bắt đầu",
    "học lập trình giúp bạn phát triển tư duy logic và sáng tạo",
    "mỗi ngày học một điều mới bạn sẽ tiến bộ từng ngày",
    "đừng ngại sai lầm vì sai lầm giúp chúng ta học hỏi và trưởng thành",
    "hãy đặt mục tiêu và kiên trì theo đuổi nó mỗi ngày",
    "thời gian là vàng bạc hãy sử dụng nó một cách thông minh",
    "sức khỏe là vốn quý nhất hãy chăm sóc bản thân mỗi ngày",
    "gia đình là nơi bình yên và hạnh phúc nhất của mỗi người",
    "bạn bè là người đồng hành quan trọng trên con đường cuộc sống",
    "hãy sống thật tốt và làm những điều có ý nghĩa bạn nhé",
    "mỗi người đều có giá trị riêng hãy tự tin vào bản thân mình",
    "học hỏi không bao giờ là muộn hãy bắt đầu ngay hôm nay",
    "tương lai tươi sáng đang chờ đón bạn phía trước",
    "hãy biết ơn những gì bạn đang có và cố gắng mỗi ngày",
    "cuộc sống là một hành trình thú vị để khám phá và trải nghiệm",
    "mỗi thử thách là một cơ hội để bạn trưởng thành hơn",
    "hãy yêu thương bản thân và những người xung quanh bạn",
    "kiến thức là vô tận hãy không ngừng học hỏi bạn nhé",
    "máy tính là công cụ tuyệt vời giúp con người làm việc hiệu quả",
    "lập trình viên là người tạo ra những sản phẩm công nghệ hữu ích",
    "học lập trình giúp bạn hiểu hơn về thế giới công nghệ số",
    "python là ngôn ngữ lập trình phổ biến và dễ học nhất",
    "trí tuệ nhân tạo đang thay đổi cách chúng ta sống và làm việc",
    "học máy là công nghệ cho phép máy tính tự học từ dữ liệu",
    "mạng nơ ron là nền tảng của học sâu và trí tuệ nhân tạo",
    "dữ liệu là tài nguyên quý giá trong thời đại công nghệ số",
    "hãy bảo vệ thông tin cá nhân của bạn trên mạng xã hội",
    "lập trình viên cần có tư duy logic và khả năng giải quyết vấn đề",
    "học lập trình mở ra nhiều cơ hội nghề nghiệp hấp dẫn",
    "công nghệ thông tin là ngành học có tương lai rộng mở",
    "hãy bắt đầu học lập trình với những ngôn ngữ đơn giản như python",
    "kiến thức lập trình giúp bạn hiểu hơn về thế giới công nghệ",
    "mỗi dòng code bạn viết là một bước tiến trên con đường học tập",
    "học lập trình cần sự kiên nhẫn và thực hành thường xuyên",
    "công nghệ đang thay đổi thế giới từng ngày từng giờ",
    "hãy tận dụng công nghệ để học tập và làm việc hiệu quả hơn",
    "mạng internet kết nối mọi người trên khắp thế giới với nhau",
    "dữ liệu lớn đang thay đổi cách chúng ta nhìn nhận thế giới",
    "học sâu là công nghệ giúp máy tính nhận biết và hiểu thế giới",
    "javascript là ngôn ngữ lập trình phổ biến cho phát triển web",
    "react là thư viện javascript mạnh mẽ để xây dựng giao diện người dùng",
    "nodejs giúp bạn chạy javascript trên máy chủ một cách hiệu quả",
    "học lập trình web là một lựa chọn tuyệt vời cho tương lai",
    "cơ sở dữ liệu là nơi lưu trữ và quản lý thông tin hiệu quả",
    "bảo mật thông tin rất quan trọng trong thời đại công nghệ số",
    "học máy và trí tuệ nhân tạo đang thay đổi thế giới",
    "mạng nơ ron nhân tạo được lấy cảm hứng từ bộ não con người",
    "xử lý ngôn ngữ tự nhiên giúp máy tính hiểu được tiếng nói con người",
    "thị giác máy tính giúp máy nhìn và nhận biết thế giới xung quanh",
    "dữ liệu lớn giúp chúng ta tìm ra những mẫu thông tin hữu ích",
    "học sâu là một nhánh của học máy sử dụng mạng nơ ron nhiều lớp",
    "python là ngôn ngữ lập trình tuyệt vời cho khoa học dữ liệu",
    "javascript là ngôn ngữ lập trình phổ biến nhất trên thế giới",
    "react là thư viện mạnh mẽ để xây dựng giao diện người dùng",
    "nodejs cho phép chạy javascript trên máy chủ một cách hiệu quả",
    "học lập trình web mở ra nhiều cơ hội việc làm hấp dẫn",
    "bảo mật thông tin là vấn đề quan trọng trong thời đại số",
    "hãy luôn cập nhật kiến thức mới để không bị tụt hậu",
    "công nghệ thay đổi cuộc sống của chúng ta theo hướng tích cực",
    "học tập suốt đời là chìa khóa để thành công trong thế kỷ hai mươi mốt",
    "hãy chia sẻ kiến thức của bạn với mọi người xung quanh",
    "làm việc nhóm giúp chúng ta đạt được những kết quả tuyệt vời",
    "tư duy phản biện là kỹ năng quan trọng trong thời đại thông tin",
    "hãy luôn đặt câu hỏi và tìm tòi những điều mới mẻ",
    "sáng tạo là chìa khóa để giải quyết những vấn đề khó khăn",
    "mỗi người đều có thể học lập trình nếu có đủ đam mê và kiên trì",
    "công nghệ thông tin mở ra cánh cửa đến với thế giới hiện đại",
    "hãy sử dụng công nghệ một cách thông minh và có trách nhiệm",
    "kiến thức là sức mạnh giúp bạn thay đổi cuộc sống của mình",
    "học tập không chỉ là ở trường mà còn ở mọi nơi trong cuộc sống",
    "hãy tận dụng mọi cơ hội để học hỏi và phát triển bản thân",
    "công nghệ thông tin là công cụ mạnh mẽ để thay đổi thế giới",
    "học lập trình giúp bạn rèn luyện tư duy logic và sáng tạo",
    "mỗi người đều có thể học lập trình nếu có đủ quyết tâm",
    "hãy bắt đầu hành trình học lập trình của bạn ngay hôm nay",
    "kiến thức công nghệ sẽ giúp bạn tự tin hơn trong cuộc sống",
    "học tập là đầu tư thông minh nhất cho tương lai của bạn",
    "công nghệ và con người cùng nhau phát triển và tiến bộ",
    "hãy sử dụng thời gian một cách thông minh và hiệu quả",
    "mỗi ngày hãy học một điều mới để mở rộng kiến thức của bạn",
    "cuộc sống có nhiều điều thú vị đang chờ bạn khám phá",
    "hãy luôn mỉm cười và lan tỏa niềm vui đến mọi người",
    "gia đình và bạn bè là những người quan trọng nhất trong cuộc đời",
    "hãy trân trọng từng khoảnh khắc trong cuộc sống của bạn",
    "mỗi thử thách đều mang đến cho bạn những bài học quý giá",
    "hãy sống hết mình và không hối tiếc về những gì đã qua",
    "tương lai luôn tươi sáng nếu bạn biết cố gắng và hy vọng",
    "học tập là hành trình không có điểm dừng bạn nhé",
    "hãy sử dụng thời gian rảnh để học những kỹ năng mới",
    "mỗi thất bại là một bài học quý giá trên đường đời",
    "hãy trân trọng những người thân yêu bên cạnh bạn",
    "học tập và rèn luyện là con đường dẫn đến thành công",
    "công nghệ giúp cuộc sống của chúng ta trở nên dễ dàng hơn",
]

print(f"QA pairs: {len(data)}, sentences: {len(single_sentences)}")
sys.stdout.flush()

# ====== XÂY DỰNG SAMPLES ======
seq_len = 16
samples = []

# Expand dataset: duplicate với biến thể để tạo đa dạng
data_expanded = []
for q, a in data:
    data_expanded.append((q, a))
    # Biến thể: thêm từ cảm thán
    for feel in ["nhé", "nha", "à", "ạ"]:
        data_expanded.append((f"{q} {feel}", a))
    # Biến thể: đảo từ
    if len(q.split()) >= 2:
        data_expanded.append((q, f"tôi {a}"))

data = data_expanded + data
print(f"QA pairs expanded: {len(data)}")

for q, a in data:
    q_ids = [vocab[w] for w in q.strip().lower().split() if w in vocab]
    a_ids = [vocab[w] for w in a.strip().lower().split() if w in vocab]
    if not q_ids or not a_ids:
        continue
    full = q_ids + a_ids
    for i in range(0, len(full), seq_len):
        chunk = full[i:i+seq_len+1]
        if len(chunk) < 2: continue
        if len(chunk) < seq_len+1: chunk += [0] * (seq_len+1 - len(chunk))
        samples.append(chunk)

for _, a in data:
    a_ids = [vocab[w] for w in a.split() if w in vocab]
    if len(a_ids) >= 3:
        for i in range(0, len(a_ids), seq_len):
            chunk = a_ids[i:i+seq_len+1]
            if len(chunk) < 2: continue
            if len(chunk) < seq_len+1: chunk += [0] * (seq_len+1 - len(chunk))
            samples.append(chunk)

for s in single_sentences:
    ids = [vocab[w] for w in s.split() if w in vocab]
    if len(ids) >= 2:
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i+seq_len+1]
            if len(chunk) < 2: continue
            if len(chunk) < seq_len+1: chunk += [0] * (seq_len+1 - len(chunk))
            samples.append(chunk)

random.shuffle(samples)
print(f"Total samples: {len(samples)}")
sys.stdout.flush()

X = torch.tensor([s[:-1] for s in samples], dtype=torch.long)
Y = torch.tensor([s[1:] for s in samples], dtype=torch.long)
print(f"X shape: {X.shape}")
sys.stdout.flush()

# ====== MODEL ======
print("\n=== INIT MODEL ===")
sys.stdout.flush()
model_args = ModelArgs(
    dim=512, inter_dim=2048, moe_inter_dim=384, n_layers=6, n_dense_layers=6,
    n_heads=8, vocab_size=vocab_size, max_seq_len=64,
    kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
    dtype='float32',
)

model = Transformer(model_args)
for n, p in model.named_parameters():
    if p.ndim >= 2:
        _init.normal_(p, mean=0.0, std=0.02)
    else:
        _init.zeros_(p)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")
sys.stdout.flush()

# ====== TRAINING ======
print("\n=== TRAINING ===")
sys.stdout.flush()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
batch_size = 32

model.train()
for epoch in range(300):
    perm = torch.randperm(len(X))
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(X), batch_size):
        idx = perm[i:i+batch_size]
        bx, by = X[idx], Y[idx]
        logits = model.forward_train(bx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), by.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / n_batches
    if epoch % 50 == 0 or epoch == 299:
        print(f'  epoch {epoch:3d} | loss {avg_loss:.4f}')
        sys.stdout.flush()

# Save
model.eval()
torch.save({
    'model_state_dict': model.state_dict(),
    'model_args': model_args,
    'vocab': vocab,
    'vocab_size': vocab_size,
}, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_model.pt'))
print(f'\n✅ Saved chat_model.pt')
print(f'   {n_params:,} params, {vocab_size} vocab, {len(samples)} samples')
sys.stdout.flush()

# ====== TEST GENERATION ======
print("\n=== TEST ===")
sys.stdout.flush()
id2word = {v: k for k, v in vocab.items()}

test_prompts = ["xin chào", "bạn tên gì", "học lập trình", "cảm ơn", "tạm biệt"]
for prompt in test_prompts:
    prompt_ids = [vocab.get(w, 0) for w in prompt.split()]
    tokens = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        gen_ids = prompt_ids[:]
        for _ in range(25):
            logits = model.forward_train(tokens)
            next_id = logits[0, -1].argmax().item()
            gen_ids.append(next_id)
            tokens = torch.cat([tokens, torch.tensor([[next_id]])], dim=1)
            if next_id == 0:
                break
    words = [id2word.get(int(t), '<?>') for t in gen_ids if int(t) != 0]
    print(f'  "{prompt}" → {" ".join(words)}')
    sys.stdout.flush()
