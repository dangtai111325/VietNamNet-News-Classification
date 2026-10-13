"""Category identifiers are stable filenames; source paths may contain slashes."""
CATEGORY_LABELS = {
    'chinh-tri': 'Chính trị', 'thoi-su': 'Thời sự', 'kinh-doanh': 'Kinh doanh',
    'dan-toc-ton-giao': 'Dân tộc - Tôn giáo', 'the-thao': 'Thể thao',
    'giao-duc': 'Giáo dục', 'the-gioi': 'Thế giới', 'doi-song': 'Đời sống',
    'van-hoa-giai-tri': 'Văn hóa - Giải trí', 'suc-khoe': 'Sức khỏe',
    'cong-nghe': 'Công nghệ', 'oto-xe-may': 'Ô tô - Xe máy', 'du-lich': 'Du lịch',
    'bat-dong-san': 'Bất động sản', 'ban-doc': 'Bạn đọc', 'tuan-viet-nam': 'Tuần Việt Nam',
    'nong-thon-moi': 'Nông thôn mới', 'noi-dung-chuyen-de': 'Nội dung chuyên đề',
    'dan-toc-thieu-so-va-mien-nui': 'Dân tộc thiểu số và miền núi',
    'cong-nghiep-ho-tro': 'Công nghiệp hỗ trợ',
}
CATEGORY_PATHS = {key: key for key in CATEGORY_LABELS}
CATEGORY_PATHS['cong-nghiep-ho-tro'] = 'hop-tac/cong-nghiep-ho-tro'
CANONICAL_CLASSES = sorted(CATEGORY_LABELS.values())
