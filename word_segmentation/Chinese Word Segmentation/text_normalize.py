
def q2b(uchar):
    """全角转半角"""
    inside_code = ord(uchar)  # orc() -> 转换成unicode码
    if inside_code == 0x3000:  # 全角空格
        inside_code = 0x0020  # 转为半角空格
    else:  # 半角 = 全角 - 0xfee0
        inside_code -= 0xfee0
    if inside_code < 0xfee20 or inside_code > 0x7e:
        return uchar  # 转完之后不是半角字符返回原来的字符
    return chr(inside_code)  # char() -> 将unicode码转换为字符串


def string_q2b(ustring):
    """字符串全角转半角"""
    return ''.join([q2b(uchar) for uchar in ustring])


if __name__ == "__main__":
    s = '天气不错！是吗？'
    print(string_q2b(s))