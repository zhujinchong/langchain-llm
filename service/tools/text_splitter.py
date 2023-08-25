from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from service.config import *
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


# 按照指定规则分割，如果过长，再按长度分割。
class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, reg_list=["\n\n(?=\d+)", "\n\n", "\n", " "], max_len=500, **kwargs):
        super().__init__(**kwargs)
        self.reg_list = reg_list
        self.max_len = max_len

    def _split_text(self, reg_list, text, max_len):
        result = []
        # print(reg_list, text, max_len)
        if len(text) <= max_len:
            return [text]
        else:
            reg = reg_list[0] if reg_list and len(reg_list) >= 1 else ""
            sent_list = [x for x in re.split(reg, text) if x]
            new_reg_list = reg_list[1:] if reg_list and len(reg_list) > 1 else None
            for cur_text in sent_list:
                result += self._split_text(new_reg_list, cur_text, max_len)
            return result
            # return self._merger_result(result, max_len)

    def _merger_result(self, str_list, max_len):
        result = []
        i = 0
        while i < len(str_list):
            if i < len(str_list) - 1:
                if len(str_list[i]) + len(str_list[i + 1]) < max_len:
                    result.append(str_list[i] + "\n" + str_list[i + 1])
                    i += 2
                else:
                    result.append(str_list[i])
                    i += 1
            else:
                result.append(str_list[i])
                i += 1
        if len(str_list) == len(result):
            return str_list
        else:
            return self._merger_result(result, max_len)

    def split_text(self, text: str) -> List[str]:
        result = self._split_text(self.reg_list, text, self.max_len)
        for i, x in enumerate(result):
            x = re.sub("\n\n\t", "\n\t", x)
            x = re.sub("\n\n", "\n", x)
            x = re.sub("\s+", " ", x)
            result[i] = x
        result = self._merger_result(result, self.max_len)
        return result


if __name__ == '__main__':
    llm = ChineseTextSplitter(reg_list=["\n\n(?=\d+)", "\n\n", "\n", " "], max_len=10)
    text = "abc de\n\n223fghmn opqr\n\nstsdu\nvwxyz"
    text = llm.split_text(text)
    print(text)
