from src.const import TRAIN_TRANS

def trans_chars(fname):
    with open(fname, encoding='utf-8') as f:
        raw_lines = f.readlines()
    trans_lines = [x.strip().split(',')[0] for x in raw_lines]
    trans = ''.join(trans_lines)
    trans_set = list(set(trans))
    trans_set.sort()
    return trans_set

chars = trans_chars(TRAIN_TRANS)
char_map = {}
char_map["'"] = 0
char_map["<SPACE>"] = 1
index = 2
for c in chars:
    if c != " ":
        char_map[c] = index
        index += 1
index_map = {v+1: k for k, v in char_map.items()}