from sentence_tokenizer import int_array_rep, unicode_rep
import pickle as pkl


char_pkl = pkl.load(open('../model/char.pkl', 'rb'))
id2char = dict([(char_pkl[char]['id'], char_pkl[char]) for char in char_pkl])
SPACE_ID = char_pkl[32]['id']
USER_ID = char_pkl[b'@user']['id']
URL_ID = char_pkl[b'!url']['id']
lower_upper_map = dict([(char_pkl[c]['id'], char_pkl[c - 32]['id']) for c in range(97, 123)])
upper_lower_map = dict([(lower_upper_map[id], id) for id in lower_upper_map])

def lower(c):
    if c in upper_lower_map:
        return upper_lower_map[c]
    else:
        return c

def extract_char_ngrams(char_array, min_n=3, max_n=10):
    length = len(char_array)
    char_array = [c for c in char_array]
    if 0 in char_array:
        char_array.remove(0)
    char_array.append(SPACE_ID)

    n_gram_count = {}

    cur_idx = 0
    while cur_idx < length:
        start_idx = cur_idx
        while char_array[cur_idx] != SPACE_ID:
            cur_idx += 1
        end_idx = cur_idx
        token_char_array = char_array[start_idx:end_idx]
        for n_gram in extract_char_ngrams_from_token(token_char_array, min_n, max_n):
            if n_gram_count.get(n_gram) is None:
                n_gram_count[n_gram] = 0
            n_gram_count[n_gram] += 1
        cur_idx += 1
    return n_gram_count

def extract_char_ngrams_from_token(token_char_array, min_n, max_n):
    result = set()
    if len(token_char_array) <= min_n:
        result.add(tuple(token_char_array))
        return result
    else:
        for n in range(min_n, max_n + 1):
            for idx in range(max(len(token_char_array) - n, 1)):
                result.add(tuple(token_char_array[idx:idx + n]))
    return result

if __name__ == '__main__':
    s = 'FREE 🔓🔓 BRO @ReesemoneySODMG Shit is FU 😤😤👿 .....👮🏽👮🏽💥💥💥🔫 #ICLR https://dd'
    arr = int_array_rep(s, option='char', debug=True)
    print(arr)
    d = extract_char_ngrams(arr)
    for key in d:
        print(str(key) + ': ' + str(d[key]))




