from preprocess import *

s = 'FREE 🔓🔓 BRO @ReesemoneySODMG Shit is FU 😤😤👿 .....👮🏽👮🏽💥💥💥🔫 #ICLR'

binary_str = preprocess(s, debug=True)
print(to_char_array(binary_str))
