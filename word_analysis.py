# import sys  # for argv etc.
#
#
# try:
#     argv1, argv2, argv3 = sys.argv
# except:
#     print("Input formart: python word_analysis.py word1 word2")
#     exit()
from scipy.sparse import *
def word_analysis(g_word, t_word):

    len_guess = len(g_word)
    len_target = len(t_word)

    char_counts = {}    #counts character number in guess word which are in the target word
    t_flag = {}
    g_flag = {}

    for i in range(len_target): #check the character in target word
        t_flag[t_word[i]] = 1

    tot = 0
    g_char = []
    for i in range(len_guess):
        if g_word[i] in t_flag: #check the guess word character appear in the target word
            tot += 1
            if g_word[i] in char_counts:
                char_counts[g_word[i]] += 1
            else:
                char_counts[g_word[i]] = 1


    tot_len = len(char_counts)
    keys = []

    for key in char_counts:
        keys.append(key)

    """
    Output
    """
    print("{} (".format(tot), end="")
    if tot_len == 1:
        for key in char_counts:
            if char_counts[key] == 1:
                print('the \"{}\"'.format(key), end="")
            else:
                print('the {} \"{}\"s'.format(char_counts[key], key), end="")

    elif tot_len == 2:
        key = keys[0]
        if char_counts[key] == 1:
            print('the \"{}\"'.format(key), end="")
        else:
            print('the {} \"{}\"s'.format(char_counts[key], key), end="")

        key = keys[1]
        print(" and ", end="")
        if char_counts[key] == 1:
            print('the \"{}\"'.format(key), end="")
        else:
            print('the {} \"{}\"s'.format(char_counts[key], key), end="")

    else:
        key = keys[0]
        if char_counts[key] == 1:
            print('the \"{}\"'.format(key), end="")
        else:
            print('the {} \"{}\"s'.format(char_counts[key], key), end="")
        for i in range(tot_len-2):
            key = keys[i+1]
            if char_counts[key] == 1:
                print(', the \"{}\"'.format(key), end="")
            else:
                print(', the {} \"{}\"s'.format(char_counts[key], key), end="")
        key = keys[tot_len-1]
        print(", and ", end="")
        if char_counts[key] == 1:
            print('the \"{}\"'.format(key), end="")
        else:
            print('the {} \"{}\"s'.format(char_counts[key], key), end="")

    print(")")

if __name__ == "__main__":
    import sys

    try:
        argv1, argv2, argv3 = sys.argv

    except:
        print("Input formart: python word_analysis.py word1 word2")
        exit()

    word_analysis(argv2, argv3)


#
# g_word = argv2
# t_word = argv3

# word_analysis(g_word, t_word)
"""
Output
"""
#
# res_str = ""
# res_str = res_str + str(tot) + " ("
#
# if tot_len == 1:
#     for key in char_counts:
#         if char_counts[key] == 1:
#             res_str = res_str + "the " + "\"" + key + "\""
#         else:
#             res_str = res_str + "the " + str(char_counts[key]) + " \"" + key + "\"s"
#
# elif tot_len == 2:
#     key = keys[0]
#     if char_counts[key] == 1:
#         res_str = res_str + "the " + "\"" + key + "\""
#     else:
#         res_str = res_str + "the " + str(char_counts[key]) + " \"" + key + "\"s"
#
#     key = keys[1]
#     res_str = res_str + " and "
#     if char_counts[key] == 1:
#         res_str = res_str + "the " + "\"" + key + "\""
#     else:
#         res_str = res_str + "the " + str(char_counts[key]) + " \"" + key + "\"s"
#
# else:
#     key = keys[0]
#     if char_counts[key] == 1:
#         res_str = res_str + "the " + "\"" + key + "\""
#     else:
#         res_str = res_str + "the " + str(char_counts[key]) + " \"" + key + "\"s"
#     for i in range(tot_len - 2):
#         key = keys[i + 1]
#         if char_counts[key] == 1:
#             res_str = res_str + ", the " + "\"" + key + "\""
#         else:
#             res_str = res_str + ", the " + str(char_counts[key]) + " \"" + key + "\"s"
#     key = keys[tot_len - 1]
#     res_str = res_str + ", and "
#
#     if char_counts[key] == 1:
#         res_str = res_str + "the " + "\"" + key + "\""
#     else:
#         res_str = res_str + "the " + str(char_counts[key]) + " \"" + key + "\"s"
#
# res_str = res_str + ")"
#
# return res_str
