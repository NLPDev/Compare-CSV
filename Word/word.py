class word:
    def __init__(self):
        import sys

        try:
            argv1, argv2, argv3 = sys.argv

        except:
            print("Input formart: python word_analysis.py word1 word2")
            exit()
        self.guess_word = argv2
        self.target_word = argv3

    def word_analyse(self):
        g_word = self.guess_word
        t_word = self.target_word

        g_word = g_word.upper()
        t_word = t_word.upper()

        len_guess = len(g_word)
        len_target = len(t_word)

        char_counts = {}  # counts character number in guess word which are in the target word
        t_flag = {}

        for i in range(len_target):  # check the character in target word
            t_flag[t_word[i]] = 1

        tot = 0

        for i in range(len_guess):
            if g_word[i] in t_flag:  # check the guess word character appear in the target word
                tot += 1
                if g_word[i] in char_counts:
                    char_counts[g_word[i]] += 1
                else:
                    char_counts[g_word[i]] = 1

        return tot

