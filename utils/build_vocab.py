

class Vocab(object):
    def __init__(self, word_list_file) -> None:
        super().__init__()
        try:
            fp = open(word_list_file, "r", encoding="utf-8")
        except:
            raise FileNotFoundError("vocab文件不存在！")
        self.word_list = [line.strip() for line in fp.readlines()]
        fp.close()

        self.word2id = self.build_word2id()
        self.id2word = self.build_id2word()
        self.vocab_size = len(self.word_list)

    def __call__(self, word):
        return self.word2id[word]

    def __len__(self):
        return self.vocab_size

    def build_word2id(self):
        return {each: i for i, each in enumerate(self.word_list)}

    def build_id2word(self):
        return {i: each for i, each in enumerate(self.word_list)}
    
    def get_word(self, _id):
        return self.id2word[_id]
    
    def get_id(self, word):
        return self.word2id[word]
    

    