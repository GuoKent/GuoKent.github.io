---
title: Tokenizer
date: 2024-12-08 00:00:00
tags:
- NLP
- LLM 
- 大模型
categories:
- 算法杂记
alias:
- deeplearning/tokenizer/
---

## 正向/逆向最大匹配法
正向最大匹配法（Forward Maximum Matching, FMM）：从左到右扫描句子，每次取最长的匹配词。
逆向最大匹配法（Backward Maximum Matching, BMM）：从右到左扫描句子，每次取最长的匹配词。
- 优点：实现简单，高效。在词典较完善时效果较好。
- 缺点：对歧义的处理能力弱。不能有效处理未登录词（OOV，Out-of-Vocabulary）
  > 歧义处理能力弱，举例：**研究生命科学**
    正向匹配：["研究生", "生命", "科学"]
    逆向匹配：["研究", "生命", "科学"]

## BPE 分词法
假设我们手头有一堆文档 $D = d_1, d_2, …$
1. 把每个文档 $d$ 变成单词列表，比如可以简单用空格分词
2. 统计每个单词 $w$ 在所有文档 $D$ 中的出现频率，并计算初始字符集 `alphabet` 作为一开始的 Vocab（包括后面的 `</w>`），字符集的意思就是所有文档 $D$ 中不同的字符集合
3. 先将每个单词划分为一个个 utf-8 字符，称为一个划分，比如 `highest -> h, i, g, h, e, s, t`
4. 然后，在每个单词的划分最后面加上 `</w>`，那么现在 `highest -> h, i, g, h, e, s, t, </w>`
5. 重复下面步骤直到满足两个条件中的任意一个：1）Vocab 达到上限。2）达到最大迭代次数
    1. 找到**最经常一起出现的 pair**，并记录这个合并规则，放在 merge vocab 里面，同时把合并之后的结果放到 Vocab 里面
    2. 更新所有单词的划分，假设我们发现 `(h, i)` 最经常一起出现，那么 `hi` 就会被添加到 Vocab 里面，同时修改划分方式为：`highest -> hi, g, h, e, s, t, </w>`

### 训练阶段
统计词频，每一个都是 1。先把每个单词变成一个个 utf-8 字符然后加上 `</w>`
```python
{
    "highest": ["h", "i", "g", "h", "e", "s", "t", "</w>"],
    "higher": ["h", "i", "g", "h", "e", "r", "</w>"],
    "lower": ["l", "o", "w", "e", "r", "</w>"],
    "lowest": ["l", "o", "w", "e", "s", "t", "</w>"],
    "cooler": ["c", "o", "o", "l", "e", "r", "</w>"],
    "collest": ["c", "o", "o", "l", "e", "s", "t", "</w>"],
}
```

找出最高频率的相邻合并词，可以看到 `(e, s)` 总共出现了 3 次，是最多次的，将 `es` 添加到 Vocab 里面，然后重新划分。注意这里 `(e, r)` 其实也有一样的出现频率，所以选 `(e, r)` 合并也是可以的
> 然后将`(e,s): "es"`合并规则加入 merge vocab

```python
{
    "highest": ["h", "i", "g", "h", "es", "t", "</w>"],
    "higher": ["h", "i", "g", "h", "e", "r", "</w>"],
    "lower": ["l", "o", "w", "e", "r", "</w>"],
    "lowest": ["l", "o", "w", "es", "t", "</w>"],
    "cooler": ["c", "o", "o", "l", "e", "r", "</w>"],
    "collest": ["c", "o", "o", "l", "es", "t", "</w>"],
}
```
接下来发现最多的是 `(es, t)`，更新划分
> 然后将`(es,t):"est"`合并规则加入 merge vocab，每次合并都需要加入，后面一样

```python
{
    "highest": ["h", "i", "g", "h", "est", "</w>"],
    "higher": ["h", "i", "g", "h", "e", "r", "</w>"],
    "lower": ["l", "o", "w", "e", "r", "</w>"],
    "lowest": ["l", "o", "w", "est", "</w>"],
    "cooler": ["c", "o", "o", "l", "e", "r", "</w>"],
    "collest": ["c", "o", "o", "l", "est", "</w>"],
}
```

接下来发现最多的是 `(est, </w>)`，更新划分

.... 以此类推，最终得到 merge vocab
```python
{('e', 's'): 'es', ('es', 't'): 'est', ('est', '</w>'): 'est</w>', ('e', 'r'): 'er', ('er', '</w>'): 'er</w>', ('h', 'i'): 'hi', ('hi', 'g'): 'hig', ('hig', 'h'): 'high', ('l', 'o'): 'lo', ('lo', 'w'): 'low', ('c', 'o'): 'co', ('co', 'o'): 'coo', ('coo', 'l'): 'cool', ('high', 'est</w>'): 'highest</w>', ('high', 'er</w>'): 'higher</w>', ('low', 'er</w>'): 'lower</w>', ('low', 'est</w>'): 'lowest</w>', ('cool', 'er</w>'): 'cooler</w>', ('cool', 'est</w>'): 'coolest</w>'}
```

### 推理阶段
先将句子划分为单个字符，然后根据 merge vocab 来进行合并。可用哈希表则减小时间复杂度。

### 代码
```python
from collections import defaultdict, Counter

class BPE:
    def __init__(
        self,
        corpus: list[str],
        vocab_size: int,
        max_iter: int = None,
        debug: bool = False,
    ):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocab = []
        self.word_freq = Counter()
        self.splits = {}  # e.g. highest: [high, est</w>]
        self.merges = {}  # e.g. [high, est</w>]: highest
        self.max_iter = max_iter
        self.debug = debug

    def train(self):
        """Train a BPE Tokenizer"""
        # count the word frequency
        for document in self.corpus:
            # split each document in corpus by whitespace
            words = document.split()
            self.word_freq += Counter(words)

        # initialize the self.splits
        for word in self.word_freq:
            self.splits[word] = list(word) + ["</w>"]

        if self.debug:
            print(f"Init splits: {self.splits}")

        alphabet = set()
        for word in self.word_freq:
            alphabet |= set(list(word))
        alphabet.add("</w>")

        self.vocab = list(alphabet)
        self.vocab.sort()

        cnt = 0
        while len(self.vocab) < self.vocab_size:
            if self.max_iter and cnt >= self.max_iter:
                break
            # find the most frequent pair
            pair_freq = self.get_pairs_freq()

            if len(pair_freq) == 0:
                print("No pair available")
                break

            pair = max(pair_freq, key=pair_freq.get)
            self.update_splits(pair[0], pair[1])

            if self.debug:
                print(f"Updated splits: {self.splits}")

            self.merges[pair] = pair[0] + pair[1]
            self.vocab.append(pair[0] + pair[1])

            if self.debug:
                print(
                    f"Most frequent pair({max(pair_freq.values())} times) "
                    f"is : {pair[0]}, {pair[1]}. Vocab size: {len(self.vocab)}"
                )
            cnt += 1

    def update_splits(self, lhs: str, rhs: str):
        """If we see lhs and rhs appear consecutively, we merge them"""
        for word, word_split in self.splits.items():
            new_split = []
            cursor = 0
            while cursor < len(word_split):
                if (
                    word_split[cursor] == lhs
                    and cursor + 1 < len(word_split)
                    and word_split[cursor + 1] == rhs
                ):
                    new_split.append(lhs + rhs)
                    cursor += 2
                else:
                    new_split.append(word_split[cursor])
                    cursor += 1
            self.splits[word] = new_split

    def get_pairs_freq(self) -> dict:
        """Compute the pair frequency"""
        pairs_freq = defaultdict(int)
        for word, freq in self.word_freq.items():
            split = self.splits[word]
            for i in range(len(split)):
                if i + 1 < len(split):
                    pairs_freq[(split[i], split[i + 1])] += freq
        return pairs_freq

    def tokenize(self, s: str) -> list[str]:
        splits = [list(t) + ["</w>"] for t in s.split()]
        for lhs, rhs in self.merges:
            for idx, split in enumerate(splits):
                new_split = []
                cursor = 0
                while cursor < len(split):
                    if (
                        cursor + 1 < len(split)
                        and split[cursor] == lhs
                        and split[cursor + 1] == rhs
                    ):
                        new_split.append(lhs + rhs)
                        cursor += 2
                    else:
                        new_split.append(split[cursor])
                        cursor += 1
                assert "".join(new_split) == "".join(split)
                splits[idx] = new_split
        return sum(splits, [])


if __name__ == "__main__":
    corpus = ["highest", "higher", "lower", "lowest", "cooler", "coolest"]
    bpe = BPE(corpus, vocab_size=50, max_iter=100, debug=False)
    bpe.train()
    print(bpe.tokenize("".join(corpus)))
    print(bpe.vocab)
    print(bpe.splits)
    print(bpe.merges)

```