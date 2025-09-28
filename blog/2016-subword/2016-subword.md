# subword model

## Neural Machine Translation of Rare Words with Subword Units
데이터 압축으로 쓰이던 bpe를 자연어에 쓴 논문이다.
단어보다 작은 subword unit을 사용하여 음운론적이고 형태학적으로 번역함으로써, open-vocabulary NMT모델을 소개한다.

<!-- truncate -->

### Introduction
agglutination and compounding 단계가 포합된 언어의 경우, word level에서 하위 수준으로 내려가는 메커니즘이 필요하다.
- subword를 통한 open-vocabulary은 back-off dictionay나 큰 large vocabularies보다 효과적이다.
- byte pair encoding (BPE)(Gage, 1994)은 가변 길이의 문자열을 고정크기의 어휘를 통해 표현 할수 있기 때문에 인공신경망에서 효과적인 방법이다.

### Neural Machine Translation
모델은 Bi-direction GRU 유닛 사용
subword로 정보를 투영할 때의 장점
- named entities
- cognates and loanwords(의미 분화 및 외래어)
- morphologically complex words(복합어)

### Bytr Pair Encoding (BPE)
알고리즘은 모든 쌍을 인덱싱하고, 데이터 구조(dictionary)를 점진적으로 업데이트 한다.
예를 들어 ('A', 'B')안에서 모든 기호 쌍을 반복적으로 계산해서 가장 빈도가 높은 쌍을 새로운 기호 'AB'로 교체한다.
허프만 인코딩 처럼 하위 단어를 기반으로 새로운 단어를 번역하고 새성한다.


```python
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] +=freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)'+bigram+r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab ={'l o w </w>' : 5, 'l o w e r </w>' : 2,
       'n e w e s t </w>' :6, 'w i d e s t </w>': 3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get) 
    vocab = merge_vocab(best, vocab) 
    print(best)
    
    

```

    ('e', 's')
    ('es', 't')
    ('est', '&lt;/w&gt;')
    ('l', 'o')
    ('lo', 'w')
    ('n', 'e')
    ('ne', 'w')
    ('new', 'est</w>')
    ('low', '</w>')
    ('w', 'i')


여기서 lower는 low, er로 분할된다.

이 논문의 task인 translation에서 bpe의 사용은 2가지가 있다.


- source 어휘와 target 각각의 어휘
- source와 target의 어휘를 합침(joint BPE) 

전자는 어휘 사이즈와, 단어가 더 compact하고 후자는 source와 target segmentation 사이에 일관성이 있다.

### result
<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./1_result.png?raw=true" itemprop="image" width="60%" />

word 단위인 Wunk와 Character 단위보다 더 나은 결과를 보여준다.

### Byte-level BPE

[GPT-2] 논문에서 BPE가 모든 unicode strings을 표현하려면, multi-symbol tokens 이 추가되기전에 voca의 갯수가 130000개나 된다고 한다.
그러나 Byte level에서는 오직 256개만의 기본 vocabulary만 필요로 한다. 그래서 어느 데이터셋에서 pre-processing, tokenization, vocab size 상관없이 empirical benefit을 가져왔다고 한다.

[GPT-2]: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

## WordPiece
 - Word: Jet makers feud over seat width with big orders at stake
 - wordpieces: _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake

BPE와 비슷한 방식의[WordPiece1]는 음성 인식에서 어느 언어에서나 적용될수 있도록(language-agnostic) multilingual vocabulary를 위해 처음 제안 됬고 [WordPiece2]에서는 번역에 적용 했다. WordPiece는 BPE처럼 2개의 units을 꺼내서 합치는 것은 같지만, 그 전에 기본적으로 language model을 만들어 놓고 새로운 유닛들 중 이 LM에서 가장 큰 likelihood를 가진 new unit을 선택한다는 점이다.  
하지만 모든 쌍을 brute-force 한다면 너무 비효율적이기 때문에 likelihood가 일정 threshold 이상일 때 새로운 단어를 생성한다.

<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./2_wpmresult.png?raw=true" itemprop="image" width="30%" />

vocabulary 32000개 일 때, 성능이 가장 좋다. ~~그 이상은 유의미한 차별성이 없나?~~

[WordPiece1]: https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
[WordPiece2]: https://arxiv.org/pdf/1609.08144.pdf

## SentencePiece
WordPiece를 에서 발전 시킨 [SentencePiece]
Normalizer, Trainer, Encoder, Decoder로 구성됨
- Normalizer는 유니코드 캐릭터를 정심 폼으로 정규화
- Trainer는 subword segmentation
- Encoder는 normalize와 subword sequence로 tokenization
- Decoder는 subword sequence를 normalized text로 변환

이중 Encoder와 Decoder를 [SentencePiece]로 부르기로 하는데, 이 때 voca를 id mapping하는 과정도 포함된다.

### Efficient subword training
binary heap(우선순위 큐)를 사용해서 BPE의 $O(N^2)$ computational cost 를 $O(Nlog(N))$ 으로 줄였다고 한다.

### Customizable character normalization
Aho-Corasick automaton을 이용하여 Unicode NFKC의 규칙을 약간 수정(Canonical Combining Class reordering의 일부 subset만 사용) 컴파일하여 정규화했다

[SentencePiece]: https://arxiv.org/pdf/1808.06226.pdf


```python

```
