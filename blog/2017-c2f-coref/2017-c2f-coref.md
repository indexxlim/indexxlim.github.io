---
title: "End-to-End Neural Coreference Resolution"
authors: [indexxlim]
tags: [NLP, coreference-resolution, neural-networks]
date: 2017-07-26
---

# c2f-coref

## Coreference Resolution

Coreferece를 찾는 NLP Task 중 하나로 coreference는 문장 속에서 Entity와 같은 의미로 언급(mention)된 span을 찾는 것을 목적.

<!-- truncate -->
Lee 아저씨가 만든 논문이 가장 유명하며, c2f-coref를 바탕으로 ELMO부터 Bert, Spanbert등의 언어모델을 이용하여 mention의 span을 찾는다.  

[cs224n]  
[End-to-end Neural Coreference Resolution]  
[Higher-order Coreference Resolution]


[cs224n]: http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture13-coref.pdf  
[End-to-end Neural Coreference Resolution]: https://arxiv.org/pdf/1707.07045.pdf  
[Higher-order Coreference Resolution]: https://www.aclweb.org/anthology/N18-2108.pdf


## 개요
- Identify all mentions that refer to the same entity in the word  

mention이라고 하면 지칭을 의미한다. 문장안에 있는 이 지칭을 찾고 같은 지칭을 표현하는 단어끼리 그룹화한다.(cluster) 
mention의 종류로는 Pronouns, Named entities, Noun phrases 등이 있다. 그 중  pronouns과 noun phrases는 pos 태그와 파서를 이용해서, named entities는 개채명인식을 통해 추출부터 한다.

<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./1_mention.png?raw=true" itemprop="image" width="80%" />

이 때 Coreference라고 함은, 2개 이상의 mentions이 같은 entity를 말하고자 하는 것을 의미한다.

<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./2_coreference.png?raw=true" itemprop="image" width="80%" />

## 방법
- Rule-based (pronominal anaphora resolution)
- Mention Pair
- Mention Ranking
- Clustering [skipping this year; see Clark and Manning (2016)]

ML 방법이 많이 사용되기 전에는 규칙에 기반한 Hobbs' naive algorithm(1976)이 많이 쓰였다.  
Mention Pair는 모든 mentions의 쌍이 coreferent인지를 binary classifier로 학습하는 것이다.
<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./3_mentionpair.png?raw=true" itemprop="image" width="80%" />

Mention Rank는 j번째 mention을 계산할 때 그 이전에 나온 mention에서 가장 확률이 높은 mention을 택한다.
<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./4_mentionranking.png?raw=true" itemprop="image" width="80%" />

Neural Network를 사용하는 최근의 sota에서는 c2f이라고 명명한 End-to-end 논문의 방법이 있다. 이 논문에서는 metion을 span의 attention sum으로 
구한뒤(representations) 각 mention과 coreferenct인지를 다음과 같이 계산한다.
<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./5_grepresentation.png?raw=true" itemprop="image" width="80%" />
<img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./6_score.png?raw=true" itemprop="image" width="80%" />

그래서 $s(i,j)$에서의 distribution을 학습합니다
$$P(y) = \frac{e^{s(x,y)}}{\sum\limits_{y'\in Y(x)} e^{s(x,y')}}$$


그외의 extra features는 최근 sota인 [Revealing the Myth of Higher-Order Inference in Coreference Resolution]을 참고!

[Revealing the Myth of Higher-Order Inference in Coreference Resolution]: https://arxiv.org/pdf/2009.12013.pdf



```python

```
