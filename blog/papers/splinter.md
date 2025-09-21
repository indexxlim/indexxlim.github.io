# Few-Shot Question Answering by Pretraining Span Selection(Splinter)

 We explore the more realistic few-shot setting, where only a few hundred training examples are available, and observe that standard models perform poorly, highlighting the discrepancy between current pretraining objectives and question answering.  
 We propose a new pretraining scheme tailored for question answering: recurring span selection. Given a passage with multiple sets of recurring spans, we mask in each set all recurring spans but one, and ask the model to select the correct span in the passage for each masked span.  
 
 본 논문의 모델인 Splinter(**span-level pointer**)는 question answering에서 새로운 학습 데이터 처리인 **recurring span selection** 을 시도했다. 이 방법은 지문에서 반복되는 정답의 위치를 모두 마스킹 하는 방법이다.
 
 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/images/splinter/1_base_size_on_SQuAD.png?raw=true" itemprop="image" width="30%">
few-shot question answering by sampling small training sets from existing question answering benchmarks.



```python

```

## Pre-Training
- Tensorflow
- Adam Optimizer
- 2.4M training steps with batches of 256 sequence of length 512
- After warming up, max learning rate 10e-4, after which it decays linearly
- 0.1 dropout rate

## Fine-Tuning
- Hyperparameters from the default conﬁguration of the HuggingFace Transformers package
- Adam Optimizer
- Fnetuning on 1024 examples or less, train for either 10 epochs or 200 steps
- For full-size datasets, train for 2 epochs
- Set the batch size to 12 and use a maximal learning rate of 3 * 10−5, which warms up in the ﬁrst 10% of the step


```python

```


```python

```
