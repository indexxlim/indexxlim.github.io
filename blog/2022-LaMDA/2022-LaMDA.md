# LaMDA: Language Models for Dialog Applications
    

 LaMDA is a family of Transformer- based neural language models specialized for dialog, which have up to 137B parameters and are pre-trained on 1.56T words of public dialog data and web text.  
 The first challenge, safety, involves ensuring that the model’s responses are consistent with a set of human values, such as preventing harmful suggestions and unfair bias.   

<!-- truncate -->
 The second challenge, factual grounding, involves enabling the model to consult external knowledge sources, such as an information retrieval system. 
 
 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./1_dialog.png?raw=true" itemprop="image" width="60%" />

 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./2_dialog.png?raw=true" itemprop="image" width="60%" />


 
 LaMDA는 transformer에 기반하여 **dialog** 즉 대화에 중점을 둔 언어모델이다. LaMDA는 1.56T단어를 학습하여 137B 파라미터를 가지고 있다. LaMDA의 크게 2가지 의 도전, safety와 factual grounding에 향상을 이끌었다.
 LaMDA는 3개의 주요 방법으로 모델 스케일에 이점을 둔다. 3가지는 Quality, Safety, Groundedness.
 
 Quality: We decompose Quality into three dimensions, Sensibleness, Specificity, and Interestingness (SSI), 
 Sensibleness는 말이되는지, Specificity는 구체적인지 Interestingness 는 통찰력있고 예상되지 않은 재치있는 말인지 구분한다.

Safety:의도 되지 않은 해로운 위험요소, 즉 해로운 고정관념이나 비방의 요소들을 피한다.

Groundedness: 정보량이 있는 근거성 있는 응답들로 구성되있는지를 정의한다


# 3. LaMDA Pre-Training  
With the objectives and metrics defined, we describe LaMDA’s two-stage training: pre-training and fine-tuning. In the pre-training stage, we first created a dataset of 1.56T words — nearly 40 times more words than what were used to train previous dialog models — from public dialog data and other public web documents. After tokenizing the dataset into 2.81T SentencePiece tokens, we pre-train the model using GSPMD to predict every next token in a sentence, given the previous tokens. The pre-trained LaMDA model has also been widely used for natural language processing research across Google, including program synthesis, zero-shot learning, style transfer, as well as in the BIG-bench workshop.

LaMDA는 코퍼스에서 the next token를 예측하는 방법으로 사전학습된다. 데이터는 대화데이터뿐만 아니라 웹데이터를 통해 학습됬다. 데이터셋은 dataset consists of 2.97B documents, 1.12B dialogs, and 13.39B dialog utterances, 총 1.56T 단어로 구성된다.
토크나이저는 sentencepiece를 이용하여 2.81T의 bpe토큰을 학습했고 사전은 32k tokens.

모델은 Transformer language model의 Decoder만 사용되었고, Large 모델은 137B non-embedding parameters, 64 layers, dmodel = 8192, df f = 65536, h = 128, dk = dv = 128, T5 처럼 relative attention, 활성화함수는 gated-GELU 사용했다.
학습은 1024 TPUv3를 통해 57.7일동안 256K tokens 의 배치사이즈 만큼 하였으며, Lingvo framework를 이용하여 123 TFLOPS/sec with 56.5% FLOPS utilization with the 2D sharding algorith으로 학습하였다.

 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./3_ pt_model.png?raw=true" itemprop="image" width="60%" />


# 4. Metric
Sensibleness, Specificity, Interestingness (SSI): Our overall quality score is an average of sensibleness, specificity, and interestingness (SSI).
sensibleness는 말이 되는지 평가하는 항목, 주의해야 할점은 모든 문장에 OK라고 대답할 경우(GenericBot) sensibleness이 70%로 측정될수 있다.
Specificity는 구체적으로 정보가 포함되는지를 평가
Interestingness는 crowdworkers를 통해 0/1로 평가 - 감성적으로 직접 평가
그 외에 safety(대화모델에서 의도하지 않은 대답의 경우) groundedness(cross-checking)가 있고, Role-specific metric으로 Helpfulness(information retrieval), role consistency(화자 정리)가 있다.

Groundedness: SSI와 safety와 유사하게 모델에 반영할 40K turns을 모았다. 정보가 5명중 3명의 crowdworker가 사실이라고 할 경우에 평가한다.

Estimating these metrics for human-generated responses: crowdworker가 외부에서 모은 데이터를 툴을 이용해서 safe, sensible, specific, interesting, grounded, and informative manner를 평가한다



# 5, 6 Fine-Tuning

LaMDA Fine-Tuning  
In the fine-tuning stage, we train LaMDA to perform a mix of generative tasks to generate natural-language responses to given contexts, and classification tasks on whether a response is safe and high-quality, resulting in a single multi-task model that can do both. The LaMDA generator is trained to predict the next token on a dialog dataset restricted to back-and-forth dialog between two authors, while the LaMDA classifiers are trained to predict the Safety and Quality (SSI) ratings for the response in context using annotated data. During a dialog, the LaMDA generator first generates several candidate responses given the current multi-turn dialog context, and the LaMDA classifiers predict the SSI and Safety scores for every response candidate. Candidate responses with low Safety scores are first filtered out. Remaining candidates are re-ranked by their SSI scores, and the top result is selected as the response. We further filter the training data used for the generation task with LaMDA classifiers to increase the density of high-quality response candidates.

몇가지 fine-turning을 pre-training(PT)에 적용한다.여기에는 Decoder로 구성되어있어 사용되는 Generative task와 문장의 quality and safety를 평가하는 dicriminator task가 있다.  
Generative 학습 입력은 " \<context\> \<sentinel\> \<response\>"으로 구성되어있다.  
• “What’s up? RESPONSE not much.”  
Discriminative 입력은 "\<context\> \<sentinel\> \<response\> \<attribute-name\> \<rating\>"으로 구성되어 있다.  
    • “What’s up? RESPONSE not much. SENSIBLE 1”   
    • “What’s up? RESPONSE not much. INTERESTING 0”   
    • “What’s up? RESPONSE not much. UNSAFE 0”  

두가지를 한번에 효과적으로 사용하기 위해서 결합해서 사용하기도 한다." P(“\<desired- rating>” | “\<context\> \<sentinel\> \<response\> \<attribute-name\>”) 이 중에서 SENSIBLE는 메트릭에서 3배의 가중치를 줬다.  
추가적으로 The toolset(TS)이나 Dialog collection등의 외부 툴이나 데이터를 모았다.
    
 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./4_finetuning.png?raw=true" itemprop="image" width="60%" />
 
Fine-tuning: 2가지 taskd에 대하여 fine-tuning했다. 
    
첫번째는 “How old is Rafael Nadal?”라는 쿼리 일때 context + base → “TS, Rafael Nadal’s age”.

두번째는 task는 TS의 단편 대화(e.g., “He is 31 years old right now” + “Rafael Nadal / Age / 35”)이다.  
context + base + query + snippet → “User, He is 35 years old right now”.  
For example, context + base + query + snippet → “TS, Rafael Nadal’s favorite song”.      



 <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./5_groundedness.png?raw=true" itemprop="image" width="60%" />
 
 
  <img src="https://github.com/indexxlim/indexxlim.github.io/blob/main/diary.py/machine_learning/paper/./6_comparing.png?raw=true" itemprop="image" width="60%" />


```python

```
