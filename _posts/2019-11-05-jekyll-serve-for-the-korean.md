---
layout: single
author_profile: true
read_time: true
comments: true
share: true
related: true
title: jekyll serve for the Korean
---

chcp 65001  
bundle exec jekyll serve

celery worker실행  
celery -A task_analysis worker --loglevel=info --concurrency=10 -n worker1@%h   
celery -A task_analysis worker -l info -c 10 -n worker1@%h 

celery multi start tasks -A app -l info

경로가 안맞을 경우 재설치  
pip3 install --upgrade pip setuptools wheel  
https://discourse.brew.sh/t/pip3-upgrade-overwrites-usr-local-bin-pip/3258/2

user mode로 설치한 pip 패키지 PATH에 등록하기
https://beomi.github.io/2018/02/12/Add-packages-installed-with-pip-usermode/
echo 'export PATH="/Users/$(whoami)/Library/Python/3.6/bin:$PATH"' >> .zshrc

nltk install 
```
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```
type 'all'

spacy -->
!python3 -m spacy download en_core_web_sm
