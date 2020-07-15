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

```
$ nohup python -u flask_app.py &
$ tail -f nohup.out
$ lsof -i :5000

COMMAND   PID  USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
python3 32258 user    3u  IPv4 5575349      0t0  TCP *:5000 (LISTEN)
python3 32260 user    3u  IPv4 5575349      0t0  TCP *:5000 (LISTEN)
python3 32260 user    4u  IPv4 5575349      0t0  TCP *:5000 (LISTEN)
$ sudo kill -9 32258
```

celery worker실행
celery -A task_analysis worker –loglevel=info –concurrency=10 -n worker1@%h
celery -A task_analysis worker -l info -c 10 -n worker1@%h

celery multi start -A task_analysis worker -l info -c 10 -n worker1@%h

celery multi start -A task_analysis -l info 10 -c 10 -n –pidfile=logs/%n.pid –logfile=logs/%n%I.log

경로가 안맞을 경우 재설치
pip3 install –upgrade pip setuptools wheel
https://discourse.brew.sh/t/pip3-upgrade-overwrites-usr-local-bin-pip/3258/2

user mode로 설치한 pip 패키지 PATH에 등록하기 https://beomi.github.io/2018/02/12/Add-packages-installed-with-pip-usermode/ echo ‘export PATH=”/Users/$(whoami)/Library/Python/3.6/bin:$PATH”’ » .zshrc

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

type ‘all’

spacy –> !python3 -m spacy download en_core_web_sm