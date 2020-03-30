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


경로가 안맞을 경우 재설치  
pip3 install --upgrade pip setuptools wheel  
https://discourse.brew.sh/t/pip3-upgrade-overwrites-usr-local-bin-pip/3258/2

