---
layout: home
author_profile: true
title: 환영합니다
---

# Max의 개발 블로그에 오신 것을 환영합니다!

이 블로그는 AI, 비즈니스 모델, 코딩 교육에 관한 제 생각과 경험을 공유하는 공간입니다. 부트아지트(bootagit) 프로젝트를 비롯한 개인 프로젝트와 기술적 인사이트를 기록하고 있습니다.

## 최근 포스트

{% for post in site.posts limit:3 %}

### [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%Y년 %m월 %d일" }}

{{ post.excerpt }}

[더 읽기]({{ post.url | relative_url }})

---

{% endfor %}
