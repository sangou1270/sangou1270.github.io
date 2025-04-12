---
layout: home
title: 환영합니다
---

# sangou1270의 블로그에 오신 것을 환영합니다!

이 블로그는 GitHub Pages와 Jekyll을 사용하여 만들어졌습니다. 여기에서 다양한 주제에 대한 글과 프로젝트를 공유할 예정입니다.

## 최근 포스트

{% for post in site.posts limit:3 %}
### [{{ post.title }}]({{ post.url | relative_url }})
{{ post.date | date: "%Y년 %m월 %d일" }}

{{ post.excerpt }}

[더 읽기]({{ post.url | relative_url }})

---
{% endfor %}