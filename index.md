---
layout: default
title: Your New Jekyll Site
---

<div id="articles">
  <h1>Recent Posts</h1>
  <ul class="posts noList">
    {% for post in site.posts %}
      <li>
       	<h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <span class="date"><em>{{ post.date | date_to_string }}</em></span>
      	<p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
      </li>
    {% endfor %}
  </ul>
</div>
