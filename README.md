<div id="articles">
  <h1>Articles</h1>
    {% for post in site.posts %}
      	<h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
      	<span class="date">{{ post.date | date_to_string }}</span>
      	<p class="description">{% if post.description %}{{ post.description  | strip_html | strip_newlines | truncate: 120 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 120 }}{% endif %}</p>
        <hr>
  </ul>
</div>
