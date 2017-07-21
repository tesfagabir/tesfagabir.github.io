---
layout: default
title: About Me
---
# About Me

![](my_photo_bw.jpg)

My name is Tesfagabir Meharizghi. I am a PhD student in Computer Science at [Florida International University](https://www.cis.fiu.edu). My research interests is machine learning/deep learning. I worked on different types of datasets such as remote sensing environmental data, genomic data, 3D MRI data, etc. Currently I am working on a project that uses both MRI scans of tumor patients and their associated genomic data. I am very passionate on applying the state-of-the-art developments on the field machine learning/deep learning.

If you want to know more or collaborate, you can [contact](../contact/) me at any time.

<html lang="{{ site.lang | default: "en-US" }}">
 <head>
    <link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/font-awesome/4.3.0/css/font-awesome.min.css">
  </head>
  <body>
     <ul id="List"> 
        {% if site.social.linkedin %}
        <li><a href="https://linkedin.com/in/{{ site.social.linkedin }}">
            <i class="fa fa-github"></i> LinkedIn
        </a></li>
        {% endif %}
       {% if site.social.github %}
        <li><a href="https://github.com/{{ site.social.github }}">
            <i class="fa fa-github"></i> Github
        </a></li>
        {% endif %} 
        {% if site.social.email %}
        <li><a href="mailto:{{ site.social.email }}">
          <i class="fa fa-envelope-square"></i> Email
          </a></li>
        {% endif %}
     </ul>
  </body>
</html>
