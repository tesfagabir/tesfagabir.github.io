---
layout: default
title: About Me
---
# About Me

![](my_photo_bw.jpg)

My name is Tesfagabir Meharizghi. I am a PhD student in Computer Science at [Florida International University](https://www.cis.fiu.edu). I am interested in making research on machine learning/deep learning. I have worked on different machine/deep learning projects. Those involve collecting data from various sources, preprocessing it and building predictive models using different machine learning algorithms. Moreover, I used different types of datasets such as geospatial, remote sensing, genomic and images. I am always passionate on the daily developments of machine learning/deep learning.

During my graduate study, I have also taken different graduate courses to build my knowledge on the field of data science. For example:
* Data Mining
* Data analysis for environmental modeling
* Machine learning
* Artificial Neural Networks
* Digital Image Processing
* Computer Vision
* Software Engineering, etc.

You can click [here]() to see details of my work.

# Contact Me
If you want to contact me, feel free drop me a message at any time:
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
