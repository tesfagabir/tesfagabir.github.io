## Table of Contents
* [Introduction](#Introduction)
* [Scraping of Weather Data](#Scraping)
    * [Weather Forecasting Data for Single City](#Single)
    * [Weather Forecasting Data for Multiple Cities](#Multiple)
* [Preprocessing and Visualizing the Data](#PreAndViz)
* [Conclusion](#Conclusion)
* [References](#References)

<a name="Introduction"></a>
## Introduction

Web Scraping is an important way of collecting data from the web. There are different types of data that are displayed systematically by web browsers. The data usually contains:

* HTML tags
* CSS styles
* javascript codes
* Images and other multimedia data

The documents in a web are organized in html tags. So, we can systematically scrape the needed data once we inspect its structure. In this post, I am going to show how to scrape weather forecasting data from [National Weather Service](http://forecast.weather.gov).

As an example in this post, we are going to select some of the most populous cities in the US. Then based on their location (latitutudes and longitudes), we will extract their corresponding weather data from the web site.

Once they extracted, we will structure the data and save in pandas data frames. Then, we will apply some preprocessing steps.

Finally we will visualize the data using some pythn visualization packages such as seaborn. The post is inspired by the post [here](https://www.dataquest.io/blog/web-scraping-tutorial-python/)

<a name="Scraping"></a>
## Scraping of Weather Data

<a name="Single"></a>
### *Weather Forecasting Data for Single City*

As a first step in scraping web data, we have to open the page on a browser and inspect how the data is structured. As an example, let's open the website and display weather data for one city, e.g., [Miami](http://forecast.weather.gov/MapClick.php?lat=25.7748&lon=-80.1977).

In this example, we are going to use google chrome to open the web page. After you open it, go to developer tools to inspect the structure of the page. Here, we are concerned only on the extended forecast data.

First let's import the necessary libraries


```python
import requests
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

**requests** package is used to download the data from a URL. **BeautifulSoup** is a package to select part of the downloaded data that is necessary for our analysis.

Download the Miami weather data


```python
# Get dat
URL = ("http://forecast.weather.gov/MapClick.php?lat=25.7748&lon=-80.1977")
page = requests.get(URL)
```

Parse the content with BeautifulSoup and extract the seven day forecast. It will then find all the items (seven days). Finally one item is displayed.


```python
soup = BeautifulSoup(page.content, 'html.parser')
seven_day = soup.find(id="seven-day-forecast")
forecast_items = seven_day.find_all(class_="tombstone-container")
first = forecast_items[0]
print(first.prettify())
```

    <div class="tombstone-container">
     <p class="period-name">
      This
      <br/>
      Afternoon
     </p>
     <p>
      <img alt="This Afternoon: Showers and thunderstorms likely, mainly after 5pm.  Mostly sunny, with a high near 90. Southeast wind 7 to 10 mph.  Chance of precipitation is 60%. New rainfall amounts of less than a tenth of an inch, except higher amounts possible in thunderstorms. " class="forecast-icon" src="newimages/medium/hi_tsra60.png" title="This Afternoon: Showers and thunderstorms likely, mainly after 5pm.  Mostly sunny, with a high near 90. Southeast wind 7 to 10 mph.  Chance of precipitation is 60%. New rainfall amounts of less than a tenth of an inch, except higher amounts possible in thunderstorms. "/>
     </p>
     <p class="short-desc">
      T-storms
      <br/>
      Likely
     </p>
     <p class="temp temp-high">
      High: 90 °F
     </p>
    </div>


From the above data, we can see that the information is organized in different html tags and classes/ids. Now, let's separate the data based on their classes for the selected item.


```python
period = first.find(class_="period-name").get_text()
short_desc = first.find(class_="short-desc").get_text()
temp = first.find(class_="temp").get_text()

print(period)
print(short_desc)
print(temp)
```

    ThisAfternoon
    T-stormsLikely
    High: 90 °F


Extract the description of the item's weather using img tag. The description is then saved in "title" value 


```python
img = first.find("img")
desc = img['title']
print(desc)
```

    This Afternoon: Showers and thunderstorms likely, mainly after 5pm.  Mostly sunny, with a high near 90. Southeast wind 7 to 10 mph.  Chance of precipitation is 60%. New rainfall amounts of less than a tenth of an inch, except higher amounts possible in thunderstorms. 


Now let's find the periods of weather forecasting data


```python
period_tags = seven_day.select(".tombstone-container .period-name")
periods = [pt.get_text() for pt in period_tags]
periods
```




    [u'ThisAfternoon',
     u'Tonight',
     u'Saturday',
     u'SaturdayNight',
     u'Sunday',
     u'SundayNight',
     u'Monday',
     u'MondayNight',
     u'Tuesday']



So, we have already seen the data for a singl city. Now let's extract the forcasting data for multiple cities and futher process the data.

<a name="Multiple"></a>
### *Weather Forecasting Data for Multiple Cities*

Let's now prepare the lists of cities and their corresponding locations. N.B. Locations can be taken from the weather data website.


```python
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
lats = [40.7146, 34.0535, 41.8843, 29.7606, 33.4483, 39.9522, 29.4246, 32.7157, 32.7782, 37.3387]
lons = [-74.0071, -118.2453, -87.6324, -95.3697, -112.0758, -75.1622, -98.4946, -117.1617, -96.7954, -121.8854]
```

Now, we will download the data for each city and save the necessary information


```python
n_cities = len(cities)

periods = None
all_temps = []

for i in range(n_cities):
    print('Extracting data of: %s' % cities[i])
    URL = "http://forecast.weather.gov/MapClick.php?lat=" + str(lats[i]) + "&lon=" + str(lons[i])
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    seven_day = soup.find(id="seven-day-forecast")
    forecast_items = seven_day.find_all(class_="tombstone-container")
    period_tags = seven_day.select(".tombstone-container .period-name")

    #Let's save only the periods and their corresponding temperatures
    periods = [pt.get_text() for pt in period_tags]
    #short_descs = [sd.get_text() for sd in seven_day.select(".tombstone-container .short-desc")]
    temps = [t.get_text() for t in seven_day.select(".tombstone-container .temp")]
    #descs = [d["title"] for d in seven_day.select(".tombstone-container img")]
    
    all_temps.append(temps)
```

    Extracting data of: New York
    Extracting data of: Los Angeles
    Extracting data of: Chicago
    Extracting data of: Houston
    Extracting data of: Phoenix
    Extracting data of: Philadelphia
    Extracting data of: San Antonio
    Extracting data of: San Diego
    Extracting data of: Dallas
    Extracting data of: San Jose


Let's now create create a pandas dataframe and save the temperature data.


```python
weather = pd.DataFrame(data=all_temps, index=cities, columns=periods)
print(weather.shape)
weather.head()
```

    (10, 9)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Today</th>
      <th>Tonight</th>
      <th>Saturday</th>
      <th>SaturdayNight</th>
      <th>Sunday</th>
      <th>SundayNight</th>
      <th>Monday</th>
      <th>MondayNight</th>
      <th>Tuesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <td>High: 83 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 74 °F</td>
      <td>Low: 62 °F</td>
      <td>High: 80 °F</td>
      <td>Low: 66 °F</td>
      <td>High: 77 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 83 °F</td>
    </tr>
    <tr>
      <th>Los Angeles</th>
      <td>High: 83 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 82 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 82 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 83 °F</td>
      <td>Low: 68 °F</td>
      <td>High: 85 °F</td>
    </tr>
    <tr>
      <th>Chicago</th>
      <td>High: 75 °F</td>
      <td>Low: 65 °F</td>
      <td>High: 76 °F</td>
      <td>Low: 66 °F</td>
      <td>High: 78 °F</td>
      <td>Low: 67 °F</td>
      <td>High: 80 °F</td>
      <td>Low: 69 °F</td>
      <td>High: 86 °F</td>
    </tr>
    <tr>
      <th>Houston</th>
      <td>High: 98 °F</td>
      <td>Low: 79 °F</td>
      <td>High: 99 °F</td>
      <td>Low: 79 °F</td>
      <td>High: 96 °F</td>
      <td>Low: 77 °F</td>
      <td>High: 95 °F</td>
      <td>Low: 76 °F</td>
      <td>High: 94 °F</td>
    </tr>
    <tr>
      <th>Phoenix</th>
      <td>High: 101 °F</td>
      <td>Low: 83 °F</td>
      <td>High: 99 °F</td>
      <td>Low: 80 °F</td>
      <td>High: 99 °F</td>
      <td>Low: 84 °F</td>
      <td>High: 103 °F</td>
      <td>Low: 85 °F</td>
      <td>High: 105 °F</td>
    </tr>
  </tbody>
</table>
</div>



<a name="PreAndViz"></a>
## Preprocessing and Visualizing the Data

Now, we will only take the numerical temperature values to be used for further processing.


```python
#Extract the numberical value of the temperature
for period in periods:
    weather[period] = weather[period].str.extract("(?P<temp_num>\d+)", expand=False)
    weather[period] = weather[period].astype('float')
weather.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Today</th>
      <th>Tonight</th>
      <th>Saturday</th>
      <th>SaturdayNight</th>
      <th>Sunday</th>
      <th>SundayNight</th>
      <th>Monday</th>
      <th>MondayNight</th>
      <th>Tuesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <td>83.0</td>
      <td>67.0</td>
      <td>74.0</td>
      <td>62.0</td>
      <td>80.0</td>
      <td>66.0</td>
      <td>77.0</td>
      <td>67.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>Los Angeles</th>
      <td>83.0</td>
      <td>67.0</td>
      <td>82.0</td>
      <td>67.0</td>
      <td>82.0</td>
      <td>67.0</td>
      <td>83.0</td>
      <td>68.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>Chicago</th>
      <td>75.0</td>
      <td>65.0</td>
      <td>76.0</td>
      <td>66.0</td>
      <td>78.0</td>
      <td>67.0</td>
      <td>80.0</td>
      <td>69.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>Houston</th>
      <td>98.0</td>
      <td>79.0</td>
      <td>99.0</td>
      <td>79.0</td>
      <td>96.0</td>
      <td>77.0</td>
      <td>95.0</td>
      <td>76.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>Phoenix</th>
      <td>101.0</td>
      <td>83.0</td>
      <td>99.0</td>
      <td>80.0</td>
      <td>99.0</td>
      <td>84.0</td>
      <td>103.0</td>
      <td>85.0</td>
      <td>105.0</td>
    </tr>
  </tbody>
</table>
</div>



We will now transpose the dataframe to enable us visualize temperature data for each city.


```python
weather = weather.T
weather.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>New York</th>
      <th>Los Angeles</th>
      <th>Chicago</th>
      <th>Houston</th>
      <th>Phoenix</th>
      <th>Philadelphia</th>
      <th>San Antonio</th>
      <th>San Diego</th>
      <th>Dallas</th>
      <th>San Jose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Today</th>
      <td>83.0</td>
      <td>83.0</td>
      <td>75.0</td>
      <td>98.0</td>
      <td>101.0</td>
      <td>85.0</td>
      <td>102.0</td>
      <td>76.0</td>
      <td>102.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>Tonight</th>
      <td>67.0</td>
      <td>67.0</td>
      <td>65.0</td>
      <td>79.0</td>
      <td>83.0</td>
      <td>69.0</td>
      <td>76.0</td>
      <td>68.0</td>
      <td>80.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>74.0</td>
      <td>82.0</td>
      <td>76.0</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>73.0</td>
      <td>103.0</td>
      <td>76.0</td>
      <td>97.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>SaturdayNight</th>
      <td>62.0</td>
      <td>67.0</td>
      <td>66.0</td>
      <td>79.0</td>
      <td>80.0</td>
      <td>63.0</td>
      <td>78.0</td>
      <td>67.0</td>
      <td>77.0</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>80.0</td>
      <td>82.0</td>
      <td>78.0</td>
      <td>96.0</td>
      <td>99.0</td>
      <td>79.0</td>
      <td>100.0</td>
      <td>77.0</td>
      <td>94.0</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
</div>



We can now visualize the data using different kinds of plots. Now we will use seaborn boxplot to display the temperature forcasting data.


```python
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(10, 7))
fig = sns.boxplot(data=weather, ax=ax, orient='h')
fig.set_title('Temperature Range')
fig.set_xlabel('Temp (Fahrenheit)')
fig.set_ylabel('Cities')
plt.show()
```


![png](https://raw.githubusercontent.com/tesfagabir/tesfagabir.github.io/master/assets/images/2017-01-20-Web-Scraping-using-Python.png)


<a name="Conclusion"></a>
## Conclusion

In this post, we have scraped weather forcasting data of different US cities. Finally, we preprocessed and visualized it. To get more information on Scraping, please go to the links given in the reference section.

<a name="References"></a>
## References

* [Web Scraping using BeautifulSoup](https://www.dataquest.io/blog/web-scraping-tutorial-python/)
* [Extracting Web Data using Python API](https://www.dataquest.io/blog/python-api-tutorial/)
* [Web Scraping Job Postings from Indeed](https://medium.com/@msalmon00/web-scraping-job-postings-from-indeed-96bd588dcb4b)
* [Scraping For Data — A Practical Guide](https://medium.com/k-folds/scraping-for-data-a-practical-guide-67cc397450b2)
