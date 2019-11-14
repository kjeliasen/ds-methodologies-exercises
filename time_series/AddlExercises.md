# Working With Time Series in Pandas

In this lesson, we will talk about how to work with time series data with pandas
dataframes.

* [Converting to DateTime Type](#converting-to-datetime-type)
* [Working with DateTime Series](#working-with-datetime-series)
* [DateTime Indexes](#datetime-indexes)
    * [Changing the Period](#changing-the-period)
    * [Filling Missing Values](#filling-missing-values)
    * [Resampling](#resampling)
    * [Rolling Windows](#rolling-windows)
    * [Lagging and Lead](#lagging-and-lead)
* [Timezones](#timezones)
* [Days Since Now](#days-since-now)
* [Exercises](#exercises)

## Converting to DateTime Type

Thus far we have discuessed numeric and string datatypes in pandas dataframes,
but pandas has a special type for dates as well.

To convert an existing string value or values to a datetime type, we can use the
`to_datetime` function from the pandas module.

```python
import pandas as pd

pd.to_datetime('Jan 1 1970')
```

By default pandas will try to infer the date format, but there may be be cases
where pandas can't figure out the format itself, and we'd have to help out.

```python
try:
    pd.to_datetime('Jan:1:1970')
except ValueError as e:
    print('ValueError', e)
```

Here we can specify a **format string** to tell pandas explicitly how to convert
this date.

```python
pd.to_datetime('Jan:1:1970', format='%b:%d:%Y')
```

In addition to clarifying date format ambiguity, adding the `format=` keyword
argument can significantly speed up the processing of larger datasets with
non-standard formats.

## Working with DateTime Series

The `.dt` accessor can be used to access various properties of a date:

| Property         | Description                                                       |
| --------         | -----------                                                       |
| year             | The year of the datetime                                          |
| month            | The month of the datetime                                         |
| day              | The days of the datetime                                          |
| hour             | The hour of the datetime                                          |
| minute           | The minutes of the datetime                                       |
| second           | The seconds of the datetime                                       |
| microsecond      | The microseconds of the datetime                                  |
| nanosecond       | The nanoseconds of the datetime                                   |
| date             | Returns datetime.date (does not contain timezone information)     |
| time             | Returns datetime.time (does not contain timezone information)     |
| timetz           | Returns datetime.time as local time with timezone information     |
| dayofyear        | The ordinal day of year                                           |
| weekofyear       | The week ordinal of the year                                      |
| week             | The week ordinal of the year                                      |
| dayofweek        | The number of the day of the week with Monday=0, Sunday=6         |
| weekday          | The number of the day of the week with Monday=0, Sunday=6         |
| weekday_name     | The name of the day in a week (ex: Friday)                        |
| quarter          | Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, etc.               |
| days_in_month    | The number of days in the month of the datetime                   |
| is_month_start   | Logical indicating if first day of month (defined by frequency)   |
| is_month_end     | Logical indicating if last day of month (defined by frequency)    |
| is_quarter_start | Logical indicating if first day of quarter (defined by frequency) |
| is_quarter_end   | Logical indicating if last day of quarter (defined by frequency)  |
| is_year_start    | Logical indicating if first day of year (defined by frequency)    |
| is_year_end      | Logical indicating if last day of year (defined by frequency)     |
| is_leap_year     | Logical indicating if the date belongs to a leap year             |

In addition to the properties listed above, we can use the `strftime` method and
give date string to format the date in a custom way.

## DateTime Indexes

Once we have a date-time column on a dataframe, we can use that column as the
index on our dataframe.

```python
df = pd.read_csv('coffee_consumption.csv')
df.date = pd.to_datetime(df.date)
df = df.set_index('date').sort_index()
df
```

!!!warning "Sort Time Data"
    You should sort the dataframe by the datetime field before doing any
    date-related manipulations, otherwise they could have undefined behavior.

Having a datetime index on our dataframe allows us to use a lot of time series
specific functionality within pandas. All of the functionality we'll discuss in
the rest of this lesson requires a dataframe with a datetime index.

First let's take a look at the start and end of our data:

```python
df.index.min(), df.index.max()
```

### Changing the Period

Notice that all of the days aren't one after another, there are some gaps in our
data. Often times it is the case that we want a uniform amount of time between
observations in each row. We can accomplish this with the `.asfreq` method.

```python
by_day = df.asfreq('D')
by_day
```

Notice here several things have changed:

- We now have a larger number of rows
- Each date is one day greater than the next
- We introduced some nulls into our data

Now our data represents an entire year, but we introduced NaNs where we were
missing data.

### Filling Missing Values

Pandas contains some special methods for filling missing values in time series
data. We can either fill the missing values with a specified value (like we've
seen in the past), or we can fill with the value from the previous observation
(a **forward fill**), or the value from the next observation (a **back fill**).

```python
by_day.assign(ffill=lambda df: df.coffee_consumption.ffill(),
              bfill=lambda df: df.coffee_consumption.bfill())
```

```python
df = df.fillna(0)
```

### Resampling

Resampling can be thought of as a sort of group by operation, but using a date
component to do the grouping. It is similar in functionality to the `.asfreq`
method, but allows more flexibility. Note that a resample only works on
dataframes with a datetime index.

Like a group by, once our data is resampled, we must specify an aggregation
operation.

For example, to get the average coffee consumption by week:

```python
df.resample('W').mean()
```

To get the total coffee consumption for each month:

```python
df.resample('M').sum()
```

TODO: insert table of common resample periods, example of, e.g. 2 weeks, or 5
months

### Rolling Windows

The `.rolling` method can be used to calculate rolling averages. A rolling
average lets us answer questions like: what was the average over the last 3
days? For every day in our dataset.

```python
rolling_df = df.resample('W').mean().assign(
    rolling_3=lambda df: df.coffee_consumption.rolling(3).mean(),
    rolling_5=lambda df: df.coffee_consumption.rolling(5).mean(),
    rolling_7=lambda df: df.coffee_consumption.rolling(7).mean(),
)
rolling_df.plot()
rolling_df.head(20)
```

We could also apply other aggregations with the .rolling method if we wanted:

```python
df.resample('W').mean().rolling(4).sum()
```

### Lagging and Lead

- `.shift`: move the data backwards and forwards by a given amount
- `.diff`: find the difference with the previous observation (or a specified
  further back observation)

```python
df['shift(-1)'] = df.coffee_consumption.shift(-1)
df['shift(1)'] = df.coffee_consumption.shift(1)
df['shift(3)'] = df.coffee_consumption.shift(3)
df['diff(1)'] = df.coffee_consumption.diff(1)
df['diff(3)'] = df.coffee_consumption.diff(3)
df
for col in ['shift(-1)', 'shift(1)', 'shift(3)', 'diff(1)', 'diff(3)']:
    del df[col]
```

## Timezones

1. working with timezones
    - add timezone -- `.tz_localize`
    - `.tz_localize(None)` to remove
    - convert to a diff `.tz_convert`
    - `s.tz is None`

## Days Since Now

How many days since today?

```python
(pd.to_datetime('now') - pd.date_range('20180101', freq='D', periods=365)) // pd.Timedelta('1d')
```

```python
weekday_order = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
(df
 # group by the first 3 letters of the weekday name, lowercased
 .groupby(df.index.weekday_name.str.lower().str[:3])
 .mean()
 # change the dates from an index into a regular column
 .reset_index()
 # turn the date column into a categorical type with a specified order
 .assign(date=lambda df: pd.Categorical(df.date, ordered=True, categories=weekday_order))
 # now that we have an ordered categorical, we can sort by it
 .sort_values(by='date')
 .set_index('date')
 .plot.bar())
```

## Exercises

For all of the datasets below, examine the data types of each column, ensure
that the dates are in the proper format, and set the dataframe's index to the
date column as appropriate.

For this exercise you'll need to install a library that will provide us access
to some more datasets:

```
pip install vega_datasets
```

You can use this library like so:

```python
from vega_datasets import data
data.sf_temps()
```

---

```python
from vega_datsets import data
data.sf_temps()
```

Use the above dataset for the exercises below:

1. Resample by the day and take the average temperature. Visualize the average
   temperature over time.
2. Write the code necessary to visualize the minimum temperature over time.
3. Write the code necessary to visualize the maximum temperature over time.
4. Which month is the coldest, on average?
5. Which month has the highest average temperature?
6. Resample by the day and calculate the min and max temp for the day (Hint:
   `.agg(['min', 'max'])`). Use this resampled dataframe to calculate the change
   in temperature for the day. Which month has the highest daily temperature
   variability?
7. **Bonus**: Visualize the daily min, average, and max temperature over time on a
   single line plot, i.e. the min, average, and maximum temperature should be 3
   seperate lines.

---

```python
from vega_datasets import data
data.seattle_weather()
```

Use the dataset to answer the following questions:

- Which year and month combination has the highest amount of precipitation?
- Visualize the amount of monthly precipitation over time.
- Visualize the amount of wind over time. Choose a time interval you think is
  appropriate.
- Which year-month combination is the windiest?
- What's the sunniest year? (Hint: which day has the highest number of days
  where weather == sun?)
- In which month does it rain the most?
- Which month has the most number of days with a non-zero amount of
  precipitation?

---

```python
data.flights_20k()
```

- Convert any negative delays to 0.
- Which hour of the day has the highest average delay?
- Does the day of the week make a difference in the delay amount?
- Does the month make a difference in the delay amount?

---

```python
from vega_datasets import data
data.iowa_electricity()
```

- For each row, calculate the percentage of the year's total that energy source
  provided.
- Lineplot of generation over time, color by source
- Display the data as table where years are columns, and energy source is rows
  (Hint: `df.pivot_table`)
- Make a line plot that shows the amount of energy generated over time. Each
  source should be a separate line?
- Is the total generation increasing over time?
    - How would you control for increasing consumption over time when
      considering how much energy is coming from each source?
    - express each number as a % of the year's total

---

1. Use the `sf_temps` dataset
1. Create 4 categories for temperature, cold, cool, warm, hot (hint: use
   `pd.cut` or `pd.qcut` for this)
1. How does the occurances of these 4 categories change month over month? i.e.
   how many days have each distinction? Visualize this and give the visual
   appropriate colors for each category.
1. Create pretty labels for time plots
1. Visualize the number of days of each month that fall into each bin by year
   (e.g. x=month, y=n_days, hue=temp_bin) or st similar

---

```python
df = data.birdstrikes()
```
