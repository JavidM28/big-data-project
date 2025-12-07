import os
import tempfile
import shutil
temp_dir = os.path.join(os.getcwd(), 'spark_temp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['SPARK_LOCAL_DIRS'] = temp_dir
os.environ['TMPDIR'] = temp_dir

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, count, when, avg, stddev, concat_ws
from pyspark.sql.functions import countDistinct, sum as spark_sum, max as spark_max, min as spark_min
import matplotlib.pyplot as plt
import seaborn as sns

# set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\nflight delay analysis using pyspark")
print("-" * 60)

# create spark session
spark = SparkSession.builder \
    .appName("Flight Delay Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# reduce spark logging noise
spark.sparkContext.setLogLevel("ERROR")

print("spark session created")

# load dataset
data_path = "Airlines.csv"
print(f"loading dataset: {data_path}")
df = spark.read.csv(data_path, header=True, inferSchema=True)

total_records = df.count()
print(f"total records: {total_records:,}\n")

# show sample
print("sample data:")
df.show(5)

# check for missing values
print("\nchecking for missing values...")
from pyspark.sql.functions import isnan, count as spark_count
missing_counts = df.select([spark_count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
missing_counts.show()

# feature engineering
print("\ncreating features...")

# time features
df = df.withColumn('Hour', (col('Time') / 100).cast('int'))
df = df.withColumn('TimeOfDay',
                   when(col('Hour').between(6, 11), 'Morning')
                   .when(col('Hour').between(12, 17), 'Afternoon')
                   .when(col('Hour').between(18, 21), 'Evening')
                   .otherwise('Night'))

# flight length categories
df = df.withColumn('FlightCategory',
                   when(col('Length') < 90, 'Short')
                   .when(col('Length').between(90, 180), 'Medium')
                   .when(col('Length').between(180, 300), 'Long')
                   .otherwise('VeryLong'))

# route identifier
df = df.withColumn('Route', concat_ws('-', col('AirportFrom'), col('AirportTo')))

print("features created: Hour, TimeOfDay, FlightCategory, Route")

# analysis 1: overall delay stats
print("\n" + "-" * 60)
print("analysis 1: overall delay statistics")
print("-" * 60)

delayed_count = df.filter(col('Delay') == 1).count()
on_time_count = df.filter(col('Delay') == 0).count()
delay_rate = (delayed_count / total_records) * 100

print(f"total flights: {total_records:,}")
print(f"delayed flights: {delayed_count:,}")
print(f"on-time flights: {on_time_count:,}")
print(f"delay rate: {delay_rate:.2f}%")

# analysis 2: airline performance
print("\n" + "-" * 60)
print("analysis 2: airline performance")
print("-" * 60)

airline_stats = df.groupBy('Airline').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate'),
    avg('Length').alias('AvgFlightLength')
).orderBy(col('DelayRate').desc())

print("\nairline delay statistics:")
airline_stats.show(20, truncate=False)

best_airline = airline_stats.orderBy('DelayRate').first()
worst_airline = airline_stats.orderBy(col('DelayRate').desc()).first()

print(f"best airline: {best_airline['Airline']} ({best_airline['DelayRate']:.2f}% delays)")
print(f"worst airline: {worst_airline['Airline']} ({worst_airline['DelayRate']:.2f}% delays)")

# analysis 3: time patterns
print("\n" + "-" * 60)
print("analysis 3: time patterns")
print("-" * 60)

# day of week
print("\ndelay by day of week:")
day_stats = df.groupBy('DayOfWeek').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).orderBy('DayOfWeek')
day_stats.show()

# time of day
print("\ndelay by time of day:")
time_stats = df.groupBy('TimeOfDay').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).orderBy(col('DelayRate').desc())
time_stats.show()

# hour
print("\ndelay by hour:")
hour_stats = df.groupBy('Hour').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).orderBy('Hour')
hour_stats.show(24)

# analysis 4: airports
print("\n" + "-" * 60)
print("analysis 4: airport performance")
print("-" * 60)

# departure airports
print("\ntop 15 departure airports by delay rate:")
departure_stats = df.groupBy('AirportFrom').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).filter(col('TotalFlights') > 1000).orderBy(col('DelayRate').desc())
departure_stats.show(15, truncate=False)

# arrival airports
print("\ntop 15 arrival airports by delay rate:")
arrival_stats = df.groupBy('AirportTo').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).filter(col('TotalFlights') > 1000).orderBy(col('DelayRate').desc())
arrival_stats.show(15, truncate=False)

# analysis 5: routes
print("\n" + "-" * 60)
print("analysis 5: route analysis")
print("-" * 60)

print("\ntop 20 routes with highest delays:")
route_stats = df.groupBy('AirportFrom', 'AirportTo').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate'),
    avg('Length').alias('AvgLength')
).filter(col('TotalFlights') > 100).orderBy(col('DelayRate').desc())
route_stats.show(20, truncate=False)

# analysis 6: flight length
print("\n" + "-" * 60)
print("analysis 6: flight length impact")
print("-" * 60)

length_stats = df.groupBy('FlightCategory').agg(
    count('*').alias('TotalFlights'),
    spark_sum('Delay').alias('DelayedFlights'),
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate'),
    avg('Length').alias('AvgLength')
).orderBy(col('DelayRate').desc())

print("\ndelay by flight length:")
length_stats.show()

# correlations
print("\n" + "-" * 60)
print("correlation analysis")
print("-" * 60)
print(f"flight length vs delay: {df.stat.corr('Length', 'Delay'):.4f}")
print(f"day of week vs delay: {df.stat.corr('DayOfWeek', 'Delay'):.4f}")
print(f"hour vs delay: {df.stat.corr('Hour', 'Delay'):.4f}")

# viz 1: delay distribution
print("creating: 1_delay_distribution.png")
delay_counts = df.groupBy('Delay').count().toPandas()
plt.figure(figsize=(8, 8))
colors = ['#2ecc71', '#e74c3c']
labels = ['On-Time', 'Delayed']
plt.pie(delay_counts['count'], labels=labels, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 14})
plt.title('Flight Delay Distribution', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('1_delay_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 2: airline comparison
print("creating: 2_airline_comparison.png")
airline_plot_data = airline_stats.orderBy(col('DelayRate').desc()).limit(15).toPandas()
plt.figure(figsize=(12, 8))
bars = plt.barh(airline_plot_data['Airline'], airline_plot_data['DelayRate'], color='steelblue')
bars[0].set_color('#e74c3c')
plt.xlabel('Delay Rate (%)', fontsize=14, fontweight='bold')
plt.ylabel('Airline', fontsize=14, fontweight='bold')
plt.title('Top 15 Airlines by Delay Rate', fontsize=18, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
for i, v in enumerate(airline_plot_data['DelayRate']):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('2_airline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 3: day of week
print("creating: 3_day_of_week_pattern.png")
day_plot_data = day_stats.toPandas()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.figure(figsize=(12, 6))
plt.plot(day_plot_data['DayOfWeek'], day_plot_data['DelayRate'], 
         marker='o', linewidth=3, markersize=10, color='#e74c3c')
plt.fill_between(day_plot_data['DayOfWeek'], day_plot_data['DelayRate'], alpha=0.3, color='#e74c3c')
plt.xlabel('Day of Week', fontsize=14, fontweight='bold')
plt.ylabel('Delay Rate (%)', fontsize=14, fontweight='bold')
plt.title('Delay Patterns by Day of Week', fontsize=18, fontweight='bold', pad=20)
plt.xticks(range(1, 8), days)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('3_day_of_week_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 4: hourly pattern
print("creating: 4_hourly_pattern.png")
hour_plot_data = hour_stats.toPandas()
plt.figure(figsize=(14, 6))
bars = plt.bar(hour_plot_data['Hour'], hour_plot_data['DelayRate'], 
               color='#3498db', edgecolor='black', linewidth=1.2)
max_delay_hour = hour_plot_data.loc[hour_plot_data['DelayRate'].idxmax(), 'Hour']
for bar in bars:
    if bar.get_x() == max_delay_hour:
        bar.set_color('#e74c3c')
plt.xlabel('Hour of Day (24-hour format)', fontsize=14, fontweight='bold')
plt.ylabel('Delay Rate (%)', fontsize=14, fontweight='bold')
plt.title('Delay Patterns by Hour of Day', fontsize=18, fontweight='bold', pad=20)
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('4_hourly_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 5: time of day
print("creating: 5_time_of_day.png")
time_plot_data = time_stats.toPandas()
plt.figure(figsize=(10, 6))
colors_time = ['#f39c12', '#3498db', '#9b59b6', '#2c3e50']
bars = plt.bar(time_plot_data['TimeOfDay'], time_plot_data['DelayRate'], 
               color=colors_time, edgecolor='black', linewidth=1.5)
plt.ylabel('Delay Rate (%)', fontsize=14, fontweight='bold')
plt.xlabel('Time of Day', fontsize=14, fontweight='bold')
plt.title('Delay Rates by Time of Day', fontsize=18, fontweight='bold', pad=20)
for i, v in enumerate(time_plot_data['DelayRate']):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('5_time_of_day.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 6: airports
print("creating: 6_airport_analysis.png")
top_departure = departure_stats.limit(10).toPandas()
top_arrival = arrival_stats.limit(10).toPandas()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.barh(top_departure['AirportFrom'], top_departure['DelayRate'], color='#e67e22')
ax1.set_xlabel('Delay Rate (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Departure Airport', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Departure Airports by Delay Rate', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax2.barh(top_arrival['AirportTo'], top_arrival['DelayRate'], color='#16a085')
ax2.set_xlabel('Delay Rate (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Arrival Airport', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Arrival Airports by Delay Rate', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
plt.tight_layout()
plt.savefig('6_airport_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 7: flight length
print("creating: 7_flight_length_impact.png")
length_plot_data = length_stats.orderBy('AvgLength').toPandas()
fig, ax1 = plt.subplots(figsize=(12, 6))
x = range(len(length_plot_data))
ax1.bar(x, length_plot_data['DelayRate'], color='#c0392b', alpha=0.7, label='Delay Rate')
ax1.set_ylabel('Delay Rate (%)', fontsize=14, fontweight='bold', color='#c0392b')
ax1.tick_params(axis='y', labelcolor='#c0392b')
ax1.set_xlabel('Flight Category', fontsize=14, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(x, length_plot_data['AvgLength'], color='#2980b9', marker='o', 
         linewidth=3, markersize=10, label='Avg Length')
ax2.set_ylabel('Average Flight Length (minutes)', fontsize=14, fontweight='bold', color='#2980b9')
ax2.tick_params(axis='y', labelcolor='#2980b9')
plt.title('Flight Length Impact on Delays', fontsize=18, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(length_plot_data['FlightCategory'])
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('7_flight_length_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# viz 8: heatmap
print("creating: 8_delay_heatmap.png")
heatmap_data = df.groupBy('DayOfWeek', 'Hour').agg(
    (spark_sum('Delay') / count('*') * 100).alias('DelayRate')
).toPandas()
heatmap_pivot = heatmap_data.pivot(index='Hour', columns='DayOfWeek', values='DelayRate')
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Delay Rate (%)'}, linewidths=0.5)
plt.title('Delay Heatmap: Day of Week vs Hour', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Day of Week', fontsize=14, fontweight='bold')
plt.ylabel('Hour of Day', fontsize=14, fontweight='bold')
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.xticks(range(len(day_labels)), day_labels)
plt.tight_layout()
plt.savefig('8_delay_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nall visualizations created successfully!")

# summary
print("\n" + "-" * 60)
print("analysis summary")
print("-" * 60)

summary_report = f"""
flight delay analysis summary

dataset:
- total flights: {total_records:,}
- airlines: {airline_stats.count()}
- routes: {route_stats.count()}

overall stats:
- delayed flights: {delayed_count:,} ({delay_rate:.2f}%)
- on-time flights: {on_time_count:,} ({100-delay_rate:.2f}%)

key findings:
- best airline: {best_airline['Airline']} ({best_airline['DelayRate']:.2f}% delays)
- worst airline: {worst_airline['Airline']} ({worst_airline['DelayRate']:.2f}% delays)
- most delays: {time_stats.first()['TimeOfDay']} ({time_stats.first()['DelayRate']:.2f}%)
- least delays: {time_stats.orderBy('DelayRate').first()['TimeOfDay']} ({time_stats.orderBy('DelayRate').first()['DelayRate']:.2f}%)

visualizations created:
1. delay_distribution.png
2. airline_comparison.png
3. day_of_week_pattern.png
4. hourly_pattern.png
5. time_of_day.png
6. airport_analysis.png
7. flight_length_impact.png
8. delay_heatmap.png
"""

print(summary_report)

# save summary
with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print("summary saved to: analysis_summary.txt")

# stop spark
spark.stop()
