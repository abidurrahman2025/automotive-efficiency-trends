# Assessing Automotive Efficiency: Technology Gains vs. Vehicle Size (1975–2025)
In previous modules, I have worked with datasets. This time I wanted a larger dataset with more than thirty years of data. I wanted to analyze whether the shift toward larger vehicles eroded the efficiency gains made by engine technology from the EPA vehicle dataset at fueleconomy.gov to negate efficiency gains. I chose the unzipped vehicles.csv containing all models and years.

Link to the dataset: https://www.fueleconomy.gov/feg/epadata/vehicles.csv

### Initial Data Cleaning and Preprocessing

I loaded the pandas, numpy, matplotlib, and seaborn libraries and imported the raw vehicle dataset from the EPA, using the `pd.read_csv` function to consolidate messy sub-categories into three clean buckets: SUV, Car, and Truck. To keep the load robust against mixed-type columns, I passed `low_memory=False`. I then filtered out missing values from critical variables like year, displacement, and combined MPG, while also categorizing transmissions into Automatic and Manual groups. Rather than looping through each row, I used NumPy's vectorized `np.select` along with pandas string `.str.contains()` to categorize the vehicle class and drivetrain in a single, efficient pass. To support the visual analysis, I calculated mean MPG and engine displacement by year and vehicle class to establish historical trends and I specifically reordered the vehicle class levels based on average efficiency. This reordering ensures that legends and plot elements appear in a logical order (from most efficient to least efficient) rather than just alphabetically, which reduces cognitive load and makes the comparison of different vehicle types more intuitive for the reader.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
# Load the dataset
vehicles = pd.read_csv(
    "https://www.fueleconomy.gov/feg/epadata/vehicles.csv", low_memory=False
)

# DATA CLEANING & RE-CATEGORIZATION
cols = ["make", "model", "year", "VClass", "cylinders", "displ", "drive",
        "trany", "comb08", "highway08", "city08", "fuelType1"]
df = vehicles[cols].copy()

# Remove rows with missing critical values
df = df.dropna(subset=["year", "comb08", "displ", "trany"])

vclass = df["VClass"].fillna("")
trany = df["trany"].fillna("")

df["vehicle_class"] = np.select(
    [
        vclass.str.contains(r"SUV|Sport Utility", case=False, regex=True),
        vclass.str.contains(r"Wagon|Sedan|Passenger Car", case=False, regex=True),
        vclass.str.contains(r"Truck|Pickup", case=False, regex=True),
    ],
    ["SUV", "Car", "Truck"],
    default="Other",
)

df["drivetrain"] = np.select(
    [
        trany.str.contains("Auto", case=False, regex=False),
        trany.str.contains("Manual", case=False, regex=False),
    ],
    ["Automatic", "Manual"],
    default="Other",
)

# Keep only the primary classes for the analysis
df = df[df["vehicle_class"].isin(["SUV", "Car", "Truck"])
        & df["drivetrain"].isin(["Automatic", "Manual"])].copy()

# AGGREGATION FOR TREND LINES
trend = (
    df.groupby(["year", "vehicle_class"], as_index=False)
      .agg(avg_mpg=("comb08", "mean"), avg_displ=("displ", "mean"), n=("comb08", "size"))
)
avg_all = df.groupby("year", as_index=False).agg(avg_mpg_all=("comb08", "mean"))

# ORGANIZE PLOTTING ORDER (classes sorted by mean of yearly average MPG, descending)
class_order = (
    trend.groupby("vehicle_class")["avg_mpg"].mean()
         .sort_values(ascending=False).index.tolist()
)
trend["vehicle_class"] = pd.Categorical(trend["vehicle_class"], categories=class_order)
df["vehicle_class"] = pd.Categorical(df["vehicle_class"], categories=class_order)
```

### Analysis of Visualizations

The following charts answer my research question: Did the shift toward larger vehicles negate efficiency gains?

#### Plot 1: Fuel Economy Distribution

I generated the density plot of MPG by mapping combined MPG to the x-axis and filling the area by vehicle class to show the relative spread of efficiency across the fleet. I utilized transparency (alpha = 0.22) and thin colored outlines to allow the overlapping distributions of Cars, SUVs, and Trucks to be visually distinguishable. By applying `sns.kdeplot`, minor grid lines were automatically managed to strip away minor grid lines, maximizing the data-ink ratio and focused the viewer’s attention on the peaks and tails of each distribution. I also moved the legend to sit horizontally above the plot area with padding on the title, so it does not obscure any of the density curves. This plot effectively visualizes the efficiency ceiling for each class, highlighting how Cars consistently peak at higher MPG values compared to the more concentrated, lower-efficiency peaks of Trucks.

```python
# Plot 1: Density plot of MPG
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.kdeplot(
    data=df, x="comb08", hue="vehicle_class", hue_order=class_order,
    fill=True, alpha=0.22, linewidth=0.8, common_norm=False, ax=ax1,
)
ax1.set(xlabel="Combined MPG", ylabel="Density")
ax1.set_title("MPG Distribution by Class", pad=40, fontweight="bold")
ax1.grid(False, axis="x")
sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, 0.99), ncol=len(class_order), title=None)
fig1.tight_layout()
plt.show()
```

**[Image: MPG Distribution by Class]**
<img width="784" height="484" alt="plot1" src="https://github.com/user-attachments/assets/f44bd9af-de73-4040-99fd-3fc0ce9d6e0f" />

This plot shows the spread of fuel efficiency across classes. It demonstrates the efficiency ceiling that SUVs face compared to smaller cars. Trucks in general have the lowest MPG out of the three, with a large majority being around 13 to 18 MPG. Cars have the highest MPG out of the three, with a peak clearly above 20 MPG and most vehicles residing in the higher-efficiency range. SUVs had the most spread curve. There are two peaks of similar heights, and they bridge the gap between Cars and Trucks. This shows that SUVs have a wide range when it comes to efficiency. The degree of overlap between the Car and SUV curves is a critical indicator of technological convergence; the more the SUV curve reaches into the Car’s territory, the more engineering has compensated for the SUV’s inherent aerodynamic and weight disadvantages.

#### Plot 2: Vehicle Mix Shifts

I constructed the stacked bar chart by first grouping the data into distinct model year periods using the `pd.cut` function and calculating the proportional share of each vehicle class within those timeframes. I reindexed the resulting pivot table to the previously defined `class_order` so that the stack segments always appear in the same order as the other charts. I used pandas' built-in stacked bar plotting with a fixed width to show the evolving composition of the automotive market over the last five decades. To prevent the legend from obscuring the chart title, I moved it into a single horizontal row above the plot area and added padding to the title. The y-axis was scaled to represent the share of vehicles rather than raw counts, allowing for a clear comparison of market dominance regardless of total annual production. This visualization captures the dramatic expansion of the SUV segment and the corresponding contraction of the traditional Car segment, setting the stage for the analysis of compositional drag on fleet-wide efficiency.

```python
# Plot 2: Stacked share of vehicles by class and model-year period
df_year = df.copy()
df_year["period"] = pd.cut(
    df_year["year"],
    bins=[1974, 1985, 1995, 2005, 2015, 2026],
    labels=["1975-1985", "1986-1995", "1996-2005", "2006-2015", "2016-2025"],
    include_lowest=True,
    right=True,
)

count_tbl = (
    df_year.groupby(["period", "vehicle_class"], observed=True)
           .size().rename("n").reset_index()
)
count_tbl["share"] = count_tbl["n"] / count_tbl.groupby("period", observed=True)["n"].transform("sum")

share_wide = (
    count_tbl.pivot(index="period", columns="vehicle_class", values="share")
             .reindex(columns=class_order).fillna(0)
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
share_wide.plot(kind="bar", stacked=True, width=0.75, ax=ax2)
ax2.set(xlabel="Model Year Period", ylabel="Share of Vehicles")
ax2.set_title("Vehicle Shifts by Classes", pad=60, fontweight="bold")
ax2.tick_params(axis="x", rotation=0)
ax2.grid(False, axis="x")
ax2.legend(title="Class", loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=len(class_order))
fig2.tight_layout()
plt.show()
```

**[Image: Vehicle Shifts by Classes]**
<img width="784" height="484" alt="plot2" src="https://github.com/user-attachments/assets/b7eba1e8-aadd-4aea-a6a5-b5eeebd4f4da" />


This plot shows the change in market share from 1975 to 2025. It shows that even if every individual SUV became more efficient over time, the sheer volume of SUVs replacing Cars creates a drag on the total fleet average. Historically, the share of SUVs was much lower than cars. However, by 2025, the data shows that SUVs have grown to dominate a massive portion of the market, while the share of traditional small cars has shrunk. From 1996 to 2025, the use of SUVs increased dramatically, overtaking cars as the primary vehicle choice and shifting the entire composition of the American fleet toward larger frames.

#### Plot 3: The Efficiency Trade-off

I produced the faceted scatter plot by plotting engine displacement against combined MPG, applying alpha transparency to the points to reveal the underlying density of the massive dataset and avoid overplotting. I layered a linear regression model over the points using `sns.lmplot` to mathematically illustrate the inverse relationship between engine size and fuel economy for each category. I disabled the confidence interval (`ci=None`) so the trend line itself stands out cleanly without a shaded band that would compete with the dense point cloud. By using `col='vehicle_class'` and forcing `sharex=True` and `sharey=True` in the facet grid, I separated the three vehicle classes into individual panels while maintaining identical scales on both axes. This design follows the principle that valid comparisons are only possible when the visual framework remains constant, allowing the audience to see that even small-displacement SUVs often struggle to match the efficiency of larger-displacement cars.

```python
# Plot 3: The Efficiency Trade-off (scatter + linear fit, faceted by class)
g = sns.lmplot(
    data=df, x="displ", y="comb08", col="vehicle_class", col_order=class_order,
    hue="vehicle_class", hue_order=class_order, ci=None, height=4, aspect=1,
    scatter_kws={"alpha": 0.15, "s": 10}, line_kws={"linewidth": 1.8},
    facet_kws={"sharex": True, "sharey": True},
)
g.set_titles("{col_name}")
g.set_axis_labels("Engine Displacement (L)", "Combined MPG")
g.figure.suptitle("Displacement vs. MPG\nHow engine size affects efficiency by class", y=0.99)
g.figure.subplots_adjust(top=0.78)
plt.show()
```

**[Image: Displacement vs. MPG]**
<img width="1184" height="398" alt="plot3" src="https://github.com/user-attachments/assets/0fab7be7-1e2e-4a51-acb7-4bcb790a88e8" />


This plot shows the relationship between engine size (displacement) and MPG, separated by class. The downward slopes (regression lines) prove that engine size remains a primary driver of consumption. By faceting, the chart shows that even if a small SUV engine often performs worse than a more efficient car engine, it illustrates the structural environmental cost of vehicle size. The slope is the steepest for Cars, meaning that as cars engines get smaller, they see a more dramatic jump in MPG compared to the heavier classes. This proves that weight and aerodynamics set a floor for how efficient a large vehicle can get, especially an SUV or Truck.

#### Plot 4: Average Combined MPG Over Time (Gasoline Only)

I constructed the final line graph by filtering the dataset to isolate pure internal combustion progress, specifically selecting regular gasoline, premium gasoline, and diesel while excluding high-MPG electric and hybrid vehicles. I mapped the year to the x-axis and the calculated average MPG to the y-axis, connecting the points with a distinct red line to signify a specific subset of the data. I manually defined the x-axis breaks and y-axis limits to provide a stable reference point for observing long-term technological growth without the skew of electric powertrains. I left-aligned the title and folded the subtitle into the same title block so the explanation of the exclusion criteria sits directly with the chart instead of floating separately. This plot isolates the impact of traditional engine advancements, such as turbocharging and variable valve timing, proving that the internal combustion engine has significantly improved despite the market's shift toward larger vehicle classes.

```python
# Gasoline/Diesel only, to see "pure" engine improvements
avg_gasoline = (
    df[df["fuelType1"].isin(["Regular Gasoline", "Premium Gasoline", "Diesel"])]
      .groupby("year", as_index=False)
      .agg(avg_mpg_gas=("comb08", "mean"))
)

# Plot 4: Gasoline-only average combined MPG over time
fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.plot(avg_gasoline["year"], avg_gasoline["avg_mpg_gas"],
         color="#E41A1C", linewidth=2, marker="o", markersize=4)
ax4.set_xticks(np.arange(1984, 2027, 5))
ax4.set_ylim(15, 30)
ax4.set_title("Pure Engine Efficiency Trends (Gasoline/Diesel Only)\n"
              "Excludes EVs and PHEVs to show internal combustion progress",
              loc="left", fontsize=12)
ax4.set(xlabel="Model Year", ylabel="Average Combined MPG")
fig4.tight_layout()
plt.show()
```

**[Image: Efficiency Trends (Gasoline/Diesel Only)]**
<img width="884" height="484" alt="plot4" src="https://github.com/user-attachments/assets/009c98e2-a39d-4097-b34b-414b0132beed" />


This line graph shows a relatively stagnant or slowly rising line through the 1990s, followed by a steady upward inflection point starting around 2005. By removing EVs, we see the pure progress of the gasoline engine. The slope is less aggressive than the all-vehicle curve, but it shows that the period from 2005 to 2025 saw the most significant gains in gasoline engine efficiency. The line clearly shows that manufacturers have optimized engines to be more efficient despite the drag created by the market shift seen in Plot 2. The line continues to climb, marking the industry's overall progress toward better efficiency despite challenging consumer tastes.

### Connecting to the Research Question

These plots suggest that the shift to larger vehicles was diluted but did not negate efficiency gains. Plot 2 and Plot 3 show that larger vehicles and bigger engines naturally return lower MPG. If the gains were truly negated, the line in Plot 4 would be flat or falling as the share of heavier vehicles increased. Instead, Plot 4 shows a clear upward trend even when filtered for gasoline only. This proves that pure engine and transmission improvements have been enough to overcome the SUV drag. While EVs certainly accelerate the trend, the internal combustion engine has evolved to become much more productive per gallon of fuel than it was thirty years ago.

### My Design Choice Reasons

I used `sns.set_theme(style="whitegrid")` to establish a consistent visual baseline across all four plots. I disabled the vertical gridlines on the first two plots so that the ink on the page is dedicated almost entirely to the data rather than competing with the category boundaries. I used aligned axes in the faceted scatter plots that allow the reader to compare the slopes of the lines across the three categories. This follows the "small multiples" design principle, which states that comparisons are only possible when scales are identical. I categorized into broad, understandable groups following years of approach to make data comprehensible to a general audience. The charts use color as a categorical identifier consistently across all four plots. If SUVs are a specific color in Plot 1, I retain that same color in Plot 2 and Plot 3, reducing the cognitive load on the reader. Finally, I placed all legends in a single horizontal row above the plotting area with additional title padding, which prevents the legends from ever competing with the data or overlapping the titles.
