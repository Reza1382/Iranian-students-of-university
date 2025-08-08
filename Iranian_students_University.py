# %% [Markdown]
# ## This file examines the population of Iranian students at various levels of education.
# %%
#: Import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import shapiro, levene, pearsonr, spearmanr

# %%
#: Load the dataset
df = pd.read_excel("Iranian students of University.xlsx")
print(df.head())

# %%
print(df.columns)

# %%
#: Changing names of columns
df.columns = ["year", "sum", "Pdiploma", "Bachelor", "MA", "Doctor"]

# %%
#: Change 'year' column by better format.
df["year"] = [
    "1386-1387",
    "1387-1388",
    "1388-1389",
    "1389-1390",
    "1390-1391",
    "1391-1392",
    "1392-1393",
    "1393-1394",
    "1394-1395",
    "1395-1396",
    "1396-1397",
]

df.index = df["year"]
del df["year"]

# %%
#: Data information
print(df.info())

# %%
#: plotting def


def bar_plot(data, title, X_label, Y_label):
    """
    data : The data set that had one row and one column
    title : name of plot
    X_label : name of 'x' axis
    Y_label : name of 'y' axis
    """
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(rotation=60)
    sns.set_style("darkgrid")
    sns.barplot(
        x=data.index,
        y=data.values,
        palette="Set2",
        width=0.8,
    )
    plt.tight_layout()
    return plt.show()


def line_plot(data, title, X_label, Y_label):
    """
    data : The data set that had one row and one column
    title : name of plot
    X_label : name of 'x' axis
    Y_label : name of 'y' axis
    """
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(rotation=60)
    sns.set_style("darkgrid")
    sns.lineplot(x=data.index, y=data.values, color="b")
    plt.tight_layout()
    return plt.show()


def scatter_plot(data, title, X_label, Y_label):
    """
    data : The data set that had one row and one column
    title : name of plot
    X_label : name of 'x' axis
    Y_label : name of 'y' axis
    """
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(rotation=60)
    sns.set_style("darkgrid")
    sns.scatterplot(x=data.index, y=data.values, color="b")
    plt.tight_layout()
    return plt.show()


def hist_plot(data, title, X_label, Y_label, bins=5):
    """
    data : The data set that had one row and one column
    title : name of plot
    X_label : name of 'x' axis
    Y_label : name of 'y' axis
    bins : distance of each band
    """
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.xticks(rotation=60)
    sns.set_style("darkgrid")
    sns.histplot(x=data.index, y=data.values, bins=bins, kde=True)
    plt.tight_layout()
    return plt.show()


# %%
#: Line plot of any grade
feature = ["Pdiploma", "Bachelor", "MA", "Doctor"]
for i in feature:
    line_plot(
        df[i],
        f"Number of {i} students of university",
        "Year",
        "Number of students",
    )

# %%
#: Yearly Growth Rate of Students by Grade

df_growth = df[["Pdiploma", "Bachelor", "MA", "Doctor"]].pct_change() * 100
df_growth.plot(kind="bar", figsize=(10, 6), colormap="Set2")
plt.title("Yearly Growth Rate of Students by Grade")
plt.ylabel("Growth Rate (%)")
plt.xlabel("Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
#: Yearly Growth Rate of Students at all

df_sum_growth = df["sum"].pct_change() * 100
bar_plot(
    df_sum_growth,
    "Yearly Growth Rate of Students at all",
    "Year",
    " Growth Rate (%)",
)
# %%
#: Add percent of any grade at all
for i in feature:
    df[f"{i}/sum"] = (df[i] * 100) / df["sum"]

#: Plot a stacked area chart showing how the percentage of students in each grade level changes over time.
df_percent = df[[f"{i}/sum" for i in feature]]
df_percent.plot.area(figsize=(10, 6), colormap="Set2", alpha=0.8)
plt.title("Percent of students per grade over years")
plt.ylabel("Percentage (%)")
plt.xlabel("Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
#: Visualize the percentage share of each education level over the years using a heatmap.
#: This highlights patterns and fluctuations in student distribution across time.

sns.heatmap(df_percent.T, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Percentage of each grade by year")
plt.show()

# %%
#: LinearRegression model for sum of students
x = np.array([range(1386, 1397)]).reshape(-1, 1)
y = df["sum"]

Ir = LinearRegression()
model = Ir.fit(x, y)
y_pred = model.predict(x)

# %%
#: Scatter plot of real data and line plot of Regression model

plt.figure(figsize=(6, 4))
plt.title("Predict model")
plt.xlabel("Year")
plt.ylabel("Number of students")
plt.xticks(rotation=60)
sns.set_style("darkgrid")
sns.scatterplot(data=df, x=df.index, y=df["sum"].values, label="Origin data")
sns.lineplot(x=df.index, y=y_pred, color="r", label="Predict model")
plt.tight_layout()
plt.legend()
plt.show()
# %%
# ## [Markdown]
# > Linear Regression cannot predict effectively.

# %%
#: Correlation between columns elements
sns.heatmap(df[["Pdiploma", "Bachelor", "MA", "Doctor"]].corr(), annot=True)

# %%
# ## [Markdown]
# > There is a significant relationship between the number of master's and doctoral students.
# > There is a significant relationship between the number of associate and bachelor's degree students.

# %%
#: Vibration in recent year in any grade
bar_plot(
    df_growth.std(), "Vibration in recent year in any grade", "Grade", "Scattering Rate"
)
# %%
# ## [Markdown]
#: Most vibration was in 'Pdiploma' & 'MA' grades.

# %%
#: Normality test for data
print("\n" + "=" * 40)
print("Normality Test (Shapiro-Wilk Test)")
print("\n" + "=" * 40)


def test_normality(data, data_column):
    stat, p_value = shapiro(data)
    print(f"** {data_column} **")
    print(f"آماره : {stat:.4f}")
    print(f"p_value : {p_value:.2f}")
    if p_value > 0.05:
        print(f"The data is Normal. (p > 0.05) \n")
    else:
        print(f"The data isn't Normal. (p ≤ 0.05) \n")


for col in feature:
    test_normality(df[col], col)


# %%
#: LinearRegression model for MA students that continue by Doctor
x = np.array(df["MA"]).reshape(-1, 1)
y = df["Doctor"]

Ir2 = LinearRegression()
model2 = Ir2.fit(x, y)
y_pred2 = model2.predict(x)

# %%
#: Scatter plot of real data and line plot of Regression model

plt.figure(figsize=(6, 4))
plt.title("Predict model")
plt.xlabel("Number of MA students")
plt.ylabel("Number of Doctor students")
plt.xticks(rotation=60)
sns.set_style("darkgrid")
sns.scatterplot(data=df, x="MA", y="Doctor", label="Origin data")
sns.lineplot(x=df["MA"], y=y_pred2, color="r", label="Predict model")
plt.tight_layout()
plt.legend()
plt.show()

# %%
# ## [Markdown]
# > Linear Regression predict but not effectively.
# %%
