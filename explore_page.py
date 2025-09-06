import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def shorten_categories(categories, cutoff):
    categorical_map = {}

    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map


def clean_education(x):
    if "Bachelor" in x:
        return "Bachelor's degree"
    elif "Master" in x:
        return "Master's degree"
    elif "Professional" in x:
        return "Post grad"
    elif "Something else" or "without" in x:
        return "No degree"

    return "Elementry"


@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "CompTotal"]]
    df = df.rename({"CompTotal": "Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed, full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 200)
    df["Country"] = df["Country"].map(country_map)
    print(df.Country.value_counts())

    df = df[df["Salary"] <= 9000000]
    df = df[df["Salary"] >= 50000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(
        lambda x: 50 if x == "More than 50 years" else 0.5 if x == "Less than 1 year" else float(x)
    )
    df["EdLevel"] = df["EdLevel"].apply(clean_education)

    return df


df = load_data("./data/survey_results_public.csv")


def show_explore_page():
    st.title("Explore Software Engineer Salaries")
    st.write("Stack Overflow Developer Survey 2024")

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)

    st.write("""#### Mean Salary Based on Country""")

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Mean Salary Based on Experience""")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
