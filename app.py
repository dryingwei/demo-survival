import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()


# 获取 API 密钥
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = api_key
)

file_path = "data/info.tsv"

# 页面标题
st.title("Demo for Survival Analysis")

# 在侧边栏中添加组件
st.sidebar.header("Options")
 
km_group_selection = st.sidebar.selectbox("Select an option for group in Kaplan-Meier Curve", 
                                          ["none","sex", "tumor_stage_category", "alcohol_category"])

# st.write("# options for COX models")
# slider_value = st.sidebar.slider("range of scrolling block", min_value=0, max_value=100, value=50)

# 主界面组件
# st.header("main interface")
# name = st.text_input("input your name：")
# if st.button("submit"):
#     st.write(f"Hello, {name}")
#     st.write(f"You choose: {selection}")
#     st.write(f"The current value is : {slider_value}")



def plot_statistics(meta_df):
    import matplotlib.pyplot as plt
    # Next Step:
    # 1) define status_counts as {prev}
    # 2) build this table (labels: living, deceased)
    # 3) set rotation to 0 and only show the y-axis grid
    # 4) show it
    status_counts = meta_df['vital_status'].value_counts()

    fig, ax = plt.subplots()
    labels = ['living','deceased']

    for i, (label, count) in enumerate(zip(labels, status_counts)):
        ax.bar(label, count, label = f'{label}, {count}',
            color = ['skyblue', 'lightcoral'][i], width = 0.5) # This loop is difficult.
        
    plt.title('Patient Count by Vitual Status')
    plt.xlabel('Vitual Status')
    plt.ylabel('Number of Patients')

    plt.xticks(rotation=0)
    plt.grid(axis = 'y') 
    ax.legend()  # easier for viewers to identify
    return fig

def km_preprocessin(meta_df):
        # preprocssing
    meta_df2 = meta_df[meta_df['vital_status'].notnull()]
    meta_df2 = meta_df2[pd.notna(meta_df['follow_up_days'])] 

    # change the deceased to true, the living to false
    meta_df2['vital_status'] =  meta_df2['vital_status'].map(lambda x: True if x == 'Deceased' else False)

    return meta_df2


def plot_km(meta_df2):
    import matplotlib.pyplot as plt
    from sksurv.nonparametric import kaplan_meier_estimator

    # plt.figure(figsize=(6,6))
    fig,ax = plt.subplots(figsize=(6,6))

    time, survival_prob, conf_int = kaplan_meier_estimator(
        meta_df2['vital_status'], meta_df2['follow_up_days'], conf_type = "log-log"
    )
    plt.step(time, survival_prob, where = "post", label=f'Overall Survival (n={meta_df2.shape[0]})') 
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0,1)


    plt.title("Kaplan Meier Curve")
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel(r"time $t$")
    plt.legend()

    # plt.savefig(out_fig_KMC_path, bbox_inches = "tight")
    return fig



def plot_kmc2(i, df):
    
    from sksurv.nonparametric import kaplan_meier_estimator
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    # if i not in ["None", "Sex"]:
        # return None
    
    df = df[df[i].notnull()]
    values = df[i].unique()

    survival_data = {}

    fig, ax = plt.subplots(figsize=(6, 6))

    for value in values:
        mask = df[i] == value
        time, survival_prob, conf_int = kaplan_meier_estimator(
            df["vital_status"][mask],
            df["follow_up_days"][mask],
            conf_type="log-log",  
        )

        survival_data[value] = (time, survival_prob)

        # plot the Kaplan-Meier curve
        plt.step(time, survival_prob, where="post",label=f"{value} (n = {mask.sum()})")
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")

    if len(values) == 2:
        results = logrank_test(df["follow_up_days"][df[i] == values[0]],
                               df["follow_up_days"][df[i] == values[1]],
                               event_observed_A=df["vital_status"][df[i] == values[0]],
                               event_observed_B=df["vital_status"][df[i] == values[1]]
                               )
        p_value = results.p_value
        plt.title(f"Kaplan Meier Curve (p-value = {p_value:.4f})")
    else:
        plt.title('Kaplan Meier Curve')

    plt.ylim(0,1)
    plt.ylabel(r"Estimated probability of survival $\hat{S}(t)$")
    plt.xlabel("Time $t$")
    plt.legend(loc="best")
    # plt.show()
    return fig

def categorize_stage(stage):
    if stage in ['Stage IA', 'Stage IB','Stage IIA', 'Stage IIB']:
        return 'stageI/II'
    elif stage in ['Stage III', 'Stage IV']:
        return 'stageIII/IV'
    else:
        return None  # 如果有其他值可以作为“未知”处理

wd = r"C:\Users\Jing\Desktop\Tianyuan\homework\PDAC"
clinical_path = os.path.join(wd, "meta/pdac_sup_1.xlsx")
clinical_df = pd.read_excel(clinical_path, sheet_name = "Clinical_data")
meta_df = clinical_df[clinical_df["histology_diagnosis"] == "PDAC"]
meta_df = meta_df[(meta_df["cause_of_death"] == "pancreatic carcinoma") | (meta_df["cause_of_death"] == "na")]
selected_columns = ['case_id', 
    'age', 
    'sex', 
    'tumor_site', 
    'tumor_focality', 
    'tumor_size_cm', 
    'tumor_necrosis', 
    'lymph_vascular_invasion', 
    'perineural_invasion', 
    'number_of_lymph_nodes_examined', 
    'number_of_lymph_nodes_positive_for_tumor', 
    'tumor_stage_pathological', 
    'bmi', 
    'alcohol_consumption', 
    'tobacco_smoking_history', 
    'follow_up_days', 
    'vital_status']

meta_df = meta_df[selected_columns]
meta_df2 = meta_df[meta_df['vital_status'].notnull()]
meta_df2 = meta_df2[pd.notna(meta_df['follow_up_days'])] 
meta_df2['vital_status'] =  meta_df2['vital_status'].map(lambda x: True if x == 'Deceased' else False)

# 使用apply函数将tumor_stage_pathological列重新分类，并生成一个新的列 'tumor_stage_category'
meta_df2['tumor_stage_category'] = meta_df2['tumor_stage_pathological'].apply(categorize_stage)

meta_df2['alcohol_consumption'].unique()

def categorize_alcohol_consumption(consumption):
    if consumption in [
        'Alcohol consumption equal to or less than 2 drinks per day for men and 1 drink or less per day for women',
        'Alcohol consumption more than 2 drinks per day for men and more than 1 drink per day for women',
        'Consumed alcohol in the past, but currently a non-drinker']:
        return 'drinker'
    elif consumption in ['Lifelong non-drinker']:
        return 'non-drinker'
    else:
        return None
meta_df2['alcohol_category'] = meta_df2['alcohol_consumption'].apply(categorize_alcohol_consumption)


# Now call plot_kmc2()
plot_kmc2('tumor_stage_category', meta_df2)
plot_kmc2('alcohol_category', meta_df2)




def plot_coefficients(coefs, n_highlight):
    fig, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    return fig



def plot_concordance_index(cv_results):
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    return fig

# Ignore (told by mother)

# 创建标签页
tabs = st.tabs(["Home", "Data", "Population","KMPlot", "COXModel","Report", "Help the people"])

populations_count = None
try:
    df = pd.read_csv(file_path, sep="\t")
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
except Exception as e:
    st.error(f"An error occurred: {e}")





# 主页标签页内容
with tabs[0]:
    st.header("Home")
    st.write("Introduction:")
    st.write("1) Survival Analysis is a statistical method used to analyze and predict the time until an event of interest (e.g., death) occurs.")
    st.write("2) There are two major methods used for this analysis: Kaplan-Meier Estimator & Cox Proportional Hazards Model.")
    st.write("3) The Kaplan-Meier Estimator gives out a graph relating survival probability to time, visualizing survival probabilities over time.")
    st.write("4) The Cox Proportional Hazards Model estimates the effect of covariates (e.g., age, treatment) on survival time.")
    st.write("5) Both models can handel censored data fairly well. They can be used in healthcare, such as predicting patient survival and evaluating treatment effectiveness.")
    st.subheader("Constrains")
    st.write("This is only a very simple project relating to Survival Analysis. There might be some errors in the Cox Proportional Hazards Model because it turns out there is only one covariate that constributes to the model.")

    

# 数据展示标签页内容
with tabs[1]:
    st.header("Data")
    if df is not None:
        # Display the dataframe in Streamlit
        st.write("Preview of the Data:")
        st.dataframe(df)


# Population
with tabs[2]:
    st.header("Population")
    populations_count = df['vital_status'].value_counts()
    st.write(populations_count)
    fig = plot_statistics(df)
    st.pyplot(fig)

with tabs[3]:

    if km_group_selection == "none":
        fig = plot_km(meta_df2)
    else:
        fig = plot_kmc2(km_group_selection, meta_df2) # That is something to be modified.
    st.pyplot(fig)     


with tabs[4]:
    st.header("COX Model")
    with st.spinner("Training the COX model..."):
        meta_path2 = os.path.join(wd, 'meta/info2.tsv')
        print(meta_path2)
        meta_df_path2 = pd.read_csv(meta_path2,sep="\t")
        meta_df_path2 = meta_df_path2.dropna()
        x = meta_df_path2.iloc[:, 1:-2]
        print(x.dtypes)

        # Fix categorical conversion
        for col in x.columns:
            if x[col].dtype == 'object':
                x[col] = x[col].astype('category')

            # if x[col].dtype.name == 'category':
            #     x[col] = x[col].cat.codes  # Convert to numerical labels


        # Ensure numerical type
        x['number_of_lymph_nodes_examined'] = x['number_of_lymph_nodes_examined'].astype('float64')

        # One-hot encoding (convert sparse to dense)
        from sksurv.preprocessing import OneHotEncoder
        x = OneHotEncoder().fit_transform(x) 
        print(x.shape)

        # Structured array for survival data
        # aux = [(bool(row['vital_status']), float(row['follow_up_days'])) for _, row in meta_df_path2.iterrows()]
        aux = [(row['vital_status'],row['follow_up_days']) for index,row in meta_df_path2.iterrows()] 

        y = np.array(aux, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

        # Train-test split
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, stratify=y['Status'], random_state=42
        )

        print(x_train.shape, x_test.shape)

        # Cox Model Training
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        cph = CoxPHSurvivalAnalysis()

        alphas = 10.0 ** np.linspace(4, -4, 50)
        coefficients = {}

        for alpha in alphas:
            cph.set_params(alpha=alpha)
            cph.fit(x, y)  # Use data to train the model
            key = round(alpha, 5)
            coefficients[key] = cph.coef_

        # coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(meta_df_path2.columns[1:-2])
        coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature", columns="alpha").set_index(x.columns)

        fig = plot_coefficients(coefficients, 5)
        st.pyplot(fig)



        import warnings
        from sklearn.exceptions import FitFailedWarning
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV, KFold





with tabs[5]:

    from openai import OpenAI

    st.title("OpenAI gpt-4o-mini")

    # 输入数据
    # data = st.text_area("请输入实验数据（如实验时间、温度、结果）：")
    data = f"""The number of patients is 108. The population of deceased is {populations_count['Deceased']}
    , the alive is {populations_count['Living']}. The average follow-up days is {int(meta_df2["follow_up_days"].mean())}. 
    In the Kaplan Meier Curve anaylis, the p-value comparing the survival curve between Male and Female is 0.7111. The Male curve has a better survival probability in the end. 
    Also, the p-value is 0.0310 for the survival curve between stage I/II and stage III/IV and 0.4524 between drinkers and non-drinkers. 
    People in stage I/II have a better survival probability than those in stage III/IV, while drinkers' curve is better than the non-drinkers' curve. 
    In the COX Proportional Hazards Model analysis, the five most important factors are: tumor_focality=Unifocal (coefficient ~ 7.5-7.2), tumor_stage_pathological=Stage IV (coefficient ~ 6.9-6.7), 
    tumor_stage_pathological=Stage IIB (coefficient ~ 5.4-5.2), tumor_stage_pathological=Stage III (coefficient ~ 5), tumor_stage_pathological=IIA (coefficient ~ 4.7-4.5).""" 

    prompt = f"The following is data ：\n{data}\n Please generate a scientific report."

    st.write(f"Prompt for the report: {prompt}")

    if st.button("Generate Report"):


        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        st.subheader("The report is：")
        st.write(completion.choices[0].message.content)


with tabs[6]:
    st.header("What Can We Do to Help the People With Cancer?")
    st.subheader("There are many ways to help someone with cancer:")
    st.write("Practical support: ")
    st.write("1) Help with errands, cooking, cleaning, or running errands.")
    st.write("2) Help with doctor visits, physical therapy, or other clinical tasks.")
    st.write("3) Donate blood to help the cancer patients.")
    st.write("Listening & Emotional support: ")
    st.write("1) Listen carefully and pay attention when the patient express thoughts or feelings.")
    st.write("2) Please respect their need of privacy.")
    st.write("3) Offer support so they know you cares about of them.")

