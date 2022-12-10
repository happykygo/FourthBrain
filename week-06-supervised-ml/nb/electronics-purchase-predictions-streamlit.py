import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Display Wal-Mart Labs logo.
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Walmart_Labs_logo.svg/1024px-Walmart_Labs_logo.svg.png")

st.markdown("<h1 style='text-align: center; color: black;'>Online Electronics Purchasing Behavior</h1>",
            unsafe_allow_html=True)

# Import train dataset to DataFrame
train_df = pd.read_csv("../dat/train.csv.gz", compression="gzip")
model_results_df = pd.read_csv("../dat/model_results.csv")
lr_global_fig = Image.open("../dat/lr_global.png")
lr_local_fig = Image.open("../dat/lr_local.png")
svm_global_fig = Image.open("../dat/svm_global.png")
svm_local_fig = Image.open("../dat/svm_local.png")
gbt_global_fig = Image.open("../dat/gbt_global.png")
gbt_local_fig = Image.open("../dat/gbt_local.png")

# Drop uniformative columns
train_df.drop(columns=["year", "month", "Weekend"], inplace=True)

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png")

    # Available models for selection
    models = ["Logistic Regression", "SVM", "Gradient Boosting Classifier"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        ("Logistic Regression", "SVM", "Gradient Boosting Classifier")
    )

    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)

    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:
    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    cols = ["tn", "fp", "fn", "tp"]

    model1_fig = model_results_df[model_results_df["model"] == model1_select]
    cm1 = model1_fig[cols].to_numpy()[0].reshape(2, 2).tolist()

    x1 = ["0", "1"]
    y1 = ["1", "0"]
    z1 = cm1
    z_text1 = [[str(y) for y in x] for x in z1]

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

        fig1 = px.imshow(z1, text_auto=True)

        # add custom x-axis title
        fig1.add_annotation(dict(font=dict(color="black", size=14),
                                 x=0.5,
                                 y=-0.00005,
                                 showarrow=False,
                                 text="Predicted value",
                                 xref="paper",
                                 yref="paper"))

        # add custom y-axis title
        fig1.add_annotation(dict(font=dict(color="black", size=14),
                                 x=-0.25,
                                 y=0.5,
                                 showarrow=False,
                                 text="True value",
                                 textangle=-90,
                                 xref="paper",
                                 yref="paper"))

        # Write plotly chart and fit to the container width.
        st.plotly_chart(fig1, use_container_width=True)

    # Build confusion matrix for second model
    with col2:
        model2_results = model_results_df[model_results_df["model"] == model2_select]
        cm2 = model2_results[cols].to_numpy()[0].reshape(2, 2).tolist()

        x2 = ["0", "1"]
        y2 = ["1", "0"]
        z2 = cm2
        z_text2 = [[str(y) for y in x] for x in z2]

        st.header(model2_select)

        fig2 = px.imshow(z2, text_auto=True)

        # add custom x-axis title
        fig2.add_annotation(dict(font=dict(color="black", size=14),
                                 x=0.5,
                                 y=-0.00005,
                                 showarrow=False,
                                 text="Predicted value",
                                 xref="paper",
                                 yref="paper"))

        # add custom y-axis title
        fig2.add_annotation(dict(font=dict(color="black", size=14),
                                 x=-0.25,
                                 y=0.5,
                                 showarrow=False,
                                 text="True value",
                                 textangle=-90,
                                 xref="paper",
                                 yref="paper"))

        # Write plotly chart and fit to the container width.
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # YOUR CODE GOES HERE!
    # Use tab2 as a guide!
    # Use columns to separate visualizations for models
    # Include a plot for local and global explanability!

    #########
    col1, col2 = st.columns(2)

    model1_fig = model_results_df[model_results_df["model"] == model1_select]

    with col1:
        st.header(model1_select)

        if model1_select == "Logistic Regression":
            st.image(lr_global_fig, caption='lr global shap')
            st.image(lr_local_fig, caption='lr local shap')

        if model1_select == "SVM":
            st.image(svm_global_fig, caption='svm global shap')
            st.image(svm_local_fig, caption='svm local shap')

        if model1_select == "Gradient Boosting Classifier":
            st.image(gbt_global_fig, caption='gbt global shap')
            st.image(gbt_local_fig, caption='gbt local shap')

    with col2:
        st.header(model2_select)

        if model1_select == "Logistic Regression":
            st.image(lr_global_fig, caption='lr global shap')
            st.image(lr_local_fig, caption='lr local shap')

        if model1_select == "SVM":
            st.image(svm_global_fig, caption='svm global shap')
            st.image(svm_local_fig, caption='svm local shap')

        if model1_select == "Gradient Boosting Classifier":
            st.image(gbt_global_fig, caption='gbt global shap')
            st.image(gbt_local_fig, caption='gbt local shap')