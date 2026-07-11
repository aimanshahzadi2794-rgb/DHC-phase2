import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


MODEL_PATH = Path("churn_pipeline.joblib")
MODEL_INFO_PATH = Path("model_info.json")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "churn_pipeline.joblib was not found."
        )

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_information():
    if not MODEL_INFO_PATH.exists():
        return {}

    with open(
        MODEL_INFO_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


try:
    model = load_model()
    model_information = load_model_information()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


st.title("Customer Churn Prediction")
st.write(
    "Enter customer information to predict whether "
    "the customer is likely to leave the telecom service."
)


with st.sidebar:
    st.header("Model Information")

    selected_model = model_information.get(
        "selected_model",
        "Not available"
    )

    sklearn_version = model_information.get(
        "scikit_learn_version",
        "Not available"
    )

    st.write(f"**Selected model:** {selected_model}")
    st.write(
        f"**Scikit-learn version:** {sklearn_version}"
    )


column_1, column_2 = st.columns(2)


with column_1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure_months = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


with column_2:
    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=10.0
    )


input_data = pd.DataFrame([{
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure_months,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
}])


expected_columns = model_information.get(
    "feature_columns",
    []
)


if expected_columns:
    missing_columns = [
        column
        for column in expected_columns
        if column not in input_data.columns
    ]

    extra_columns = [
        column
        for column in input_data.columns
        if column not in expected_columns
    ]

    if missing_columns:
        st.error(
            f"Application is missing columns: "
            f"{missing_columns}"
        )
        st.stop()

    if extra_columns:
        st.warning(
            f"Application contains additional columns: "
            f"{extra_columns}"
        )

    input_data = input_data[
        expected_columns
    ]


if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):
    try:
        prediction = int(
            model.predict(input_data)[0]
        )

        churn_probability = float(
            model.predict_proba(
                input_data
            )[0][1]
        )

        st.subheader("Prediction Result")

        probability_column, result_column = st.columns(2)

        with probability_column:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2%}"
            )

        with result_column:
            if prediction == 1:
                st.error(
                    "The customer is likely to churn."
                )
            else:
                st.success(
                    "The customer is likely to stay."
                )

        st.progress(
            min(
                max(churn_probability, 0.0),
                1.0
            )
        )

        with st.expander("View submitted data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as error:
        st.error(
            f"Prediction error: {error}"
        )