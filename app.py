# app.py
# -----------------------------------------------
# Student Micro-Learning Habit Recommendation System using AI
# Streamlit Web App (single file)
# -----------------------------------------------

import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st

# ---------------------------
# 1. Data generation
# ---------------------------

def generate_sample(n_samples=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    records = []
    learning_styles = ["visual", "reading", "practice", "mixed"]
    activities = [
        "PRACTICE_CODING_QUESTIONS",
        "WATCH_SHORT_VIDEO",
        "REVISE_NOTES",
        "SOLVE_MCQ_SET",
        "READ_ARTICLE_OR_BLOG",
    ]

    for _ in range(n_samples):
        study_hours = round(np.random.uniform(0.5, 5.0), 1)
        subject_difficulty = np.random.randint(1, 6)
        academic_perf = round(np.random.uniform(40, 95), 1)
        pref_style = random.choice(learning_styles)
        available_time = np.random.randint(10, 90)  # minutes
        motivation = np.random.randint(1, 6)

        # Rule-based pseudo ground-truth for labels
        if available_time <= 20:
            if pref_style in ["visual", "mixed"]:
                activity = "WATCH_SHORT_VIDEO"
            else:
                activity = "READ_ARTICLE_OR_BLOG"
        elif motivation <= 2:
            activity = "WATCH_SHORT_VIDEO"
        elif subject_difficulty >= 4 and available_time >= 30:
            activity = "PRACTICE_CODING_QUESTIONS"
        elif academic_perf < 60:
            activity = "REVISE_NOTES"
        else:
            activity = random.choice(activities)

        records.append({
            "study_hours_per_day": study_hours,
            "subject_difficulty": subject_difficulty,
            "academic_performance": academic_perf,
            "preferred_learning_style": pref_style,
            "available_time_mins": available_time,
            "motivation_level": motivation,
            "recommended_activity": activity,
        })

    return pd.DataFrame(records)

# ---------------------------
# 2. Model training (cached)
# ---------------------------

@st.cache_resource
def train_model():
    df = generate_sample()

    X = df.drop(columns=["recommended_activity"])
    y = df["recommended_activity"])

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    numeric_features = [
        "study_hours_per_day",
        "subject_difficulty",
        "academic_performance",
        "available_time_mins",
        "motivation_level",
    ]
    categorical_features = ["preferred_learning_style"]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model.fit(X_train, y_train)

    # Simple accuracy for display
    accuracy = model.score(X_test, y_test)

    return model, label_encoder, accuracy

# ---------------------------
# 3. Explanation helper
# ---------------------------

def explain_recommendation(activity_label):
    explanations = {
        "PRACTICE_CODING_QUESTIONS": {
            "title": "Practice 2–3 Coding Questions",
            "details": [
                "Focus on problems from today's topic or a weak concept.",
                "Spend ~20–30 minutes actively coding, not just reading solutions.",
                "Note down patterns and common mistakes."
            ]
        },
        "WATCH_SHORT_VIDEO": {
            "title": "Watch a Short Concept Video (10–15 min)",
            "details": [
                "Pick one focused video on a single concept.",
                "Pause and take quick notes when something is new.",
                "After the video, write 2–3 key points in your own words."
            ]
        },
        "REVISE_NOTES": {
            "title": "Revise Notes (15–20 min)",
            "details": [
                "Revisit your class notes or previous revision notes.",
                "Highlight key formulas, steps, or definitions.",
                "Mark any area that still feels confusing for deeper study later."
            ]
        },
        "SOLVE_MCQ_SET": {
            "title": "Solve a Short MCQ Set",
            "details": [
                "Attempt 10–15 topic-based MCQs under a small time limit.",
                "Mark questions you guessed or got wrong.",
                "Review explanations immediately to reinforce learning."
            ]
        },
        "READ_ARTICLE_OR_BLOG": {
            "title": "Read an Article / Blog on the Topic",
            "details": [
                "Choose a short article (5–10 minutes read) on your subject.",
                "Try to explain the article to yourself in 3–4 bullet points.",
                "Relate what you read to a problem or example you already know."
            ]
        }
    }

    return explanations.get(activity_label, {
        "title": "Custom Learning Task",
        "details": [
            "Do a short, focused activity aligned with your current goal.",
            "Keep it small enough to finish in one sitting.",
            "Reflect for 2 minutes after finishing: What did I learn?"
        ]
    })

# ---------------------------
# 4. Streamlit UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="Student Micro-Learning Habit Recommender",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Student Micro-Learning Habit Recommendation System")
    st.caption("AI-powered micro-learning suggestions for consistent, stress-free study habits.")

    # Train/load model (cached)
    with st.spinner("Preparing AI model..."):
        model, label_encoder, accuracy = train_model()

    # Layout: two columns
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🔍 Enter Your Study Context")

        with st.form("input_form"):
            st.markdown("**Study Profile**")
            study_hours = st.slider(
                "Average study hours per day",
                min_value=0.5,
                max_value=8.0,
                value=2.0,
                step=0.5
            )

            subject_difficulty = st.slider(
                "How difficult is this subject for you? (1 = very easy, 5 = very hard)",
                min_value=1,
                max_value=5,
                value=3
            )

            academic_performance = st.slider(
                "Current academic performance in this subject (%)",
                min_value=0,
                max_value=100,
                value=65
            )

            st.markdown("---")
            st.markdown("**Preferences & Time**")

            preferred_learning_style = st.selectbox(
                "Preferred learning style",
                options=["visual", "reading", "practice", "mixed"]
            )

            available_time_mins = st.slider(
                "Available time for studying today (minutes)",
                min_value=10,
                max_value=120,
                value=30,
                step=5
            )

            motivation_level = st.slider(
                "Motivation level today (1 = very low, 5 = very high)",
                min_value=1,
                max_value=5,
                value=3
            )

            submitted = st.form_submit_button("💡 Get Micro-Learning Recommendation")

        if submitted:
            # Prepare input dataframe
            input_dict = {
                "study_hours_per_day": [study_hours],
                "subject_difficulty": [subject_difficulty],
                "academic_performance": [academic_performance],
                "preferred_learning_style": [preferred_learning_style],
                "available_time_mins": [available_time_mins],
                "motivation_level": [motivation_level],
            }
            input_df = pd.DataFrame(input_dict)

            # Prediction
            y_pred = model.predict(input_df)
            predicted_label = label_encoder.inverse_transform(y_pred)[0]

            info = explain_recommendation(predicted_label)

            st.markdown("### 🎯 Recommended Micro-Learning Task for Today")
            st.success(f"**{info['title']}**")

            st.markdown("**How to do it:**")
            for point in info["details"]:
                st.markdown(f"- {point}")

            # Small reflection tip
            st.markdown("---")
            st.markdown("✅ **Habit Tip:** After finishing this task, take **2 minutes** to write what you learned. This reflection strengthens your memory and builds long-term habits.")

    with col_right:
        st.subheader("📈 About This AI System")

        st.markdown(
            """
            This system uses a **machine learning model** trained on synthetic student scenarios.

            It considers:
            - ⏱️ Study hours per day  
            - 🎓 Subject difficulty & current performance  
            - 🧠 Preferred learning style  
            - 📅 Time available today  
            - 💪 Motivation level  
            """
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Snapshot")
        st.metric(label="Estimated Model Accuracy", value=f"{accuracy * 100:.1f}%")

        with st.expander("What is a micro-learning task?"):
            st.write(
                """
                Micro-learning means **small, focused study tasks** that you can finish in one sitting  
                (usually 10–30 minutes).  

                Instead of planning a 3-hour study marathon, this system helps you:
                - Study a little **every day**
                - Avoid feeling overwhelmed
                - Build a **strong, consistent habit** over time
                """
            )

        with st.expander("How could this be extended in the future?"):
            st.write(
                """
                - Track whether you **completed** the task and adapt future suggestions  
                - Use real student data instead of synthetic data  
                - Send daily reminders via web / mobile app  
                - Integrate with LMS / coding platforms to pick tasks automatically  
                """
            )

    st.markdown("---")
    st.caption("Made with ❤️ for students who want to build consistent self-learning discipline.")

if __name__ == "__main__":
    main()
