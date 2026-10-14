
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="What Should I Build?",
    page_icon="💡",
    layout="wide"
)

# ----------------------------------
# 1. SAMPLE PROJECT KNOWLEDGE BASE
# ----------------------------------

projects = [
    {
        "title": "AI-Based Face Mask Detection",
        "branch": "CSE / IT / AI",
        "skills": "Python Machine Learning Deep Learning OpenCV",
        "interests": "AI Computer Vision Healthcare",
        "level": "Intermediate",
        "budget": "Low",
        "career": "AI/ML Engineer",
        "duration": "3-4 weeks",
        "tools": "Python, OpenCV, TensorFlow",
        "description": "Classify face images into mask and no-mask categories.",
        "steps": [
            "Find a suitable public face-mask image dataset.",
            "Clean and split the dataset into training and testing sets.",
            "Train an image classification model.",
            "Evaluate accuracy and test unseen images.",
            "Build a Streamlit interface for predictions."
        ]
    },
    {
        "title": "Student Performance Prediction",
        "branch": "All Engineering",
        "skills": "Python Pandas Machine Learning Statistics",
        "interests": "AI Education Data Analysis",
        "level": "Beginner",
        "budget": "Low",
        "career": "Data Scientist",
        "duration": "2-3 weeks",
        "tools": "Python, Pandas, Scikit-learn, Streamlit",
        "description": "Explore student data and predict performance categories.",
        "steps": [
            "Obtain a public or synthetic student dataset.",
            "Explore and clean the data.",
            "Select useful features and define the target.",
            "Train and evaluate a classification model.",
            "Build a dashboard with predictions and charts."
        ]
    },
    {
        "title": "AI-Powered Study Assistant",
        "branch": "CSE / IT / AI",
        "skills": "Python NLP APIs Web Development",
        "interests": "AI Education Automation",
        "level": "Intermediate",
        "budget": "Low",
        "career": "AI/ML Engineer",
        "duration": "3-5 weeks",
        "tools": "Python, Streamlit, NLP, optional LLM API",
        "description": "Help students organize notes and generate practice questions.",
        "steps": [
            "Define the notes and document formats to support.",
            "Extract text from uploaded study material.",
            "Implement keyword search or question generation.",
            "Add summaries and practice questions.",
            "Test the output against sample notes."
        ]
    },
    {
        "title": "Expense Tracker and Analytics",
        "branch": "All Engineering",
        "skills": "Python Pandas SQL Data Visualization",
        "interests": "Finance Analytics Web Development",
        "level": "Beginner",
        "budget": "Low",
        "career": "Data Analyst",
        "duration": "1-2 weeks",
        "tools": "Python, Pandas, Streamlit, SQLite",
        "description": "Track expenses and visualize spending patterns.",
        "steps": [
            "Design an expense record format.",
            "Add expense entry and category selection.",
            "Clean and aggregate transaction data.",
            "Create category and monthly spending charts.",
            "Add CSV export and input validation."
        ]
    },
    {
        "title": "Smart Irrigation Monitoring",
        "branch": "ECE / EEE / ECE / CSE",
        "skills": "Python IoT Sensors Arduino Data Analysis",
        "interests": "IoT Agriculture Sustainability",
        "level": "Intermediate",
        "budget": "Medium",
        "career": "IoT Engineer",
        "duration": "3-5 weeks",
        "tools": "Arduino, sensors, Python, optional dashboard",
        "description": "Monitor soil moisture and simulate irrigation decisions.",
        "steps": [
            "Define soil moisture and environmental inputs.",
            "Create a simulated sensor dataset or use hardware.",
            "Set irrigation thresholds and safety conditions.",
            "Display readings and system status on a dashboard.",
            "Test decisions under different simulated conditions."
        ]
    },
    {
        "title": "Intelligent Traffic Analysis",
        "branch": "CSE / IT / ECE / AI",
        "skills": "Python Computer Vision Machine Learning",
        "interests": "AI Smart Cities Transportation",
        "level": "Advanced",
        "budget": "Low",
        "career": "AI/ML Engineer",
        "duration": "4-6 weeks",
        "tools": "Python, OpenCV, public traffic datasets",
        "description": "Analyze traffic footage or images to estimate vehicle counts.",
        "steps": [
            "Select suitable public traffic footage.",
            "Process frames using computer vision.",
            "Detect or track vehicles using a suitable model.",
            "Estimate counts and evaluate errors.",
            "Visualize results and document limitations."
        ]
    },
    {
        "title": "Cloud Cost Monitoring Dashboard",
        "branch": "CSE / IT / Cloud",
        "skills": "Python SQL APIs Data Visualization",
        "interests": "Cloud DevOps Automation Analytics",
        "level": "Intermediate",
        "budget": "Low",
        "career": "Cloud Engineer",
        "duration": "2-4 weeks",
        "tools": "Python, Pandas, Streamlit, CSV or mock cloud data",
        "description": "Analyze sample cloud usage and flag unusual costs.",
        "steps": [
            "Create or obtain sample cloud billing data.",
            "Calculate service-wise and daily costs.",
            "Set budget limits and anomaly rules.",
            "Build cost charts and warning indicators.",
            "Test the dashboard using simulated billing scenarios."
        ]
    },
    {
        "title": "Smart Parking Availability System",
        "branch": "CSE / IT / ECE / EEE",
        "skills": "Python IoT Sensors Computer Vision",
        "interests": "IoT Smart Cities Automation",
        "level": "Intermediate",
        "budget": "Medium",
        "career": "IoT Engineer",
        "duration": "3-5 weeks",
        "tools": "Python, simulated sensor data, optional hardware",
        "description": "Track available parking slots using simulated sensor readings.",
        "steps": [
            "Design a parking slot data model.",
            "Generate or collect occupancy readings.",
            "Calculate occupied and available spaces.",
            "Build a live-style parking dashboard.",
            "Test edge cases such as missing sensor readings."
        ]
    }
]

df = pd.DataFrame(projects)

# ----------------------------------
# 2. APP HEADER
# ----------------------------------

st.title("💡 What Should I Build?")
st.subheader("AI-powered project recommendations for engineering students")
st.write(
    "Tell us about your skills, interests, budget, and career goals. "
    "Discover projects you can realistically build and showcase."
)

# ----------------------------------
# 3. STUDENT INPUTS
# ----------------------------------

st.sidebar.header("Your Student Profile")

branch = st.sidebar.selectbox(
    "Engineering branch",
    [
        "CSE", "IT", "AI", "ECE", "EEE",
        "Mechanical", "Civil", "Other"
    ]
)

skill_options = [
    "Python", "Machine Learning", "Deep Learning",
    "Pandas", "SQL", "Data Visualization",
    "NLP", "APIs", "Web Development",
    "Computer Vision", "IoT", "Sensors",
    "Arduino", "Statistics"
]

skills = st.sidebar.multiselect(
    "Skills you know",
    skill_options,
    default=["Python"]
)

interests = st.sidebar.multiselect(
    "Your interests",
    [
        "AI", "Data Analysis", "Education",
        "Healthcare", "Computer Vision", "IoT",
        "Agriculture", "Cloud", "DevOps",
        "Automation", "Finance", "Smart Cities",
        "Sustainability", "Web Development"
    ],
    default=["AI", "Education"]
)

level = st.sidebar.selectbox(
    "Preferred difficulty",
    ["Beginner", "Intermediate", "Advanced"]
)

budget = st.sidebar.selectbox(
    "Available budget",
    ["Low", "Medium", "High"]
)

career = st.sidebar.selectbox(
    "Career goal",
    [
        "AI/ML Engineer",
        "Data Scientist",
        "Data Analyst",
        "Software Developer",
        "Cloud Engineer",
        "IoT Engineer",
        "Not sure yet"
    ]
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=8,
    value=5
)

# ----------------------------------
# 4. RECOMMENDATION ENGINE
# ----------------------------------

def recommend_projects():
    profile_text = " ".join([
        branch,
        " ".join(skills),
        " ".join(interests),
        level,
        budget,
        career
    ])

    # Build text documents from project metadata.
    project_text = (
        df["branch"] + " " +
        df["skills"] + " " +
        df["interests"] + " " +
        df["level"] + " " +
        df["budget"] + " " +
        df["career"] + " " +
        df["description"]
    ).tolist()

    # Include the student profile as a document.
    documents = project_text + [profile_text]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(documents)

    # Similarity between the student profile and each project.
    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    results = df.copy()
    results["similarity"] = similarities

    # Small, transparent preference adjustments.
    # Branch and career compatibility are rules, not learned models.
    def branch_match(project_branch):
        if branch in ["CSE", "IT", "AI"]:
            return (
                branch in project_branch
                or "All Engineering" in project_branch
            )
        if branch in ["ECE", "EEE"]:
            return (
                branch in project_branch
                or "All Engineering" in project_branch
            )
        return "All Engineering" in project_branch

    results["score"] = results["similarity"]

    results.loc[
        results["branch"].apply(branch_match),
        "score"
    ] += 0.15

    results.loc[
        results["level"] == level,
        "score"
    ] += 0.12

    results.loc[
        results["budget"] == budget,
        "score"
    ] += 0.08

    results.loc[
        (results["career"] == career) |
        (career == "Not sure yet"),
        "score"
    ] += 0.12

    # Put the strongest matches first.
    results = results.sort_values(
        "score",
        ascending=False
    ).head(top_n)

    # Normalize the displayed scores to a 0-100 scale.
    maximum = results["score"].max()

    if maximum > 0:
        results["match_percent"] = (
            results["score"] / maximum * 100
        ).round().astype(int)
    else:
        results["match_percent"] = 0

    return results


# ----------------------------------
# 5. DISPLAY RECOMMENDATIONS
# ----------------------------------

if st.button("✨ Find My Project Ideas", type="primary"):
    recommendations = recommend_projects()

    st.success(
        f"Found {len(recommendations)} project ideas for your profile."
    )

    st.info(
        "Match percentages are relative ranking scores for this "
        "recommendation run, not probabilities of project success."
    )

    for rank, (_, project) in enumerate(
        recommendations.iterrows(), start=1
    ):
        with st.container(border=True):
            st.markdown(
                f"### {rank}. {project['title']}"
            )

            st.progress(
                int(project["match_percent"]),
                text=f"Relative match score: {project['match_percent']}%"
            )

            st.write(project["description"])

            c1, c2, c3 = st.columns(3)

            c1.metric("Difficulty", project["level"])
            c2.metric("Estimated time", project["duration"])
            c3.metric("Budget", project["budget"])

            st.write("**Suitable branches:**", project["branch"])
            st.write("**Technologies:**", project["tools"])
            st.write("**Career path:**", project["career"])

            with st.expander("View implementation roadmap"):
                for step_no, step in enumerate(
                    project["steps"], start=1
                ):
                    st.write(f"{step_no}. {step}")

            with st.expander("View project details"):
                st.write("**Skills:**", project["skills"])
                st.write("**Interests:**", project["interests"])

    st.subheader("Compare your recommendations")
    st.dataframe(
        recommendations[
            [
                "title", "level", "budget",
                "duration", "career", "match_percent"
            ]
        ].rename(columns={
            "title": "Project",
            "level": "Difficulty",
            "budget": "Budget",
            "duration": "Estimated duration",
            "career": "Career goal",
            "match_percent": "Relative match (%)"
        }),
        hide_index=True,
        use_container_width=True
    )

else:
    st.markdown("### How it works")
    st.write("1. Select your branch and current technical skills.")
    st.write("2. Choose your interests, budget, and career goal.")
    st.write("3. Click **Find My Project Ideas**.")
    st.write("4. Compare recommendations and follow a project roadmap.")

st.divider()
st.caption(
    "Educational prototype. Recommendations depend on the sample "
    "catalog and selected preferences. Review project requirements "
    "before choosing a final-year or safety-critical project."
)
