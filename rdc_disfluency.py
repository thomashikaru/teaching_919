import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


def compute_score(pi0e, pi0ag, gamma, rgag, alpha, pi0agIe):
    # Define your scoring function here
    val = np.log(pi0e / pi0ag) - alpha * (1 - gamma) * rgag + gamma * np.log(pi0agIe)
    return np.exp(val)


def plot_data(score):
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(
        x=["Um", "Target"],
        height=np.array([score, 1.0]) / np.linalg.norm([score, 1.0]),
        color=["lightblue", "orange"],
    )
    plt.xlabel("Action", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot(fig)


def main():

    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

    with col1:
        st.header("Parameters")

        # Define sliders
        pi0e = st.slider(
            "Predictability of 'Um'", min_value=0.0, max_value=1.0, value=0.5
        )
        st.write("---")
        pi0ag = st.slider(
            "Predictability of Target Word", min_value=0.0, max_value=1.0, value=0.5
        )
        st.write("---")
        gamma = st.slider(
            "Gamma (Discounting)", min_value=0.0, max_value=1.0, value=0.5
        )
        st.write("---")

    with col2:
        st.header(".")
        rgag = st.slider(
            "Reward of Target Word", min_value=0.0, max_value=1.0, value=0.5
        )
        st.write("---")
        alpha = st.slider("Alpha (Control)", min_value=0.0, max_value=1.0, value=0.5)
        st.write("---")
        pi0agIe = st.slider(
            "Predictability of Target Word, given 'Um'",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
        )
        st.write("---")

    # Compute score
    score = compute_score(pi0e, pi0ag, gamma, rgag, alpha, pi0agIe)

    with col3:
        # Display score
        st.header(f"Bias Towards 'Um': {score:.2f}")
        plot_data(score)


if __name__ == "__main__":
    main()
