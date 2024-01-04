
# Employee Attrition using unsupervised learning

This project utilizes various unsupervised learning techniques to analyze and understand employee attrition within a company. It employs clustering algorithms, outlier detection methods, and predictive models to gain insights into employee behaviors and potential attrition risk factors.



## Table of contents

- Introduction
- Setup
- Usage
- Visualizations
- Models and Analysis
- Contributing
- License

## Introduction

The goal of this project is to identify patterns and potential indicators of employee attrition. It employs Python libraries such as Pandas, Scikit-learn, Matplotlib, Plotly, and Streamlit for data manipulation, visualization, and modeling. The project primarily focuses on:
- Exploratory Data Analysis (EDA) of employee-related datasets.
- Utilization of K-means clustering, Local Outlier Factor (LOF), Isolation Forest, Gaussian Mixture Models (GMM), and predictive models for analysis.
- Visualization of clusters, outliers, and predictive insights using Streamlit and Matplotlib/Plotly.

## Setup

- Dependencies: Ensure you have Python installed. Install required libraries using `pip install -r requirements.txt`.
- Data: The dataset used for analysis can be found in `dataset.csv`.

## Usage

- Clone the repository: `git clone https://github.com/yourusername/employe_attrition.git`
- Run `streamlit run app.py` to access the Streamlit application for interactive visualizations and analysis.

##  Visualizations

- Elbow Method Plot: Identifies optimal cluster count using the elbow method.
- K-means Clusters with PCA/t-SNE: Visualizes K-means clusters in reduced dimensions.
- LOF Outliers with PCA: Identifies outliers using LOF algorithm and PCA.
- Isolation Forest Outliers: Detects outliers using Isolation Forest.
- Gaussian Mixture Model Clusters: Clustering based on GMM focusing on specific features.

## Models and Analysis

- Outlier Detection: Utilizes Isolation Forest and LOF to detect outliers in employee data.
- Predictive Model: Uses Isolation Forest for predicting potential employee attrition based on selected features.

## Contribution

Contributions are welcome! If you have any suggestions, improvements, or new features, feel free to open an issue or create a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)

This project is licensed under the [MIT License](LICENSE).
