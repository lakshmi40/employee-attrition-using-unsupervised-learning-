import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import BernoulliRBM
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go
import streamlit as st

# Load the CSV file into a DataFrame
file_path = "D:\\DCS\\SEM- 5\\5. HR Lab\\Project\\dataset.csv"
df = pd.read_csv(file_path)

# Check for null values in the DataFrame
null_values = df.isnull().sum()
print("Count of null values in each column:")
print(null_values)

# Handling missing values (as shown in the previous code)
# Streamlit code
st.title("Employee Attrition with Unsupervised Learning")
st.sidebar.title("Employee Attrition Model")
visualization_option = st.sidebar.selectbox("Select", ["Elbow Method Plot", "K-means Clusters with PCA", "K-means Clusters with t-SNE", "LOF Outliers with PCA",
                                                       "Isolation Forest Outliers", "Gaussian Mixture Model Clusters","Prediction"])
# Descriptions for the selected visualization
descriptions = {
    "Elbow Method Plot": "Determine the optimal number of clusters using within-cluster sum of squares.",
    "K-means Clusters with PCA": "Visualizes K-means clustering results in two-dimensional space using Principal Component Analysis (PCA) for dimensionality reduction.",
    "K-means Clusters with t-SNE": "Displays K-means clustering results in two-dimensional space using t-distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.",
    "LOF Outliers with PCA": "Identifies and visualizes outliers in the data using Local Outlier Factor (LOF) algorithm and presents the results in two-dimensional space using Principal Component Analysis (PCA).",
    "Isolation Forest Outliers": "Detects outliers in the data using the Isolation Forest model",
    "Gaussian Mixture Model Clusters": "Applies Gaussian Mixture Model (GMM) clustering to the data and visualizes the resulting clusters, specifically focusing on certain selected features like age, job level, and monthly income.",
    "Prediction" :"The Isolation Forest model is trained and utilized for predicting whether a specific employee is likely to leave the company based on the provided input features."
}
# Display the selected visualization description in the sidebar
st.sidebar.subheader(descriptions.get(visualization_option, ""))

# Select columns for clustering
columns_for_clustering = ['Age', 'MonthlyIncome', 'JobLevel']  # Add more columns as needed

# Select data for clustering
data_for_clustering = df[columns_for_clustering]

# Perform feature scaling if required (optional but recommended for K-means)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)
# Choosing the number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Performing K-means clustering
if visualization_option == "Elbow Method Plot":
    st.subheader("Elbow Method Plot")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    st.pyplot(plt)


# Choosing the optimal number of clusters
optimal_clusters = 3  # Change this according to your elbow plot

# Performing K-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(scaled_data)


# Adding cluster labels to the DataFrame
df['Cluster_KMeans'] = kmeans.labels_

# Visualizing K-means clusters
# Dimensionality reduction using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(scaled_data)

# Visualizing K-means clusters with PCA
if visualization_option == "K-means Clusters with PCA":
    st.subheader("K-means Clusters with PCA")
    # code to create the scatter plot with K-means clusters and PCA
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=df['Cluster_KMeans'], cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Components with K-means Clusters')
    plt.show()
    st.pyplot(plt)

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(scaled_data)

# Visualizing t-SNE components with K-means clusters
if visualization_option == "K-means Clusters with t-SNE":
        st.subheader("K-means Clusters with t-SNE")
        # code to create the scatter plot with K-means clusters and t-SNE
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=df['Cluster_KMeans'], cmap='viridis')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Components with K-means Clusters')
        plt.show()
        st.pyplot(plt)


# Novelty and Outlier Detection using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_labels = lof.fit_predict(scaled_data)

# Visualizing LOF Outliers
if visualization_option == "LOF Outliers with PCA":
    st.subheader("LOF Outliers with PCA")
    # Your code to create the scatter plot with LOF outliers and PCA
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=outlier_labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('LOF Outliers')
    plt.show()
    st.pyplot(plt)

# Splitting X and Y
# X contains the features, and y contains the target variable
# Split the data into training and testing sets
# And X_test and y_test for evaluating its performance
X = df[['Age','BusinessTravel','DailyRate','Department','DistanceFromHome',  'Education','EducationField','EmployeeCount','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel' ,'JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
Y = df[['Attrition', 'EmployeeNumber', 'Over18','Gender']]  # 'Attrition' is the target variable
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

# Outlier Analysis
# Isolation Forest
# Assuming 'df' is your DataFrame
# Identify categorical columns
categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
# Identify numerical columns
numerical_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
# Split data into features (X) and target variable (y)
x = df[numerical_columns+categorical_columns]
y = df['Attrition']  # Assuming 'Attrition' is your target variable

# Create a ColumnTransformer to apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'  # Pass through the non-categorical columns
)


# Create a pipeline with preprocessing and Isolation Forest
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('isolation_forest', IsolationForest(contamination=0.1, random_state=42))  # Adjust contamination based on your data
])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Print information about x_train
# Print column names in x_train
print("Column names in x_train:", x_train.columns)

# Fit the pipeline on the training data
clf.fit(x_train)

# Predict on the test set
y_pred = clf.predict(x_test)
print(y_pred)
y_true_numeric = np.where(y_test == 'No', 1, -1)

# Visualize Isolation Forest Outliers without PCA
if visualization_option == "Isolation Forest Outliers":
    st.subheader("Isolation Forest Outliers")
    fig, ax = plt.subplots()
    # Choose two numerical features for the plot, e.g., 'Age' and 'MonthlyIncome'
    feature1 = 'Age'
    feature2 = 'MonthlyIncome'
    # Scatter plot
    scatter = ax.scatter(x_test[feature1], x_test[feature2], c=y_pred, cmap='viridis')
    # Legend
    legend = ax.legend(*scatter.legend_elements(), title="Outliers")
    ax.add_artist(legend)
    # Axis labels and title
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_title('Isolation Forest Outliers')
    st.pyplot(fig)
    # Print classification report
    st.subheader("Classification Report")
    classification_rep = classification_report(y_true_numeric, y_pred, output_dict=True)
    st.table(classification_rep)

if visualization_option == "Prediction":
    st.subheader("Prediction Model")
    # Selecting only the relevant columns
    selected_columns = ['Age', 'MonthlyIncome', 'JobLevel', 'JobSatisfaction']

    # Creating a new DataFrame with only the selected columns
    selected_data = df[selected_columns]

    # Split data into features (X) and target variable (y)
    X = selected_data
    y = df['Attrition']  # Assuming 'Attrition' is your target variable

    # Create a ColumnTransformer to apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), [])  # No categorical columns in this case
        ],
        remainder='passthrough'  # Pass through the non-categorical columns
    )

    # Create a pipeline with preprocessing and Isolation Forest
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('isolation_forest', IsolationForest(contamination=0.1, random_state=42))  # Adjust contamination based on your data
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline on the training data
    clf.fit(X_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_true_numeric = np.where(y_test == 'No', 1, -1)

   # Collect user input
    user_age_input = st.text_input("Enter Age", 25)
    user_monthly_income_input = st.text_input("Enter Monthly Income", 5000)
    user_job_level_input = st.selectbox("Select Job Level", [1, 2, 3, 4, 5], index=2)
    user_job_satisfaction_input = st.selectbox("Select Job Satisfaction", [1, 2, 3, 4, 5], index=3)

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'Age': [user_age_input],
        'MonthlyIncome': [user_monthly_income_input],
        'JobLevel': [user_job_level_input],
        'JobSatisfaction': [user_job_satisfaction_input],
    })

    # Make predictions
    user_pred = clf.predict(user_data)

    # Display prediction result
    st.subheader("Prediction Result")
    if user_pred[0] == -1:
        st.error("The model predicts that the employee is likely to leave the company.")
    else:
        st.success("The model predicts that the employee is likely to stay with the company.")


# Gaussian Mixture model
selected_columns = ['Age', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                     'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'WorkLifeBalance']

selected_data = df[selected_columns]
def fit_and_plot_gmm(selected_columns, n_components=4):
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(selected_columns)

    # Predict cluster labels
    labels = gmm.predict(selected_columns)

    # Add cluster labels to the DataFrame
    selected_columns['cluster'] = labels

    # Plot clusters
    plot_clusters(selected_columns)

def plot_clusters(selected_columns):
    colors = ['blue', 'green', 'cyan', 'black']

    # Scatter plot for each cluster
    for k in range(selected_columns['cluster'].nunique()):
        cluster_data = selected_columns[selected_columns['cluster'] == k]
        plt.scatter(cluster_data['Age'], cluster_data['MonthlyIncome'], c=colors[k], label=f'Cluster {k}')
print("Before GMM:", selected_data.shape)
fit_and_plot_gmm(selected_data)
print("After GMM:", selected_data.shape)

# Fit and plot GMM with the selected features
# Add labels and title
if visualization_option == "Gaussian Mixture Model Clusters":
    st.subheader("Gaussian Mixture Model Clusters")
    plt.xlabel('Age')
    plt.ylabel('MonthlyIncome')
    plt.title('Gaussian Mixture Model Clustering')
    plt.legend()
    plt.show()
    st.pyplot(plt)