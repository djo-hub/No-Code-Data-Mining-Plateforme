import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, silhouette_score, calinski_harabasz_score
)

# =====================================================
# DATA LOADING FUNCTIONS
# =====================================================
def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file"""
    try:
        if uploaded_file is None:
            raise ValueError("No file uploaded")

        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)

        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

        return df
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")

# =====================================================
# PREPROCESSING FUNCTIONS
# =====================================================
def preprocess_data(df, operation, method=None):
    """Preprocess data based on selected operation"""
    df_processed = df.copy()

    if operation in ['missing', 'all']:
        if method == "Drop rows with missing values" or method == "drop":
            df_processed = df_processed.dropna()
        elif method == "Fill with mean":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif method == "Fill with median":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif method == "Fill with mode":
            for col in df_processed.columns:
                if not df_processed[col].mode().empty:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    if operation in ['duplicates', 'all']:
        df_processed = df_processed.drop_duplicates()

    return df_processed

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================
def create_eda_plots(df, column, plot_type):
    """Create exploratory data analysis plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'histogram':
        ax.hist(df[column].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    elif plot_type == 'boxplot':
        ax.boxplot(df[column].dropna(), vert=True)
        ax.set_ylabel(column, fontsize=12)
        ax.set_title(f'Box Plot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    elif plot_type == 'correlation':
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                    square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

    elif plot_type == 'scatter':
        x_col, y_col = column
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color='coral')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{y_col} vs {x_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_results(y_true, y_pred, task_type, plot_type, feature_names=None):
    """Plot model results"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'confusion_matrix':
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    elif plot_type == 'regression_plot':
        ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif plot_type == 'clustering_plot':
        unique_labels = np.unique(y_pred)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = y_pred == label
            ax.scatter(y_true[mask, 0], y_true[mask, 1], 
                      c=[color], label=f'Cluster {label}', alpha=0.6, s=50)

        ax.set_xlabel(feature_names[0] if feature_names else 'Feature 1', fontsize=12)
        ax.set_ylabel(feature_names[1] if feature_names else 'Feature 2', fontsize=12)
        ax.set_title('Clustering Results', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# =====================================================
# MACHINE LEARNING FUNCTIONS
# =====================================================
def train_model(df, features, target, algorithm, task_type, test_size=0.2, params=None):
    """Train machine learning model"""
    if task_type in ["Classification", "Regression"]:
        X = df[features].dropna()
        y = df.loc[X.index, target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if task_type == "Classification":
            if algorithm == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=5)
            elif algorithm == "Support Vector Machine":
                model = SVC(kernel='rbf', random_state=42)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42, max_depth=5)
            elif algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Support Vector Regression":
                model = SVR(kernel='rbf')
            elif algorithm == "Decision Tree Regression":
                model = DecisionTreeRegressor(random_state=42, max_depth=5)

        model.fit(X_train_scaled, y_train)
        return model, X_test_scaled, X_test, y_train, y_test

    else:  # Clustering
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if algorithm == "K-Means":
            model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
        elif algorithm == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=params['n_clusters'])
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

        labels = model.fit_predict(X_scaled)
        return model, X_scaled, labels

def evaluate_model(model, X_test, y_test, task_type):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    if task_type == "Classification":
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    else:
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

    return metrics

# =====================================================
# GLOBAL STATE
# =====================================================
state = {
    'data': None,
    'processed_data': None,
    'model': None
}

# =====================================================
# INTERFACE FUNCTIONS WITH AUTO-UPDATING DROPDOWNS
# =====================================================
def load_data_interface(file):
    """Load data and update all dropdowns"""
    try:
        df = load_data(file)
        state['data'] = df

        # Get numeric columns for dropdowns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        info_text = f"""✅ Data loaded successfully!

📊 Dataset Information:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""

        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })

        # Return updates for all dropdown components
        return (
            info_text, 
            df.head(20), 
            col_info,
            gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # hist_col
            gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # box_col
            gr.update(choices=numeric_cols, value=numeric_cols[0] if len(numeric_cols) > 0 else None),  # scatter_x
            gr.update(choices=numeric_cols, value=numeric_cols[1] if len(numeric_cols) > 1 else None),  # scatter_y
            gr.update(choices=numeric_cols, value=numeric_cols[0] if numeric_cols else None),  # target_col
            gr.update(choices=numeric_cols, value=numeric_cols[1:4] if len(numeric_cols) > 1 else []),  # feature_cols
            gr.update(choices=numeric_cols, value=numeric_cols[:3] if len(numeric_cols) >= 2 else []),  # feature_cols_clust
        )
    except Exception as e:
        return (
            f"❌ Error: {str(e)}", 
            None, 
            None,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
        )

def show_statistics():
    """Show descriptive statistics with labeled rows and explanation"""
    if state['data'] is None:
        return "⚠️ Please load data first", None

    # Get statistics
    stats = state['data'].describe()

    # Reset index to make the statistic names a column
    stats_with_labels = stats.reset_index()

    # Rename the index column to "Statistic"
    stats_with_labels.rename(columns={'index': 'Statistic'}, inplace=True)

    # Add friendly descriptions
    stat_descriptions = {
        'count': 'count (Total Values)',
        'mean': 'mean (Average)',
        'std': 'std (Standard Deviation)',
        'min': 'min (Minimum)',
        '25%': '25% (Q1 - First Quartile)',
        '50%': '50% (Median)',
        '75%': '75% (Q3 - Third Quartile)',
        'max': 'max (Maximum)'
    }

    # Replace statistic names with descriptions
    stats_with_labels['Statistic'] = stats_with_labels['Statistic'].map(
        lambda x: stat_descriptions.get(x, x)
    )

    # Explanation text
    explanation = """📊 Statistical Summary Explanation:

• count: Total number of non-missing values in each column
• mean: Average value (sum of all values ÷ number of values)
• std: Standard deviation - measures how spread out the data is (high = diverse, low = similar)
• min: Minimum value (smallest value in the column)
• 25% (Q1): First quartile - 25% of data is below this value
• 50% (Median): Middle value - 50% above and 50% below
• 75% (Q3): Third quartile - 75% of data is below this value
• max: Maximum value (largest value in the column)

📈 Use these statistics to understand your data distribution before training models.
"""

    return explanation, stats_with_labels

def show_missing_info():
    if state['data'] is None:
        return "⚠️ Please load data first"

    df = state['data']
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Missing': df.isnull().sum().values,
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
    })
    return info_df

def create_histogram(column):
    if state['data'] is None or column is None:
        return None
    return create_eda_plots(state['data'], column, 'histogram')

def create_correlation():
    if state['data'] is None:
        return None
    numeric_cols = state['data'].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    return create_eda_plots(state['data'][numeric_cols], None, 'correlation')

def create_boxplot(column):
    if state['data'] is None or column is None:
        return None
    return create_eda_plots(state['data'], column, 'boxplot')

def create_scatterplot(x_col, y_col):
    if state['data'] is None or x_col is None or y_col is None:
        return None
    return create_eda_plots(state['data'], [x_col, y_col], 'scatter')

def clean_data(operation, method):
    if state['data'] is None:
        return "⚠️ Please load data first", None

    try:
        df = state['data'].copy()

        if operation == "Handle Missing Values":
            df_cleaned = preprocess_data(df, 'missing', method)
        elif operation == "Remove Duplicates":
            df_cleaned = preprocess_data(df, 'duplicates', None)
        else:
            df_cleaned = preprocess_data(df, 'all', 'drop')

        state['processed_data'] = df_cleaned

        result_text = f"""✅ Data cleaned successfully!

Original shape: {df.shape}
New shape: {df_cleaned.shape}
Rows removed: {df.shape[0] - df_cleaned.shape[0]}
"""
        return result_text, df_cleaned.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def train_supervised_model(task_type, algorithm, target, features, test_size):
    if state['data'] is None:
        return "⚠️ Please load data first"
    if not target or not features:
        return "⚠️ Please select target and features"

    try:
        df = state['processed_data'] if state['processed_data'] is not None else state['data']
        model, X_test, X_test_orig, y_train, y_test = train_model(
            df, features, target, algorithm, task_type, test_size/100
        )

        state['model'] = {
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_train': y_train,
            'task_type': task_type,
            'algorithm': algorithm,
            'features': features,
            'target': target
        }
        return "✅ Model trained successfully! Go to Results & Evaluation tab."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def train_clustering_model(algorithm, features, n_clusters, eps, min_samples):
    if state['data'] is None:
        return "⚠️ Please load data first"
    if not features or len(features) < 2:
        return "⚠️ Please select at least 2 features"

    try:
        df = state['processed_data'] if state['processed_data'] is not None else state['data']

        if algorithm == "K-Means":
            params = {'n_clusters': int(n_clusters)}
        elif algorithm == "DBSCAN":
            params = {'eps': eps, 'min_samples': int(min_samples)}
        else:
            params = {'n_clusters': int(n_clusters)}

        model, X_scaled, labels = train_model(df, features, None, algorithm, "Clustering", params=params)

        state['model'] = {
            'model': model,
            'X_scaled': X_scaled,
            'labels': labels,
            'task_type': "Clustering",
            'algorithm': algorithm,
            'features': features
        }
        return "✅ Clustering model trained successfully! Go to Results & Evaluation tab."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def show_results():
    """Display model results with detailed information"""
    if state['model'] is None:
        return "⚠️ Please train a model first", None, None

    model_info = state['model']
    task_type = model_info['task_type']

    # Build detailed model information
    if task_type in ["Classification", "Regression"]:
        # For supervised learning
        features_list = ', '.join(model_info['features'])

        # Calculate test set size percentage
        total_samples = len(model_info['X_test']) + len(model_info.get('y_train', []))
        test_percentage = (len(model_info['X_test']) / total_samples * 100) if total_samples > 0 else 0

        info_text = f"""📊 Model Information:
- Algorithm: {model_info['algorithm']}
- Task Type: {task_type}
- Target Variable: {model_info['target']}
- Features: {features_list}
- Number of Features: {len(model_info['features'])}
- Test Set Size: {len(model_info['X_test'])} samples ({test_percentage:.1f}%)
- Training Set Size: {len(model_info.get('y_train', []))} samples
"""

        metrics = evaluate_model(model_info['model'], model_info['X_test'], model_info['y_test'], task_type)

        if task_type == "Classification":
            metrics_text = f"""📈 Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1']:.4f}
"""
            fig = plot_results(model_info['y_test'], model_info['model'].predict(model_info['X_test']), 
                             task_type, 'confusion_matrix')
        else:
            metrics_text = f"""📈 Performance Metrics:
- R² Score: {metrics['r2']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
"""
            fig = plot_results(model_info['y_test'], model_info['model'].predict(model_info['X_test']), 
                             task_type, 'regression_plot')

        return info_text, metrics_text, fig
    else:
        # For clustering
        features_list = ', '.join(model_info['features'])

        silhouette = silhouette_score(model_info['X_scaled'], model_info['labels'])
        calinski = calinski_harabasz_score(model_info['X_scaled'], model_info['labels'])
        n_clusters = len(np.unique(model_info['labels']))

        info_text = f"""📊 Model Information:
- Algorithm: {model_info['algorithm']}
- Task Type: {task_type}
- Features: {features_list}
- Number of Features: {len(model_info['features'])}
- Total Samples: {len(model_info['X_scaled'])}
- Clusters Found: {n_clusters}
"""

        metrics_text = f"""📈 Clustering Metrics:
- Silhouette Score: {silhouette:.4f}
- Calinski-Harabasz: {calinski:.2f}
- Number of Clusters: {n_clusters}
"""
        fig = plot_results(model_info['X_scaled'], model_info['labels'], task_type, 
                         'clustering_plot', feature_names=model_info['features'][:2])

        return info_text, metrics_text, fig

# =====================================================
# BUILD GRADIO INTERFACE WITH AUTO-UPDATING DROPDOWNS
# =====================================================
# Custom CSS for footer
custom_css = """
footer {
    background-color: #1a1a1a !important;
    color: white !important;
    padding: 20px !important;
    text-align: center !important;
    margin-top: 40px !important;
    border-top: 3px solid #4a90e2 !important;
}
"""

with gr.Blocks(title="No-Code Data Mining Platform", theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("""
    # 📊 No-Code Data Mining Platform
    """)

    with gr.Tabs():
        # TAB 1: DATA LOADING
        with gr.Tab("📂 Data Loading"):
            gr.Markdown("## Upload Your Dataset")
            file_input = gr.File(label="Upload CSV or Excel file", file_types=[".csv", ".xlsx", ".xls"])
            load_btn = gr.Button("Load Data", variant="primary", size="lg")
            info_output = gr.Textbox(label="Status", lines=8)
            preview_output = gr.Dataframe(label="Dataset Preview", interactive=False)
            columns_output = gr.Dataframe(label="Column Information", interactive=False)

        # TAB 2: EDA & PREPROCESSING
        with gr.Tab("🔍 EDA & Preprocessing"):
            gr.Markdown("## Exploratory Data Analysis & Preprocessing")

            with gr.Tabs():
                with gr.Tab("📊 Exploration"):
                    stat_btn = gr.Button("Show Statistics", variant="primary")
                    stat_explanation = gr.Textbox(label="What These Statistics Mean", lines=12)
                    stat_output = gr.Dataframe(label="Statistical Summary")
                    stat_btn.click(show_statistics, outputs=[stat_explanation, stat_output])

                    missing_btn = gr.Button("Show Missing Values Info", variant="primary")
                    missing_output = gr.Dataframe(label="Missing Values Information")
                    missing_btn.click(show_missing_info, outputs=[missing_output])

                with gr.Tab("📈 Visualization"):
                    gr.Markdown("### Distribution Plot")
                    hist_col = gr.Dropdown(label="Select Column", choices=[], value=None)
                    hist_btn = gr.Button("Create Histogram", variant="primary")
                    hist_plot = gr.Plot(label="Histogram")
                    hist_btn.click(create_histogram, inputs=[hist_col], outputs=[hist_plot])

                    gr.Markdown("---")
                    gr.Markdown("### Correlation Matrix")
                    corr_btn = gr.Button("Create Correlation Matrix", variant="primary")
                    corr_plot = gr.Plot(label="Correlation Matrix")
                    corr_btn.click(create_correlation, outputs=[corr_plot])

                    gr.Markdown("---")
                    gr.Markdown("### Box Plot")
                    box_col = gr.Dropdown(label="Select Column", choices=[], value=None)
                    box_btn = gr.Button("Create Box Plot", variant="primary")
                    box_plot = gr.Plot(label="Box Plot")
                    box_btn.click(create_boxplot, inputs=[box_col], outputs=[box_plot])

                    gr.Markdown("---")
                    gr.Markdown("### Scatter Plot")
                    with gr.Row():
                        scatter_x = gr.Dropdown(label="X-axis", choices=[], value=None)
                        scatter_y = gr.Dropdown(label="Y-axis", choices=[], value=None)
                    scatter_btn = gr.Button("Create Scatter Plot", variant="primary")
                    scatter_plot = gr.Plot(label="Scatter Plot")
                    scatter_btn.click(create_scatterplot, inputs=[scatter_x, scatter_y], outputs=[scatter_plot])

                with gr.Tab("🧹 Cleaning"):
                    clean_op = gr.Radio(
                        ["Handle Missing Values", "Remove Duplicates", "Apply All Cleaning"],
                        label="Select Cleaning Operation", value="Handle Missing Values"
                    )
                    clean_method = gr.Dropdown(
                        ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode"],
                        label="Method for Missing Values", value="Drop rows with missing values"
                    )
                    clean_btn = gr.Button("Apply Cleaning", variant="primary", size="lg")
                    clean_output = gr.Textbox(label="Status", lines=6)
                    clean_preview = gr.Dataframe(label="Cleaned Data Preview")
                    clean_btn.click(clean_data, inputs=[clean_op, clean_method], 
                                  outputs=[clean_output, clean_preview])

        # TAB 3: MACHINE LEARNING
        with gr.Tab("🤖 Machine Learning"):
            gr.Markdown("## Train Machine Learning Models")

            task_type = gr.Radio(["Classification", "Regression", "Clustering"], 
                               label="Select ML Task", value="Classification")

            with gr.Column(visible=True) as supervised_section:
                gr.Markdown("### Supervised Learning Configuration")
                algorithm_sup = gr.Dropdown(
                    ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"],
                    label="Select Algorithm", value="K-Nearest Neighbors"
                )
                target_col = gr.Dropdown(label="Target Variable (what to predict)", choices=[], value=None)
                feature_cols = gr.Dropdown(
                    label="Feature Variables (predictors) - Select multiple", 
                    choices=[], 
                    multiselect=True, 
                    value=[]
                )
                test_size = gr.Slider(10, 40, value=20, step=5, label="Test Set Size (%)")
                train_sup_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
                train_sup_output = gr.Textbox(label="Training Status", lines=4)
                train_sup_btn.click(train_supervised_model, 
                                  inputs=[task_type, algorithm_sup, target_col, feature_cols, test_size],
                                  outputs=[train_sup_output])

            with gr.Column(visible=False) as clustering_section:
                gr.Markdown("### Unsupervised Learning Configuration (Clustering)")
                algorithm_clust = gr.Dropdown(
                    ["K-Means", "Hierarchical Clustering", "DBSCAN"],
                    label="Select Algorithm", value="K-Means"
                )
                feature_cols_clust = gr.Dropdown(
                    label="Feature Variables - Select at least 2", 
                    choices=[], 
                    multiselect=True, 
                    value=[]
                )
                with gr.Row():
                    n_clusters = gr.Slider(2, 10, value=3, step=1, label="Number of Clusters (K-Means/Hierarchical)")
                with gr.Row():
                    eps_param = gr.Slider(0.1, 5.0, value=0.5, step=0.1, label="Epsilon (DBSCAN)")
                    min_samples = gr.Slider(2, 10, value=5, step=1, label="Min Samples (DBSCAN)")
                train_clust_btn = gr.Button("🚀 Train Clustering Model", variant="primary", size="lg")
                train_clust_output = gr.Textbox(label="Training Status", lines=4)
                train_clust_btn.click(train_clustering_model,
                                    inputs=[algorithm_clust, feature_cols_clust, n_clusters, eps_param, min_samples],
                                    outputs=[train_clust_output])

            def update_task_interface(task):
                if task == "Clustering":
                    return gr.update(visible=False), gr.update(visible=True)
                return gr.update(visible=True), gr.update(visible=False)

            def update_algorithm_choices(task):
                if task == "Classification":
                    return gr.update(choices=["K-Nearest Neighbors", "Support Vector Machine", 
                                             "Decision Tree", "Random Forest"], value="K-Nearest Neighbors")
                return gr.update(choices=["Linear Regression", "Support Vector Regression", 
                                        "Decision Tree Regression"], value="Linear Regression")

            task_type.change(update_task_interface, inputs=[task_type], 
                           outputs=[supervised_section, clustering_section])
            task_type.change(update_algorithm_choices, inputs=[task_type], outputs=[algorithm_sup])

        # TAB 4: RESULTS & EVALUATION
        with gr.Tab("📈 Results & Evaluation"):
            gr.Markdown("## Model Performance & Evaluation")
            eval_btn = gr.Button("Show Results", variant="primary", size="lg")
            model_info_output = gr.Textbox(label="Model Information", lines=10)
            metrics_output = gr.Textbox(label="Performance Metrics", lines=8)
            plot_output = gr.Plot(label="Visualization")
            eval_btn.click(show_results, outputs=[model_info_output, metrics_output, plot_output])

    # FOOTER
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; background-color: #1a1a1a; color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
        <h3 style='margin: 0; color: #4a90e2;'>Université Abbas Laghrour - Khenchela | Master 2 IA | 2025-2026</h3>
        <p style='margin: 10px 0 0 0; font-size: 16px;'>👨‍💻 Developed by: <strong>Djoghlal Abid</strong></p>
    </div>
    """)

    # Connect load button to update all dropdowns
    load_btn.click(
        load_data_interface,
        inputs=[file_input],
        outputs=[
            info_output, preview_output, columns_output,
            hist_col, box_col, scatter_x, scatter_y,
            target_col, feature_cols, feature_cols_clust
        ]
    )

if __name__ == "__main__":
    app.launch()
