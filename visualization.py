import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def create_eda_plots(df, column, plot_type):
    """
    Create exploratory data analysis plots

    Args:
        df: pandas DataFrame
        column: column name(s) for plotting
        plot_type: type of plot to create

    Returns:
        matplotlib figure
    """
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
    """
    Plot model results

    Args:
        y_true: true values or features (for clustering)
        y_pred: predicted values or labels
        task_type: 'Classification', 'Regression', or 'Clustering'
        plot_type: type of result plot
        feature_names: names of features (for clustering)

    Returns:
        matplotlib figure
    """
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

        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif plot_type == 'clustering_plot':
        # y_true is X_scaled, y_pred is labels
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
