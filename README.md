# 📊 No-Code Data Mining Platform

No-code Data Mining and Machine Learning platform developed as part of the Master 2 Artificial Intelligence program at **Université Abess Laghrour – Khenchela**, academic year **2025–2026**.

**Developer:** Djoghlal Abid

---

## 🎯 Main Features

### 1. Data Loading
- Supports CSV and Excel formats: `.csv`, `.xls`, `.xlsx`
- File upload directly from the local computer through the web interface
- Automatic preview of the first rows of the dataset in a table
- Column information table: data type, non-null count, null count, and number of unique values

### 2. Exploration and Preprocessing (EDA)

#### Exploration
- Descriptive statistics for numerical columns (mean, std, min, max, quartiles)
- Summary of missing values by column with percentages

#### Visualizations
- Histograms for a selected numerical column
- Box plots for a selected numerical column
- Scatter plots between two selected numerical columns
- Correlation matrix heatmap for all numerical features

#### Cleaning
- Handle missing values:
  - Drop rows with missing values
  - Fill with mean, median, or mode for numeric columns
- Remove duplicate rows
- "Apply All Cleaning" option: drop missing values and remove duplicates in one step

**Note:** All visualization and cleaning selectors use dropdown lists that are automatically populated with the dataset's column names after loading the data, so users never need to type column names manually.

### 3. Machine Learning

#### Supervised Learning

**Classification Algorithms:**
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

**Regression Algorithms:**
- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regression

**Configuration:**
- Target variable (what to predict) selected via dropdown from numeric columns
- Feature variables (predictors) selected via multi-select dropdown
- Test set size configurable via slider (10–40%)

#### Unsupervised Learning

**Clustering Algorithms:**
- K-Means
- Hierarchical Clustering (CAH)
- DBSCAN

**Configuration:**
- Feature variables (minimum 2) chosen via multi-select dropdown
- Parameters controlled via sliders:
  - `n_clusters` (for K-Means and Hierarchical)
  - `eps` and `min_samples` for DBSCAN

### 4. Evaluation and Visualization

After training, go to the **"Results & Evaluation"** tab to see metrics and plots.

**Classification:**
- Metrics: Accuracy, Precision, Recall, F1-score (weighted)
- Confusion matrix heatmap

**Regression:**
- Metrics: R², RMSE, MAE
- "Predictions vs Actual" scatter plot with perfect prediction reference line

**Clustering:**
- Metrics: Silhouette Score, Calinski–Harabasz index, number of clusters
- 2D cluster visualization using the first two selected features

---

## 🛠 Technology Stack

- **Frontend / UI:** Gradio (Blocks API, tabs, dropdowns, plots)
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (classification, regression, clustering, metrics)
- **Visualization:** Matplotlib, Seaborn (static plots rendered into Gradio)
- **File Formats:** CSV, Excel via `pandas.read_csv` and `pandas.read_excel`

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or later
- `pip` package manager
- Git (optional, for cloning from GitHub)

### Option 1: Clone from GitHub (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/djo-hub/No-Code-Data-Mining-Plateforme.git
   ```

2. Navigate to the project folder:
   ```bash
   cd No-Code-Data-Mining-Plateforme
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Manual Download

1. Download the repository as ZIP from GitHub
2. Extract the ZIP file
3. Navigate to the extracted folder:
   ```bash
   cd No-Code-Data-Mining-Plateforme
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies (requirements.txt)
```text
gradio==4.44.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
openpyxl==3.1.2
```

---

## 💻 How to Run

### Option 1 – From Terminal / Command Prompt
```bash
python app.py
```
The application will start and open in your browser at:
`http://127.0.0.1:7860`

### Option 2 – From Spyder IDE
1. Open `app.py` in Spyder
2. Press **F5** or click the **Run** button
3. The Gradio interface will launch in your default browser at `http://127.0.0.1:7860`

### Option 3 – From IPython Console
```python
%run app.py
```

---

## 📖 Usage Guide

### Step 1 – Load Data
- Go to the **"Data Loading"** tab
- Upload a CSV or Excel file
- Click **"Load Data"**
- Check the preview table and column information
- **All dropdowns throughout the app will automatically populate with your column names**

### Step 2 – EDA & Preprocessing
- Use **"Exploration"** to view stats and missing values
- Use **"Visualization"** to generate:
  - Histograms (select column from dropdown)
  - Box plots (select column from dropdown)
  - Scatter plots (select X and Y columns from dropdowns)
  - Correlation matrix (automatic for all numeric columns)
- Use **"Cleaning"** to handle missing values and duplicates
- Cleaned data is stored for later steps

### Step 3 – Machine Learning
- Open the **"Machine Learning"** tab
- Choose the task type: Classification, Regression, or Clustering

**For Supervised Learning (Classification/Regression):**
- Select algorithm from dropdown
- Select target variable from dropdown (what you want to predict)
- Select feature variables from multi-select dropdown (predictors)
- Adjust test size with slider
- Click **"🚀 Train Model"**

**For Clustering:**
- Select algorithm from dropdown
- Select at least 2 features from multi-select dropdown
- Adjust parameters with sliders (K for K-Means, eps/min_samples for DBSCAN)
- Click **"🚀 Train Clustering Model"**

### Step 4 – Results & Evaluation
- Go to **"Results & Evaluation"** tab
- Click **"Show Results"**
- Review performance metrics
- View visualizations (confusion matrix, predictions plot, or cluster plot)

---

## 📁 Project Structure

```
No-Code-Data-Mining-Plateforme/
├── __init__.py           # Package initialization
├── app.py                # Main Gradio application entry point
├── data_loader.py        # Data loading and file handling module
├── ml_models.py          # Machine learning algorithms and training
├── preprocessing.py      # Data preprocessing and cleaning functions
├── visualization.py      # EDA plots and results visualization
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation (this file)
```

### Module Descriptions

- **`__init__.py`**: Python package initialization file
- **`app.py`**: Main application file containing the Gradio interface, UI components, tabs, and user interaction logic
- **`data_loader.py`**: Handles file uploads (CSV/Excel), data parsing, and initial data validation
- **`ml_models.py`**: Contains all machine learning algorithms, model training functions, and evaluation metrics
- **`preprocessing.py`**: Data cleaning functions including missing value handling, duplicate removal, and data transformation
- **`visualization.py`**: EDA visualizations (histograms, box plots, scatter plots, correlation matrices) and model results plotting

**Architecture:** The application follows a modular design with clear separation of concerns, making the code maintainable, testable, and easy to extend.

---

## 🔗 GitHub Repository

**Repository Name:** `No-Code-Data-Mining-Plateforme`

**Clone URL:**
```bash
git clone https://github.com/djo-hub/No-Code-Data-Mining-Plateforme.git
```

---

## ✨ Key Advantages

✅ **No-Code Interface:** Users don't need to write any Python code  
✅ **Auto-Populated Dropdowns:** Column names automatically appear in all selectors  
✅ **Runs in Spyder:** Can be executed directly with F5, no terminal commands needed  
✅ **Complete Workflow:** From data loading to model evaluation in one app  
✅ **Multiple Algorithms:** 10+ ML algorithms for classification, regression, and clustering  
✅ **Interactive Visualizations:** All EDA plots and results rendered in the browser  
✅ **Modular Design:** Clean architecture with separated modules for easy maintenance  

---

## 🐛 Troubleshooting

### Error: "Module not found"
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use
If port 7860 is busy, modify the port in `app.py`:
```python
app.launch(server_port=7861)  # Change 7860 to another port
```

### Visualization Issues
Ensure matplotlib is properly installed:
```bash
pip install matplotlib --upgrade
```

### Running from Spyder
Make sure you're running the entire `app.py` file (F5) and not just selected lines (F9).

### Git Clone Issues
If you encounter SSL or connection errors:
```bash
git config --global http.sslVerify false
git clone https://github.com/djo-hub/No-Code-Data-Mining-Plateforme.git
```

### Import Errors from Modules
If you get import errors like "No module named 'data_loader'":
1. Make sure `__init__.py` exists in the project folder
2. Run the app from the project root directory
3. Check that all module files are in the same directory as `app.py`

---

## 📧 Contact

**Developer:** Djoghlal Abid 

**Email:** djoghlal.abid@univ-khenchela.dz

---

## 📝 License

This project is developed for educational purposes as part of the Master 2 AI curriculum.

---

## 🙏 Acknowledgments


- **Framework:** Gradio for rapid ML interface development
- **Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 🚀 Quick Start

For a quick start after cloning:
```bash
# Clone the repository
git clone https://github.com/djo-hub/No-Code-Data-Mining-Plateforme.git

# Navigate to folder
cd No-Code-Data-Mining-Plateforme

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser and go to `http://127.0.0.1:7860`

---

