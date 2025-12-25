import pandas as pd
import io

def load_data(uploaded_file):
    """
    Load data from uploaded CSV or Excel file

    Args:
        uploaded_file: File object or file path

    Returns:
        pandas DataFrame
    """
    try:
        if uploaded_file is None:
            raise ValueError("No file uploaded")

        # Get file name
        if hasattr(uploaded_file, 'name'):
            file_name = uploaded_file.name
        else:
            file_name = str(uploaded_file)

        # Read file based on extension
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

        return df

    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")
