"""LA City Payroll Data Loader and Analyzer

This script downloads, cleans, and loads LA City payroll data from Kaggle into a SQLite database.
It performs data cleaning on monetary values and percentages, creates optimized database indices,
and runs various analyses on the dataset including salary statistics and benefits analysis.

Dataset Source: City of LA payroll data from Kaggle
Database: SQLite (city_payroll.db)

Main Features:
- Automatic dataset download via kagglehub
- Data cleaning for monetary values and percentages
- SQLite database creation with optimized indices
- Comprehensive data validation and summary statistics
- Benefits and compensation analysis
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import kagglehub
import re

def clean_monetary_value(value):
    """Convert monetary strings to float values.
    
    Handles various input formats including:
    - String format: '$70,386.48' -> 70386.48
    - Already numeric: 70386.48 -> 70386.48
    - Missing/invalid: None -> None
    
    Parameters:
    -----------
    value : str, int, float, or None
        The monetary value to clean
    
    Returns:
    --------
    float or None
        Cleaned numeric value, or None if conversion fails
    
    Examples:
    ---------
    >>> clean_monetary_value('$70,386.48')
    70386.48
    >>> clean_monetary_value(70386.48)
    70386.48
    >>> clean_monetary_value(None)
    None
    """
    # Handle missing values
    if pd.isna(value):
        return None

    # If it's already a number, just return it as float
    if isinstance(value, (int, float)):
        return float(value)

    # If it's a string, clean it up
    if isinstance(value, str):
        # Remove all non-numeric characters except decimal point and minus sign
        # This handles: '$', ',', and any other formatting characters
        value = re.sub(r'[^\d.-]', '', value)
        
        # Attempt conversion to float
        try:
            return float(value)
        except ValueError:
            # If conversion fails, return None (handles malformed strings)
            return None

    # If it's some other type we don't expect, return None
    return None


def clean_percentage(value):
    """Convert percentage strings to float values.
    
    Removes the '%' symbol and converts to numeric format.
    Note: Returns the percentage value itself (e.g., '23.67%' -> 23.67), 
    not the decimal representation (0.2367).
    
    Parameters:
    -----------
    value : str or None
        The percentage string to clean (e.g., '23.67%')
    
    Returns:
    --------
    float or None
        Numeric percentage value, or None if conversion fails
    
    Examples:
    ---------
    >>> clean_percentage('23.67%')
    23.67
    >>> clean_percentage(None)
    None
    """
    # Only process non-missing string values
    if pd.isna(value) or not isinstance(value, str):
        return None
    
    # Remove the '%' symbol
    value = value.replace('%', '')
    
    # Attempt conversion to float
    try:
        return float(value)
    except ValueError:
        # Return None for malformed percentage strings
        return None

def load_payroll_data(csv_path, db_path='city_payroll.db'):
    """
    Load, clean, and import LA City payroll data into SQLite database.
    
    This function performs several key operations:
    1. Reads CSV file with pandas (handles large datasets)
    2. Standardizes column names (replaces spaces with underscores)
    3. Cleans monetary and percentage columns
    4. Creates SQLite database with optimized indices
    5. Validates data with summary queries
    
    Parameters:
    -----------
    csv_path : str
        Path to the source CSV file containing payroll data
    db_path : str, optional
        Path where SQLite database will be created (default: 'city_payroll.db')
        Will overwrite if file already exists
    
    Returns:
    --------
    pandas.DataFrame
        The loaded and cleaned dataframe
    
    Database Schema:
    ----------------
    Creates 'payroll' table with cleaned data and the following indices:
    - idx_year: Index on Year column for temporal queries
    - idx_dept: Index on Department_Title for department-based queries
    - idx_job: Index on Job_Class_Title for job-based queries
    """
    # Read the CSV file using pandas
    print(f"Reading CSV file: {csv_path}")
    # Set low_memory=False to avoid DtypeWarning when pandas infers column types
    # This prevents issues with mixed-type columns in large datasets
    df = pd.read_csv(csv_path, low_memory=False)

    # Display basic dataset information
    print(f"Dataset shape: {df.shape}")  # Shows (rows, columns)

    # Standardize column names: replace spaces with underscores
    # This makes SQL queries easier and avoids quoting issues
    # Example: "Base Pay" -> "Base_Pay"
    df.columns = [col.replace(' ', '_') for col in df.columns]

    # Clean monetary columns: convert string representations to float values
    # These columns typically contain values like '$70,386.48' that need cleaning
    # List includes all known monetary columns in the LA City payroll dataset
    monetary_columns = [
        'Hourly_or_Event_Rate',
        'Projected_Annual_Salary', 'Q1_Payments', 'Q2_Payments', 'Q3_Payments',
        'Q4_Payments', 'Payments_Over_Base_Pay', 'Total_Payments', 'Base_Pay',
        'Permanent_Bonus_Pay', 'Longevity_Bonus_Pay', 'Temporary_Bonus_Pay',
        'Lump_Sum_Pay', 'Overtime_Pay', 'Other_Pay_&_Adjustments',
        'Other_Pay_(Payroll_Explorer)', 'Average_Health_Cost', 'Average_Dental_Cost',
        'Average_Basic_Life', 'Average_Benefit_Cost'
    ]

    # Apply monetary cleaning to each column if it exists in the dataset
    # Some columns may not be present in all versions of the dataset
    for col in monetary_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_monetary_value)

    # Clean the percentage column if it exists
    # Converts values like '23.67%' to float 23.67
    if '%_Over_Base_Pay' in df.columns:
        df['%_Over_Base_Pay'] = df['%_Over_Base_Pay'].apply(clean_percentage)

    # Connect to SQLite database (creates file if it doesn't exist)
    print(f"Creating/connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)

    # Create 'payroll' table and insert all data
    # if_exists='replace': drops existing table and recreates it
    # index=False: don't write pandas DataFrame index as a column
    df.to_sql('payroll', conn, if_exists='replace', index=False)

    # Create database indices to optimize query performance
    # Indices speed up WHERE, JOIN, and ORDER BY operations on these columns
    print("Creating indices...")
    cursor = conn.cursor()
    try:
        # Index on Year: optimizes temporal queries and year-based aggregations
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON payroll(Year)')
        # Index on Department_Title: speeds up department-based filtering and grouping
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dept ON payroll(Department_Title)')
        # Index on Job_Class_Title: optimizes job title searches and classifications
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_job ON payroll(Job_Class_Title)')
        conn.commit()
    except sqlite3.OperationalError as e:
        # Handle cases where indices might already exist or table structure issues
        print(f"Warning: Could not create some indices: {e}")
        conn.rollback()

    # Run validation queries to ensure data loaded correctly
    # and provide useful summary statistics
    print("\nData Summary:")

    # Count records by year to verify temporal distribution
    # Helps identify which years are covered and data completeness
    cursor.execute('SELECT Year, COUNT(*) FROM payroll GROUP BY Year')
    years_data = cursor.fetchall()
    print("\nRecords by Year:")
    for year, count in years_data:
        print(f"  {year}: {count} records")

    # Department distribution: shows which departments have most employees
    # Limited to top 10 to keep output manageable
    # Useful for understanding city workforce composition
    cursor.execute(
        'SELECT Department_Title, COUNT(*) as count FROM payroll GROUP BY Department_Title ORDER BY count DESC LIMIT 10')
    dept_data = cursor.fetchall()
    print("\nTop 10 Department Distribution:")
    for dept, count in dept_data:
        print(f"  {dept}: {count} records")

    # Compute basic salary statistics across all records
    # Provides quick overview of compensation ranges
    # Uses cleaned monetary values for accurate calculations
    cursor.execute('''
        SELECT 
            AVG(Total_Payments) as avg_total,
            MIN(Total_Payments) as min_total,
            MAX(Total_Payments) as max_total
        FROM payroll
    ''')
    salary_stats = cursor.fetchone()
    print("\nSalary Statistics:")
    print(f"  Average Total Payment: ${salary_stats[0]:.2f}")
    print(f"  Minimum Total Payment: ${salary_stats[1]:.2f}")
    print(f"  Maximum Total Payment: ${salary_stats[2]:.2f}")

    # Clean up: close database connection
    conn.close()
    print("\nDatabase created successfully!")

    # Return the cleaned dataframe for further analysis in the main script
    return df


# Main execution block
# Only runs when script is executed directly (not when imported)
if __name__ == "__main__":
    # Download LA City payroll dataset from Kaggle using kagglehub
    # The dataset is cached locally after first download
    # Dataset contains employee payroll information from City of Los Angeles
    path = kagglehub.dataset_download("cityofLA/city-payroll-data")
    print("Path to dataset files:", path)

    # Locate the CSV file in the downloaded dataset
    # Try default filename first
    csv_file = os.path.join(path, 'data.csv')

    # If the file exists at the default location, load it
    if os.path.exists(csv_file):
        df = load_payroll_data(csv_file)
    else:
        # If not found with default name, search recursively for any CSV file
        # This handles cases where the dataset structure might change
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    df = load_payroll_data(csv_file)
                    break  # Use first CSV file found

    # Display first few rows of the loaded dataset
    # Useful for quick verification of data structure and content
    print("\nData Preview:")
    print(df.head())

    # Generate a detailed summary of the dataset
    print("\nDetailed Dataset Summary:")
    
    # Analyze job title distribution
    # Shows most common positions in city government
    job_titles = df['Job_Class_Title'].value_counts().head(10)
    print(f"Top 10 Job Titles (out of {df['Job_Class_Title'].nunique()} unique):")
    for title, count in job_titles.items():
        print(f"  {title}: {count}")

    # Analyze the difference between total payments and base pay
    # This reveals overtime, bonuses, and other additional compensation
    df['Pay_Difference'] = df['Total_Payments'] - df['Base_Pay']

    # Calculate percentage difference safely, avoiding division by zero
    # Initialize as empty float series to avoid dtype issues
    df['Pay_Difference_%'] = pd.Series(dtype='float64')

    # Only calculate percentage for valid rows (where Base Pay exists and is positive)
    # This prevents division by zero and NaN propagation
    valid_base_pay = (df['Base_Pay'] > 0) & (~df['Base_Pay'].isna())
    if valid_base_pay.any():
        # Calculate: (Total - Base) / Base * 100
        # Shows what percentage over base pay employees receive on average
        df.loc[valid_base_pay, 'Pay_Difference_%'] = (
                df.loc[valid_base_pay, 'Pay_Difference'] / df.loc[valid_base_pay, 'Base_Pay'] * 100
        )

    # Prepare data for reporting by removing missing values
    # This prevents numpy warnings about NaN in statistical operations
    pay_diff = df['Pay_Difference'].dropna()
    pay_diff_pct = df['Pay_Difference_%'].dropna()

    # Report average difference between total and base pay
    # Shows typical additional compensation beyond base salary
    if not pay_diff.empty:
        print("\nAverage Pay Difference: ${:.2f}".format(pay_diff.mean()))
    else:
        print("\nAverage Pay Difference: No valid data")

    # Report average percentage increase over base pay
    # Indicates how much more employees earn beyond their base salary
    if not pay_diff_pct.empty:
        print("Average Pay Difference %: {:.2f}%".format(pay_diff_pct.mean()))
    else:
        print("Average Pay Difference %: No valid data")

    # Analyze benefits costs if available in the dataset
    # Not all versions of the dataset include this information
    if 'Average_Benefit_Cost' in df.columns:
        print("\nBenefits Analysis:")
        
        # Ensure the column contains valid numeric data
        # Convert any string representations to numeric, coerce errors to NaN
        df['Average_Benefit_Cost'] = pd.to_numeric(df['Average_Benefit_Cost'], errors='coerce')

        # Remove missing values (NaN) that couldn't be converted
        benefit_costs = df['Average_Benefit_Cost'].dropna()

        # Remove infinite values that could cause calculation errors
        # Can occur from malformed data or division errors
        benefit_costs = benefit_costs[~np.isinf(benefit_costs)]

        # If we have valid benefit cost data, compute statistics
        if not benefit_costs.empty:
            # Display basic statistics about benefits costs
            print("Average Benefits Cost: ${:.2f}".format(benefit_costs.mean()))
            print("Min Benefits Cost: ${:.2f}".format(benefit_costs.min()))
            print("Max Benefits Cost: ${:.2f}".format(benefit_costs.max()))
            
            # Data quality reporting
            print("Number of records with valid benefit data: {}".format(len(benefit_costs)))
            print("Number of records with invalid or missing benefit data: {}".format(df.shape[0] - len(benefit_costs)))

            # Data validation: check for anomalies
            # Negative benefit costs would indicate data quality issues
            low_benefits = benefit_costs[benefit_costs < 0].count()
            if low_benefits > 0:
                print(f"Warning: {low_benefits} records have negative benefit costs")

            # Zero benefit costs might indicate part-time or contract workers
            zero_benefits = benefit_costs[benefit_costs == 0].count()
            if zero_benefits > 0:
                print(f"Note: {zero_benefits} records have zero benefit costs")
        else:
            # No valid benefit data found in the dataset
            print("No valid benefit cost data available.")