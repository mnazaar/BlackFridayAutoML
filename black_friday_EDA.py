import logging
import os
from dataprep.clean import clean_headers, clean_df

from dataprep.eda import create_report

from ydata_profiling import ProfileReport

# Configure logging for EDA
eda_log_file = "eda_analysis.log"


def eda_pandas_profiling(df, name):
    """Generate Pandas Profiling EDA report."""
    try:
        os.makedirs("reports", exist_ok=True)
        logging.info("Generating Pandas Profiling report...")

        profile = ProfileReport(df, title="Black Friday EDA - Pandas Profiling", explorative=True)
        output_path = f"reports/black_friday_eda_pandas_{name}.html"
        profile.to_file(output_path)

        logging.info(f"Pandas Profiling report saved at {output_path}")
    except Exception as e:
        logging.error(f"Error generating Pandas Profiling report: {e}")



def eda_dataprep(df, report_name):
    print("Before cleaning:", type(df))  # Check type before any processing

    df = clean_headers(df)
    print("After clean_headers:", type(df))  # Should still be a DataFrame

    _, df = clean_df(df)
    print("After clean_df:", type(df))

    # Print each column name and its type
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    print(df.dtypes)

    report = create_report(df)  # If df is not a DataFrame, error occurs here
    report.save(f"reports/dataprep_eda_report_{report_name}.html")