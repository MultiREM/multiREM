import pandas as pd

def list_to_csv(list, csv_file_path):
    # Read python list
    df = pd.DataFrame(list)
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
