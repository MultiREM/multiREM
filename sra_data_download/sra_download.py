#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import threading
import ntpath
import concurrent.futures

import boto3
from boto3.s3.transfer import TransferConfig

from tqdm import tqdm
import pandas as pd
import awswrangler as wr
from hurry.filesize import size, si 
from pathlib import Path

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        # self._size = float(os.path.getsize(filename))
        self._size = float(Path(filename).stat().st_size)
        self._seen_so_far = 0
        self._lock = threading.Lock()
 
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write("\r%s  %s / %s  (%.2f%%)" % (ntpath.basename(self._filename), size(self._seen_so_far), size(self._size), percentage))
            sys.stdout.flush()

# Globals 
GB = 1024 ** 3
MP_THRESHOLD = 1
MP_CONCURRENCY = 5
MAX_RETRY_COUNT = 3

# Create the arguments parser
parser = argparse.ArgumentParser(description="Download SRA sequences")
parser.add_argument('csv', help="Path to the input CSV file")
args = parser.parse_args()

# Path to data
data_path = args.csv
output_path = os.path.join(os.getcwd(), 'SRA_sequences')

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the data into a DataFrame
df = pd.read_csv(data_path)

# Filter the DataFrame for transcriptomic data
df_transcriptomic = df[df['run_library_source'] == 'transcriptomic']

# Create a new 'file_name' column
df_transcriptomic['file_name'] = df_transcriptomic['run_accession'] + '_' + df_transcriptomic['organism_taxonomy_id'].astype(str)

# S3 Bucket details
bucket_name = 'methanotrophs'  # Replace with your S3 bucket name

# Create an S3 client
s3 = boto3.client('s3')

# Existing uris in S3
existing_uris = wr.s3.list_objects('s3://methanotrophs/SRA_sequences')

def download_and_upload_file(row):
    url = row['run_download_fastq_url']
    file_name = row['file_name'] + '.fastq.gz'
    file_path = os.path.join(output_path, file_name)

    try: 
        print(f"Starting download for {file_name}", flush=True)

        if os.path.exists(file_path): 
            print(f'{file_name} already exists locally. Skipping download...', flush=True)
        else: 
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure the request was successful

            file_size = int(response.headers.get('Content-Length', 0))  # Get the total file size
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, file=sys.stdout)  # Create a progress bar

            # Download the file in chunks and write it to local file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))  # Update the progress bar
                    if chunk:  # If you have a chunk, write it to file
                        f.write(chunk)

            progress_bar.close()  # Close the progress bar

    except Exception as e: 
        print(f"Error downloading {file_name}: {e}", flush=True)

    try: 
        # Once the file is fully downloaded, upload it to S3
        # with open(file_path, 'rb') as data:
        # s3_client.upload_fileobj(data, bucket_name, os.path.join("SRA_sequences", file_name))
        print(f"Starting upload for {file_name}", flush=True)

        key = f's3://{bucket}/SRA_sequences/{file_name}'
        if key in existing_uris: 
            print(f'{file_name} already exists in S3. Skipping upload...', flush=True)
        else: 
            config = TransferConfig(multipart_threshold=MP_THRESHOLD*GB, use_threads=True, max_concurrency=MP_CONCURRENCY)
            s3.upload_file(file_path, bucket_name, os.path.join("SRA_sequences", file_name), Config=config, Callback=ProgressPercentage(file_path))

        # Remove file if upload was successful
        os.remove(file_path)
    except Exception as e: 
        print(f"Error uploading {file_name}: {e}", flush=True)

# # Maximum number of concurrent downloads
# max_workers = 5

# # Use a ThreadPoolExecutor to download the files concurrently and upload them to S3
# with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#     executor.map(download_and_upload_file, df_transcriptomic.to_dict('records'))

for i, row in df.iterrows(): 
    download_and_upload_file(row)
