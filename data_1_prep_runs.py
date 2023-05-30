import json
import os
from utils.list_to_json import list_to_json

def prep_runs(data):    
    rows = []
    # get samples
    for sample in data:
        row = dict()
        # run info
        row['run_accession'] = sample.get('Run')
        row['run_library_source'] = sample.get('LibrarySource').lower()
        row['run_library_selection'] = sample.get('LibrarySelection').lower()
        row['run_bases'] = sample.get('bases')
        row['run_size_mb'] = sample.get('size_MB')
        row['run_download_lite_url'] = sample.get('download_path')
        row['run_download_fasta_url'] = 'https://trace.ncbi.nlm.nih.gov/Traces/sra-reads-be/fasta?acc=' + sample.get('Run')
        # experiment info
        row['experiment_accession'] = sample.get('Experiment')
        # sample info
        row['biosample_accession'] = sample.get('BioSample')
        # project info
        row['bioproject_accession'] = sample.get('BioProject')
        row['bioproject_accession_id'] = str(sample.get('ProjectID'))
        rows.append(row)
    # list -> json
    list_to_json(rows, './data_transforms/runs.json')


if __name__ == "__main__":
    def run():
        dna_json_file_path = os.getcwd() + '/data_sources/run_dna.json'
        rna_json_file_path = os.getcwd() + '/data_sources/run_rna.json'
        with open(dna_json_file_path, 'r') as dna_file:
            with open(rna_json_file_path, 'r') as rna_file:
                try:
                    dna_json_data = json.load(dna_file)
                    rna_json_data = json.load(rna_file)
                    json_data = dna_json_data + rna_json_data
                    data = prep_runs(json_data)
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to load JSON file. {e}")
                    return None
        if data:
            print("JSON loaded successfully into a dictionary.")
            print(data)
    run()
