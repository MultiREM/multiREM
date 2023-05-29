import json
import os

from utils.list_to_json import list_to_json

def merge_biosamples_into_runs():
    with open(os.getcwd() + '/data_transforms/biosamples+projects.json', 'r') as samples_file:
        with open(os.getcwd() + '/data_transforms/runs.json', 'r') as runs_file:
            try:
                # load data
                biosamples_data = json.load(samples_file)
                runs_data = json.load(runs_file)

                # key projects on id w/ dict as value
                biosamples_dict = { bp['biosample_accession']: bp for bp in biosamples_data }

                # map over biosamples
                def extend_run_with_sample(run):
                    run_with_sample_data = { **run, **biosamples_dict.get(run.get('biosample_accession'), {}) }
                    return run_with_sample_data
                extended_runs_data = list(map(extend_run_with_sample, runs_data))

                # list -> json
                list_to_json(extended_runs_data, './data_transforms/runs_samples.json')
                
            except json.JSONDecodeError as e:
                print(f"Error: Failed to load JSON file. {e}")
                return None


if __name__ == "__main__":
    merge_biosamples_into_runs()
