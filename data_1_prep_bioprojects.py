import json
import os
from utils.list_to_csv import list_to_csv
from utils.list_to_json import list_to_json

def convert_bio_project_csv(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            rows = []

            # get samples
            for sample in data.get('Document').get('DocumentSummary'):
                row = dict()
                # bio project lable/id
                row['bioproject_accession'] = sample.get('Project').get('ProjectID').get('ArchiveID').get('@accession')
                row['bioproject_accession_id'] = sample.get('Project').get('ProjectID').get('ArchiveID').get('@id')
                row['bioproject_url'] = 'https://www.ncbi.nlm.nih.gov/bioproject/' + row['bioproject_accession_id']
                row['project_tilte'] = sample.get('Project').get('ProjectDescr').get('Title')
                row['project_description'] = sample.get('Project').get('ProjectDescr').get('Description')
                # organism (name/strain... kind of messy bc can be multiple samples. just relate back on id)
                # relevances
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Agricultural', None) == 'yes':
                    row['is_relevance_agricultural'] = True
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Environmental', None) == 'yes':
                    row['is_relevance_environmental'] = True
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Evolution', None) == 'yes':
                    row['is_relevance_evolution'] = True
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Industrial', None) == 'yes':
                    row['is_relevance_industrial'] = True
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('ModelOrganism', None) == 'yes':
                    row['is_relevance_model_organism'] = True
                if sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Other') != None:
                    row['relevance_other'] = sample.get('Project').get('ProjectDescr').get('Relevance', {}).get('Other')
                rows.append(row)

            # list -> json
            list_to_json(rows, './data_transforms/bioprojects.json')
            # list -> csv
            list_to_csv(rows, './data_transforms/bioprojects.csv')

        except json.JSONDecodeError as e:
            print(f"Error: Failed to load JSON file. {e}")
            return None
        

if __name__ == "__main__":
    json_file_path = os.getcwd() + '/data_sources/bioproject_result.json'
    json_data = convert_bio_project_csv(json_file_path)
    if json_data:
        print("JSON loaded successfully into a dictionary.")
        print(json_data)
