import json
import os

from utils.list_to_json import list_to_json

def init_merge_bioprojects_into_biosamples():
    with open(os.getcwd() + '/data_transforms/biosamples.json', 'r') as samples_file:
        with open(os.getcwd() + '/data_transforms/bioprojects.json', 'r') as projects_file:
            try:
                # load data
                biosamples_data = json.load(samples_file)
                bioprojects_data = json.load(projects_file)

                # key projects on id w/ dict as value
                bioprojects_dict = { bp['bioproject_label']: bp for bp in bioprojects_data }

                # map over biosamples
                def extend_biosample_with_project_info(biosample):
                    if biosample.get('bioproject_label', None) != None and bioprojects_dict.get(biosample.get('bioproject_label'), None) != None:
                        biosample['bioproject_title'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('project_title')
                        biosample['bioproject_description'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('project_description')
                        biosample['is_relevance_agricultural'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('is_relevance_agricultural')
                        biosample['is_relevance_environmental'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('is_relevance_environmental')
                        biosample['is_relevance_evolution'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('is_relevance_evolution')
                        biosample['is_relevance_industrial'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('is_relevance_industrial')
                        biosample['is_relevance_model_organism'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('is_relevance_model_organism')
                        biosample['relevance_other'] = bioprojects_dict.get(biosample.get('bioproject_label')).get('relevance_other')
                    return biosample
                extended_biosamples_data = list(map(extend_biosample_with_project_info, biosamples_data))

                # list -> json
                list_to_json(extended_biosamples_data, './data_transforms/biosamples+projects.json')
                
            except json.JSONDecodeError as e:
                print(f"Error: Failed to load JSON file. {e}")
                return None
    print('ok')

if __name__ == "__main__":
    init_merge_bioprojects_into_biosamples()
