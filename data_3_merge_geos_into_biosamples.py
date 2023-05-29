import json
import os
import requests
from bs4 import BeautifulSoup

from utils.list_to_json import list_to_json

def scrape_protocol_table(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # If the GET request is successful, the status code will be 200
    if response.status_code == 200:
        # Get the content of the response
        page_content = response.content
        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(page_content, 'html.parser')
        # Find the first table in the HTML w/ organism reference
        tables = soup.find_all('table', recursive=True)
        for table in tables:
            data = []
            for row in table.find_all('tr'):
                rowData = []
                for td in row.find_all('td'):
                    rowData.append(td.text.strip())
                if len(rowData) > 1:
                    data.append(rowData)
            # Create a dict from data
            data_dict = { sublist[0]: sublist[1] for sublist in data }
            # Return just a trimmed down version
            data_dict_trimmed = {
                'gene_expression_omnibus_data_processing': data_dict.get('Data processing', None),
                'gene_expression_omnibus_extraction_protocol': data_dict.get('Extraction protocol', None),
                'gene_expression_omnibus_growth_protocol': data_dict.get('Growth protocol', None),
                'gene_expression_omnibus_treatment_protocol': data_dict.get('Treatment protocol', None),
            }
            # Return the DataFrame
            return data_dict_trimmed
        raise 'No table found'
    else:
        print("Failed to retrieve page")

def scrape_biosample_geos():
    # read all geos from samples
    with open(os.getcwd() + '/data_transforms/biosamples+projects.json', 'r') as samples_file:
        biosamples_data = json.load(samples_file)
        geos_data = []
        # loop over samples that have geo ids, thus meaning protocols
        for sample in biosamples_data:
            if sample.get('gene_expression_omnibus_id') != None:
                print(sample.get('gene_expression_omnibus_id'))
                # --- scrape
                sample_geo_data = scrape_protocol_table('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + sample.get('gene_expression_omnibus_id'))
                # --- append data
                sample_geo_data_with_id = { **sample_geo_data, 'gene_expression_omnibus_id': sample.get('gene_expression_omnibus_id') }
                geos_data.append(sample_geo_data_with_id)
        # write to file
        list_to_json(geos_data, './data_transforms/geos.json')

def merge_geos_into_biosamples():
    with open(os.getcwd() + '/data_transforms/biosamples+projects.json', 'r') as samples_file:
        with open(os.getcwd() + '/data_transforms/geos.json', 'r') as geos_file:
            try:
                # load data
                biosamples_data = json.load(samples_file)
                geos_data = json.load(geos_file)
                # key projects on id w/ dict as value
                geos_dict = { geo['gene_expression_omnibus_id']: geo for geo in geos_data }
                # map over biosamples
                def extend_sample_with_geo(sample):
                    run_with_sample_data = { **sample, **geos_dict.get(sample.get('gene_expression_omnibus_id'), {}) }
                    return run_with_sample_data
                extended_runs_data = list(map(extend_sample_with_geo, biosamples_data))
                # list -> json
                list_to_json(extended_runs_data, './data_transforms/biosamples+projects+geos.json')
            except json.JSONDecodeError as e:
                print(f"Error: Failed to load JSON file. {e}")
                return None


if __name__ == "__main__":
    scrape_biosample_geos()
    merge_geos_into_biosamples()
