import json
import os
from utils.common import to_list
from utils.align_attribute_name import align_attribute_name
from utils.list_to_csv import list_to_csv
from utils.list_to_json import list_to_json

def convert_bio_sample_csv(file_path):
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            rows = []

            # get samples
            for sample in data.get('BioSampleSet').get('BioSample'):
                # .. for each sample
                row = dict()
                # --- organism (Description property)
                row['organism_taxonomy_id'] = sample.get('Description').get('Organism').get('@taxonomy_id')
                row['organism_taxonomy_name'] = sample.get('Description').get('Organism').get('@taxonomy_name')
                row['organism_title'] = sample.get('Description').get('Title')
                # --- ids (mostly biosamples but some others exist)
                if sample.get('Ids') != None:
                    id_list = to_list(sample.get("Ids").get("Id"))
                    for id in id_list:
                        # ------ accession
                        if id.get('@db') == 'BioSample':
                            row['biosample_accession'] = id.get('#text') # matches '@accession' key.
                            row['biosample_url'] = 'https://www.ncbi.nlm.nih.gov/biosample/' + row['biosample_accession']
                        # ------ sample name
                        if id.get('@db') == 'Sample name':
                            row['sample_name'] = id.get('#text')
                        # ------ SRA (sequence read archive. These are not the final runs, that's in sra_result)
                        if id.get('@db') == 'SRA':
                            row['sra_id'] = id.get('#text')
                            row['sra_url'] = 'https://ncbi.nlm.nih.gov/sra/' + id.get('#text') # ex: https://ncbi.nlm.nih.gov/sra/SRX20019450
                        # ------ GEO
                        if id.get('@db') == 'GEO':
                            row['geo_id'] = id.get('#text')
                            row['geo_url'] = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=' + id.get('#text')
                # --- links ()
                if sample.get('Links') != None:
                    link_list = to_list(sample.get('Links').get('Link'))
                    # ------ defaults
                    row['is_bioproject'] = False
                    for link in link_list:
                        # ------ bio project (Link property. Going to make a column for both label and text)
                        if link.get('@target') == 'bioproject':
                            row['is_bioproject'] = True
                            row['bioproject_label'] = link.get('@label')
                            row['bioproject_label_id'] = link.get('#text')
                            row['bioproject_url'] = 'https://www.ncbi.nlm.nih.gov/bioproject/' + link.get('#text')
                        # ------ other link: pubmed -- eh only 1 record has this
                        # ------ other link: url
                        if link.get('@type') == 'url' and 'Goldstamp' in link.get('@label'):
                            row['goldstamp_label'] = link.get('@label')
                            row['goldstamp_url'] = link.get('#text')
                        if link.get('@type') == 'url' and 'DOI' in link.get('@label'):
                            row['doi_label'] = link.get('@label')
                            row['doi_url'] = link.get('#text')
                        if link.get('@type') == 'url' and 'GEO Sample' in link.get('@label'):
                            row['ncbi_label'] = link.get('@label')
                            row['ncbi_url'] = link.get('#text')
                # --- urls (constructing for convenience if we need to scrape/download or ??? built from ids)
                # --- all attributes (Attributes -> Attribute) gonna prefix w/ attr so we don't collide with any cleaned up attrs
                if sample.get('Attributes'):
                    attr_list = to_list(sample.get('Attributes').get('Attribute'))
                    for attr in attr_list:
                        attr_key = align_attribute_name(attr.get('@attribute_name'))
                        # there are a ton of 'Missing' answers so not including to declutter. the empty col will make it clear
                        if attr_key != None and attr.get('#text').lower().startswith('missing') == False:
                            # if we're doing taxa, break down each into a column
                            if attr_key == 'gtdb_taxonomy':
                                split = attr.get('#text').split(';')
                                row['attr_' + attr_key + '_kingdom'] = split[0][3:] # grabbing after the d__ or c__ etc
                                row['attr_' + attr_key + '_phylum'] = split[1][3:] if len(split) > 2 else None
                                row['attr_' + attr_key + '_class'] = split[2][3:] if len(split) > 3 else None
                                row['attr_' + attr_key + '_order'] = split[3][3:] if len(split) > 4 else None
                                row['attr_' + attr_key + '_family'] = split[4][3:] if len(split) > 5 else None
                                row['attr_' + attr_key + '_genus'] = split[5][3:] if len(split) > 6 else None
                                row['attr_' + attr_key + '_species'] = split[6][3:] if len(split) > 7 else None
                            else:
                                row['attr_' + attr_key] = attr.get('#text')
                # --- consolidate attributes (ex: lat/long need to be split into independent geographic lat longs and countries)
                # gucci to go
                rows.append(row)
            
            # list -> json
            list_to_json(rows, './data_transforms/biosamples.json')
            # list -> csv
            list_to_csv(rows, './data_transforms/biosamples.csv')

        except json.JSONDecodeError as e:
            print(f"Error: Failed to load JSON file. {e}")
            return None


if __name__ == "__main__":
    json_file_path = os.getcwd() + '/data_sources/biosample_result.json'
    json_data = convert_bio_sample_csv(json_file_path)
    if json_data:
        print("JSON loaded successfully into a dictionary.")
        print(json_data)
