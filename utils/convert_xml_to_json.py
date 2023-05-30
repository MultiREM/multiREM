import json
import xmltodict
import os
import argparse

def convert_xml_to_json(xml_file_path):
    # Derive JSON file path from XML file path
    base = os.path.splitext(xml_file_path)[0]
    json_file_path = base + '.json'

    with open(xml_file_path, 'r') as xml_file:
        xml_dict = xmltodict.parse(xml_file.read())  # Convert XML to Python dict
    json_data = json.dumps(xml_dict)  # Convert Python dict to JSON
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

    print(f"JSON file has been saved as {json_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML to JSON.')
    parser.add_argument('xml_file_path', type=str, help='The path to the XML file to convert.')

    args = parser.parse_args()

    convert_xml_to_json(args.xml_file_path)