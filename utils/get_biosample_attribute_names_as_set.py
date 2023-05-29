import json

def get_biosample_attribute_names_as_set(json_dict):
    # Initialize an empty list to store attribute names
    attribute_names_set = set()

    # Iterate over bio samples
    for sample in json_dict["BioSampleSet"]["BioSample"]:
        if sample.get('Attributes', None) != None:
            # Iterate over the "Attribute" list in the "Attributes" dictionary
            for attribute in sample["Attributes"]["Attribute"]:
                # some are strs
                if type(attribute) == dict:
                    # # Append the value of "@attribute_name" to the list
                    attribute_names_set.add(attribute["@attribute_name"])

    # Print the list of attribute names
    attribute_names_set_sorted = sorted(attribute_names_set)
    return attribute_names_set_sorted
