#!/usr/bin/env python
import pyportal
from collections import Counter
import time

# only if auth is needed (usually not)
api_key = 'your-api-key'
# otherwise
api_key = None

# create an API class instance
api = pyportal.API(api_key)
from pyportal.constants import resources

# these examples use the specimens dataset
res_id = resources.specimens

def Get_images(record):
    image_list = list()
    assoc_media = record['associatedMedia']
    for i in assoc_media:
        image_list.append(i['identifier'])
    return(image_list)



def Get_species_dicts(species_string):
    search = api.records(res_id, query=species_string)
    count=0
    temp_list = list()
    Example_dict = dict()
    #loop through returned records
    for record in search.all():
        count+=1
        # create a list of names/aberrations to use for calculation of frequency
        temp_list.append(record['scientificName'])
        #print(record)
        # create a dictionary that only includes each name once and extract the first record as an example for image search
        #check if has images
        if 'associatedMedia' in record:
             #check if this is the first occurence of a name
            if record['scientificName'] not in Example_dict:

                Example_dict[record['scientificName']] = list()
                Example_dict[record['scientificName']].append(record['catalogNumber'])
                Example_dict[record['scientificName']].append(Get_images(record))
                #print(record)
            elif len(Example_dict[record['scientificName']]) <6:

                Example_dict[record['scientificName']].append(record['catalogNumber'])
                Example_dict[record['scientificName']].append(Get_images(record))
                #print(record['scientificName'])

    Frequency_dict=Counter(temp_list)
    return(Frequency_dict,Example_dict)



#https://species.nbnatlas.org/species/NHMSYS0000502377# classification tab download species
# use csv file for uk ennominae
file="UKSI_ennominae.csv"
fh = open(file)

for line in fh:
    line2 = line.split(",")
    #print(line2[2])
    #add time delay to avoid upsetting NHM
    time.sleep(1)
    #Querying database
    #print(count)
    Frequency_dict, Example_dict = Get_species_dicts(line2[2])
    # counter gets the frequencies from the list constructed above as a counter/dict object

    for key in Frequency_dict:
        if key in Frequency_dict and key in Example_dict:
            print(key,Frequency_dict[key], Example_dict[key], sep="\t")
    #print(Var_count_dict)

    #Rep_dict is a dict containing an example catalogue ID example for each species/abberation
    #print(Rep_dict)
