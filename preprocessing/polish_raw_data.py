import os
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor

from dicts import *

def read_yearly_exchanges(country, year):
    """
    Polish raw csv-s on yearly exchanges of single countries. An
    edgelist is created as a dictionary; the outer dictionary has
    "Import" and "Export" as keys, each of them referring to a
    nested dictionary whose keys are the import/export partners,
    and the values the yearly aggregated import/export exchanges.

    Given a generic row from the csv, row[12] refers to the partner
    country, row[7] to whether the exchange is of import/export,
    row[31] to the aggregated yearly exchange with the partner.

    Incorrect csv-s are disregarded.

    Args:
      - country: index of the country by alphabetic order starting
        from 0;
      - year: corresponding year

    Returns:
      - edgelist: dictionary containing the edge list for the country
        and year.
    """
    edgelist = {"Import" : {}, "Export" : {}}
    data_folder = f"../data/{countries_names[country][0]}/"
    if not os.path.exists(data_folder):
        raise NameError("NameError: Invalid country name.")
    data_file = os.path.join(data_folder, f"{year}.csv")
    if os.path.exists(data_file):
        f_in = open(data_file, 'r')
        trades = list(csv.reader(f_in, delimiter=','))
        if trades[1][1] != f'{year}':
            print(f"ValueError: year value for {country}, {year} doesn\'t match. "
                    f"File will be disregarded.")
        elif trades[1][9] != country:
            print(f"NameError: country name for {country}, {year} doesn\'t match. "
                    f"File will be disregarded.")
        else:
            for row in trades[1:]:
                if row[12] in countries_names.keys():
                    edgelist[row[7]][row[12]] = int(row[31])
        f_in.close()
    return edgelist

def read_gdp(country):
    """
    Polish raw csv-s on yearly gdp of single countries. A dictionary
    is created with keys given as the years analysed, and values a
    nested dictionary with a single key, "gdp", and the country's gdp
    as a value.

    Incorrect csv-s are disregarded.

    Args:
      - country: index of the country by alphabetic order starting
        from 0;

    Returns:
      - country_dict: dictionary of yearly gdp values of given country.
    """
    gdp_series = {}
    data_folder = f"../data/{countries_names[country][0]}/"
    if not os.path.exists(data_folder):
        raise NameError(f"NameError: Invalid country name: {country}.")
    gdp_file = os.path.join(data_folder, "gdp.csv")
    f_in = open(gdp_file, 'r')
    gdp = csv.reader(f_in, delimiter=',')
    for row in list(gdp)[1:]:
        gdp_series[int(row[0])] = {"gdp" : float(row[1].replace(',', ''))} if row[1] != "N/A" else {"gdp" : 0.}
    f_in.close()
    return gdp_series

def read_data(country):
    """
    Merges dictionaries on import/export with the ones about gdp.
    The final dictionary is made by an outer dictionary with countries
    as keys and nested dictionaries as values; each of these has in
    turn years as keys and finally other nested dictionaries as values;
    the latter have as keys "Import", "Export" and "gdp", corresponding
    to the dictionaries in read_yearly_exchanges and read_gdp.
    """
    country_dict = {}
    country_dict[country] = read_gdp(country)
    for year in years:
        country_dict[country][year].update(read_yearly_exchanges(country, year))
    return country_dict

def read_all():
    """
    Reads data from csv files, polish it and dumps it into a json file.
    """
    start = time.time()
    countries_dict = [read_data(country) for country in countries_names]
    print(f"Time taken: {(time.time()-start):.1f}")
    folder = os.path.join("..", "final_data")
    if not os.path.exists(folder):
        os.makedirs(folder)
    data_json = os.path.join(folder, "data.json")
    with open(data_json, 'w') as f:
        json.dump(countries_dict, f)

if __name__ == '__main__':
    read_all()
