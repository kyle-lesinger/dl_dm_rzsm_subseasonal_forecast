#!/usr/bin/env python

import cdsapi

dataset = "satellite-soil-moisture"
request = {
    "variable": ["volumetric_surface_soil_moisture"],
    "time_aggregation": ["day_average"],
    "year": ["2000"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "type_of_record": ["cdr"],
    "version": ["v202312"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
