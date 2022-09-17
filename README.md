# Setup
This is the general file structure for the project. 
```
.
└── MESA-safegraph-sim/
    ├── Data/
    │   ├── Shapefiles/
    │   │   └── Census_2010_Blockgroup.shp
    │   ├── CBG_POI_Probs
    │   ├── CBG_Visits
    │   ├── Monthly_CBG_Topic_Dist
    │   ├── Monthly_NAICs
    │   ├── Monthly_POIs
    │   ├── Topic_JSONs
    │   └── Training_Splits
    ├── Data_Analysis/
    │   ├── jaccard_runner.py
    │   ├── kendalls_tau.py
    │   ├── naics_plotter.py
    │   ├── plotter.py
    │   └── rsme.py
    ├── Data_Parsing/
    │   ├── main.py
    │   ├── CBG_Parser.py
    │   ├── POI_Parser.py
    │   ├── Spatial_Join_Analysis.py
    │   ├── Yearly_Average.py
    │   └── utils.py
    ├── Logging/
    │   ├── NAICS_Visits
    │   └── Patterns 
    └── model.py
```

If you are using Anaconda, a environment.yml has been provided with all of the necessary packages. To install, either launch the anaconda GUI and import using the import environment wizard or by running the following from a command line. 

```conda env create -f environment.yml```


# Required Data

### <b> SafeGraph </b>

The majority of the data used for this project is from SafeGraph, the following is a comprehensive list of the data required to run. 

You can request access to SafeGraph data through their data for academics program <a href='https://www.safegraph.com/academics'>here.</a>

All data must be stored as noted within a folder called "safegraph-data". I reccomend using the AWS CLI referenced by SafeGraph in their tutorial as it will set up a file stucture automatically for you. 

Example Call for the Open Census Data: 

```aws s3 sync s3://sg-c19-response/open-census-data/ ./Users/Justin/Mesa-safegraph-sim/safegraph-data/ --profile safegraphws --endpoint https://s3.wasabisys.com```


The folder containing the datasets can be stored anywhere on your computer, you just need to update the utils.py function <i>get_data_path(),</i> which serves as a reference to this folder. 


1.	Social Distancing Metrics v2.1 (formerly Physical Distancing Metrics)
    - ./safegraph-data/safegraph_social_distancing_metrics/

2.  Open Census Data
    - ./safegraph-data/safegraph_open_census_data/

3. 	Core Places US (Pre-Nov-2020)
    - ./safegraph-data/safegraph_core_places/

4.  Core Places US (Nov 2020 - Present)
    - ./safegraph-data/safegraph_core_places_new/

5.  Weekly Places Patterns v2 (until 2020-06-15)
    - ./safegraph-data/safegraph_weekly_patterns_v2/

6. 	Weekly Places Patterns (for data from 2020-06-15 to 2020-11-30)
    - ./safegraph-data/safegraph_weekly_patterns_v3/

7.  Weekly Places Patterns (for data from 2020-11-30 to Present)
    - ./safegraph-data/safegraph_weekly_patterns_v3.1/

### <b> Additional Data </b>

1. Fairfax Census Block Groups Shapefile
    
    - Available <a href='https://data-fairfaxcountygis.opendata.arcgis.com/datasets/census-2010-blockgroup?geometry=-78.596%2C38.645%2C-75.982%2C39.019'>here.</a>
    - The files must be extracted from the compressed folder and stored in:
        - ./MESA-safegraph-sim/Data/Shapefiles/


<!-- # Usage
After downloading all of the required  -->