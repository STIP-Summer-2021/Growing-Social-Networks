# Setup

# Generated network with K = 20
<img src="Figs/network.svg" >


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

