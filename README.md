# These are the steps to generate the data files that serve as our dataset. These are already generated and provided @:

https://purr.purdue.edu/projects/ridesharing/files

This project is in python 3.7 and tesnorflow 2.0
## Setup
Below you will find step-by-step instructions to set up the NYC taxi simulation using 2016-05 trips for training and 2016-06 trips for evaluation.
### 1. Download OSM Data
```commandline
wget https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf -P osrm
```

### 2. Preprocess OSM Data
```commandline
cd osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/NewYork.osm.pbf
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-partition /data/NewYork.osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-customize /data/NewYork.osrm
```

### 3. Download Trip Data
```commandline
mkdir data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-06.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-06.csv -P data/trip_records
```

### 4. Build Docker image
```commandline
docker-compose build sim
```

### 5. Preprocess Trip Records
```commandline
docker-compose run --no-deps sim python src/preprocessing/preprocess_nyc_dataset.py ./data/trip_records/ --month 2016-05
docker-compose run --no-deps sim python src/preprocessing/preprocess_nyc_dataset.py ./data/trip_records/ --month 2016-06
```

### 6. Snap origins and destinations of all trips to OSM
```commandline
docker-compose run sim python src/preprocessing/snap_to_road.py ./data/trip_records/trips_2016-05.csv ./data/trip_records/mm_trips_2016-05.csv
docker-compose run sim python src/preprocessing/snap_to_road.py ./data/trip_records/trips_2016-06.csv ./data/trip_records/mm_trips_2016-06.csv
```

### 7. Create trip database for Simulation
```commandline
docker-compose run --no-deps sim python src/preprocessing/create_db.py ./data/trip_records/mm_trips_2016-06.csv
```

### 8. Prepare statistical demand profile using training dataset
```commandline
docker-compose run --no-deps sim python src/preprocessing/create_profile.py ./data/trip_records/mm_trips_2016-05.csv
```

### 9. Precompute trip time and trajectories by OSRM
```commandline
docker-compose run sim python src/preprocessing/create_tt_map.py ./data
```
The tt_map needs to be recreated when you change simulation settings such as MAX_MOVE.

### 10. Change simulation settings
You can find simulation setting files in `src/config/settings` and `src/simulator/settings`.
