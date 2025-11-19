# story_telling_mit_daten
# ammar taha ata4447, benjamin hartmann

The data for this project is from kaggle by user Emin Serkan Erdonmez:

https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset

# About Dataset
The aforementioned data were compiled from the ships transiting the Kattegat Strait between January 1st and March 10th, 2022. AIS (Automatic Identification System) data is published as open source in some countries.
Two types of data, static and dynamic, are kept in the AIS device:

# Static Information :

1. The ship's IMO number
2. The ship's MMSI number
3. The ship's Call Sign
4. The ship's name
5. The ship's type
6. What type of destination this message was received from (like Class A / Class B)
7. Width of ship
8. Length of ship
9. Draft of ship
10. Type of GPS device
11. Length from GPS to bow (Length A)
12. Length from GPS to stern (Size B)
13. Length from GPS to starboard (Size C)
14. Length from GPS to port side (Dimension D)

# Dynamic Data:

1. Time information (31/12/2015 in 23:59:59 format)
2. Latitude
3. Longitude
4. Navigational status (For example: 'Fishing', Anchored, etc.)
5. Rate of Turn (ROT)
6. Speed Over Ground (SOG)
7. Course Over Ground (COG)
8. Heading
9. Type of cargo
10. Port of Destination
11. Estimated Time of Arrival (ETA)
12. Data source type, eg. AIS

# Data only includes:

,mmsi,navigationalstatus,sog,cog,heading,shiptype,width,length,draught

mmsi: Maritime Mobile Service Identity: unique vessel id for radio and AIS communication.

navigationalstatus: current state of the vessel, Under way, Moored, At anchor, Fishing, Unknown value, etc.

sog: Speed Over Ground in knots.

cog: Course Over Ground in degrees relative to true north.

heading: Direction of the bow of the ship, difference from cog does not sccount for drift.

shiptype: vessel type/description

width: max width of vessel beam in meters.

length: full vessel length in meters

drought: distance between waterline and deepest point hull/ship in meters, shows how deep ship is, affected by cargo weight. 

# requirements, how to run:

```
pip install -r requirements.txt

python analyze_ais.py
```

# ais_data.csv needs to be in same directory as script



