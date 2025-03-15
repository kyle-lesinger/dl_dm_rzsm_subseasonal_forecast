#!/bin/bash

#cdo operators will not work on this data unless this in included

#TODO: Only run this command once. The next time, it will append more data that makes 
#the file unreadable. If so, just change the lon,a,c to lon,o,c for all variables (and for lat)
for f in *.nc;do
ncatted -a standard_name,lon,a,c,"longitude" $f
ncatted -a long_name,lon,a,c,"longitude" $f
ncatted -a units,lon,a,c,"degrees_east" $f
ncatted -a axis,lon,a,c,"lon" $f

ncatted -a standard_name,lat,a,c,"latitude" $f
ncatted -a long_name,lat,a,c,"latitude" $f
ncatted -a units,lat,a,c,"degrees_north" $f
ncatted -a axis,lat,a,c,"lat" $f;
done

