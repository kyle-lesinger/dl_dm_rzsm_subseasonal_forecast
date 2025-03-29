#!/user/bin/env python3

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def d2_plot(xarray_obj):
    # Create a figure and axis with Cartopy projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add features like coastlines and borders
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Plot the data using pcolormesh
    xarray_obj.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=True)
    
    # Set title
    ax.set_title("2D Geographic Plot")
    
    # Show the plot
    plt.show()