import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
import seaborn as sns

# Read the CSV file for Chicago crime data
crime_data = pd.read_csv("CrimeLocation.csv")

# Corrected variable names
latitude_col = 'latitude'
longitude_col = 'longitude'
primary_type_col = 'primary_type'

# Limit to the first 1000 entries
crime_data = crime_data.head(1000)

# Create a folium map centered around the mean of coordinates
m = folium.Map(location=[crime_data[latitude_col].mean(), crime_data[longitude_col].mean()], zoom_start=10)

# Create a MarkerCluster for individual observations
marker_cluster = MarkerCluster().add_to(m)

# Generate a color palette based on the number of unique crime types
n_crime_types = len(crime_data[primary_type_col].unique())
color_palette = sns.color_palette("husl", n_crime_types).as_hex()

# Create a dictionary mapping crime types to colors
crime_type_colors = dict(zip(crime_data[primary_type_col].unique(), color_palette))

# Add circles and labels for each observation
for index, row in crime_data.iterrows():
    folium.CircleMarker(
        location=[row[latitude_col], row[longitude_col]],
        radius=5,  # Adjust the radius as needed
        color=crime_type_colors.get(row[primary_type_col], 'gray'),  # Use gray for unknown crime types
        fill=True,
        fill_color=crime_type_colors.get(row[primary_type_col], 'gray'),
        fill_opacity=0.7,
        tooltip=row[primary_type_col]  # Add tooltip with the crime type
    ).add_to(marker_cluster)

# Perform KMeans clustering based on latitude and longitude
X = crime_data[[latitude_col, longitude_col]]
kmeans = KMeans(n_clusters=n_crime_types, random_state=42).fit(X)
crime_data['cluster'] = kmeans.labels_

# Predict for multiple sets of coordinates
new_coordinates_list = [
    [41.9, -87.7],
    [41.8, -87.6],
    [41.7, -87.8],
    [41.9, -87.6],
    [41.8, -87.7]
]

# Add markers for all predicted coordinates with the predicted crime type
for new_coordinates in new_coordinates_list:
    # Predict the cluster for the new coordinates
    predicted_cluster = kmeans.predict([new_coordinates])[0]

    # Get crime types in the predicted cluster
    predicted_crime_types = crime_data[crime_data['cluster'] == predicted_cluster][primary_type_col].unique()

    # Customize the icon for predicted coordinates
    icon_color = crime_type_colors.get(predicted_cluster, 'gray')
    icon = folium.Icon(color=icon_color, prefix='fa', icon='circle')

    # Add a marker for the predicted coordinates with the predicted crime types
    folium.Marker(
        location=new_coordinates,
        popup=f"Predicted Cluster: {predicted_cluster}<br>Predicted Crimes: {', '.join(predicted_crime_types)}",
        icon=icon
    ).add_to(m)

# Display the map
m.save("chicago_crime_map_with_predictions_aesthetic.html")
