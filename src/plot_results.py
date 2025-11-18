import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os


def plot_2d_suitability_map(csv_file_path, output_image_path):
    print(f"Loading final map from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    df_plot = df[df['suitability_score'] >= 0.5]

    print(f"Plotting {len(df_plot)} suitable grid cells...")

    # --- FIX: Removed dark background style ---
    # plt.style.use('dark_background') # This line is removed.
    # Matplotlib's default is a white background.

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.scatter(
        df_plot['longitude'],
        df_plot['latitude'],
        c=df_plot['suitability_score'],
        cmap='RdYlGn',  # Red-Yellow-Green (Green is high)
        s=1,  # Small marker size
        vmin=0.5,  # Set color minimum
        vmax=1.0  # Set color maximum
    )

    # --- FIX: Removed all 'color='white'' parameters ---
    # The default color is black, which is what we want.
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Suitability Score (Probability of Success)')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    # --- FIX: Set map limits and remove explicit label colors ---
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (0-360° East)")
    ax.set_ylabel("Latitude")
    ax.set_title('Safe Harbour - Predicted Landing Suitability', fontsize=16)

    # --- FIX: Set tick colors to black (or remove for default) ---
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    fig.tight_layout()
    # --- FIX: Ensure figure background is saved as white ---
    fig.savefig(output_image_path, dpi=300, facecolor='white', transparent=False)
    print(f"Saved 2D map to {output_image_path}")
    plt.close(fig)  # Close the figure to free up memory


def convert_lat_lon_to_xyz(lat, lon, radius=1):
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate (x, y, z)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z


def plot_3d_globe_map(csv_file_path, output_html_path):
    print(f"Loading final map from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Filter for better visibility.
    df_plot = df[df['suitability_score'] >= 0.5]
    print(f"Plotting {len(df_plot)} high-suitability sites on 3D globe...")

    # 1. Convert our data points to (x, y, z)
    radius_data = 1.01
    x_data, y_data, z_data = convert_lat_lon_to_xyz(
        df_plot['latitude'],
        df_plot['longitude'],
        radius=radius_data
    )

    # 2. Create the data trace for our landing sites
    data_trace = go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=dict(
            color=df_plot['suitability_score'],
            colorscale='RdYlGn',  # Red-Yellow-Green
            cmin=0.5,
            cmax=1.0,
            colorbar_title="Suitability Score",
            size=1.5
        ),
        text=df_plot.apply(
            lambda r: f"Lat: {r['latitude']:.1f}<br>Lon: {r['longitude']:.1f}<br>Score: {r['suitability_score']:.2f}",
            axis=1
        ),
        hoverinfo='text'  # Show only our custom text on hover
    )

    # 3. Create the sphere (globe) surface for Mars
    radius_globe = 1.0
    u = np.linspace(0, 360, 100)  # Longitude
    v = np.linspace(-90, 90, 100)  # Latitude

    x_globe, y_globe, z_globe = convert_lat_lon_to_xyz(
        np.outer(v, np.ones_like(u)),
        np.outer(np.ones_like(v), u),
        radius=radius_globe
    )

    mars_surface = go.Surface(
        x=x_globe,
        y=y_globe,
        z=z_globe,
        colorscale=[[0, 'rgb(180, 80, 50)'], [1, 'rgb(180, 80, 50)']],  # Solid reddish-brown
        showscale=False,
        opacity=0.8,
        hoverinfo='none'
    )

    # 4. Create the Figure
    fig = go.Figure(data=[mars_surface, data_trace])

    # --- FIX: Changed layout to a light theme ---
    fig.update_layout(
        title='Safe Harbour - Predicted Global Suitability',
        scene=dict(
            # Set background to white
            bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False, title=''),
            aspectmode='data'  # This ensures it's a perfect sphere
        ),
        # Use the default light theme
        template="plotly_white"
    )
    # --- END FIX ---

    # Save the interactive plot as an HTML file
    fig.write_html(output_html_path)
    print(f"Saved interactive 3D globe to {output_html_path}")


def main():
    # Example usage
    input_map_file = "../output/final/final_suitability_map.csv"

    # Check if the input file exists before running
    if os.path.exists(input_map_file):
        output_image = "../output/final/final_map_2D_2.png"
        plot_2d_suitability_map(input_map_file, output_image)

        output_3d_html = "../output/final/final_map_3D_globe2.html"
        plot_3d_globe_map(input_map_file, output_3d_html)
    else:
        print(f"Error: Input file not found at '{input_map_file}'")
        print("Please run your analysis script first to generate this file.")


if __name__ == "__main__":
    main()