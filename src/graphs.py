import graphviz
import os

def create_pipeline_graph(output_filename='../output/project_pipeline'):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Initialize a new Directed Graph
    dot = graphviz.Digraph(comment='Project: Safe Harbour Pipeline')
    dot.attr(rankdir='TB', label='Project: Safe Harbour - Full ML Pipeline', fontsize='20')
    dot.attr(splines='ortho')  # Use straight/orthogonal lines

    # --- Define Node Styles ---
    style_data = {'shape': 'ellipse', 'style': 'filled', 'color': 'skyblue'}
    style_process = {'shape': 'box', 'style': 'filled', 'color': 'palegreen'}
    style_output = {'shape': 'note', 'style': 'filled', 'color': 'lightgrey'}
    style_model = {'shape': 'cylinder', 'style': 'filled', 'color': 'orange'}
    style_final = {'shape': 'diamond', 'style': 'filled', 'color': 'gold'}

    # --- 1. Raw Data Nodes ---
    with dot.subgraph(name='cluster_0_raw_data') as c:
        c.attr(label='Raw Data Sources', style='filled', color='lightyellow')
        c.node('mola_raw', 'MOLA Global Map (Raw)', **style_data)
        c.node('gcm_raw', 'GCM (24 "Monthly" Files)', **style_data)
        c.node('missions_raw', 'Past Missions (16 missions)', **style_data)
        c.attr(rank='same')  # Keep them aligned

    # --- 2. Preprocessing Steps ---
    with dot.subgraph(name='cluster_1_preprocessing') as c:
        c.attr(label='Part 1: Data Preprocessing', style='filled', color='whitesmoke')
        c.node('mola_process', '1a. Terrain Analysis\n(Slope, Elevation, etc.)', **style_process)
        c.node('gcm_process', '1b. GCM Aggregation\n(Calculate Yearly min/max/mean)', **style_process)

        c.node('mola_file', 'MOLA Suitability Map\n(mola_map.csv)', **style_output)
        c.node('gcm_file', 'GCM Yearly Stats Map\n(gcm_yearly_stats.csv)', **style_output)

        dot.edge('mola_raw', 'mola_process')
        dot.edge('gcm_raw', 'gcm_process')
        dot.edge('mola_process', 'mola_file')
        dot.edge('gcm_process', 'gcm_file')

    # --- 3. Model Training Branch ---
    with dot.subgraph(name='cluster_2_training') as c:
        c.attr(label='Part 2: Model Training (on Past Missions)', style='filled', color='whitesmoke')
        c.node('merge_train', '2a. Merge Training Data\n(Missions + MOLA + GCM)', **style_process)
        c.node('train_model', '2b. Train Suitability Model\n(RandomForestClassifier)', **style_model)

        dot.edge('missions_raw', 'merge_train')
        dot.edge('mola_file', 'merge_train')
        dot.edge('gcm_file', 'merge_train')
        dot.edge('merge_train', 'train_model')

    # --- 4. Global Prediction Branch ---
    with dot.subgraph(name='cluster_3_prediction') as c:
        c.attr(label='Part 3: Global Prediction (on All Mars)', style='filled', color='whitesmoke')
        c.node('merge_predict', '3a. Prepare Global Data\n(MOLA + GCM)', **style_process)
        c.node('predict', '3b. Predict Suitability\n(model.predict_proba)', **style_process)
        c.node('final_map', 'Final Suitability Map\n(suitability_score: 0.0 to 1.0)', **style_final)

        dot.edge('mola_file', 'merge_predict')
        dot.edge('gcm_file', 'merge_predict')
        dot.edge('merge_predict', 'predict')
        dot.edge('train_model', 'predict')  # The trained model is an input to prediction
        dot.edge('predict', 'final_map')

    # --- Render the graph ---
    try:
        # This line saves the file
        dot.render(output_filename, format='png', cleanup=True)
        print(f"Success! Pipeline flowchart saved to {output_filename}.png")
        print("Please check your '../output/' folder to see the image.")
    except Exception as e:
        print(f"Error generating graph: {e}")
        print("Please ensure you have 'graphviz' installed.")
        print("Run: pip install graphviz")


if __name__ == "__main__":
    create_pipeline_graph()