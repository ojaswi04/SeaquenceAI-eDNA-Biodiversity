#!/usr/bin/env python3
"""
SeaquenceAI - Interactive Plotly Dashboard
Web interface for deep-sea eDNA analysis with interactive visualizations
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import base64
import io
from datetime import datetime
import threading
import time

# Import your main SeaquenceAI pipeline
from seaquence_ai_main import SeaquenceAI
from Bio import SeqIO

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

# Global variables for storing analysis results
analysis_results = {}
current_pipeline = None
analysis_status = {"running": False, "progress": 0, "message": "Ready"}

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SeaquenceAI - Deep-sea eDNA Analysis</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
            }
            .main-header {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                padding: 1rem;
                margin-bottom: 2rem;
            }
            .card {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem;
                box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
                backdrop-filter: blur(4px);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                background: rgba(102, 126, 234, 0.1);
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: rgba(118, 75, 162, 0.1);
            }
            .progress-bar {
                background: linear-gradient(90deg, #667eea, #764ba2);
                height: 8px;
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            .metric-card {
                text-align: center;
                padding: 1rem;
                margin: 0.5rem;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border-radius: 10px;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-ready { background-color: #28a745; }
            .status-running { background-color: #ffc107; animation: pulse 1.5s infinite; }
            .status-error { background-color: #dc3545; }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-water", style={'margin-right': '10px'}),
            "SeaquenceAI"
        ], style={
            'color': 'white', 
            'text-align': 'center', 
            'margin': '0',
            'font-size': '2.5rem',
            'text-shadow': '2px 2px 4px rgba(0,0,0,0.3)'
        }),
        html.P("Deep-sea eDNA Analysis & Biodiversity Assessment Platform", 
               style={'color': 'rgba(255,255,255,0.9)', 'text-align': 'center', 'margin': '0.5rem 0 0 0'})
    ], className="main-header"),
    
    # Main content
    html.Div([
        # Control Panel
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-upload", style={'margin-right': '10px'}),
                    "Data Upload & Analysis"
                ]),
                
                # File upload area
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt", 
                                  style={'font-size': '3rem', 'margin-bottom': '1rem', 'color': '#667eea'}),
                            html.Br(),
                            'Drag and Drop FASTA files or ',
                            html.A('Select Files', style={'color': '#667eea', 'text-decoration': 'underline'})
                        ]),
                        className="upload-area",
                        multiple=False
                    ),
                ], style={'margin-bottom': '1rem'}),
                
                # Analysis controls
                html.Div([
                    html.Button([
                        html.I(className="fas fa-play", style={'margin-right': '8px'}),
                        "Start Analysis"
                    ], id='run-analysis', n_clicks=0, 
                    style={
                        'background': 'linear-gradient(135deg, #667eea, #764ba2)',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 24px',
                        'border-radius': '8px',
                        'font-size': '1rem',
                        'cursor': 'pointer',
                        'margin-right': '1rem'
                    }),
                    
                    html.Button([
                        html.I(className="fas fa-robot", style={'margin-right': '8px'}),
                        "Demo with AI Data"
                    ], id='demo-analysis', n_clicks=0,
                    style={
                        'background': 'linear-gradient(135deg, #28a745, #20c997)',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 24px',
                        'border-radius': '8px',
                        'font-size': '1rem',
                        'cursor': 'pointer'
                    })
                ], style={'text-align': 'center'}),
                
                # Status indicator
                html.Div(id='analysis-status', style={'margin-top': '1rem'}),
                
                # Progress bar
                html.Div([
                    html.Div(id='progress-bar', className='progress-bar', 
                            style={'width': '0%'})
                ], style={
                    'background': 'rgba(0,0,0,0.1)', 
                    'border-radius': '4px', 
                    'margin-top': '1rem'
                })
                
            ], className="card")
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Results Panel
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-chart-bar", style={'margin-right': '10px'}),
                    "Analysis Results"
                ]),
                html.Div(id='results-content')
            ], className="card")
        ], style={'width': '68%', 'display': 'inline-block', 'vertical-align': 'top'})
        
    ], style={'max-width': '1400px', 'margin': '0 auto'}),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'}),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    )
])

def run_analysis_background(use_demo_data=True, fasta_content=None):
    """Run analysis in background thread"""
    global current_pipeline, analysis_results, analysis_status
    
    try:
        analysis_status["running"] = True
        analysis_status["message"] = "Initializing pipeline..."
        analysis_status["progress"] = 10
        
        # Initialize pipeline
        current_pipeline = SeaquenceAI(device='cpu')
        
        if use_demo_data:
            analysis_status["message"] = "Generating demo sequences..."
            analysis_status["progress"] = 20
            sequences = None
        else:
            analysis_status["message"] = "Processing uploaded sequences..."
            analysis_status["progress"] = 20
            # Parse FASTA content
            sequences = []
            if fasta_content:
                fasta_io = io.StringIO(fasta_content)
                for record in SeqIO.parse(fasta_io, "fasta"):
                    sequences.append(str(record.seq))
        
        # Load sequences
        analysis_status["message"] = "Loading and preprocessing sequences..."
        analysis_status["progress"] = 30
        current_pipeline.load_sequences(sequences=sequences)
        current_pipeline.preprocess_sequences()
        
        # Generate embeddings
        analysis_status["message"] = "Training embedding model..."
        analysis_status["progress"] = 50
        current_pipeline.generate_embeddings()
        
        # Classification
        analysis_status["message"] = "Classifying sequences..."
        analysis_status["progress"] = 70
        current_pipeline.classify_sequences()
        
        # Novel taxa discovery
        analysis_status["message"] = "Discovering novel taxa..."
        analysis_status["progress"] = 85
        current_pipeline.discover_novel_clusters()
        
        # Calculate biodiversity
        analysis_status["message"] = "Calculating biodiversity indices..."
        analysis_status["progress"] = 95
        current_pipeline.calculate_biodiversity_indices()
        
        # Store results
        analysis_results = current_pipeline.results
        analysis_results['timestamp'] = datetime.now().isoformat()
        analysis_results['sequence_count'] = len(current_pipeline.sequences)
        analysis_results['reference_taxa'] = current_pipeline.reference_taxa
        analysis_results['classifications'] = current_pipeline.classifications
        
        analysis_status["running"] = False
        analysis_status["message"] = "Analysis complete!"
        analysis_status["progress"] = 100
        
    except Exception as e:
        analysis_status["running"] = False
        analysis_status["message"] = f"Error: {str(e)}"
        analysis_status["progress"] = 0

@app.callback(
    Output('analysis-status', 'children'),
    Output('progress-bar', 'style'),
    Input('interval-component', 'n_intervals')
)
def update_status(n):
    """Update analysis status and progress"""
    global analysis_status
    
    if analysis_status["running"]:
        status_class = "status-running"
        status_text = "Running"
    elif analysis_status["progress"] == 100:
        status_class = "status-ready"
        status_text = "Complete"
    elif analysis_status["progress"] == 0 and "Error" in analysis_status["message"]:
        status_class = "status-error"
        status_text = "Error"
    else:
        status_class = "status-ready"
        status_text = "Ready"
    
    status_div = html.Div([
        html.Span(className=f"status-indicator {status_class}"),
        html.Strong(f"{status_text}: "),
        html.Span(analysis_status["message"])
    ])
    
    progress_style = {
        'width': f'{analysis_status["progress"]}%',
        'background': 'linear-gradient(90deg, #667eea, #764ba2)',
        'height': '8px',
        'border-radius': '4px',
        'transition': 'width 0.3s ease'
    }
    
    return status_div, progress_style

@app.callback(
    Output('stored-data', 'children'),
    Input('run-analysis', 'n_clicks'),
    Input('demo-analysis', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def trigger_analysis(run_clicks, demo_clicks, uploaded_file):
    """Trigger analysis based on button clicks"""
    global analysis_status
    
    if analysis_status["running"]:
        return ""
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'demo-analysis':
        # Start demo analysis
        thread = threading.Thread(target=run_analysis_background, args=(True, None))
        thread.daemon = True
        thread.start()
    elif button_id == 'run-analysis' and uploaded_file:
        # Parse uploaded file
        content_type, content_string = uploaded_file.split(',')
        decoded = base64.b64decode(content_string)
        fasta_content = decoded.decode('utf-8')
        
        # Start analysis with uploaded data
        thread = threading.Thread(target=run_analysis_background, args=(False, fasta_content))
        thread.daemon = True
        thread.start()
    
    return ""

@app.callback(
    Output('results-content', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_results(n):
    """Update results display"""
    global analysis_results
    
    if not analysis_results or not analysis_results.get('biodiversity_indices'):
        return html.Div([
            html.I(className="fas fa-info-circle", 
                  style={'font-size': '3rem', 'color': '#ccc', 'margin-bottom': '1rem'}),
            html.P("No analysis results yet. Upload data or run demo analysis to see results.",
                  style={'color': '#666', 'text-align': 'center'})
        ], style={'text-align': 'center', 'padding': '3rem'})
    
    # Extract data for visualization
    biodiversity = analysis_results['biodiversity_indices']
    
    # Create metrics cards
    metrics_cards = html.Div([
        html.Div([
            html.H4(f"{biodiversity['total_sequences']}", style={'margin': '0', 'font-size': '2rem'}),
            html.P("Total Sequences", style={'margin': '0.5rem 0 0 0'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H4(f"{biodiversity['species_richness']}", style={'margin': '0', 'font-size': '2rem'}),
            html.P("Species Richness", style={'margin': '0.5rem 0 0 0'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H4(f"{biodiversity['shannon_diversity']:.2f}", style={'margin': '0', 'font-size': '2rem'}),
            html.P("Shannon Diversity", style={'margin': '0.5rem 0 0 0'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block'}),
        
        html.Div([
            html.H4(f"{analysis_results['novel_taxa_count']}", style={'margin': '0', 'font-size': '2rem'}),
            html.P("Novel Clusters", style={'margin': '0.5rem 0 0 0'})
        ], className="metric-card", style={'width': '23%', 'display': 'inline-block'})
    ], style={'margin-bottom': '2rem'})
    
    # Create visualizations
    visualizations = create_visualizations()
    
    return html.Div([
        metrics_cards,
        visualizations
    ])

def create_visualizations():
    """Create interactive Plotly visualizations"""
    global analysis_results
    
    if not analysis_results:
        return html.Div()
    
    # 1. Taxonomic Distribution Pie Chart
    classifications = analysis_results.get('classifications', [])
    reference_taxa = analysis_results.get('reference_taxa', {})
    
    # Count classifications
    known_counts = {}
    unknown_count = 0
    
    for c in classifications:
        if c == -1:
            unknown_count += 1
        else:
            taxa_name = reference_taxa.get(c, f"Unknown_{c}")
            known_counts[taxa_name] = known_counts.get(taxa_name, 0) + 1
    
    if unknown_count > 0:
        known_counts["Unclassified/Novel"] = unknown_count
    
    pie_fig = go.Figure(data=[go.Pie(
        labels=list(known_counts.keys()),
        values=list(known_counts.values()),
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    pie_fig.update_layout(
        title="Taxonomic Distribution",
        font=dict(size=12),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    # 2. Biodiversity Indices Bar Chart
    biodiversity = analysis_results['biodiversity_indices']
    
    bar_fig = go.Figure(data=[
        go.Bar(
            x=['Shannon Diversity', 'Simpson Diversity', 'Evenness'],
            y=[biodiversity['shannon_diversity'], biodiversity['simpson_diversity'], biodiversity['evenness']],
            marker=dict(color=['#667eea', '#764ba2', '#28a745'])
        )
    ])
    
    bar_fig.update_layout(
        title="Biodiversity Indices",
        yaxis_title="Index Value",
        margin=dict(t=50, b=50, l=50, r=0)
    )
    
    # 3. Analysis Summary Heatmap (mock data for embeddings similarity)
    np.random.seed(42)
    similarity_matrix = np.random.rand(10, 10)
    np.fill_diagonal(similarity_matrix, 1)
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        colorscale='Viridis',
        showscale=True
    ))
    
    heatmap_fig.update_layout(
        title="Sequence Similarity Matrix (Sample)",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # 4. Novel Taxa Discovery Scatter (mock t-SNE data)
    np.random.seed(42)
    n_points = min(100, len(classifications))
    tsne_x = np.random.randn(n_points)
    tsne_y = np.random.randn(n_points)
    colors = ['Known' if c != -1 else 'Novel' for c in classifications[:n_points]]
    
    scatter_fig = go.Figure()
    
    for category in ['Known', 'Novel']:
        mask = [c == category for c in colors]
        scatter_fig.add_trace(go.Scatter(
            x=[x for x, m in zip(tsne_x, mask) if m],
            y=[y for y, m in zip(tsne_y, mask) if m],
            mode='markers',
            name=category,
            marker=dict(
                size=8,
                color='#667eea' if category == 'Known' else '#e74c3c',
                opacity=0.7
            )
        ))
    
    scatter_fig.update_layout(
        title="Sequence Embedding Visualization (t-SNE)",
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Create layout with subplots
    return html.Div([
        html.Div([
            html.Div([
                dcc.Graph(figure=pie_fig)
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(figure=bar_fig)
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=scatter_fig)
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(figure=heatmap_fig)
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'})
        ]),
        
        # Additional insights
        html.Div([
            html.H4("🔬 Analysis Insights", style={'color': '#667eea', 'margin-bottom': '1rem'}),
            html.Ul([
                html.Li(f"Identified {len(set([c for c in classifications if c != -1]))} distinct known taxa"),
                html.Li(f"Discovered {analysis_results['novel_taxa_count']} potential novel species clusters"),
                html.Li(f"Shannon diversity index of {biodiversity['shannon_diversity']:.3f} indicates {'high' if biodiversity['shannon_diversity'] > 2 else 'moderate' if biodiversity['shannon_diversity'] > 1 else 'low'} biodiversity"),
                html.Li(f"Species evenness of {biodiversity['evenness']:.3f} shows {'balanced' if biodiversity['evenness'] > 0.7 else 'unbalanced'} community structure")
            ], style={'color': '#555'})
        ], style={
            'background': 'rgba(102, 126, 234, 0.1)',
            'padding': '1.5rem',
            'border-radius': '10px',
            'margin-top': '2rem'
        })
    ])

if __name__ == '__main__':
    print("🌊 Starting SeaquenceAI Plotly Dashboard...")
    print("🚀 Open your browser to: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)