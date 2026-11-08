#!/usr/bin/env python3
"""
SeaquenceAI - Interactive Plotly Dashboard
Stunning web interface for deep-sea eDNA analysis with 3D animations and ocean effects
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
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
import sys
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


# Add sklearn imports with error handling
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except ImportError:
    print("Warning: scikit-learn not available. TSNE visualization will be disabled.")
    TSNE = None
    PCA = None

# Import BioPython with error handling
try:
    from Bio import SeqIO
except ImportError:
    print("Warning: BioPython not available. FASTA parsing will use fallback method.")
    SeqIO = None

# Import your main SeaquenceAI pipeline with error handling
try:
    from seaquence_ai_main import SeaquenceAI
except ImportError:
    print("Warning: seaquence_ai_main.py not found. Creating mock SeaquenceAI class.")
    
    # Mock SeaquenceAI class for testing
    class SeaquenceAI:
        def __init__(self, device='cpu'):
            self.device = device
            self.sequences = []
            self.sequence_ids = []
            self.embeddings = None
            self.classifications = []
            self.confidence_scores = []
            self.cluster_labels = []
            self.reference_taxa = {
                0: "Foraminifera sp.",
                1: "Radiolaria sp.", 
                2: "Dinoflagellate sp.",
                3: "Diatom sp.",
                4: "Copepod sp.",
            }
            self.results = {
                'embeddings': None,
                'classifications': None,
                'clusters': None,
                'biodiversity_indices': {},
                'novel_taxa_count': 0
            }
        
        def load_sequences(self, sequences=None):
            if sequences:
                self.sequences = sequences
            else:
                # Generate mock sequences
                bases = ['A', 'T', 'G', 'C']
                self.sequences = [''.join(np.random.choice(bases, 150)) for _ in range(100)]
            self.sequence_ids = [f"seq_{i}" for i in range(len(self.sequences))]
        
        def preprocess_sequences(self):
            pass
        
        def generate_embeddings(self):
            self.embeddings = np.random.rand(len(self.sequences), 128)
            self.results['embeddings'] = self.embeddings
        
        def classify_sequences(self):
            self.classifications = [np.random.randint(-1, 5) for _ in range(len(self.sequences))]
            self.confidence_scores = [np.random.rand() for _ in range(len(self.sequences))]
            self.results['classifications'] = self.classifications
        
        def discover_novel_clusters(self):
            self.cluster_labels = [np.random.randint(-1, 3) for _ in range(len(self.sequences))]
            self.results['clusters'] = self.cluster_labels
            self.results['novel_taxa_count'] = len(set([c for c in self.cluster_labels if c != -1]))
        
        def calculate_biodiversity_indices(self):
            self.results['biodiversity_indices'] = {
                'shannon_diversity': np.random.uniform(1.5, 3.0),
                'simpson_diversity': np.random.uniform(0.6, 0.9),
                'species_richness': len(set([c for c in self.classifications if c != -1])),
                'evenness': np.random.uniform(0.4, 0.8),
                'total_sequences': len(self.sequences),
                'known_taxa': len([c for c in self.classifications if c != -1]),
                'novel_clusters': self.results.get('novel_taxa_count', 0)
            }

# Helper function for JSON serialization
def to_serializable(obj):
    """Convert non-serializable objects to serializable ones."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)
        
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
    'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap'
])

# Global variables for storing analysis results
analysis_results = {}
current_pipeline = None
analysis_status = {"running": False, "progress": 0, "message": "Ready", "start_time": None}
analysis_lock = threading.Lock()

# Custom CSS styling with ocean theme and animations (kept same as your friend's snippet)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>SeaquenceAI - Deep-sea eDNA Analysis</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                overflow-x: hidden;
                background: #0a0e27;
                position: relative;
            }
            
            /* Animated Ocean Background */
            .ocean-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(45deg, #0a0e27, #1e3c72, #2a5298, #1e3c72, #0a0e27);
                background-size: 400% 400%;
                animation: oceanWave 12s ease-in-out infinite;
                z-index: -2;
            }
            
            /* Floating Bubbles */
            .bubbles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }
            
            .bubble {
                position: absolute;
                background: radial-gradient(circle, rgba(64, 224, 255, 0.3), rgba(0, 191, 255, 0.1));
                border: 1px solid rgba(64, 224, 255, 0.2);
                border-radius: 50%;
                animation: float 6s ease-in-out infinite;
            }
            
            .bubble:nth-child(1) { width: 40px; height: 40px; left: 10%; animation-delay: -0.5s; }
            .bubble:nth-child(2) { width: 20px; height: 20px; left: 20%; animation-delay: -1s; }
            .bubble:nth-child(3) { width: 60px; height: 60px; left: 35%; animation-delay: -2s; }
            .bubble:nth-child(4) { width: 80px; height: 80px; left: 50%; animation-delay: -3s; }
            .bubble:nth-child(5) { width: 30px; height: 30px; left: 70%; animation-delay: -4s; }
            .bubble:nth-child(6) { width: 50px; height: 50px; left: 85%; animation-delay: -5s; }
            
            /* DNA Helix Animation */
            .dna-container {
                position: fixed;
                top: 20%;
                right: 5%;
                width: 100px;
                height: 200px;
                z-index: 0;
                opacity: 0.3;
            }
            
            .dna-helix {
                position: absolute;
                width: 4px;
                height: 200px;
                background: linear-gradient(to bottom, #40e0ff, #00bfff, #1e90ff);
                border-radius: 2px;
                animation: rotate3d 4s linear infinite;
                transform-origin: center center;
            }
            
            .dna-helix:nth-child(1) { left: 20px; }
            .dna-helix:nth-child(2) { right: 20px; animation-delay: -2s; }
            
            .dna-base {
                position: absolute;
                width: 60px;
                height: 2px;
                background: linear-gradient(90deg, #40e0ff, #00bfff);
                left: -28px;
                animation: dnaPulse 2s ease-in-out infinite;
            }
            
            .main-header {
                background: rgba(15, 32, 60, 0.9);
                backdrop-filter: blur(20px);
                border-bottom: 2px solid rgba(64, 224, 255, 0.3);
                padding: 2rem 1rem;
                margin-bottom: 2rem;
                position: relative;
                overflow: hidden;
            }
            
            .main-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(64, 224, 255, 0.1), transparent);
                animation: shimmer 3s infinite;
            }
            
            .logo-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1rem;
            }
            
            .logo-icon {
                font-size: 4rem;
                background: linear-gradient(45deg, #40e0ff, #00bfff, #1e90ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-right: 1rem;
                animation: iconFloat 3s ease-in-out infinite;
            }
            
            .main-title {
                font-family: 'Orbitron', monospace;
                font-size: 3.5rem;
                font-weight: 900;
                background: linear-gradient(45deg, #40e0ff, #00bfff, #1e90ff, #4169e1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 0 30px rgba(64, 224, 255, 0.5);
                animation: titleGlow 2s ease-in-out infinite alternate;
            }
            
            .subtitle {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.2rem;
                font-weight: 300;
                text-align: center;
                margin-top: 0.5rem;
                letter-spacing: 2px;
            }
            
            .card {
                background: rgba(15, 32, 60, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2rem;
                margin: 1rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 20px rgba(64, 224, 255, 0.1);
                border: 1px solid rgba(64, 224, 255, 0.2);
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4), 0 0 30px rgba(64, 224, 255, 0.2);
            }
            
            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, #40e0ff, transparent);
            }
            
            .card-title {
                color: #40e0ff;
                font-family: 'Orbitron', monospace;
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
            }
            
            .card-title i {
                margin-right: 0.5rem;
                animation: iconPulse 2s ease-in-out infinite;
            }
            
            .upload-area {
                border: 2px dashed #40e0ff;
                border-radius: 15px;
                padding: 3rem 2rem;
                text-align: center;
                background: rgba(64, 224, 255, 0.05);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .upload-area:hover {
                border-color: #00bfff;
                background: rgba(64, 224, 255, 0.1);
                transform: scale(1.02);
            }
            
            .upload-area::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: radial-gradient(circle, rgba(64, 224, 255, 0.1), transparent);
                transition: all 0.3s ease;
                transform: translate(-50%, -50%);
            }
            
            .upload-area:hover::before {
                width: 300px;
                height: 300px;
            }
            
            .upload-icon {
                font-size: 4rem;
                color: #40e0ff;
                margin-bottom: 1rem;
                animation: uploadFloat 3s ease-in-out infinite;
            }
            
            .upload-text {
                color: rgba(255, 255, 255, 0.9);
                font-size: 1.1rem;
                z-index: 1;
                position: relative;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #40e0ff, #00bfff, #1e90ff);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                margin: 0.5rem;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
                box-shadow: 0 10px 25px rgba(64, 224, 255, 0.3);
            }
            
            .btn-primary:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 15px 35px rgba(64, 224, 255, 0.4);
            }
            
            .btn-primary::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                transition: left 0.5s ease;
            }
            
            .btn-primary:hover::before {
                left: 100%;
            }
            
            .btn-demo {
                background: linear-gradient(135deg, #00ff88, #00cc6a, #00994d);
                box-shadow: 0 10px 25px rgba(0, 255, 136, 0.3);
            }
            
            .btn-demo:hover {
                box-shadow: 0 15px 35px rgba(0, 255, 136, 0.4);
            }
            
            .progress-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 25px;
                padding: 5px;
                margin-top: 1rem;
                overflow: hidden;
            }
            
            .progress-bar {
                background: linear-gradient(90deg, #40e0ff, #00bfff, #1e90ff);
                height: 15px;
                border-radius: 20px;
                transition: width 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .progress-bar::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 30px;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                animation: progressShine 2s linear infinite;
            }
            
            .metric-card {
                background: linear-gradient(135deg, rgba(64, 224, 255, 0.9), rgba(30, 144, 255, 0.8));
                color: white;
                text-align: center;
                padding: 1.5rem;
                margin: 0.5rem;
                border-radius: 15px;
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
                box-shadow: 0 10px 25px rgba(64, 224, 255, 0.2);
            }
            
            .metric-card:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 15px 35px rgba(64, 224, 255, 0.3);
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.1), transparent);
                animation: cardShine 4s linear infinite;
            }
            
            .metric-number {
                font-family: 'Orbitron', monospace;
                font-size: 2.5rem;
                font-weight: 900;
                margin: 0;
                text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            }
            
            .metric-label {
                font-size: 0.9rem;
                margin: 0.5rem 0 0 0;
                opacity: 0.9;
                font-weight: 600;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                box-shadow: 0 0 10px currentColor;
            }
            
            .status-ready { 
                background-color: #00ff88; 
                color: #00ff88;
                box-shadow: 0 0 15px #00ff88;
            }
            
            .status-running { 
                background-color: #ffaa00; 
                color: #ffaa00;
                animation: pulse 1.5s infinite;
                box-shadow: 0 0 15px #ffaa00;
            }
            
            .status-error { 
                background-color: #ff4444; 
                color: #ff4444;
                box-shadow: 0 0 15px #ff4444;
            }
            
            .status-text {
                color: rgba(255, 255, 255, 0.9);
                font-weight: 600;
            }
            
            .insights-panel {
                background: rgba(0, 255, 136, 0.1);
                border: 1px solid rgba(0, 255, 136, 0.3);
                padding: 2rem;
                border-radius: 15px;
                margin-top: 2rem;
            }
            
            .insights-title {
                color: #00ff88;
                font-family: 'Orbitron', monospace;
                margin-bottom: 1rem;
            }
            
            .insights-list {
                color: rgba(255, 255, 255, 0.8);
                line-height: 1.8;
            }
            
            .insights-list li {
                margin-bottom: 0.5rem;
                padding-left: 1rem;
                position: relative;
            }
            
            .insights-list li::before {
                content: '▶';
                position: absolute;
                left: 0;
                color: #00ff88;
            }
            
            .no-results {
                text-align: center;
                padding: 4rem;
                color: rgba(255, 255, 255, 0.6);
            }
            
            .no-results-icon {
                font-size: 5rem;
                color: rgba(64, 224, 255, 0.3);
                margin-bottom: 2rem;
                animation: iconFloat 3s ease-in-out infinite;
            }
            
            /* Animations */
            @keyframes oceanWave {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(100vh) scale(0); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100px) scale(1); opacity: 0; }
            }
            
            @keyframes rotate3d {
                0% { transform: rotateY(0deg) rotateX(0deg); }
                25% { transform: rotateY(90deg) rotateX(45deg); }
                50% { transform: rotateY(180deg) rotateX(0deg); }
                75% { transform: rotateY(270deg) rotateX(-45deg); }
                100% { transform: rotateY(360deg) rotateX(0deg); }
            }
            
            @keyframes dnaPulse {
                0%, 100% { opacity: 0.3; transform: scaleX(0.8); }
                50% { opacity: 0.8; transform: scaleX(1.2); }
            }
            
            @keyframes shimmer {
                0% { left: -100%; }
                100% { left: 100%; }
            }
            
            @keyframes iconFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes titleGlow {
                0% { text-shadow: 0 0 30px rgba(64, 224, 255, 0.5); }
                100% { text-shadow: 0 0 40px rgba(64, 224, 255, 0.8), 0 0 60px rgba(64, 224, 255, 0.3); }
            }
            
            @keyframes iconPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            @keyframes uploadFloat {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-15px); }
            }
            
            @keyframes progressShine {
                0% { left: -30px; }
                100% { left: 100%; }
            }
            
            @keyframes cardShine {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(1.1); }
            }
            
            /* Scrollbar Styling */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(0, 0, 0, 0.3);
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(45deg, #40e0ff, #00bfff);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(45deg, #00bfff, #1e90ff);
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .main-title { font-size: 2.5rem; }
                .logo-icon { font-size: 3rem; }
                .card { margin: 0.5rem; padding: 1.5rem; }
                .dna-container { display: none; }
            }
        </style>
    </head>
    <body>
        <!-- Animated Ocean Background -->
        <div class="ocean-bg"></div>
        
        <!-- Floating Bubbles -->
        <div class="bubbles">
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
        </div>
        
        <!-- DNA Helix Animation -->
        <div class="dna-container">
            <div class="dna-helix">
                <div class="dna-base" style="top: 10px;"></div>
                <div class="dna-base" style="top: 30px;"></div>
                <div class="dna-base" style="top: 50px;"></div>
                <div class="dna-base" style="top: 70px;"></div>
                <div class="dna-base" style="top: 90px;"></div>
                <div class="dna-base" style="top: 110px;"></div>
                <div class="dna-base" style="top: 130px;"></div>
                <div class="dna-base" style="top: 150px;"></div>
                <div class="dna-base" style="top: 170px;"></div>
            </div>
            <div class="dna-helix">
                <div class="dna-base" style="top: 20px;"></div>
                <div class="dna-base" style="top: 40px;"></div>
                <div class="dna-base" style="top: 60px;"></div>
                <div class="dna-base" style="top: 80px;"></div>
                <div class="dna-base" style="top: 100px;"></div>
                <div class="dna-base" style="top: 120px;"></div>
                <div class="dna-base" style="top: 140px;"></div>
                <div class="dna-base" style="top: 160px;"></div>
                <div class="dna-base" style="top: 180px;"></div>
            </div>
        </div>
        
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
        html.Div([
            html.I(className="fas fa-dna logo-icon"),
            html.H1("SeaquenceAI", className="main-title")
        ], className="logo-container"),
        html.P("Advanced Deep-Sea eDNA Analysis & Marine Biodiversity Discovery Platform", 
               className="subtitle")
    ], className="main-header"),
    
    # Main content
    html.Div([
        # Control Panel
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-rocket"),
                    "Analysis Control Center"
                ], className="card-title"),
                
                # File upload area
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt upload-icon"),
                            html.Div([
                                'Drop your FASTA sequences here or ',
                                html.Span('browse files', style={
                                    'color': '#40e0ff', 
                                    'text-decoration': 'underline',
                                    'cursor': 'pointer'
                                })
                            ], className="upload-text")
                        ]),
                        className="upload-area",
                        multiple=False
                    ),
                ], style={'margin-bottom': '2rem'}),
                
                # Analysis controls
                html.Div([
                    html.Button([
                        html.I(className="fas fa-play", style={'margin-right': '10px'}),
                        "Launch Analysis"
                    ], id='run-analysis', n_clicks=0, className="btn-primary"),
                    
                    html.Br(),
                    
                    html.Button([
                        html.I(className="fas fa-magic", style={'margin-right': '10px'}),
                        "Demo with AI Dataset"
                    ], id='demo-analysis', n_clicks=0, className="btn-primary btn-demo")
                ], style={'text-align': 'center', 'margin': '2rem 0'}),
                
                # Status indicator
                html.Div(id='analysis-status', className="status-text", 
                        style={'margin-bottom': '1rem', 'text-align': 'center'}),
                
                # Progress bar
                html.Div([
                    html.Div(id='progress-bar', className='progress-bar', 
                            style={'width': '0%'})
                ], className="progress-container")
                
            ], className="card")
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Results Panel
        html.Div([
            html.Div([
                html.H3([
                    html.I(className="fas fa-microscope"),
                    "Analysis Dashboard"
                ], className="card-title"),
                html.Div(id='results-content')
            ], className="card")
        ], style={'width': '68%', 'display': 'inline-block', 'vertical-align': 'top'})
        
    ], style={'max-width': '1400px', 'margin': '0 auto', 'padding': '0 1rem'}),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'}),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # Update every 2 seconds
        n_intervals=0
    )
])

# ---------- Helper functions ----------

def parse_fasta_fallback(fasta_content):
    """Fallback FASTA parser when BioPython is not available.
    fasta_content: text string (decoded) of FASTA file
    Returns: list of sequences (strings)
    """
    sequences = []
    current_seq = []
    for line in fasta_content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current_seq:
                sequences.append(''.join(current_seq))
                current_seq = []
            # header ignored
        else:
            current_seq.append(line)
    if current_seq:
        sequences.append(''.join(current_seq))
    return sequences

def parse_upload(contents):
    """Parse dcc.Upload contents into a list of sequences."""
    if not contents:
        return []
    try:
        header, encoded = contents.split(',', 1)
    except Exception:
        return []
    data = base64.b64decode(encoded)
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        text = data.decode('latin-1')
    # If SeqIO available, try to parse
    if SeqIO is not None:
        try:
            handle = io.StringIO(text)
            seqs = [str(rec.seq) for rec in SeqIO.parse(handle, 'fasta')]
            if seqs:
                return seqs
        except Exception:
            # fallback to manual parser
            pass
    return parse_fasta_fallback(text)

def create_scatter_from_embeddings(embeddings, labels=None, hovertext=None):
    """Return a dcc.Graph scatter figure (2D). If embeddings are higher-dim, reduce to 2D (TSNE if available)."""
    X = np.array(embeddings)
    if X.shape[1] > 2:
        if TSNE is not None:
            try:
                reducer = TSNE(n_components=2, random_state=42, init='random', perplexity=30, learning_rate='auto')
                X2 = reducer.fit_transform(X)
            except Exception:
                # fallback to PCA
                if PCA is not None:
                    reducer = PCA(n_components=2, random_state=42)
                    X2 = reducer.fit_transform(X)
                else:
                    X2 = X[:, :2]
        elif PCA is not None:
            reducer = PCA(n_components=2, random_state=42)
            X2 = reducer.fit_transform(X)
        else:
            X2 = X[:, :2]
    else:
        X2 = X
    
    df = pd.DataFrame({'x': X2[:, 0], 'y': X2[:, 1]})
    if labels is not None:
        df['label'] = labels
    if hovertext is not None:
        df['hover'] = hovertext
    
    fig = px.scatter(df, x='x', y='y', color='label' if 'label' in df.columns else None,
                     hover_name='hover' if 'hover' in df.columns else None,
                     title='Embedding visualization (2D)')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font_color='white',
        legend_title_text='Cluster'
    )
    return dcc.Graph(figure=fig, style={'height': '500px'})

def build_results_layout(results):
    """Construct HTML children for results-content from analysis results dict"""
    if not results:
        return html.Div([
            html.Div("No results yet. Launch analysis or upload sequences.", className="no-results")
        ])
    
    # build metric cards
    biodiv = results.get('biodiversity_indices', {})
    total_sequences = biodiv.get('total_sequences', '—')
    species_richness = biodiv.get('species_richness', '—')
    shannon = biodiv.get('shannon_diversity', '—')
    simpson = biodiv.get('simpson_diversity', '—')
    novel_clusters = biodiv.get('novel_clusters', results.get('novel_taxa_count', 0))
    
    cards = html.Div([
        html.Div([
            html.Div([
                html.Div(f"{total_sequences}", className="metric-number"),
                html.Div("Total Sequences", className="metric-label")
            ], className="metric-card")
        ], style={'display': 'inline-block', 'width': '23%'}),
        html.Div([
            html.Div([
                html.Div(f"{species_richness}", className="metric-number"),
                html.Div("Species Richness", className="metric-label")
            ], className="metric-card")
        ], style={'display': 'inline-block', 'width': '23%'}),
        html.Div([
            html.Div([
                html.Div(f"{shannon:.2f}" if isinstance(shannon, float) else shannon, className="metric-number"),
                html.Div("Shannon Diversity", className="metric-label")
            ], className="metric-card")
        ], style={'display': 'inline-block', 'width': '23%'}),
        html.Div([
            html.Div([
                html.Div(f"{novel_clusters}", className="metric-number"),
                html.Div("Novel Clusters", className="metric-label")
            ], className="metric-card")
        ], style={'display': 'inline-block', 'width': '23%'}),
    ], style={'margin-bottom': '1rem'})
    
    # embedding scatter (if present)
    graph_children = []
    embeddings = results.get('embeddings')
    labels = results.get('classifications') or results.get('clusters')
    hover = None
    if embeddings is not None:
        try:
            graph_children.append(create_scatter_from_embeddings(np.array(embeddings), labels=labels, hovertext=hover))
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            # if something fails, skip the figure
            pass
    
    # top taxa / insights
    insights_list = []
    classification = results.get('classifications') or []
    ref = results.get('reference_taxa') or {}
    # make a small tally
    if classification:
        counts = {}
        for c in classification:
            counts[c] = counts.get(c, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])[:6]
        for label, cnt in sorted_counts:
            name = ref.get(label, f"Unknown ({label})")
            insights_list.append(html.Li(f"{name}: {cnt} sequences"))
    if not insights_list:
        insights_list.append(html.Li("No classification results available."))
    
    insights_panel = html.Div([
        html.H4("Insights & Top taxa", className="insights-title"),
        html.Ul(insights_list, className="insights-list")
    ], className="insights-panel")
    
    return html.Div([cards] + graph_children + [insights_panel])

# ---------- Background analysis pipeline ----------

def run_analysis_background(sequences, demo=False):
    """Run the SeaquenceAI pipeline in a background thread.
    Update analysis_status and analysis_results (thread-safe).
    """
    global current_pipeline, analysis_results, analysis_status
    with analysis_lock:
        if analysis_status.get('running'):
            return  # already running
        analysis_status['running'] = True
        analysis_status['progress'] = 1
        analysis_status['message'] = 'Starting analysis...'
        analysis_status['start_time'] = datetime.utcnow().isoformat()
    try:
        pipeline = SeaquenceAI()
        current_pipeline = pipeline
        pipeline.load_sequences(sequences if sequences else None)
        with analysis_lock:
            analysis_status['progress'] = 5
            analysis_status['message'] = 'Preprocessing sequences...'
        pipeline.preprocess_sequences()
        time.sleep(0.5)  # simulate progress
        with analysis_lock:
            analysis_status['progress'] = 20
            analysis_status['message'] = 'Generating embeddings...'
        pipeline.generate_embeddings()
        time.sleep(0.5)
        with analysis_lock:
            analysis_status['progress'] = 50
            analysis_status['message'] = 'Classifying sequences...'
        pipeline.classify_sequences()
        time.sleep(0.5)
        with analysis_lock:
            analysis_status['progress'] = 75
            analysis_status['message'] = 'Discovering clusters and novel taxa...'
        pipeline.discover_novel_clusters()
        time.sleep(0.5)
        with analysis_lock:
            analysis_status['progress'] = 90
            analysis_status['message'] = 'Calculating biodiversity indices...'
        pipeline.calculate_biodiversity_indices()
        time.sleep(0.5)
        # collect results
        result_obj = {
            'embeddings': pipeline.results.get('embeddings') if pipeline.results.get('embeddings') is not None else pipeline.embeddings,
            'classifications': pipeline.results.get('classifications') if pipeline.results.get('classifications') is not None else pipeline.classifications,
            'clusters': pipeline.results.get('clusters') if pipeline.results.get('clusters') is not None else pipeline.cluster_labels,
            'biodiversity_indices': pipeline.results.get('biodiversity_indices', {}),
            'novel_taxa_count': pipeline.results.get('novel_taxa_count', 0),
            'reference_taxa': getattr(pipeline, 'reference_taxa', {})
        }
        with analysis_lock:
            analysis_results = result_obj
            analysis_status['progress'] = 100
            analysis_status['message'] = 'Analysis complete.'
    except Exception as e:
        with analysis_lock:
            analysis_status['message'] = f'Error: {str(e)}'
            analysis_status['running'] = False
            analysis_status['progress'] = 0
    finally:
        # keep running flag True briefly so UI shows results - set to False on next poll after completion
        time.sleep(0.2)
        with analysis_lock:
            analysis_status['running'] = False

# ---------- Callbacks ----------

# Single callback to trigger analysis (both Upload + Demo buttons) and update immediate status/progress bar.
# This callback DOES NOT attempt to write stored-data or results; it only toggles the background thread and returns status bar values.
@app.callback(
    Output('analysis-status', 'children'),
    Output('progress-bar', 'style'),
    Input('run-analysis', 'n_clicks'),
    Input('demo-analysis', 'n_clicks'),
    State('upload-data', 'contents'),
    prevent_initial_call=True
)
def trigger_analysis(run_n, demo_n, upload_contents):
    triggered = callback_context.triggered
    if not triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = triggered[0]['prop_id'].split('.')[0]
    # If already running, return current status
    with analysis_lock:
        if analysis_status.get('running'):
            status_msg = f"⏳ {analysis_status.get('message', 'Running...')}"
            progress_style = {'width': f"{analysis_status.get('progress', 0)}%"}
            return status_msg, progress_style
    # Determine sequences to run
    sequences = []
    if trigger_id == 'demo-analysis':
        sequences = None  # pipeline will generate mock sequences if None
    elif trigger_id == 'run-analysis':
        sequences = parse_upload(upload_contents)
        if not sequences:
            # If no sequences were uploaded, inform user
            with analysis_lock:
                analysis_status['message'] = 'No FASTA uploaded — please upload a FASTA file or use Demo.'
                analysis_status['progress'] = 0
            status_msg = f"⚠️ {analysis_status['message']}"
            return status_msg, {'width': '0%'}
    # Start background thread
    t = threading.Thread(target=run_analysis_background, args=(sequences, trigger_id == 'demo-analysis'), daemon=True)
    t.start()
    with analysis_lock:
        status_msg = "🚀 Analysis launched..."
        analysis_status['message'] = status_msg
        analysis_status['progress'] = 2
    return status_msg, {'width': '2%'}

# Interval callback: poll the global analysis_status and analysis_results and write the results to UI and stored-data.
# This is the ONLY callback that writes the heavy results (stored-data and results-content) — avoids child callback conflicts.
@app.callback(
    Output('results-content', 'children'),
    Output('stored-data', 'children'),
    Input('interval-component', 'n_intervals')
)
def poll_status_and_update(n_intervals):
    with analysis_lock:
        status_copy = analysis_status.copy()
        results_copy = analysis_results.copy() if isinstance(analysis_results, dict) else {}

    status_block = html.Div([
        html.Div([
            html.Span(className="status-indicator " + ("status-running" if status_copy.get('running') else "status-ready")),
            html.Span(status_copy.get('message', 'Ready'), style={'marginLeft': '8px'})
        ])
    ], style={'marginBottom': '1rem'})

    results_layout = build_results_layout(results_copy)
    full = html.Div([status_block, results_layout])

    # ✅ safe serialization now
    stored_json = json.dumps({'status': status_copy, 'results': results_copy}, default=to_serializable)
    return full, stored_json


# ---------- Main ----------

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=True, port=port)