#!/usr/bin/env python3
"""
SeaquenceAI - Deep-sea eDNA Analysis Pipeline
Main ML/DL pipeline for taxonomic classification and novel taxa discovery
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Bio import SeqIO
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class DNASequenceDataset(Dataset):
    """Custom dataset for DNA sequences"""
    
    def __init__(self, sequences, labels=None, max_length=300):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
        # Create nucleotide vocabulary
        self.vocab = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 5, 'PAD': 0}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.sequences)
    
    def encode_sequence(self, sequence):
        """Encode DNA sequence to numerical representation"""
        sequence = sequence.upper()[:self.max_length]
        encoded = [self.vocab.get(base, self.vocab['N']) for base in sequence]
        
        # Pad sequence to max_length
        if len(encoded) < self.max_length:
            encoded.extend([self.vocab['PAD']] * (self.max_length - len(encoded)))
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_seq = self.encode_sequence(sequence)
        
        if self.labels is not None:
            return encoded_seq, self.labels[idx]
        return encoded_seq

class DNAEmbeddingModel(nn.Module):
    """Deep learning model for generating DNA sequence embeddings"""
    
    def __init__(self, vocab_size=6, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(DNAEmbeddingModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim * 2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Final embedding
        embedding = self.fc(self.dropout(pooled))  # (batch_size, embedding_dim)
        
        return embedding

class TaxonomicClassifier(nn.Module):
    """Neural network for taxonomic classification"""
    
    def __init__(self, embedding_dim=128, num_classes=100):
        super(TaxonomicClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class SeaquenceAI:
    """Main SeaquenceAI pipeline for deep-sea eDNA analysis"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🌊 SeaquenceAI initialized on device: {self.device}")
        
        self.embedding_model = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.clusterer = None
        
        # Reference database (mock data for demo)
        self.reference_taxa = self._create_mock_reference_db()
        
        # Results storage
        self.results = {
            'embeddings': None,
            'classifications': None,
            'clusters': None,
            'biodiversity_indices': {},
            'novel_taxa_count': 0
        }
    
    def _create_mock_reference_db(self):
        """Create mock reference database for known taxa"""
        taxa_db = {
            0: "Foraminifera sp.",
            1: "Radiolaria sp.",
            2: "Dinoflagellate sp.",
            3: "Diatom sp.",
            4: "Copepod sp.",
            5: "Nematode sp.",
            6: "Polychaete sp.",
            7: "Cnidarian sp.",
            8: "Porifera sp.",
            9: "Echinoderm sp.",
        }
        return taxa_db
    
    def load_sequences(self, fasta_file=None, sequences=None):
        """Load DNA sequences from FASTA file or list"""
        if fasta_file and os.path.exists(fasta_file):
            print(f"📁 Loading sequences from {fasta_file}")
            sequences = []
            sequence_ids = []
            
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
            
            self.sequences = sequences
            self.sequence_ids = sequence_ids
            
        elif sequences:
            print(f"📁 Loading {len(sequences)} sequences from input list")
            self.sequences = sequences
            self.sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        else:
            # Generate mock eDNA sequences for demo
            print("🧬 Generating mock eDNA sequences for demonstration")
            self.sequences = self._generate_mock_sequences(500)
            self.sequence_ids = [f"seq_{i}" for i in range(len(self.sequences))]
        
        print(f"✅ Loaded {len(self.sequences)} sequences")
        return self.sequences
    
    def _generate_mock_sequences(self, num_sequences=500):
        """Generate mock DNA sequences for demonstration"""
        sequences = []
        bases = ['A', 'T', 'G', 'C']
        
        # Create sequences with different patterns (simulating different taxa)
        patterns = [
            ['A', 'T'] * 75,  # AT-rich sequences
            ['G', 'C'] * 75,  # GC-rich sequences
            ['A', 'G', 'C', 'T'] * 37 + ['A', 'G', 'C'],  # Mixed
            ['T', 'A', 'G', 'C'] * 37 + ['T', 'A', 'G'],  # Mixed variant
            ['C', 'G', 'A', 'T'] * 37 + ['C', 'G', 'A'],  # Mixed variant 2
        ]
        
        for i in range(num_sequences):
            if i < len(patterns) * 50:  # 50 sequences per pattern
                pattern_idx = i // 50
                base_pattern = patterns[pattern_idx]
                # Add some random mutations
                sequence = []
                for base in base_pattern:
                    if np.random.random() < 0.05:  # 5% mutation rate
                        sequence.append(np.random.choice(bases))
                    else:
                        sequence.append(base)
                sequences.append(''.join(sequence))
            else:
                # Random sequences for novel taxa
                length = np.random.randint(100, 300)
                sequence = ''.join(np.random.choice(bases) for _ in range(length))
                sequences.append(sequence)
        
        return sequences
    
    def preprocess_sequences(self):
        """Preprocess DNA sequences"""
        print("🔄 Preprocessing sequences...")
        
        # Filter sequences by length and quality
        filtered_sequences = []
        filtered_ids = []
        
        for seq, seq_id in zip(self.sequences, self.sequence_ids):
            # Remove sequences that are too short or contain too many Ns
            if len(seq) >= 50 and seq.count('N') / len(seq) < 0.1:
                filtered_sequences.append(seq)
                filtered_ids.append(seq_id)
        
        self.sequences = filtered_sequences
        self.sequence_ids = filtered_ids
        
        print(f"✅ Preprocessing complete. {len(self.sequences)} sequences retained")
    
    def train_embedding_model(self):
        """Train the DNA embedding model"""
        print("🚀 Training DNA embedding model...")
        
        # Create dataset
        dataset = DNASequenceDataset(self.sequences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.embedding_model = DNAEmbeddingModel().to(self.device)
        optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=0.001)
        
        # Simple autoencoder training (unsupervised)
        self.embedding_model.train()
        total_loss = 0
        
        for epoch in range(5):  # Quick training for demo
            epoch_loss = 0
            for batch in dataloader:
                sequences = batch.to(self.device)
                
                # Generate embeddings
                embeddings = self.embedding_model(sequences)
                
                # Simple reconstruction loss (using cosine similarity)
                loss = 1 - torch.mean(torch.cosine_similarity(
                    embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
                ).diagonal())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss = epoch_loss / len(dataloader)
            print(f"   Epoch {epoch+1}/5, Loss: {total_loss:.4f}")
        
        print("✅ Embedding model training complete")
    
    def generate_embeddings(self):
        """Generate embeddings for all sequences"""
        print("🧠 Generating sequence embeddings...")
        
        if self.embedding_model is None:
            self.train_embedding_model()
        
        dataset = DNASequenceDataset(self.sequences)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        embeddings = []
        self.embedding_model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                sequences = batch.to(self.device)
                batch_embeddings = self.embedding_model(sequences)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        self.embeddings = np.vstack(embeddings)
        self.results['embeddings'] = self.embeddings
        
        print(f"✅ Generated embeddings: {self.embeddings.shape}")
    
    def classify_sequences(self):
        """Classify sequences into known taxa"""
        print("🔬 Classifying sequences into known taxa...")
        
        if self.embeddings is None:
            self.generate_embeddings()
        
        # Mock classification using simple clustering for demo
        # In reality, this would use a trained classifier
        from sklearn.cluster import KMeans
        
        n_known_taxa = len(self.reference_taxa)
        kmeans = KMeans(n_clusters=n_known_taxa, random_state=42, n_init=10)
        
        # Use subset of embeddings for "known" classifications
        subset_size = min(200, len(self.embeddings))
        classification_labels = kmeans.fit_predict(self.embeddings[:subset_size])
        
        # Extend to all sequences with confidence scores
        all_classifications = []
        confidence_scores = []
        
        for i, embedding in enumerate(self.embeddings):
            if i < subset_size:
                pred_label = classification_labels[i]
                confidence = np.random.uniform(0.7, 0.95)  # Mock confidence
            else:
                # These are potential novel taxa
                pred_label = -1  # Unknown
                confidence = np.random.uniform(0.1, 0.3)  # Low confidence
            
            all_classifications.append(pred_label)
            confidence_scores.append(confidence)
        
        self.classifications = all_classifications
        self.confidence_scores = confidence_scores
        self.results['classifications'] = self.classifications
        
        # Count classifications
        classification_counts = Counter([c for c in self.classifications if c != -1])
        print("📊 Classification Results:")
        for label, count in classification_counts.items():
            taxa_name = self.reference_taxa.get(label, f"Unknown_{label}")
            print(f"   {taxa_name}: {count} sequences")
        
        unknown_count = self.classifications.count(-1)
        print(f"   Unclassified (potential novel taxa): {unknown_count} sequences")
    
    def discover_novel_clusters(self):
        """Discover novel taxa using unsupervised clustering"""
        print("🔍 Discovering novel taxa clusters...")
        
        if self.embeddings is None:
            self.generate_embeddings()
        
        # Focus on unclassified sequences
        unclassified_indices = [i for i, c in enumerate(self.classifications) if c == -1]
        
        if len(unclassified_indices) < 5:
            print("   Not enough unclassified sequences for clustering")
            return
        
        unclassified_embeddings = self.embeddings[unclassified_indices]
        
        # Apply DBSCAN clustering
        self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = self.clusterer.fit_predict(unclassified_embeddings)
        
        # Store cluster results
        self.cluster_labels = cluster_labels
        self.results['clusters'] = cluster_labels
        
        # Count novel clusters
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Remove noise points
        
        self.results['novel_taxa_count'] = len(unique_clusters)
        
        print(f"🆕 Discovered {len(unique_clusters)} potential novel taxa clusters")
        
        # Print cluster statistics
        cluster_counts = Counter(cluster_labels)
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                print(f"   Noise points: {count}")
            else:
                print(f"   Novel cluster {cluster_id}: {count} sequences")
        
        return cluster_labels
    
    def calculate_biodiversity_indices(self):
        """Calculate biodiversity indices"""
        print("📈 Calculating biodiversity indices...")
        
        # Shannon diversity index
        all_labels = [c for c in self.classifications if c != -1]
        all_labels.extend([f"novel_{c}" for c in self.cluster_labels if c != -1])
        
        label_counts = Counter(all_labels)
        total = sum(label_counts.values())
        
        # Shannon diversity
        shannon = -sum((count/total) * np.log(count/total) 
                      for count in label_counts.values())
        
        # Simpson diversity
        simpson = 1 - sum((count/total)**2 for count in label_counts.values())
        
        # Richness (number of different taxa)
        richness = len(label_counts)
        
        # Evenness
        evenness = shannon / np.log(richness) if richness > 1 else 0
        
        self.results['biodiversity_indices'] = {
            'shannon_diversity': shannon,
            'simpson_diversity': simpson,
            'species_richness': richness,
            'evenness': evenness,
            'total_sequences': len(self.sequences),
            'known_taxa': len([c for c in self.classifications if c != -1]),
            'novel_clusters': self.results['novel_taxa_count']
        }
        
        print("📊 Biodiversity Indices:")
        print(f"   Shannon Diversity: {shannon:.3f}")
        print(f"   Simpson Diversity: {simpson:.3f}")
        print(f"   Species Richness: {richness}")
        print(f"   Evenness: {evenness:.3f}")
        print(f"   Total Sequences: {len(self.sequences)}")
        print(f"   Known Taxa: {len([c for c in self.classifications if c != -1])}")
        print(f"   Novel Clusters: {self.results['novel_taxa_count']}")
    
    def visualize_results(self):
        """Create visualizations of the analysis results"""
        print("📊 Creating visualizations...")
        
        if self.embeddings is None:
            return
        
        # t-SNE visualization of embeddings
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_embeddings = tsne.fit_transform(self.embeddings[:200])  # Subset for speed
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: t-SNE colored by classification
        plt.subplot(1, 3, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.reference_taxa)))
        
        for i, (label, color) in enumerate(zip(range(len(self.reference_taxa)), colors)):
            mask = np.array(self.classifications[:200]) == label
            if np.any(mask):
                plt.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], 
                           c=[color], label=self.reference_taxa[label], alpha=0.7)
        
        # Plot unclassified points
        unclassified_mask = np.array(self.classifications[:200]) == -1
        if np.any(unclassified_mask):
            plt.scatter(tsne_embeddings[unclassified_mask, 0], 
                       tsne_embeddings[unclassified_mask, 1], 
                       c='red', label='Unclassified', alpha=0.7, marker='x')
        
        plt.title('Sequence Embeddings (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Taxonomic distribution
        plt.subplot(1, 3, 2)
        classification_counts = Counter([c for c in self.classifications if c != -1])
        taxa_names = [self.reference_taxa[label] for label in classification_counts.keys()]
        counts = list(classification_counts.values())
        
        plt.pie(counts, labels=taxa_names, autopct='%1.1f%%', startangle=90)
        plt.title('Taxonomic Distribution')
        
        # Plot 3: Biodiversity metrics
        plt.subplot(1, 3, 3)
        indices = self.results['biodiversity_indices']
        metrics = ['Shannon', 'Simpson', 'Richness', 'Evenness']
        values = [indices['shannon_diversity'], indices['simpson_diversity'], 
                 indices['species_richness'], indices['evenness']]
        
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        plt.title('Biodiversity Indices')
        plt.ylabel('Index Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('seaquence_ai_results.png', dpi=300, bbox_inches='tight')
        print("💾 Visualization saved as 'seaquence_ai_results.png'")
        plt.show()
    
    def save_results(self, output_file='seaquence_ai_results.json'):
        """Save analysis results to JSON file"""
        print(f"💾 Saving results to {output_file}")
        
        # Prepare results for JSON serialization
        output_results = {
            'timestamp': datetime.now().isoformat(),
            'sequence_count': len(self.sequences),
            'biodiversity_indices': self.results['biodiversity_indices'],
            'novel_taxa_count': self.results['novel_taxa_count'],
            'taxonomic_summary': {},
            'processing_summary': {
                'total_sequences': len(self.sequences),
                'classified_sequences': len([c for c in self.classifications if c != -1]),
                'novel_clusters_found': self.results['novel_taxa_count']
            }
        }
        
        # Add taxonomic summary
        classification_counts = Counter([c for c in self.classifications if c != -1])
        for label, count in classification_counts.items():
            taxa_name = self.reference_taxa.get(label, f"Unknown_{label}")
            output_results['taxonomic_summary'][taxa_name] = count
        
        with open(output_file, 'w') as f:
            json.dump(output_results, f, indent=2)
        
        print(f"✅ Results saved successfully")
    
    def run_full_analysis(self, fasta_file=None, sequences=None):
        """Run the complete SeaquenceAI analysis pipeline"""
        print("🌊 Starting SeaquenceAI Deep-sea eDNA Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Load and preprocess sequences
            self.load_sequences(fasta_file, sequences)
            self.preprocess_sequences()
            
            # Generate embeddings
            self.generate_embeddings()
            
            # Classify sequences
            self.classify_sequences()
            
            # Discover novel taxa
            self.discover_novel_clusters()
            
            # Calculate biodiversity indices
            self.calculate_biodiversity_indices()
            
            # Create visualizations
            self.visualize_results()
            
            # Save results
            self.save_results()
            
            print("=" * 60)
            print("🎉 SeaquenceAI Analysis Complete!")
            print(f"📊 Summary:")
            print(f"   • Total sequences analyzed: {len(self.sequences)}")
            print(f"   • Known taxa identified: {len(set([c for c in self.classifications if c != -1]))}")
            print(f"   • Novel clusters discovered: {self.results['novel_taxa_count']}")
            print(f"   • Shannon diversity: {self.results['biodiversity_indices']['shannon_diversity']:.3f}")
            print(f"   • Species richness: {self.results['biodiversity_indices']['species_richness']}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run SeaquenceAI"""
    print("🌊 Welcome to SeaquenceAI - Deep-sea eDNA Analysis Platform")
    
    # Initialize the pipeline
    pipeline = SeaquenceAI(device='cpu')  # Use 'cuda' if GPU available
    
    # Run full analysis with mock data
    results = pipeline.run_full_analysis()
    
    if results:
        print("\n🔬 Analysis completed successfully!")
        print("📁 Check the generated files:")
        print("   • seaquence_ai_results.json - Detailed results")
        print("   • seaquence_ai_results.png - Visualizations")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()