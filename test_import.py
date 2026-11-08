import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Try importing your SeaquenceAI
try:
    from seaquence_ai_main import SeaquenceAI
    print("Successfully imported SeaquenceAI!")
    
    # Test creating an instance
    pipeline = SeaquenceAI()
    print("Successfully created SeaquenceAI instance!")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}")