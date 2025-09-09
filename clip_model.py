"""
Phase 2: Load Pretrained CLIP Model
On-Device Photo Search Prototype - CLIP Model Implementation
"""

import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
from typing import Union
import os

class CLIPEmbeddingExtractor:
    """CLIP model wrapper for extracting image and text embeddings"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model and processor
        
        Args:
            model_name: HuggingFace model identifier for CLIP
        """
        print(f"🔄 Loading CLIP model: {model_name}")
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🎯 Using device: cuda")
        else:
            self.device = torch.device("cpu") 
            print(f"🎯 Using device: cpu")
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ CLIP model loaded successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def get_clip_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract CLIP embedding from an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            512-dimensional embedding as NumPy array
        """
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Open image
            image = Image.open(image_path).convert('RGB')
            
            # Process image (224x224, normalize)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize embeddings (important for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and return
            embedding = image_features.cpu().numpy().squeeze()
            
            print(f"📸 Extracted image embedding: {image_path} -> {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            raise
    
    def get_clip_text_embedding(self, query: str) -> np.ndarray:
        """
        Extract CLIP embedding from a text query
        
        Args:
            query: Text query string
            
        Returns:
            512-dimensional embedding as NumPy array
        """
        try:
            # Process text
            inputs = self.processor(text=query, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
            # Normalize embeddings (important for cosine similarity)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy and return
            embedding = text_features.cpu().numpy().squeeze()
            
            print(f"📝 Extracted text embedding: '{query}' -> {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"❌ Error processing text '{query}': {e}")
            raise
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure embeddings are 2D for sklearn
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.reshape(1, -1)
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
            
        # Compute cosine similarity
        similarity = np.dot(image_embedding, text_embedding.T).item()
        return max(0, similarity)  # Ensure non-negative
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings"""
        return self.model.config.projection_dim


def test_clip_model():
    """Test the CLIP model implementation"""
    print("\n" + "="*60)
    print("🧪 Testing CLIP Model Implementation")
    print("="*60)
    
    try:
        # Initialize CLIP extractor
        clip = CLIPEmbeddingExtractor()
        
        # Test text embedding
        print("\n📝 Testing text embedding...")
        test_queries = [
            "a red car on the street",
            "a cute dog playing in the park", 
            "a sunset over the ocean",
            "a person reading a book"
        ]
        
        text_embeddings = []
        for query in test_queries:
            embedding = clip.get_clip_text_embedding(query)
            text_embeddings.append(embedding)
            print(f"   Query: '{query}' -> Embedding shape: {embedding.shape}")
        
        print(f"\n📊 Embedding dimension: {clip.get_embedding_dimension()}")
        print(f"📊 Actual embedding shape: {text_embeddings[0].shape}")
        
        # Test similarity between different text queries
        print("\n🔍 Testing text-to-text similarity:")
        sim1 = clip.compute_similarity(text_embeddings[0], text_embeddings[1])
        sim2 = clip.compute_similarity(text_embeddings[0], text_embeddings[0])  # Self-similarity
        print(f"   'red car' vs 'cute dog': {sim1:.3f}")
        print(f"   'red car' vs 'red car': {sim2:.3f}")
        
        # Create a test image (since we don't have sample images yet)
        print("\n🖼️ Creating test image for embedding...")
        test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red image
        test_image_path = "test_image.jpg"
        test_image.save(test_image_path)
        
        # Test image embedding
        image_embedding = clip.get_clip_image_embedding(test_image_path)
        print(f"   Test image embedding shape: {image_embedding.shape}")
        
        # Test image-text similarity
        print("\n🔗 Testing image-text similarity:")
        text_red = clip.get_clip_text_embedding("a red image")
        text_blue = clip.get_clip_text_embedding("a blue image")
        
        sim_red = clip.compute_similarity(image_embedding, text_red)
        sim_blue = clip.compute_similarity(image_embedding, text_blue)
        
        print(f"   Red image vs 'red image': {sim_red:.3f}")
        print(f"   Red image vs 'blue image': {sim_blue:.3f}")
        
        # Cleanup
        os.remove(test_image_path)
        
        print(f"\n✅ CLIP model test completed successfully!")
        print(f"🎉 Phase 2 complete - ready for Phase 3 (SQLite Database)!")
        
        return clip
        
    except Exception as e:
        print(f"❌ CLIP model test failed: {e}")
        raise


if __name__ == "__main__":
    # Test the CLIP model implementation
    clip_extractor = test_clip_model()
