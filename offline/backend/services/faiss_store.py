"""
FAISS Vector Store Module
Handles FAISS index management, persistence, and retrieval
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class FAISSCollection:
    """FAISS-based vector store with ChromaDB-like interface."""
    
    def __init__(self, name: str, persist_dir: str, embedding_model: SentenceTransformer):
        """
        Initialize FAISS collection.
        
        Args:
            name: Collection name
            persist_dir: Directory to persist index and metadata
            embedding_model: SentenceTransformer model for embeddings
        """
        self.name = name
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Storage
        self.index = None
        self.documents = []  # Store document texts
        self.metadatas = []  # Store metadata dicts
        self.ids = []  # Store document IDs
        
        # Paths
        self.index_path = os.path.join(persist_dir, f"{name}_index.faiss")
        self.meta_path = os.path.join(persist_dir, f"{name}_metadata.pkl")
        
        # Load or create
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata['documents']
                    self.metadatas = metadata['metadatas']
                    self.ids = metadata['ids']
                print(f"✅ Loaded FAISS index '{self.name}' with {len(self.documents)} documents")
            except Exception as e:
                print(f"⚠️  Failed to load index '{self.name}': {e}, creating new")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        print(f"✅ Created new FAISS index '{self.name}'")
    
    def _persist(self):
        """Save index and metadata to disk."""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            print(f"      📁 Index saved: {self.index_path} (ntotal={self.index.ntotal})")
            
            with open(self.meta_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids
                }, f)
            print(f"      📁 Metadata saved: {self.meta_path} ({len(self.ids)} docs)")
        except Exception as e:
            print(f"      ⚠️  Failed to persist index '{self.name}': {e}")
    
    def add(self, ids: list[str], documents: list[str], metadatas: list[dict] = None):
        """
        Add documents to the collection.
        
        Args:
            ids: List of unique IDs
            documents: List of document texts
            metadatas: List of metadata dicts
        """
        if not documents:
            print(f"⚠️  No documents to add to {self.name}")
            return
        
        print(f"\n🔧 FAISS Collection '{self.name}' - Adding documents")
        print(f"   Documents count: {len(documents)}")
        print(f"   IDs count: {len(ids)}")
        
        # Generate embeddings
        print(f"   📊 Generating embeddings using {self.embedding_model.__class__.__name__}...")
        try:
            embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
            print(f"   ✅ Generated embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"   ❌ Error generating embeddings: {e}")
            raise
        
        # Add to FAISS index
        print(f"   📥 Adding {len(embeddings)} embeddings to FAISS index...")
        try:
            self.index.add(embeddings.astype('float32'))
            print(f"   ✅ FAISS index updated. Current size: {self.index.ntotal}")
        except Exception as e:
            print(f"   ❌ Error adding to FAISS: {e}")
            raise
        
        # Store metadata
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas if metadatas else [{} for _ in documents])
        
        print(f"   💾 Collection metadata: Total documents now = {len(self.documents)}")
        
        # Persist changes
        print(f"   💾 Persisting to disk...")
        try:
            self._persist()
            print(f"   ✅ Persisted to {self.index_path}")
        except Exception as e:
            print(f"   ❌ Error persisting: {e}")
            raise
    
    def get(self, where: dict = None):
        """
        Get documents matching metadata filter.
        
        Args:
            where: Metadata filter dict (e.g., {"source": "file.pdf"})
        
        Returns:
            Dict with 'ids', 'documents', 'metadatas' keys
        """
        if not where:
            return {
                'ids': self.ids,
                'documents': self.documents,
                'metadatas': self.metadatas
            }
        
        # Filter by metadata
        matching_ids = []
        matching_docs = []
        matching_metas = []
        
        for i, meta in enumerate(self.metadatas):
            if all(meta.get(k) == v for k, v in where.items()):
                matching_ids.append(self.ids[i])
                matching_docs.append(self.documents[i])
                matching_metas.append(meta)
        
        return {
            'ids': matching_ids,
            'documents': matching_docs,
            'metadatas': matching_metas
        }
    
    def query(self, query_texts: list[str], n_results: int = 5):
        """
        Query the collection for similar documents.
        
        Args:
            query_texts: List of query strings (typically single query)
            n_results: Number of results to return
        
        Returns:
            Dict with 'documents', 'distances', 'metadatas' keys
        """
        if not query_texts or len(self.documents) == 0:
            print(f"⚠️  Query on empty collection '{self.name}'")
            return {'documents': [[]], 'distances': [[]], 'metadatas': [[]]}
        
        print(f"\n🔍 Querying FAISS collection '{self.name}'")
        print(f"   Query text: '{query_texts[0][:100]}...'")
        print(f"   Collection size: {len(self.documents)} documents, FAISS index size: {self.index.ntotal}")
        
        # Encode query
        try:
            query_embedding = self.embedding_model.encode(query_texts, convert_to_numpy=True)
            print(f"   ✅ Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            print(f"   ❌ Error encoding query: {e}")
            raise
        
        # Search FAISS
        k = min(n_results, len(self.documents))
        print(f"   🔎 Searching for {k} results...")
        try:
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            print(f"   ✅ Found {len(indices[0])} results, distances: {distances[0][:3]}...")
        except Exception as e:
            print(f"   ❌ Error searching FAISS: {e}")
            raise
        
        # Gather results
        result_docs = []
        result_metas = []
        
        for idx_list in indices:
            docs = [self.documents[i] for i in idx_list if i < len(self.documents)]
            metas = [self.metadatas[i] for i in idx_list if i < len(self.metadatas)]
            result_docs.append(docs)
            result_metas.append(metas)
            print(f"   📄 Top result preview: {docs[0][:150]}..." if docs else "   ❌ No results")
        
        return {
            'documents': result_docs,
            'distances': distances.tolist(),
            'metadatas': result_metas
        }
    
    def count(self):
        """Return number of documents in collection."""
        return len(self.documents)
