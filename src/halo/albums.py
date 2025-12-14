"""Automatic album generation using clustering and LLM."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from halo.config import get_config
from halo.embeddings import embed_image
from halo.llm_utils import _generate_text
from halo.search import PhotoSearcher


@dataclass
class Album:
    """Represents a photo album with metadata."""
    id: str
    title: str
    description: str
    story: str
    photo_paths: List[str]
    created_at: str
    cluster_method: str
    num_photos: int


class AlbumGenerator:
    """Generates photo albums using clustering and LLM."""
    
    def __init__(self):
        self.cfg = get_config()
        self.searcher = PhotoSearcher()
        
    def generate_albums(
        self,
        method: Literal["visual", "temporal", "hybrid"] = "visual",
        n_albums: int = 5,
        min_photos: int = 3,
    ) -> List[Album]:
        """
        Generate albums using specified clustering method.
        
        Args:
            method: Clustering strategy ("visual", "temporal", or "hybrid")
            n_albums: Target number of albums
            min_photos: Minimum photos per album
            
        Returns:
            List of Album objects
        """
        # Get all photos from vector store
        photos = self._get_all_photos()
        
        if len(photos) < min_photos:
            return []
        
        # Cluster photos based on method
        if method == "visual":
            clusters = self._cluster_by_visual_similarity(photos, n_albums)
        elif method == "temporal":
            clusters = self._cluster_by_time(photos, n_albums)
        else:  # hybrid
            clusters = self._cluster_hybrid(photos, n_albums)
        
        # Filter out small clusters
        valid_clusters = [c for c in clusters if len(c) >= min_photos]
        
        # Generate album metadata for each cluster
        albums = []
        for idx, cluster in enumerate(valid_clusters):
            album = self._create_album(cluster, idx, method)
            albums.append(album)
        
        return albums
    
    def _get_all_photos(self) -> List[Tuple[str, np.ndarray, dict]]:
        """
        Retrieve all photos from vector store.
        
        Returns:
            List of (path, embedding, metadata) tuples
        """
        collection = self.searcher.image_collection
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        # Check if results exist and have data
        if results is None:
            return []
        
        embeddings = results.get('embeddings')
        documents = results.get('documents')
        
        if embeddings is None or documents is None:
            return []
            
        if len(documents) == 0:
            return []
        
        photos = []
        for i, path in enumerate(documents):
            embedding = np.array(embeddings[i])
            metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
            photos.append((path, embedding, metadata))
        
        return photos
    
    def _cluster_by_visual_similarity(
        self,
        photos: List[Tuple[str, np.ndarray, dict]],
        n_clusters: int
    ) -> List[List[Tuple[str, np.ndarray, dict]]]:
        """Cluster photos by CLIP embedding similarity using K-means."""
        embeddings = np.array([photo[1] for photo in photos])
        
        # Use K-means clustering
        n_clusters = min(n_clusters, len(photos))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Group photos by cluster
        clusters = [[] for _ in range(n_clusters)]
        for photo, label in zip(photos, labels):
            clusters[label].append(photo)
        
        return [c for c in clusters if c]  # Remove empty clusters
    
    def _cluster_by_time(
        self,
        photos: List[Tuple[str, np.ndarray, dict]],
        n_clusters: int
    ) -> List[List[Tuple[str, np.ndarray, dict]]]:
        """Cluster photos by capture date/time."""
        # Extract timestamps
        photos_with_time = []
        photos_without_time = []
        
        for photo in photos:
            metadata = photo[2]
            dt_iso = metadata.get('datetime_iso')
            
            if dt_iso:
                try:
                    dt = datetime.fromisoformat(dt_iso)
                    timestamp = dt.timestamp()
                    photos_with_time.append((photo, timestamp))
                except (ValueError, TypeError):
                    photos_without_time.append(photo)
            else:
                photos_without_time.append(photo)
        
        if not photos_with_time:
            # Fallback to visual clustering if no timestamps
            return self._cluster_by_visual_similarity(photos, n_clusters)
        
        # Sort by time
        photos_with_time.sort(key=lambda x: x[1])
        
        # Create time-based clusters (simple binning)
        cluster_size = max(1, len(photos_with_time) // n_clusters)
        clusters = []
        
        for i in range(0, len(photos_with_time), cluster_size):
            cluster = [p[0] for p in photos_with_time[i:i + cluster_size]]
            if cluster:
                clusters.append(cluster)
        
        # Add photos without timestamps to the last cluster
        if photos_without_time and clusters:
            clusters[-1].extend(photos_without_time)
        elif photos_without_time:
            clusters.append(photos_without_time)
        
        return clusters
    
    def _cluster_hybrid(
        self,
        photos: List[Tuple[str, np.ndarray, dict]],
        n_clusters: int
    ) -> List[List[Tuple[str, np.ndarray, dict]]]:
        """Cluster photos using both visual similarity and temporal information."""
        # Extract features
        features_list = []
        valid_photos = []
        
        for photo in photos:
            embedding = photo[1]
            metadata = photo[2]
            
            # Get timestamp feature
            dt_iso = metadata.get('datetime_iso')
            timestamp_feature = 0.0
            
            if dt_iso:
                try:
                    dt = datetime.fromisoformat(dt_iso)
                    # Normalize timestamp to [0, 1] range
                    timestamp_feature = dt.timestamp() / 1e10
                except (ValueError, TypeError):
                    pass
            
            # Get GPS features
            lat = metadata.get('latitude', 0.0) or 0.0
            lon = metadata.get('longitude', 0.0) or 0.0
            
            # Combine features: embedding + timestamp + GPS
            # Weight embeddings more heavily (0.7) vs metadata (0.3)
            combined = np.concatenate([
                embedding * 0.7,
                [timestamp_feature * 0.15, lat * 0.075, lon * 0.075]
            ])
            
            features_list.append(combined)
            valid_photos.append(photo)
        
        features = np.array(features_list)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use K-means on combined features
        n_clusters = min(n_clusters, len(valid_photos))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Group photos by cluster
        clusters = [[] for _ in range(n_clusters)]
        for photo, label in zip(valid_photos, labels):
            clusters[label].append(photo)
        
        return [c for c in clusters if c]
    
    def _create_album(
        self,
        cluster: List[Tuple[str, np.ndarray, dict]],
        idx: int,
        method: str
    ) -> Album:
        """Create an Album object with LLM-generated title and description."""
        ordered_cluster = self._order_cluster_photos(cluster)
        paths = [photo[0] for photo in ordered_cluster]

        # Get representative metadata
        metadata_summary = self._summarize_cluster_metadata(ordered_cluster)

        captions = [
            (photo[2].get("caption") or Path(photo[0]).stem).strip() or Path(photo[0]).stem
            for photo in ordered_cluster
        ]
        story = self._generate_album_story_text(captions)

        # Generate title and description using LLM
        title, description = self._generate_album_story(
            ordered_cluster, metadata_summary, method
        )

        return Album(
            id=f"album_{method}_{idx}_{datetime.now().timestamp()}",
            title=title,
            description=description,
            story=story,
            photo_paths=paths,
            created_at=datetime.now().isoformat(),
            cluster_method=method,
            num_photos=len(paths)
        )
    
    def _summarize_cluster_metadata(
        self,
        cluster: List[Tuple[str, np.ndarray, dict]]
    ) -> str:
        """Create a text summary of cluster metadata."""
        dates = []
        locations = []
        
        for photo in cluster:
            metadata = photo[2]
            
            # Collect dates
            dt_iso = metadata.get('datetime_iso')
            if dt_iso:
                dates.append(dt_iso)
            
            # Collect GPS
            lat = metadata.get('latitude')
            lon = metadata.get('longitude')
            if lat and lon:
                locations.append(f"{lat:.2f},{lon:.2f}")
        
        summary_parts = []
        
        if dates:
            dates.sort()
            summary_parts.append(f"Dates: {dates[0]} to {dates[-1]}")
        
        if locations:
            summary_parts.append(f"Locations: {len(set(locations))} unique GPS coordinates")
        
        summary_parts.append(f"Total photos: {len(cluster)}")
        
        return " | ".join(summary_parts)

    def _order_cluster_photos(
        self, cluster: List[Tuple[str, np.ndarray, dict]]
    ) -> List[Tuple[str, np.ndarray, dict]]:
        """Order photos for storytelling: prefer chronological, fallback to PCA on embeddings."""
        with_time = []
        without_time = []
        for photo in cluster:
            dt_iso = photo[2].get('datetime_iso')
            if dt_iso:
                try:
                    with_time.append((datetime.fromisoformat(dt_iso), photo))
                except ValueError:
                    without_time.append(photo)
            else:
                without_time.append(photo)

        if with_time:
            with_time.sort(key=lambda item: item[0])
            ordered = [p for _, p in with_time] + without_time
            return ordered

        if len(cluster) > 1:
            try:
                embeddings = np.vstack([p[1] for p in cluster])
                ordering = np.argsort(
                    PCA(n_components=1, random_state=42).fit_transform(embeddings).flatten()
                )
                return [cluster[i] for i in ordering]
            except Exception:
                pass

        return cluster
    
    def _generate_album_story(
        self,
        cluster: List[Tuple[str, np.ndarray, dict]],
        metadata_summary: str,
        method: str
    ) -> Tuple[str, str]:
        """Use LLM to generate album title and description."""
        # Sample up to 5 representative photos for context
        sample_size = min(5, len(cluster))
        sample_indices = np.linspace(0, len(cluster) - 1, sample_size, dtype=int)
        sample_photos = [cluster[i] for i in sample_indices]
        
        # Build context for LLM
        context_parts = [
            f"Album clustering method: {method}",
            f"Metadata: {metadata_summary}",
            f"Sample photo filenames: {', '.join([Path(p[0]).name for p in sample_photos[:3]])}"
        ]
        context = "\n".join(context_parts)
        
        instruction = (
            "You are creating a photo album. Based on the clustering method and metadata, "
            "generate a creative album title (max 6 words) and a brief description (2-3 sentences) "
            "that tells a story about this collection. "
            "Return ONLY a JSON object with 'title' and 'description' keys."
        )
        
        # Try to get LLM response
        response = _generate_text(instruction, context, max_tokens=150)
        
        if response:
            try:
                # Parse JSON response
                data = json.loads(response.strip().replace("```json", "").replace("```", ""))
                title = data.get('title', f'Album {method.title()}')
                description = data.get('description', 'A collection of photos.')
                return title, description
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Fallback titles if LLM fails
        fallback_titles = {
            'visual': 'Visual Memories',
            'temporal': 'Timeline Collection',
            'hybrid': 'Mixed Moments'
        }
        
        return (
            fallback_titles.get(method, 'Photo Album'),
            f"A collection of {len(cluster)} photos grouped by {method} similarity. {metadata_summary}"
        )

    def _generate_album_story_text(self, captions: List[str]) -> str:
        """Generate a cohesive short story tying together ordered captions."""
        if not captions:
            return "No story available."

        scene_list = [f"{i + 1}. {cap}" for i, cap in enumerate(captions)]
        scenes_formatted = "\n".join(scene_list)
        instruction = (
            "You are a narrative-generation model. You are given a sequence of image captions that represent visual "
            "scenes taken in order. Write a coherent, engaging short story (150-250 words) that directly addresses "
            "the user as the protagonist. Connect the scenes naturally, preserve the visual facts, and ensure a clear "
            "arc: beginning -> development -> resolution. Avoid introducing contradictions or new locations not "
            "suggested by the scenes."
        )
        context = f"Scenes:\n{scenes_formatted}\n\nProduce the story now."
        response = _generate_text(instruction, context, max_tokens=None)
        if response:
            return response
        return "A story could not be generated (set LLM API keys to enable this feature)."
    
    def save_albums(self, albums: List[Album], output_path: Optional[Path] = None) -> Path:
        """Save albums to JSON file."""
        if output_path is None:
            output_path = self.cfg.photo_root / "albums.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        albums_data = [
            {
                'id': album.id,
                'title': album.title,
                'description': album.description,
                'story': album.story,
                'photo_paths': album.photo_paths,
                'created_at': album.created_at,
                'cluster_method': album.cluster_method,
                'num_photos': album.num_photos
            }
            for album in albums
        ]
        
        output_path.write_text(json.dumps(albums_data, indent=2))
        return output_path
    
    def load_albums(self, input_path: Optional[Path] = None) -> List[Album]:
        """Load albums from JSON file."""
        if input_path is None:
            input_path = self.cfg.photo_root / "albums.json"
        
        if not input_path.exists():
            return []
        
        albums_data = json.loads(input_path.read_text())
        
        return [
            Album(
                id=data['id'],
                title=data['title'],
                description=data['description'],
                story=data.get('story', ''),
                photo_paths=data['photo_paths'],
                created_at=data['created_at'],
                cluster_method=data['cluster_method'],
                num_photos=data['num_photos']
            )
            for data in albums_data
        ]
