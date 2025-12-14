# Halo

Search by meaning, not metadata

Halo is a multimodal Retrieval-Augmented Generation (RAG) system for semantic photo search with **automatic album generation**. Users can search their photo library by describing vibes and moods, or let AI automatically organize photos into intelligent albums.

## Features

### Semantic Photo Search
- **CLIP-based embeddings** for images, text queries, and BLIP-generated captions
- **Optional BLIP captioning** and hybrid scoring for better vibe/mood recall
- **LLM-powered query expansion** plus optional explanation mode for search hits
- **Search-by-example**: upload a reference photo and retrieve visually similar shots
- **Metadata filters** (date range + GPS bounding box) to narrow the search space
- **Local-only vector store** (ChromaDB) for privacy-preserving retrieval

### Automatic Album Generation
- **AI-powered clustering**: Automatically organizes photos into meaningful albums
- **Three clustering methods**:
  - **Visual**: Groups photos by appearance similarity using K-means on CLIP embeddings
  - **Temporal**: Groups photos by time periods based on capture dates
  - **Hybrid**: Combines visual similarity + timestamps + GPS location for intelligent grouping
- **LLM-generated titles**: Creative album names and descriptions powered by Gemini/GPT
- **Customizable parameters**: Adjust target number of albums and minimum photos per album
- **Persistent storage**: Albums save to JSON and reload on app restart

---

## Getting Started

### Installation

```bash
# Clone and navigate to project
git clone https://github.com/PeterMcMaster/Halo.git
cd Halo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode so `halo` is on your Python path
pip install -e .
```

### Configuration

Create a `.env` file (or copy `.env.example`) and set your preferred LLM provider:

**For Gemini (Free tier available):**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-google-ai-studio-key
GEMINI_MODEL=gemini-1.5-flash
```

**For OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

**Get API keys:**
- Gemini: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys

### Run the Application

```bash
streamlit run src/halo/ui.py
```

The app will open in your browser at `http://localhost:8501`

---

## How to Use

### 1. Index Your Photos

1. Go to the **"Index Photos"** section in the sidebar
2. Enter the path to your photo folder
3. Toggle **"Generate BLIP captions"** (recommended for better search)
4. Click **"Index Photos"**
5. Wait for completion (~1-2 seconds per photo with BLIP)

### 2. Search for Photos

**Text Search:**
1. Navigate to **"Text Search"** tab
2. Enter a description: "moody nighttime cityscapes", "cozy indoor warm lighting"
3. Toggle **"LLM query expansion"** for richer descriptions
4. Optional: Apply date range or GPS filters
5. Click **"Run text search"**

**Search by Example:**
1. Navigate to **"Search by Example"** tab
2. Upload a reference photo
3. Click **"Find similar"**
4. View visually similar images with similarity scores

### 3. Generate Albums

1. Go to the **"Albums"** tab
2. Choose **Clustering Method**:
   - **Visual**: Groups similar-looking photos (beaches, mountains, portraits)
   - **Temporal**: Groups by time periods (trips, events, seasons)
   - **Hybrid**: Smart grouping using visual + temporal + location data (recommended)
3. Set **Target # of Albums** (2-10)
4. Set **Min Photos/Album** (2-10)
5. Click **"🎬 Generate Albums"**
6. Browse generated albums with AI-generated titles

**Album Features:**
- Each album has a creative title and description
- Photos displayed in grid layout
- Albums automatically saved to `photos/albums.json`
- Load previously generated albums with "Load Saved Albums"

---

### React Components 

The Streamlit UI can load a custom React results grid. Build once and Streamlit will serve the static assets:

```bash
cd react_components/result-grid
npm install
npm run build
```

For live development, run the dev server and point Streamlit to it:
```bash
npm run dev  # at react_components/result-grid (defaults to http://localhost:5173)
export RESULT_GRID_DEV_URL=http://localhost:5173
streamlit run src/halo/ui.py
```

If `RESULT_GRID_DEV_URL` is unset, Streamlit will load the built bundle from `react_components/result-grid/dist`.

---

## Album Generation: Technical Details

### How It Works

**1. Feature Extraction**
- Each photo is represented as a 512-dimensional CLIP embedding vector
- Optional temporal features from EXIF timestamps
- Optional spatial features from GPS coordinates

**2. Clustering Algorithms**

**Visual Clustering:**
- Uses K-means algorithm on CLIP embeddings
- Groups photos with similar visual content
- Ideal for collections with distinct visual themes

**Temporal Clustering:**
- Extracts capture timestamps from EXIF metadata
- Groups photos taken within similar time periods
- Simple time-based binning approach
- Ideal for organizing by trips and events

**Hybrid Clustering:**
- Combines multiple features into unified feature space:
  - CLIP embeddings (70% weight)
  - Normalized timestamp (15% weight)
  - GPS coordinates (15% weight)
- Uses K-means on standardized combined features
- Most intelligent method for general photo libraries

**3. LLM-Powered Naming**
- Analyzes cluster metadata (dates, locations, photo count)
- Generates creative album titles (max 6 words)
- Creates descriptive 2-3 sentence summaries
- Falls back to generic names if LLM unavailable

**4. Persistence**
- Albums saved as JSON to `photos/albums.json`
- Includes all metadata: titles, descriptions, photo paths
- Reloadable across sessions

### Algorithm Comparison

| Method | Best For | Algorithm | Features Used |
|--------|----------|-----------|---------------|
| **Visual** | Similar-looking photos | K-means | CLIP embeddings only |
| **Temporal** | Events & trips | Time binning | Timestamps only |
| **Hybrid** | Smart organization | K-means | CLIP + Time + GPS |

**Hybrid clustering is recommended** as it produces the most meaningful albums by considering both visual content and contextual metadata.

---

## Testing with Sample Data

If you need quick test photos:

```bash
source venv/bin/activate

# Download 40 sample photos from Picsum
python scripts/download_sample_photos.py --clean --limit 40

# Run smoke test
python scripts/smoke_test.py --folder photos/sample_dataset
```

**Script Options:**
- `--limit`: Number of images (1-100)
- `--width/--height`: Image dimensions
- `--clean`: Remove old photos first
- `--no-expand`: Skip LLM query expansion

Results written to `smoke_results.json`

---

## Evaluation

### Performance Metrics

See `notebooks/evaluation.ipynb` for:
- Ablation studies (CLIP-only vs hybrid scoring)
- Latency measurements across collection sizes
- UMAP visualization of embedding space
- Qualitative assessments for reports

Launch via Jupyter/VS Code after activating the virtual environment:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

### Expected Album Generation Results

```
Collection Size | Albums Generated | Processing Time
----------------|------------------|----------------
40 photos       | 3-5 albums      | ~10 seconds
100 photos      | 5-10 albums     | ~20 seconds
500 photos      | 15-25 albums    | ~60 seconds
```

*Processing time includes clustering and LLM name generation*

---

## Dependencies

### Core Libraries
- `torch`, `torchvision` - PyTorch for neural networks
- `transformers` - Hugging Face models (CLIP, BLIP)
- `chromadb` - Vector database for embeddings
- `pillow` - Image processing
- `streamlit` - Web UI framework

### LLM Providers
- `openai` - OpenAI API (GPT models)
- `google-generativeai` - Google Gemini API

### Data Science
- `numpy` - Numerical computing
- `scikit-learn` - Clustering algorithms (K-means, StandardScaler)
- `umap-learn` - Dimensionality reduction for visualization
- `matplotlib` - Plotting and visualization

### Utilities
- `python-dotenv` - Environment variable management
- `exifread` - Extract EXIF metadata from photos
- `tqdm` - Progress bars

See `requirements.txt` for complete dependency list with versions.