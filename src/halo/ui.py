"""Streamlit UI for Halo."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st


from halo.components.results_grid import render_results_grid
from halo.ingestion import PhotoIndexer
from halo.llm_utils import explain_match
from halo.search import PhotoSearcher
from halo.albums import AlbumGenerator


def _maybe_parse_geo_box(raw: str) -> Optional[Tuple[float, float, float, float]]:
    if not raw.strip():
        return None
    try:
        min_lat, max_lat, min_lon, max_lon = [float(part.strip()) for part in raw.split(",")]
    except ValueError:
        st.warning("Provide four comma-separated values for lat/lon bounds.")
        return None
    return (min_lat, max_lat, min_lon, max_lon)


def _describe_metadata(meta: dict) -> str:
    lines = []
    if meta.get("caption"):
        lines.append(f"**Caption:** {meta['caption']}")
    if meta.get("datetime_iso"):
        lines.append(f"**Taken:** {meta['datetime_iso']}")
    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if lat is not None and lon is not None:
        lines.append(f"**GPS:** {lat:.3f}, {lon:.3f}")
    if meta.get("year"):
        lines.append(f"**Year:** {meta['year']} (month {meta.get('month')})")
    if not lines:
        return "No metadata captured."
    return "\n".join(lines)


def run_app() -> None:
    st.set_page_config(page_title="Halo", layout="wide")
    st.title("Halo – Multimodal Photo Search")

    with st.sidebar:
        st.header("Ingestion")
        folder = st.text_input("Photo folder path", value=str(Path.cwd()))
        use_captions = st.checkbox("Generate BLIP captions", value=True)
        if st.button("Index Photos"):
            indexer = PhotoIndexer(enable_captions=use_captions)
            with st.spinner("Indexing photos..."):
                result = indexer.index_folder(folder)
            st.success(f"Indexed {result.indexed} photos · skipped {result.skipped}")

        st.header("Search Controls")
        top_k = st.slider("Results per query", min_value=3, max_value=48, value=12, step=3)
        expand_query_opt = st.checkbox("LLM query expansion", value=True)
        explain_opt = st.checkbox("Explain matches", value=False)
        use_react_grid = st.checkbox("Use React results grid (if built)", value=True)

        st.subheader("Filters")
        use_dates = st.checkbox("Filter by capture date", value=False)
        start_date = end_date = None
        if use_dates:
            today = dt.date.today()
            default_start = today - dt.timedelta(days=365 * 5)
            start_date = st.date_input("Start date", value=default_start)
            end_date = st.date_input("End date", value=today)

        use_geo = st.checkbox("Filter by GPS bounding box", value=False)
        geo_box = None
        if use_geo:
            geo_box = _maybe_parse_geo_box(
                st.text_input("min_lat,max_lat,min_lon,max_lon", value="")
            )

    searcher = PhotoSearcher()
    tabs = st.tabs(["Text Search", "Search by Example", "Albums"])

    with tabs[0]:
        query = st.text_input("Describe the vibe", "moody nighttime cityscapes")
        if st.button("Run text search") and query:
            with st.spinner("Retrieving matches..."):
                results = searcher.search_text(
                    query=query,
                    k=top_k,
                    expand=expand_query_opt,
                    start_date=start_date if use_dates else None,
                    end_date=end_date if use_dates else None,
                    geo_box=geo_box if use_geo else None,
                )
            _render_results(results, explain_opt, query, use_react_grid)

    with tabs[1]:
        uploaded = st.file_uploader("Upload a reference photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if st.button("Find similar") and uploaded:
            with st.spinner("Finding visually similar photos..."):
                results = searcher.search_by_image(
                    uploaded,
                    k=top_k,
                    start_date=start_date if use_dates else None,
                    end_date=end_date if use_dates else None,
                    geo_box=geo_box if use_geo else None,
                )
            _render_results(results, explain_opt, "reference image", use_react_grid)
    
    with tabs[2]:
        st.subheader("🎨 Automatic Album Generation")
        st.markdown("""
        Generate photo albums automatically using AI clustering:
        - **Visual**: Groups visually similar photos
        - **Temporal**: Groups photos by time periods
        - **Hybrid**: Combines visual similarity + time + location
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cluster_method = st.selectbox(
                "Clustering Method",
                ["visual", "temporal", "hybrid"],
                help="How to group photos into albums"
            )
        
        with col2:
            n_albums = st.slider("Target # of Albums", min_value=2, max_value=10, value=5)
        
        with col3:
            min_photos = st.slider("Min Photos/Album", min_value=2, max_value=10, value=3)
        
        if st.button("🎬 Generate Albums", type="primary"):
            generator = AlbumGenerator()
            
            with st.spinner(f"Generating albums using {cluster_method} clustering..."):
                albums = generator.generate_albums(
                    method=cluster_method,
                    n_albums=n_albums,
                    min_photos=min_photos
                )
            
            if not albums:
                st.warning("No albums generated. Try indexing more photos or adjusting parameters.")
            else:
                st.success(f" Generated {len(albums)} albums!")
                
                # Save albums
                save_path = generator.save_albums(albums)
                st.info(f" Albums saved to: `{save_path}`")
                
                # Display each album
                for album in albums:
                    with st.expander(f"📸 {album.title} ({album.num_photos} photos)", expanded=True):
                        st.markdown(f"**{album.description}**")
                        if album.story:
                            st.info(album.story)
                        st.caption(f"Method: {album.cluster_method} | Created: {album.created_at[:19]}")
                        
                        # Display photos in grid
                        cols = st.columns(min(4, album.num_photos))
                        for idx, path in enumerate(album.photo_paths[:12]):  # Show max 12
                            with cols[idx % 4]:
                                try:
                                    st.image(path, use_container_width=True)
                                    st.caption(Path(path).name)
                                except Exception:
                                    st.error(f"Cannot load: {Path(path).name}")
                        
                        if album.num_photos > 12:
                            st.info(f"... and {album.num_photos - 12} more photos")
        
        # Load existing albums
        st.divider()
        st.subheader(" Saved Albums")
        
        if st.button(" Load Saved Albums"):
            generator = AlbumGenerator()
            saved_albums = generator.load_albums()
            
            if not saved_albums:
                st.info("No saved albums found. Generate some first!")
            else:
                for album in saved_albums:
                    with st.expander(f" {album.title} ({album.num_photos} photos)"):
                        st.markdown(f"**{album.description}**")
                        if album.story:
                            st.info(album.story)
                        st.caption(f"Method: {album.cluster_method} | Created: {album.created_at[:19]}")


def _render_results(results, explain_opt: bool, query: str, use_react_grid: bool) -> None:
    if not results:
        st.warning("No matches found. Consider indexing more photos or relaxing filters.")
        return

    rendered_with_react = False
    if use_react_grid:
        selection = render_results_grid(results, columns=3, thumb_height=240, show_score=True)
        if selection is not None:
            rendered_with_react = True
            if selection:
                st.info(f"Selected {len(selection)} items (via React grid).")
            if explain_opt:
                st.divider()
                st.subheader("Why these matched")
                for hit in results:
                    reason = explain_match(query, hit.metadata)
                    st.markdown(f"**{Path(hit.path).name}**")
                    st.caption(_describe_metadata(hit.metadata))
                    st.info(reason)

    if rendered_with_react:
        return

    cols = st.columns(3)
    for idx, hit in enumerate(results):
        with cols[idx % 3]:
            st.image(hit.path, caption=f"Score {hit.score:.3f}")
            st.markdown(_describe_metadata(hit.metadata))
            if explain_opt:
                reason = explain_match(query, hit.metadata)
                st.info(reason)


if __name__ == "__main__":
    run_app()
