Stage 0 — Prep & dependencies (run once)

# Project setup: create a new Python module `relationship_mapping.py` and add required dependencies.
# Write code that:
# - Imports: numpy, pandas, sqlite3, sklearn.metrics.pairwise.cosine_similarity, sklearn.cluster.DBSCAN, networkx, json, time, typing.
# - Provides a `requirements.txt` snippet:
#   numpy
#   pandas
#   scikit-learn
#   networkx
#   matplotlib
# - Adds a short README header in the file explaining "All processing is on-device; face labeling requires explicit user consent".
# - Create helper functions file `utils_io.py` with:
#   - `read_db_rows(db_path: str) -> List[Dict]` : reads `photos` table rows (id, path, timestamp, faces_json).
#   - `save_json(obj, path)` and `load_json(path)`.
# - Print sample CLI usage at bottom of the file.


Important notes to include in generated code:

Use float32 for embeddings; compress to base64 when saving JSON.

Normalize embeddings to unit vectors for cosine similarity.


Stage 2 — Identity clustering (face → person clusters)

# Add clustering code to relationship_mapping.py:
# - Implement function `cluster_face_embeddings(embeddings: np.ndarray, eps: float=0.5, min_samples: int=3) -> np.ndarray`
#   - Use DBSCAN with metric='cosine' or apply cosine distance (1 - cosine_similarity).
#   - Return cluster labels array aligned to embeddings input.
# - Implement `build_face_index(face_files_dir: str) -> Tuple[np.ndarray, List[Dict]]`:
#   - Loads all `{photo_id}_faces.json`, concatenates all embeddings into an array E (N x D) and a parallel list meta[] with {"photo_id", "face_index", "bbox"}.
#   - Runs `cluster_face_embeddings` on E.
#   - Returns E and meta.
# - Implement `save_clusters_to_db(db_path: str, meta: List[Dict], labels: np.ndarray)`:
#   - Create new tables:
#     CREATE TABLE person_clusters (person_id TEXT PRIMARY KEY, cluster_label INTEGER, repr_embedding BLOB);
#     CREATE TABLE photo_faces (photo_id TEXT, face_index INTEGER, person_id TEXT, embedding BLOB, PRIMARY KEY(photo_id, face_index));
#   - person_id = "person_{cluster_label}".
#   - Compute repr_embedding = mean of cluster embeddings (normalized).
#   - Insert rows into photo_faces and person_clusters.
# - Provide a CLI command: `--cluster-faces` which runs the full pipeline and prints number of clusters found.
# - Add recommended hyperparameter guidance: try eps in [0.35,0.6] for cosine.


Expected artifact: DB tables photo_faces and person_clusters populated.
Test: --cluster-faces on sample photos produces labeled person_0, person_1, etc.


Stage 3 — Build co-occurrence graph (people ↔ people)

# Extend relationship_mapping.py with graph construction functions:
# - Implement `build_cooccurrence_graph(db_path: str) -> networkx.Graph`:
#   - Query `photo_faces` table grouped by photo_id to get list of person_ids per photo.
#   - For each photo, for every unordered pair (p,q) increment edge weight w(p,q) by 1.
#   - Node attributes: person_id, repr_embedding (vector), n_photos (count of photos person appears in).
#   - Edge attributes: weight (co-occurrence count), last_seen_timestamp (max of photo timestamps where pair appears).
# - Implement `save_graph(graph: nx.Graph, path: str)` (write as GraphML or JSON).
# - Implement `load_graph(path: str)`.
# - Provide CLI flag `--build-graph` that constructs and saves `person_graph.graphml`.
# - Include an example visualization function `plot_graph(graph)` using networkx + matplotlib which sizes nodes by n_photos and colors by community.


Design choices to call out: normalize edge weight by min(n_photos(p), n_photos(q)) to avoid bias toward very active people.


Stage 4 — Relationship inference heuristics & community detection

# Add relationship inference code:
# - Implement `detect_communities(graph: nx.Graph) -> Dict[person_id, community_id]`:
#   - Use Louvain or networkx.community.greedy_modularity_communities (choose one).
#   - Return mapping person_id -> community_id.
# - Implement `infer_relationship_type(graph: nx.Graph, person_id: str, neighbor_stats: Dict) -> str`:
#   - Heuristic rules:
#     - If person co-occurs with target user frequently across many different event timestamps and co-occurrence fraction > 0.6 => label "family".
#     - If co-occurs mainly in photos with object tags indicating "office, laptop, id_badge" and co-occurrence fraction in work events > 0.5 => label "coworker".
#     - If community size small (2-4) and high mutual co-occurrence => "close_friend".
#     - Fallback => "acquaintance".
# - Implement `auto_label_relationships(graph: nx.Graph, db_path: str) -> None`:
#   - For each person node, compute heuristics and write a `relationships` table:
#     CREATE TABLE relationships (person_id TEXT PRIMARY KEY, inferred_type TEXT, confidence REAL).
# - Provide CLI `--infer-relationships` and print top 10 inferred relationships as sample output.
# - Add comments: these heuristics should be treated as suggestions and always allow user override.


Notes: include thresholds (co-occurrence fraction > 0.6, min co-occurrence count >= 5). Use event clustering (next stage) for context.


Stage 5 — Event / temporal context to support inference


# Implement event clustering and temporal weighting:
# - Add function `group_photos_into_events(db_path: str, window_hours: int=48) -> Dict[event_id, List[photo_id]]`:
#   - Use timestamps to cluster photos into events (photos within `window_hours` belong to the same event).
# - Modify co-occurrence graph construction to track in how many distinct events two people co-occur (edge attribute events_count).
# - Update heuristics in Stage 4 to use events_count / person_events_count as a signal (more robust than raw photo counts).
# - Add CLI `--build-events` and `--rebuild-graph-with-events`.


Why: families co-occur across many distinct events; coworkers often co-occur in fewer event types (meetings).


Stage 6 — User labeling flow (UI/CLI) & supervised refinement

# Add user-in-the-loop labeling and persistence:
# - Implement CLI commands:
#   --list-clusters  # show person_id, sample photo paths, cluster size
#   --label-person <person_id> "<label>"  # e.g., label "person_3" as "Alice"
#   --create-group <group_name> <person_id1> <person_id2> ...
#   --list-groups
# - Persist user labels in DB table:
#   CREATE TABLE person_labels (person_id TEXT PRIMARY KEY, label TEXT, labeled_at INTEGER);
#   CREATE TABLE groups (group_name TEXT PRIMARY KEY, members TEXT JSON, created_at INTEGER);
# - Implement a function `propagate_label_by_similarity(person_id, threshold=0.85)`:
#   - Finds other person_clusters whose repr_embedding cosine similarity >= threshold and offers to apply same label (prompt).
# - Implement `train_small_relationship_classifier(db_path: str)`:
#   - Optional: if user labels >= N (e.g., 30), train a simple logistic/regression model using features (cooccurrence_count, events_fraction, avg_edge_weight, embedding_similarity to user-labeled cluster prototypes).
#   - Save model to disk for local predictions.
# - CLI: `--train-relationship-model` (local, on-device).
# - Emphasize privacy: save only model weights locally; do not upload.


UX notes: require explicit confirmation before auto-propagation. Provide undo.


Stage 7 — Integrate groups into search & query engine

# Modify final_photo_search.py (or search backend) to accept group filters:
# - Add support for flags:
#   --group "family"
#   --group "coworkers"
# - Implement search flow:
#   - Resolve group -> list of person_ids (from groups table or inferred relationships table).
#   - For multi-person queries, find intersection or co-occurrence photos where all person_ids appear.
#   - Combine with CLIP/text similarity and time filter: e.g.,
#     score = w_text * text_sim + w_people * people_presence_score + w_time * time_score
# - Provide example CLI:
#   python final_photo_search.py --group "family" --search "beach" --limit 20
# - Add test: run a search for a labeled group and confirm returned photos include group members.


Ranking detail: people_presence_score = fraction of requested persons present in photo (0..1) or 1 if all present.


Stage 8 — Visualization & debugging tools

# Add developer utilities:
# - `visualize_person_samples(person_id, k=6)` : shows k sample images with bounding boxes and highlights for that person.
# - `plot_relationship_graph(person_id)` : plots ego-network (neighbors up to depth=2) with edge weights annotated.
# - `export_relationship_summary_csv(path)` : writes CSV with columns (person_id, label, inferred_type, n_photos, top_cooccurring_persons).
# - CLI flags:
#   --visualize-person person_3
#   --export-relationships out.csv


Goal: makes it easy to present to judges and debug mis-classifications.


Stage 9 — Evaluation & metrics
# Add evaluation utilities to measure quality and tune hyperparams:
# - `compute_clustering_stats(labels, true_labels=None)` returns: n_clusters, avg_cluster_size, silhouette_score (if true_labels not provided), average_intra_cluster_similarity, average_inter_cluster_similarity.
# - `evaluate_relationship_inference(db_path: str, gold_file: Optional[str])`:
#   - If gold labels exist, compute precision/recall/F1 for inferred types (family/coworker/etc).
#   - Otherwise compute proxy metrics: agreement with user labels, consistency over time.
# - Provide a short notebook or script `evaluate_relationships.py` to run these metrics and print recommended thresholds.


# - Provide a short notebook or script `evaluate_relationships.py` to run these metrics and print recommended thresholds.

Tuning tips: vary DBSCAN eps, min_samples, event window sizes and keep a simple config file.

Stage 10 — Mobile / On-device considerations & optimizations

# Produce a mobile-optimized variant summary and code hints:
# - Replace heavy in-memory arrays with streaming DB reads to avoid OOM.
# - Use quantized float16 embeddings stored as BLOBs; load person repr embeddings only when needed.
# - Run expensive clustering/graph building as a background job when device is idle/charging.
# - Provide Kotlin/Java pseudo-API signatures to integrate:
#   - fun getPersonSamples(personId: String): List<String>
#   - fun queryPhotosByGroup(groupName: String, queryEmbedding: FloatArray, topK: Int) : List<PhotoResult>
# - Suggest using incremental clustering: when a new embedding arrives, assign to nearest cluster if cosine_sim > 0.85, else buffer for periodic recluster.
# - Add note: require explicit opt-in for face label persistence and group sync (if any).
Deliverable: short mobile integration README appended to relationship_mapping.py.


Stage 11 — Advanced / optional ML improvements
# Add optional advanced features prompts for Copilot (only if time permits):
# - Semi-supervised label propagation: implement Label Spreading on the person graph using embedding similarity + co-occurrence edges.
# - Temporal embedding: include time features per co-occurrence edge to differentiate "always together" vs "one-off".
# - Supervised classifier: train a small tree-based classifier (LightGBM) on features (events_fraction, avg_object_context_score, embedding_similarity_to_user, cluster_size) to predict relationship type.
# - Provide calibration & confidence outputs; persist confidence in relationships table.
# - Add function `explain_relationship(person_id)` which outputs which signals led to the inference (cooccurrence_count, events_fraction, object-context).


Stage 12 — Final integration checklist & tests
# Provide a final checklist function `run_relationship_pipeline_all(db_path, photos_dir, out_dir)` that executes:
# 1) index faces -> out_dir faces
# 2) cluster embeddings -> update DB
# 3) build events -> rebuild graph
# 4) infer relationships -> write relationships table
# 5) start a simple Flask local web UI for labeling & viewing (optional)
# - Also include unit/integration test commands:
#   - pytest tests/test_relationship_mapping.py
# - Include sample small dataset (6 people, 100 photos) for CI.


Final privacy & UX guidance (include in each prompt)

Add this paragraph to the top of every prompt / generated file (so Copilot includes it):
# PRIVACY & USER CONSENT: This pipeline processes biometric face data and must run entirely on-device by default.
# - Require explicit user opt-in before enabling face clustering or labeling.
# - Store labels & clusters locally only; encrypt DB at rest if possible.
# - Provide UI to delete a person's data and undo automatic groupings.


Short example CLI usage (to include with generated code)


# Build face index and clusters
python relationship_mapping.py --index-faces --out ./face_index

# Cluster faces into persons
python relationship_mapping.py --cluster-faces --face-index ./face_index

# Build event clusters & cooccurrence graph
python relationship_mapping.py --build-events --build-graph

# Infer relationships
python relationship_mapping.py --infer-relationships

# List inferred relationships
python relationship_mapping.py --list-relationships

# Label a person manually
python relationship_mapping.py --label-person person_3 "Alice"
