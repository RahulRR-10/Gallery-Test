🟢 Phase 2 Roadmap — Copilot Prompts
Stage 1: Temporal Intelligence

Prompt for Copilot:

# Add a new section to README called "Phase 2 Roadmap".
# First, describe "Temporal Intelligence":
# - Explain that EXIF metadata (timestamps) is used to filter queries.
# - Give examples: "photos from last Christmas", "pictures from my college years".
# - Mention implementation details: parse EXIF timestamps, map human time expressions to ranges using dateparser.
# - Show usage example in CLI (e.g., --search "photos from 2020").


Stage 2: Face Recognition + Relationship Mapping

Prompt for Copilot:

# Extend the Phase 2 Roadmap section with "Face Recognition + Relationship Mapping".
# - Explain that face embeddings (MobileFaceNet/ArcFace) will be used to identify and cluster people across photos.
# - Mention that clusters can be labeled by the user (Mom, Sarah, Coworkers).
# - Show how co-occurrence graphs help map relationships (e.g., family, colleagues).
# - Provide usage examples: 
#   --search "photos with Sarah"
#   --search "pictures with my coworkers"


Stage 3: Activity Recognition

Prompt for Copilot:

# Extend the Phase 2 Roadmap section with "Activity Recognition".
# - Explain that activities are detected beyond objects.
# - Start with rule-based detection using YOLO tags (e.g., cake + candles = birthday, backpack + mountain = hiking).
# - Later allow small action recognition models.
# - Provide examples of queries:
#   --search "cooking photos"
#   --search "hiking with friends"
#   --search "celebration photos"


Stage 4: Mood & Emotion Detection

Prompt for Copilot:

# Extend the Phase 2 Roadmap section with "Mood & Emotion Detection".
# - Explain that facial expression classifiers (smile, neutral, sad) will be used on cropped faces.
# - Mention fusion with scene context (smiles + balloons = celebration).
# - Provide examples of queries:
#   --search "happy moments"
#   --search "sad memories"
#   --search "celebration photos"

Stage 5: Final Summary

Prompt for Copilot:

# Add a closing note to the Phase 2 Roadmap section:
# - Summarize that Phase 2 expands search from objects/semantics to human memory style (time, people, activities, emotions).
# - Emphasize privacy-first: all features still run 100% on-device.
# - Position it as unique vs Apple/Google: more contextual, but private.
