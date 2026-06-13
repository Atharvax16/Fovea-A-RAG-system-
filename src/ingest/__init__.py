"""Ingestion lanes for Fovea.

Each "lane" handles one source *type* (prose, tables, code, figures).
Step 1 builds the prose lane only. Keeping lanes separate is a deliberate
research discipline: if retrieval breaks later, we want the cause to be
obvious instead of tangled across formats.
"""
