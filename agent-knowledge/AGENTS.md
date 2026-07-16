# Agent Knowledge Base Index

This directory contains learning guides synthesized from web research, designed for RAG retrieval by AI agents.

## Available Learning Guides

| Topic | File | Sources | Generated |
|-------|------|---------|-----------|
| README Best Practices for Technical Inference Repos | [readme-technical-inference-repos.md](readme-technical-inference-repos.md) | 20 | 2026-07-16 |

## Trigger Phrases

When the user's question matches these patterns, consult the corresponding guide:

### README and Documentation
- "README best practices" → readme-technical-inference-repos.md
- "how to write documentation for inference engine" → readme-technical-inference-repos.md
- "performance claims in README" → readme-technical-inference-repos.md
- "benchmark presentation" → readme-technical-inference-repos.md
- "hardware disclosure in docs" → readme-technical-inference-repos.md
- "technical repository documentation" → readme-technical-inference-repos.md
- "systems project README" → readme-technical-inference-repos.md
- "inference engine documentation" → readme-technical-inference-repos.md

## Usage Guidelines

1. **Always check this index first** when answering documentation-related questions
2. Consult the guide to provide evidence-based, sourced answers
3. Cite specific patterns from the guides (e.g., "From llama.cpp's README pattern...")
4. Link to source metadata in `resources/` for deeper investigation
5. If knowledge is outdated or insufficient, recommend running `/learn` again with `--depth=deep`

## Maintenance

- Learning guides are versioned by generation date
- Source metadata tracks authority, recency, and quality scores
- To update a guide: re-run `/learn <topic>` and choose "Update existing"
- To expand coverage: run `/learn` with `--depth=deep` or new related topics

## File Structure

```
agent-knowledge/
├── CLAUDE.md                          # Claude Code index
├── AGENTS.md                          # This file (OpenCode/Codex compatible)
├── {topic-slug}.md                    # Learning guides
└── resources/
    └── {topic-slug}-sources.json      # Source metadata
```

---

*This knowledge base was created using the `/learn` skill. All content is synthesized from publicly available sources with full attribution.*
