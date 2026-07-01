This is where I'll document my changes made so you/I can better see what changes along the way (most recent first)

2026 July 1 
- Connected QDrant DB from ChromaDB
    - For some reason, QDrant DB doesn't return any vectors (always very low similarity scores) ----->>> Lowered require confidence values, Chroma DB uses L2 Loss (always very high) while QDrant uses Cosine Similarity (Which is much harder to get high)
- Artificially modified user personas to make them more accurate

2026 June 30 
- Switched to QDrant Vector Database from ChromaDB 
    - Haven't yet completed making my database, so I wanted to switch as early as possible before building out the complete database and then needing to swap it out afterwards. 
    - Made changes to the node data stored to make it easy to query 
    - QDrant is faster than ChromaDB (Rust vs Python), better at querying, stored on disk with memory mapping so I don't need to spend a lot of money on RAM 
- Better env setting up so I don't have to modify code while its local and when its on the droplet