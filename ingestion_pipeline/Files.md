##### consumer.py
This is what monitors the redis buffer and pulls from it to begin processing in the QDrant vector database

##### Create_Claim.py
The biomedical entity and claim extraction utility. Used during the ingestion phase to process raw academic text, identify critical biological relationships, and format them into structured claims or graph medoids before they are vectorized and injected into the database.

##### ingestion_worker.py
Uses Modal CPUs to pull biomedical research papers from the Europe PMC API. Uses API pagination to get more sources. Redis cache is used to ensure no duplicate processing occurs.