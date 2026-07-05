import sqlite3
import urllib.request
import urllib.parse
import json
import redis
from Create_Claim import ExtractionEngine, app as modal_app
import os
from dotenv import load_dotenv

# uses dotenv for loading env variables
load_dotenv()

# I set up Redis here to act as a gatekeeper so I don't process the same URLs twice across different runs, also is a buffer so droplet doesn't get too much info 
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=6379,
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# cursor pagination from Europe PMC
def search_europe_pmc(query_string, max_results_per_page=50, max_pages=100):
    target_urls = []
    headers = {'User-Agent': 'AgenticResearcherBot/2.0'}
    encoded_query = urllib.parse.quote(query_string)

    # this asterisk is just the starting bookmark Europe PMC expects when you begin a new search
    cursor_mark = "*" 
    page_count = 0

    print(f"Searching: {query_string}")

    while page_count < max_pages:
        # injecting the cursor mark here so the API knows exactly which page of results to give me next
        api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&resultType=lite&pageSize={max_results_per_page}&cursorMark={urllib.parse.quote(cursor_mark)}"

        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = data.get('resultList', {}).get('result', [])

                # if there's nothing left, break out of the loop
                if not results:
                    break 

                new_urls_found = 0
                for item in results:
                    url = None
                    if pmcid := item.get('pmcid'):
                        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    elif pmid := item.get('pmid'):
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                    if url:
                        # checking my redis database here to see if I've already processed this paper before
                        if not redis_client.sismember("visited_urls", url):
                            target_urls.append(url)
                            redis_client.sadd("visited_urls", url)
                            new_urls_found += 1

                print(f"   📄 Page {page_count + 1}: Found {new_urls_found} NEW URLs (out of {len(results)} total on page)")

                # for the next batch of results
                next_cursor = data.get('nextCursorMark')
                
                # if the API stops giving me new bookmarks, it means I've hit the end of the line for this timeframe
                if not next_cursor or next_cursor == cursor_mark:
                    print("Reached the end of the results for this timeframe.")
                    break 

                cursor_mark = next_cursor
                page_count += 1

        except Exception as e:
            # catching exceptions so the whole pipeline doesn't crash if Europe PMC drops my connection
            print(f"⚠️ API Search failed on page {page_count + 1} for '{query_string}': {e}")
            break 

    return target_urls

def run_orchestrator(domain="immunology", batch_size=4):
    db_file = f"query_queue_{domain}.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # local sqlite database of time-chunked queries
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # target specific quarters of the year (this ensures I get as many sources as possible, because otherwise I get a maximum of a couple results from the PMC API)
    cursor.execute("SELECT COUNT(*) FROM search_queries")
    if cursor.fetchone()[0] == 0:
        print("🌱 Database is empty. Seeding with bulk timeframe API queries")
        bulk_queries = [
            ('KW:"Immunology" AND OPEN_ACCESS:Y AND FIRST_PDATE:[2024-01-01 TO 2024-03-31]',),
            ('KW:"Immunology" AND OPEN_ACCESS:Y AND FIRST_PDATE:[2024-04-01 TO 2024-06-30]',),
            ('KW:"Immunology" AND OPEN_ACCESS:Y AND FIRST_PDATE:[2024-07-01 TO 2024-09-30]',),
            ('KW:"Immunology" AND OPEN_ACCESS:Y AND FIRST_PDATE:[2024-10-01 TO 2024-12-31]',)
        ]
        cursor.executemany("INSERT INTO search_queries (query_text) VALUES (?)", bulk_queries)
        conn.commit()
    
    # next batch of timeframes
    cursor.execute("SELECT id, query_text FROM search_queries WHERE status = 'PENDING' LIMIT ?", (batch_size,))
    rows = cursor.fetchall()
    
    if not rows:
        print("Queue empty. All timeframes processed.")
        conn.close()
        return False 

    query_ids = [row[0] for row in rows]
    query_texts = [row[1] for row in rows]
    
    cursor.executemany("UPDATE search_queries SET status = 'PROCESSING' WHERE id = ?", [(qid,) for qid in query_ids])
    conn.commit()
    
    # sending the queries to the search function to get all the actual article URLs back
    print(f"{len(query_texts)} timeframes. Searching Europe PMC")
    
    all_target_urls = []
    for query in query_texts:
        all_target_urls.extend(search_europe_pmc(query))
    
    if not all_target_urls:
        print("No new URLs found in these timeframes")
        cursor.executemany("UPDATE search_queries SET status = 'COMPLETED' WHERE id = ?", [(qid,) for qid in query_ids])
        conn.commit()
        # returning True to keep the orchestrator running for the next batch
        return True 
    
    # it doesn't matter how many urls are found, we need to check how many are actually added to the database. Modal is what collects all the information (they have a maximum of 100 CPU containers, so it can scrape much faster than I can, not to mention they can do it while my laptop is off if I put the calling code on the droplet, which it is)
    print(f"Redis allowed {len(all_target_urls)} new URLs through. Modal starting")
    
    # Modal scrapes in parallel
    with modal_app.run():
        engine = ExtractionEngine()
        results = list(engine.process_and_push.map(
            all_target_urls, 
            # I just map all these back to the first query ID so I can track them in the logs
            [str(query_ids[0])] * len(all_target_urls) 
        ))

    print(f"Modal execution finished. Results added to QDrant Database")
    cursor.executemany("UPDATE search_queries SET status = 'COMPLETED' WHERE id = ?", [(qid,) for qid in query_ids])
    conn.commit()
    conn.close()
    return True
    
if __name__ == "__main__":
    import time
    print("Orchestrator starting")
    while True:
        try:
            has_more_data = run_orchestrator(domain="immunology", batch_size=1)
            if not has_more_data:
                # if there is not more data left in the queue, it shuts down so I stop paying money on CPUs
                print("Orchestrator shutting down cleanly.")
                break 
            
            # sleeping to allow droplet to process some of the buffer just in case
            print("Batch complete. 10 seconds before next batch begins")
            time.sleep(10)
        except Exception as e:
            # if it fails due to a time related issue, this is called
            print(f"Error: {e}")
            time.sleep(30)