import sys
import argparse
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "scrapers"))

from ikman_used_gpus_scraper import (
    scrape_ikman_multiple_pages,
    clean_gpu_data as clean_ikman_data,
    build_train_ready_records
)
from md_used_gpus_scraper import (
    scrape_md_multiple_pages,
    clean_gpu_data as clean_md_data
)
from msk_used_gpus_scraper import (
    scrape_msk_multiple_pages,
    clean_gpu_data as clean_msk_data
)

from storage_utils import (
    utc_now_iso,
    attach_scrape_metadata,
    record_identity,
    write_json_records,
    load_json_records,
    dated_snapshot_path,
    merge_and_deduplicate_records
)

ALL_SCRAPED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned" / "all_scraped_data.json"

def process_site(source, scrape_func, clean_func, max_pages=None, is_ikman=False):
    print(f"\n{'='*40}")
    print(f" Running scraper for {source.upper()}")
    print(f"{'='*40}")
    
    # 1. Scrape
    if source == "ikman":
        raw_df = scrape_func(max_pages=max_pages) if max_pages else scrape_func()
    else:
        raw_df = scrape_func(max_pages=max_pages)
        
    if raw_df.empty:
        print(f"No data returned for {source}.")
        return [], []
        
    scraped_at_utc = utc_now_iso()
    raw_records = raw_df.to_dict(orient="records")
    raw_records = attach_scrape_metadata(raw_records, source=source, scraped_at_utc=scraped_at_utc)
    
    # 2. Clean
    cleaned_df = clean_func(pd.DataFrame(raw_records))
    if is_ikman:
        cleaned_records = build_train_ready_records(cleaned_df)
    else:
        cleaned_records = cleaned_df.to_dict(orient="records")
        
    cleaned_records = attach_scrape_metadata(cleaned_records, source=source, scraped_at_utc=scraped_at_utc)
    
    # Add 'Source' key to easily track where items came from
    for r in raw_records:
        r["Source"] = source
    for r in cleaned_records:
        r["Source"] = source
        
    # Save raw snapshot
    raw_snapshot = dated_snapshot_path(
        project_root=PROJECT_ROOT,
        dataset_kind="raw",
        source=source,
        stem=f"{source}_gpus_raw",
        scraped_at_utc=scraped_at_utc,
    )
    write_json_records(raw_snapshot, raw_records)
    print(f"[SUCCESS] Exported raw snapshot to {raw_snapshot.name}")
    
    # Save cleaned snapshot
    clean_snapshot = dated_snapshot_path(
        project_root=PROJECT_ROOT,
        dataset_kind="cleaned",
        source=source,
        stem=f"{source}_gpus_cleaned",
        scraped_at_utc=scraped_at_utc,
    )
    write_json_records(clean_snapshot, cleaned_records)
    print(f"[SUCCESS] Exported cleaned snapshot to {clean_snapshot.name}")
    
    # Update latest raw and cleaned for each site
    raw_all_output = PROJECT_ROOT / "data" / "raw" / f"{source}_gpus_raw_all.json"
    clean_all_output = PROJECT_ROOT / "data" / "cleaned" / f"{source}_gpus_cleaned_all.json"
    
    _, raw_dropped = merge_and_deduplicate_records(
        source=source,
        output_path=raw_all_output,
        new_records=raw_records,
        allow_legacy_ikman_fallback=is_ikman
    )
    
    _, clean_dropped = merge_and_deduplicate_records(
        source=source,
        output_path=clean_all_output,
        new_records=cleaned_records,
        allow_legacy_ikman_fallback=is_ikman
    )
    print(f"[SUCCESS] Merged {source} dataset. Dropped duplicates: {raw_dropped} raw, {clean_dropped} cleaned.")

    return raw_records, cleaned_records

def main():
    parser = argparse.ArgumentParser(description="Universal Pipeline to scrape, clean, and consolidate GPU data.")
    parser.add_argument("--ikman-pages", type=int, default=250, help="Max pages for Ikman scraper")
    parser.add_argument("--md-pages", type=int, default=None, help="Max pages for MD scraper")
    parser.add_argument("--msk-pages", type=int, default=None, help="Max pages for MSK scraper")
    args = parser.parse_args()

    print("\nStarting Universal GPU Scraper Pipeline...")
    
    # Load existing all_scraped_data.json if exists
    all_data = []
    if ALL_SCRAPED_DATA_PATH.exists():
        all_data = load_json_records(ALL_SCRAPED_DATA_PATH)
        print(f"Loaded {len(all_data)} records from {ALL_SCRAPED_DATA_PATH.name}")
        
    # Process each site
    sites = [
        ("ikman", scrape_ikman_multiple_pages, clean_ikman_data, True, args.ikman_pages),
        ("md", scrape_md_multiple_pages, clean_md_data, False, args.md_pages),
        ("msk", scrape_msk_multiple_pages, clean_msk_data, False, args.msk_pages)
    ]
    
    new_cleaned_records = []
    
    for source, scrape_func, clean_func, is_ikman, max_pages in sites:
        _, cleaned_records = process_site(source, scrape_func, clean_func, max_pages, is_ikman)
        new_cleaned_records.extend(cleaned_records)
        
    print(f"\n{'='*40}")
    print(" Consolidating All Data")
    print(f"{'='*40}")
    
    known_identities = set()
    for record in all_data:
        s = record.get("Source", "ikman")
        iden = record_identity(s, record, allow_legacy_ikman_fallback=True)
        if iden:
            known_identities.add(iden)
            
    added_count = 0
    for record in new_cleaned_records:
        s = record.get("Source", "ikman")
        iden = record_identity(s, record, allow_legacy_ikman_fallback=True)
        if iden and iden in known_identities:
            continue
        if iden:
            known_identities.add(iden)
        all_data.append(record)
        added_count += 1
        
    write_json_records(ALL_SCRAPED_DATA_PATH, all_data)
    print(f"\n[SUCCESS] Pipeline complete! Added {added_count} new unique records.")
    print(f"[SUCCESS] Total records in {ALL_SCRAPED_DATA_PATH.name}: {len(all_data)}")

if __name__ == "__main__":
    main()
