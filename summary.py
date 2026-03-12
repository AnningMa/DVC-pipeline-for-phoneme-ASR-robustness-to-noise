import os
import glob
import pandas as pd

def main():
    metrics_dir = 'data/metrics'
    all_files = glob.glob(os.path.join(metrics_dir, '*_evaluation.csv'))
    
    if not all_files:
        print(f"No files found in {metrics_dir}")
        return

    data_frames = []
    
    for file in all_files:
        #extract language code from filename
        basename = os.path.basename(file)
        lang = basename.split('_')[0]
        
        df = pd.read_csv(file)
        df['Language'] = lang
        data_frames.append(df)
  
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    combined_df['snr_db'] = pd.to_numeric(combined_df['snr_db'])

    print("===Table 1: Language Level Comparison===")
    table1 = combined_df.pivot(index='Language', columns='snr_db', values='per_percentage')
    table1 = table1[sorted(table1.columns, reverse=True)]
    print(table1.to_markdown())
    print("\n")

    print("=== Table 2: Noise Level Comparison===")
    table2 = combined_df.pivot(index='snr_db', columns='Language', values='per_percentage')
    table2 = table2.sort_index(ascending=False)
    print(table2.to_markdown())
    print("\n")
    
 
    with open('data/metrics/global_summary.txt', 'w') as f:
        f.write("=== Table 1 ===\n")
        f.write(table1.to_markdown())
        f.write("\n\n=== Table 2 ===\n")
        f.write(table2.to_markdown())

if __name__ == "__main__":
    main()