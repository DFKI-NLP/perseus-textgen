import requests
import argparse
import pandas as pd
import json
from tqdm import tqdm

def call_llm(input_str, params, api_url):
    body = {'inputs': input_str, 'parameters': params}
    x = requests.post(api_url, json = body)
    x.raise_for_status()  # raises exception when not a 2xx response
    if x.status_code != 204:
        return x.json()
    

def batch_process(args):
    
    df = pd.read_csv(args.input_file, dtype=str)
    if args.input_column not in df.columns:
        raise ValueError(f"Input csv must have an {args.input_column} column")
    # Create a column if it doesn't exist already
    df[args.output_column] = df.get(args.output_column, None)

    with open(args.parameters, 'r') as f:
        params = json.load(f)

    processed = 0
    try:
        with tqdm(total=df.shape[0], ascii=True, desc=f"processing {args.input_file}", unit="rows", position=0, leave = True) as pbar:
            for i, row in enumerate(df.itertuples()):
                #Skip rows which already have an output
                if not pd.isnull(row.__getattribute__(args.output_column)):
                    pbar.update()
                    continue 
                input_str = str(row.__getattribute__(args.input_column))
                resp = call_llm(input_str, params, args.api)
                # print(resp)
                df.loc[row.Index, args.output_column] = resp['generated_text']
                df.to_csv(args.output_file, index=False)
                processed += 1
                pbar.update()
    except KeyboardInterrupt:
        pbar.close()
    
    timer = pbar.format_dict["elapsed"]
    print(f"Processed {processed} rows in {timer:.4} seconds. ({processed/timer:.4} rows/second)")
    print(f"Saved results into {args.output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="batch_processor",
                                 description="Process a csv data in api")
    
    ap.add_argument("--api", type=str, help= "API URL")
    ap.add_argument("-i", "--input_file", default="input.csv", type=str, help = "input file path; defaults to 'input.csv'")
    ap.add_argument("-o", "--output_file", default="output.csv", type=str, help = "output file path; defaults to 'output.csv'")
    ap.add_argument("-ic", "--input_column", default="input", type=str, help = "name of the input data column; defaults to 'input'")
    ap.add_argument("-oc", "--output_column", default="output", type=str, help = "name of the output data column; defaults to 'output'")

    # #params
    # ap.add_argument("--max_new_tokens", type=int, default=100, help = "maximum model return size in tokens; defaults to 100")
    # ap.add_argument("--temperature", type=float, default=1., help = "model output temperature between 0 and 2, defines how random is output; defaults to 1")
    ap.add_argument("-p", "--parameters", default="parameters.json", type=str, help = "parameters for the model; defaults to 'parameters.json'" )
    args = ap.parse_args()

    try:
        batch_process(args)
    except KeyboardInterrupt:
        pass