# batch processor
Tool for processing CSV files with llm. 
### Usage
Help:
```
$ python batch_processor.py -h
usage: batch_processor [-h] [--api API] [-i INPUT_FILE] [-o OUTPUT_FILE] [-ic INPUT_COLUMN] [-oc OUTPUT_COLUMN] [--max_new_tokens MAX_NEW_TOKENS] [--temperature TEMPERATURE]

Process a csv data in api

options:
  -h, --help            show this help message and exit
  --api API             API URL;
  -i INPUT_FILE, --input_file INPUT_FILE
                        input file path; defaults to 'input.csv'
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file path; defaults to 'output.csv'
  -ic INPUT_COLUMN, --input_column INPUT_COLUMN
                        name of the input data column; defaults to 'input'
  -oc OUTPUT_COLUMN, --output_column OUTPUT_COLUMN
                        name of the output data column; defaults to 'output'
  -p PARAMS, --parameters 
                        name of the json with parameters; defaults to 'parameters.json'
```
Each entry in an INPUT_COLUMN of INPUT_FILE will be sent to API.

OUTPUT_FILE is a copy of an input file with a new column OUTPUT_COLUMN (default = 'output') which contains API responses. 

Make sure to set MAX_NEW_TOKENS as needed. 100 Tokens is only 1 paragraph maximum.

If input file already has an 'output' column - only rows with empty output will be processed. Processing may be interrupted(via CTRL+C in UNIX) at any point and continued later by using an output file as an input. 

Example usage
```
$ python3.11 batch_processor.py -i output.csv
processing output.csv: 100%|########################################| 60/60 [00:00<00:00, 673.91rows/s]
Processed 60 rows in 0.08912 seconds. (673.3 rows/second)
Saved results into output.csv
```
