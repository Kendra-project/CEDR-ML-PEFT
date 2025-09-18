import csv
import argparse
import statistics

def generate_argparser():
    parser = argparse.ArgumentParser(description="JSON profiling modifier - Gives API costs from a CEDR timing log")

    parser.add_argument("-p", "--profiling-results", help="The timing log from CEDR that gives the profiling results for each node")

    return parser

parser = generate_argparser()
args = parser.parse_args()


costs={}
costs['FFT']={}
costs['FFT']['cpu']=[]
costs['FFT']['fft']=[]
costs['FFT']['gpu']=[]
costs['GEMM']={}
costs['GEMM']['cpu']=[]
costs['GEMM']['gemm']=[]
costs['GEMM']['gpu']=[]
costs['ZIP']={}
costs['ZIP']['cpu']=[]
costs['ZIP']['zip']=[]
costs['ZIP']['gpu']=[]

with open(args.profiling_results) as app_logs:
    for line in app_logs:
        #print(line.split(', ')[-1].split(': ')[1].split('\n')[0])
        if 'DASH_FFT' in line:
            if 'cpu' in line:
                costs['FFT']['cpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            elif 'gpu' in line:
                costs['FFT']['gpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            else:
                costs['FFT']['fft'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
        elif 'DASH_GEMM' in line:
            if 'cpu' in line:
                costs['GEMM']['cpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            elif 'gpu' in line:
                costs['GEMM']['gpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            else:
                costs['GEMM']['gemm'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
        elif 'DASH_ZIP' in line:
            if 'cpu' in line:
                costs['ZIP']['cpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            elif 'gpu' in line:
                costs['ZIP']['gpu'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))
            else:
                costs['ZIP']['zip'].append(int(line.split(', ')[-1].split(': ')[1].split('\n')[0]))


for key,value in costs.items():
    for key1,value1 in value.items():
        print(key, end='\t')
        print(key1, end='\t')
        if len(value1) == 0:
            print(0)
        else:
            #print(statistics.median(value1))
            print(statistics.mean(value1))
