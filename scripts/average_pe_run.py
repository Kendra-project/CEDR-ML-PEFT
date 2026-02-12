import  sys
from statistics import mean
from statistics import median

if (len(sys.argv) < 2):
    print("Input list:")
    print("1. Input Filename")

filename = sys.argv[1]
print("Reading File:", end=' ')
print(filename)
with open(filename) as f:
    lines = f.readlines()
    tasks={}
    tasks["CPU"]={}
    tasks["FFT"]=[]
    tasks["ZIP"]=[]
    tasks["CONV"]=[]
    tasks["MULT"]=[]
    tasks["CPU"]["FFT"]=[]
    tasks["CPU"]["ZIP"]=[]
    tasks["CPU"]["CONV"]=[]
    tasks["CPU"]["MULT"]=[]
    counter=0
    for line in lines:
        if "resource_name: fft" in line:
            counter+=1
            l=line.split(",")[7].split(":")[1].split(" ")[1]
            tasks["FFT"].append(float(l))
        elif "resource_name: cpu" in line:
            l=line.split(",")[7].split(":")[1].split(" ")[1]
            if "DASH_FFT" in line:
                tasks["CPU"]["FFT"].append(float(l))
            elif "DASH_ZIP" in line:
                tasks["CPU"]["ZIP"].append(float(l))
            elif "DASH_CONV" in line:
                tasks["CPU"]["CONV"].append(float(l))
            elif "DASH_GEMM" in line:
                tasks["CPU"]["MULT"].append(float(l))
        elif "resource_name: zip" in line:
            l=line.split(",")[7].split(":")[1].split(" ")[1]
            tasks["ZIP"].append(float(l))
        elif "resource_name: conv" in line:
            l=line.split(",")[7].split(":")[1].split(" ")[1]
            tasks["CONV"].append(float(l))
        elif "resource_name: gemm" in line:
            l=line.split(",")[7].split(":")[1].split(" ")[1]
            tasks["MULT"].append(float(l))
#for key, val in tasks.items():
#    print(key, end=': ')
#    print(val/counter)
print("Average Results:")
print("\t\\\"DASH API Costs\\\": {")
if(len(tasks["CPU"]["FFT"])!=0):
#    print("CPU-FFT:",end=" ")
    print("\t\t\\\"DASH_FFT_cpu\\\": " + str(int(mean(tasks["CPU"]["FFT"]))) + ",")
else:
    print("\t\t\\\"DASH_FFT_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["FFT"])!=0):
#    print("FFT:",end=" ")
    print("\t\t\\\"DASH_FFT_fft\\\": " + str(int(mean(tasks["FFT"]))) + ",")
else:
    print("\t\t\\\"DASH_FFT_fft\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["ZIP"])!=0):
#    print("CPU-ZIP:",end=" ")
    print("\t\t\\\"DASH_ZIP_cpu\\\": " + str(int(mean(tasks["CPU"]["ZIP"]))) + ",")
else:
    print("\t\t\\\"DASH_ZIP_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["ZIP"])!=0):
#    print("ZIP:",end=" ")
    print("\t\t\\\"DASH_ZIP_zip\\\": " + str(int(mean(tasks["ZIP"]))) + ",")
else:
    print("\t\t\\\"DASH_ZIP_zip\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["CONV"])!=0):
#    print("CPU-CONV:",end=" ")
    print("\t\t\\\"DASH_CONV_2D_cpu\\\": " + str(int(mean(tasks["CPU"]["CONV"]))) + ",")
else:
    print("\t\t\\\"DASH_CONV_2D_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["CONV"])!=0):
#    print("CONV:",end=" ")
    print("\t\t\\\"DASH_CONV_2D_conv_2d\\\": " + str(int(mean(tasks["CONV"]))))
else:
    print("\t\t\\\"DASH_CONV_2D_conv_2d\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["MULT"])!=0):
#    print("CPU-CONV:",end=" ")
    print("\t\t\\\"DASH_GEMM_cpu\\\": " + str(int(mean(tasks["CPU"]["MULT"]))) + ",")
else:
    print("\t\t\\\"DASH_GEMM_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["MULT"])!=0):
#    print("CONV:",end=" ")
    print("\t\t\\\"DASH_GEMM_mmult\\\": " + str(int(mean(tasks["MULT"]))))
else:
    print("\t\t\\\"DASH_GEMM_mmult\\\": " + str(int(100)))
print("\t},")
print("\n")
print("Median Results:")
print("\t\\\"DASH API Costs\\\": {")
if(len(tasks["CPU"]["FFT"])!=0):
#    print("CPU-FFT:",end=" ")
    print("\t\t\\\"DASH_FFT_cpu\\\": " + str(int(median(tasks["CPU"]["FFT"]))) + ",")
else:
    print("\t\t\\\"DASH_FFT_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["FFT"])!=0):
#    print("FFT:",end=" ")
    print("\t\t\\\"DASH_FFT_fft\\\": " + str(int(median(tasks["FFT"]))) + ",")
else:
    print("\t\t\\\"DASH_FFT_fft\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["ZIP"])!=0):
#    print("CPU-ZIP:",end=" ")
    print("\t\t\\\"DASH_ZIP_cpu\\\": " + str(int(median(tasks["CPU"]["ZIP"]))) + ",")
else:
    print("\t\t\\\"DASH_ZIP_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["ZIP"])!=0):
#    print("ZIP:",end=" ")
    print("\t\t\\\"DASH_ZIP_zip\\\": " + str(int(median(tasks["ZIP"]))) + ",")
else:
    print("\t\t\\\"DASH_ZIP_zip\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["CONV"])!=0):
#    print("CPU-CONV:",end=" ")
    print("\t\t\\\"DASH_CONV_2D_cpu\\\": " + str(int(median(tasks["CPU"]["CONV"]))) + ",")
else:
    print("\t\t\\\"DASH_CONV_2D_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["CONV"])!=0):
#    print("CONV:",end=" ")
    print("\t\t\\\"DASH_CONV_2D_conv_2d\\\": " + str(int(median(tasks["CONV"]))))
else:
    print("\t\t\\\"DASH_CONV_2D_conv_2d\\\": " + str(int(100)) + ",")
if(len(tasks["CPU"]["MULT"])!=0):
#    print("CPU-CONV:",end=" ")
    print("\t\t\\\"DASH_GEMM_cpu\\\": " + str(int(median(tasks["CPU"]["MULT"]))) + ",")
else:
    print("\t\t\\\"DASH_GEMM_cpu\\\": " + str(int(100)) + ",")
if(len(tasks["MULT"])!=0):
#    print("CONV:",end=" ")
    print("\t\t\\\"DASH_GEMM_mmult\\\": " + str(int(median(tasks["MULT"]))))
else:
    print("\t\t\\\"DASH_GEMM_mmult\\\": " + str(int(100)))
print("\t},")
#print(counter)
#print(tasks)
