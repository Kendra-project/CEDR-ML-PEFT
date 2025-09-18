# FlushTime

FlushTime is a Linux kernel defense to mitigating flush-based cache attacks via collaborating flush instructions and timers on ARMv8-A.
We are only using the PoC codes of the cache-timing vulernabilities for our anomaly detection research. Not interested in enabling this kernel patch currently.
Tested on all of Jetson, Rock5B, and ZCU102, the secret fails to be predicted on ZCU102.  

Credits:
The concept and PoC for FlushTime are based on the original work by Ruhrie et al. (see: https://github.com/gejingquan/FlushTime, if available). We use only the PoC codes for cache-timing vulnerability research and anomaly detection.