# ECE 506 CEDR Tutorial

## Tutorial #0: Getting Started

[📖 Tutorial #0: Requirements and Initial Setup](./tutorial_README/tutorial0.md)

1. CEDR requirements installation
     * Docker-based setup
     * Native Linux-based setup
2. Initial build of CEDR for testing requirements
3. Links to the hardware image that would be used

## Class Tutorial #1: API Integration and Verification

[📖 Step-by-Step Guide for Tutorial #1](./tutorial_README/tutorial1.md)

1. Introduce the existing FFT API in a sample application
     1. Demonstrate functional verification of the FFT API
2. Extend the system with a new ZIP API signature
3. Update the sample application to use the new ZIP API
     1. Perform functional verification
     2. Validate API behavior using Gantt chart analysis

## Class Tutorial #2: FPGA Integration and Accelerator Testing

[📖 Step-by-Step Guide for Tutorial #2](./tutorial_README/tutorial2.md)

1. Start with an existing FPGA image containing FFT and ZIP accelerators
   1. Cross Compilation:
      * Build CEDR for the FPGA
      * Build sample application for the FPGA
   2. Access the FPGA hardware and testing CEDR on the FPGA using CPUs only
2. Add a new FFT accelerator to CEDR
   1. Rebuild CEDR with the updated configuration
   2. Test the FFT accelerator with CEDR
      * Validate execution using Gantt charts

## Class Tutorial #3: Scheduling and Design Space Exploration

[📖 Step-by-Step Guide for Tutorial #3](./tutorial_README/tutorial3.md)

1. Running multiple applications with CEDR
   1. Running multiple instances of a single application
   2. Running multiple applications with multiple instances
2. Integration and Evaluation of EFT Scheduler
   1. Initialization
   2. EFT heuristic - Task-to-PE mapping
   3. EFT heuristic - Task-to-PE assignment
   4. Final checks
   5. Full EFT in C/C++
   6. Adding EFT as a scheduling option
   7. Enabling EFT for CEDR
   8. Running CEDR with EFT Scheduler
3. Perform simple design space exploration (DSE) on x86
