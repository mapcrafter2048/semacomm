# Gemini Agent Instructions

This document outlines the operating procedures for the Gemini agent within this project.

## 1. Post-Execution Logging

After every significant execution or task, I will record the following in a private log:

*   **Key Changes:** A summary of the modifications made to the codebase.
*   **Lessons Learned:** Observations about what worked well and what didn't. This includes identifying inefficient code, problematic dependencies, or successful refactoring strategies.
*   **Key Code Structure:** Notes on the project's architecture, main components, and data flow to maintain a high-level understanding of the codebase.

## 2. Coding Best Practices

When writing or modifying code, I will adhere to the following best practices:

### 2.1. Debugging

*   I will use the `logging` module for structured logging instead of `print()` statements.
*   I will implement different log levels (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`, `MODIFIABLE`, `CRITICAL`, `NOTIMPORTANT` etc) to control the verbosity of the output.

### 2.2. Monitoring and Logging

*   I will use the PyTorch Profiler to identify performance bottlenecks.
*   I will use pytorch compatible TensorBoard to visualize metrics, model graphs, and profiling data.
*   I will use wandb Weights and Biases wherever it suitable.

### 2.3. Pipeline Visualization and Performance Analysis

To visualize data flow and analyze performance, especially concerning CPU/GPU memory transfers, I will use the following tools and techniques:

*   **PyTorch Profiler:** I will use the `torch.profiler.profile` context manager to wrap code blocks and record performance data. I will configure it to record CPU and CUDA activity, as well as memory usage.
*   **TensorBoard:** I will export the profiler's output to a trace file and launch TensorBoard to visualize the execution timeline. This will allow us to see:
    *   Which operations are running on the CPU vs. the GPU.
    *   The duration of each operation.
    *   Memory allocation and deallocation events.
    *   The timing of data transfers between the CPU and GPU.
*   **Code Implementation:** When writing functions that process tensors, I will add the necessary boilerplate code to enable profiling. This includes:
    *   Wrapping the main execution loop with the `torch.profiler.profile` context manager.
    *   Adding `torch.profiler.record_function` blocks to profile specific sections of code.
    *   Using the `schedule` and `on_trace_ready` arguments to control when the profiler is active and how the data is saved.

By following these instructions, I will be able to provide more detailed insights into the performance of your code and help you identify and resolve bottlenecks.
