# GPU Performance Engineering for AI Infrastructure

> A comprehensive guide to GPU kernel programming and optimization, following the structure of [Elicit's machine-learning-list](https://github.com/elicit/machine-learning-list). Curated for practitioners building high-performance AI systems.

**Legend:** ✨ = Recently added | 🔥 = Community favorite | 📄 = Paper | 📝 = Blog/Tutorial | 🎥 = Video | 📚 = Book | 💻 = Code

---

## Table of Contents

1. [Fundamentals](#1-fundamentals)
2. [Matrix Multiplication (The Gateway)](#2-matrix-multiplication-the-gateway)
3. [Tensor Cores & Mixed Precision](#3-tensor-cores--mixed-precision)
4. [Attention & Memory-Bound Kernels](#4-attention--memory-bound-kernels)
5. [Compiler & DSL Approaches](#5-compiler--dsl-approaches)
6. [Profiling & Optimization](#6-profiling--optimization)
7. [AMD & Alternative Hardware](#7-amd--alternative-hardware)
8. [Production Inference Systems](#8-production-inference-systems)
9. [LLM-Generated Kernels](#9-llm-generated-kernels)
10. [Distributed & Multi-GPU](#10-distributed--multi-gpu)
11. [The Big Picture](#11-the-big-picture)

---

## 1. Fundamentals

### Tier 1: Start Here

📚 **Programming Massively Parallel Processors (PMPP)** - Hwu, Kirk, El Hajj
- The canonical textbook, 4th edition covers Ampere/Hopper
- Covers GPU architecture, memory hierarchy, parallel patterns

🎥 **GPU Mode Lectures** - [github.com/gpu-mode/lectures](https://github.com/gpu-mode/lectures) 🔥
- Community-driven lecture series: profiling → kernels → CUTLASS → SASS
- Active Discord community (23k+ members): [discord.gg/gpumode](https://discord.gg/gpumode)

📝 **NVIDIA CUDA Programming Guide** - [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- Official documentation, essential reference for programming model

### Tier 2: Architecture Deep Dives

📝 **NVIDIA Hopper Architecture In-Depth** - [developer.nvidia.com/blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- TMA, Thread Block Clusters, Distributed Shared Memory, WGMMA

📝 **Chips and Cheese: Blackwell** - [chipsandcheese.com](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu) ✨
- Microbenchmarking analysis of GB202, memory latency comparisons

📄 **Dissecting the NVIDIA Hopper GPU Architecture** - [arxiv.org/abs/2402.13499](https://arxiv.org/abs/2402.13499)
- Academic microbenchmarking of H100

📄 **Dissecting the NVIDIA Blackwell Architecture** - [arxiv.org/abs/2507.10789](https://arxiv.org/abs/2507.10789) ✨
- Microbenchmarks covering tcgen05, TMEM, 2SM MMA

### Tier 3: Low-Level Details

📝 **PTX ISA Documentation** - [docs.nvidia.com/cuda/parallel-thread-execution](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- Official PTX instruction set reference

📝 **Understanding PTX** - [developer.nvidia.com/blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing)
- Introduction to CUDA's virtual assembly language

💻 **DocumentSASS** - [github.com/0xD0GF00D/DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS)
- Unofficial SASS instruction documentation extracted from nvdisasm

📝 **JEB SASS Disassembler** - [pnfsoftware.com](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/)
- Reverse engineering GPU binaries (Volta → Blackwell)

---

## 2. Matrix Multiplication (The Gateway)

### Tier 1: Essential Tutorials

📝 **How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance** - siboehm 🔥
- [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)
- The canonical starting tutorial. Covers tiling, shared memory, vectorized loads

📝 **Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels** - Aleksa Gordić ✨
- [aleksagordic.com/blog/matmul](https://www.aleksagordic.com/blog/matmul)
- 47 figures. Covers PTX/SASS, wave quantization, ILP, roofline model, warp tiling

📝 **Outperforming cuBLAS on H100: A Worklog** - cudaforfun 🔥
- [cudaforfun.substack.com](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- Real optimization journey using WGMMA and TMA

💻 **Fast CUDA GEMM with Tensor Cores** - lezcano
- [github.com/lezcano/gemm](https://github.com/lezcano/gemm)
- Practical tensor core implementation

### Tier 2: Advanced Implementations

📝 **Advanced Matrix Multiplication Optimization** - salykova
- [salykova.github.io/sgemm-gpu](https://salykova.github.io/sgemm-gpu)
- Detailed optimization techniques following CUTLASS approach

📝 **CUDA Matrix Multiplication Optimization** - Lei Mao
- [leimao.github.io](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- Systematic optimization progression

📝 **Optimizing SGEMV for cuBLAS-like Performance** - Maharshi
- [maharshi.bearblog.dev](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/)
- Matrix-vector multiplication optimization worklog

💻 **DeepGEMM** - DeepSeek
- [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- Clean FP8 GEMM implementation for Hopper, ~300 lines

### Tier 3: cuBLAS Internals

📝 **New cuBLAS 12.0 Features** - [developer.nvidia.com/blog](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/)
- Hopper-specific optimizations and performance

📝 **cuBLAS 12.9 Floating Point Emulation** - [developer.nvidia.com/blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/) ✨
- FP32 emulation with BF16 tensor cores

---

## 3. Tensor Cores & Mixed Precision

### Tier 1: Fundamentals

📄 **NVIDIA Tensor Core Evolution: Volta to Blackwell** - SemiAnalysis 🔥
- [semianalysis.com](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)
- Comprehensive evolution: WMMA → MMA → WGMMA → tcgen05

📝 **Deep Dive on Hopper TMA Unit for FP8 GEMMs** - PyTorch
- [pytorch.org/blog](https://pytorch.org/blog/hopper-tma-unit/)
- TMA programming model and FP8 integration

📝 **CUTLASS Tutorial: Mastering TMA** - Colfax Research
- [research.colfax-intl.com](https://research.colfax-intl.com/tutorial-hopper-tma/)
- Tensor Memory Accelerator programming

### Tier 2: Precision Formats

📝 **Introducing FP8 for Efficient AI Training** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- E4M3 vs E5M2 formats, scaling strategies

📝 **Introducing NVFP4 for Low-Precision Inference** - NVIDIA ✨
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- Blackwell FP4 with microscaling (MXFP4)

💻 **NVIDIA Transformer Engine** - [github.com/NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
- Library for FP8/FP4 training and inference

📝 **Per-Tensor and Per-Block Scaling for FP8** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- Scaling strategies for quantization

### Tier 3: Blackwell-Specific

📝 **Matrix Multiplication on Blackwell: Part 1** - Modular ✨
- [modular.com/blog](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)
- tcgen05, TMEM, 2SM MMA programming

📝 **Blackwell Pipelining with CuTeDSL** - Simon Veitner ✨
- LinkedIn post on advanced Blackwell kernel patterns

---

## 4. Attention & Memory-Bound Kernels

### Tier 1: FlashAttention

📄 **FlashAttention: Fast and Memory-Efficient Attention** - Dao et al. 🔥
- [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- Original paper: IO-aware exact attention

📄 **FlashAttention-2** - Dao
- [arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- Better parallelization, work partitioning

📄 **FlashAttention-3: Fast and Accurate Attention with Asynchrony** - Dao et al. ✨
- [arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)
- Hopper-specific: warp specialization, WGMMA pipelining

📄 **A Case Study in CUDA Kernel Fusion: FlashAttention-2 on Hopper** - Jay Shah et al.
- [arxiv.org/abs/2312.11918](https://arxiv.org/abs/2312.11918)
- CUTLASS implementation details

### Tier 2: PagedAttention & Serving

📄 **Efficient Memory Management for LLM Serving with PagedAttention** - vLLM team 🔥
- [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- Virtual memory for KV cache

💻 **FlashInfer** - [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) ✨
- Kernel library for LLM serving (MLSys 2025 Best Paper)
- PagedAttention, FlashAttention-3, MLA support

📝 **Accelerating Self-Attentions with FlashInfer**
- [flashinfer.ai](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- Architecture and design decisions

### Tier 3: KV Cache Optimization

📝 **Mastering LLM Techniques: Inference Optimization** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- Comprehensive guide: GQA, MQA, KV cache compression

📄 **GQA: Training Generalized Multi-Query Transformer Models** - Google
- Grouped Query Attention for memory efficiency

📄 **Multi-Head Latent Attention (MLA)** - DeepSeek
- Low-rank KV compression, 8x cache reduction

📄 **A Survey on LLM Acceleration based on KV Cache Management**
- [arxiv.org/abs/2412.19442](https://arxiv.org/abs/2412.19442)
- Comprehensive taxonomy of KV cache techniques

---

## 5. Compiler & DSL Approaches

### Tier 1: Triton

📝 **Introducing Triton** - OpenAI 🔥
- [openai.com/index/triton](https://openai.com/index/triton)
- Original announcement and motivation

💻 **Triton Language** - [github.com/triton-lang/triton](https://github.com/triton-lang/triton)
- Development repository

📝 **Deep Dive into Triton Internals (Parts 1-3)** - Kapil Sharma 🔥
- [kapilsharma.dev](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)
- Compiler pipeline: Python → MLIR → PTX → CUBIN

🎥 **GPU Mode: Triton Internals Talk** - Kapil Sharma
- [kapilsharma.dev](https://www.kapilsharma.dev/posts/gpu-mode-triton-internals-talk/)
- Video + slides from the lecture

### Tier 2: CUTLASS & CuTe

📝 **Learn CUTLASS the Hard Way** - Lei Mao 🔥
- [leimao.github.io](https://leimao.github.io/article/Learn-CUTLASS-The-Hard-Way/)
- Naive GEMM → real CUTLASS progression

📝 **CUTLASS Tutorial: GEMM Kernel Design with Pipelining** - Colfax Research
- [research.colfax-intl.com](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- Warp specialization, producer-consumer patterns

💻 **NVIDIA CUTLASS** - [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- CUDA Templates for Linear Algebra Subroutines

💻 **cuTile (CUDA Tile)** - [github.com/NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python) ✨
- New tile-level programming model in CUDA 13.1

### Tier 3: Other DSLs

💻 **TileLang** - [github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang) ✨
- Composable tiled programming, 1075x speedup over PyTorch on H100

💻 **ThunderKittens** - Stanford Hazy Research
- [github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- DSL for writing fast GPU kernels

📝 **Apache TVM** - [tvm.apache.org](https://tvm.apache.org/)
- End-to-end ML compiler with auto-tuning (Ansor)

📝 **MLIR GPU Dialect** - [mlir.llvm.org](https://mlir.llvm.org/)
- Compiler infrastructure for heterogeneous compute

💻 **Mojo** - [modular.com/mojo](https://www.modular.com/mojo) ✨
- MLIR-based language targeting GPU/CPU, SIMD-first design

---

## 6. Profiling & Optimization

### Tier 1: NVIDIA Tools

📝 **Nsight Compute Roofline Analysis** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
- Roofline modeling for bottleneck analysis

📝 **CUDA Occupancy Calculator** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API

📝 **Hopper Tuning Guide** - [docs.nvidia.com/cuda/hopper-tuning-guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- Official optimization guide for H100

### Tier 2: Optimization Techniques

📝 **Memory Coalescing and Bank Conflicts** - Medium
- [medium.com/@dhanushg295](https://medium.com/@dhanushg295/mastering-cuda-matrix-multiplication-an-introduction-to-shared-memory-tile-memory-coalescing-and-d7979499b9c5)
- Shared memory optimization, padding tricks

📝 **Understanding CUDA Occupancy** - Medium
- [medium.com/@manisharadwad](https://medium.com/@manisharadwad/unlocking-gpu-potential-understanding-and-optimizing-cuda-occupancy-2f43ee01ad7e)
- Thread block configuration

📝 **The Roofline Model** - NERSC
- [docs.nersc.gov](https://docs.nersc.gov/tools/performance/roofline/)
- Arithmetic intensity, compute vs memory bound

📝 **Understanding the Top-K CUDA Kernel with PTX** - alpindale ✨
- [blog.alpindale.net](https://blog.alpindale.net/posts/top_k_cuda/)
- 10x speedup over torch.topk, PTX-level optimization

### Tier 3: Advanced Topics

📝 **CUDA Graphs for Reduced Launch Overhead** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/cuda-graphs/)
- Batch kernel launches, 5x speedup for small kernels

📄 **Kernel Batching with CUDA Graphs** - [arxiv.org/abs/2501.09398](https://arxiv.org/abs/2501.09398) ✨
- Optimal batch sizes (50-100 nodes), 1.4x improvement

📝 **Warp Specialization in PyTorch** - [pytorch.org/blog](https://pytorch.org/blog/warp-specialization/)
- Producer-consumer patterns, async execution

📄 **Tawa: Automatic Warp Specialization** - [arxiv.org/abs/2510.14719](https://arxiv.org/abs/2510.14719) ✨
- Matches FlashAttention-3 performance with less effort

---

## 7. AMD & Alternative Hardware

### Tier 1: ROCm Fundamentals

📝 **Developing Triton Kernels on AMD GPUs** - AMD ROCm Blog
- [rocm.blogs.amd.com](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- Triton for MI300X

📝 **Triton Kernel Optimizations on AMD** - AMD ROCm Blog ✨
- [rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)
- Performance tuning for CDNA

💻 **HipKittens** - ThunderKittens for AMD ✨
- Tile programming abstraction for MI300X

### Tier 2: CDNA Architecture

📝 **Chips and Cheese: AMD CDNA 3** - [chipsandcheese.com](https://chipsandcheese.com)
- MI300X architecture analysis, chiplet design

📝 **Chips and Cheese: RDNA 4** - [chipsandcheese.com](https://chipsandcheese.com/p/amds-rdna4-gpu-architecture-at-hot) ✨
- Dynamic register allocation, cache strategies

📝 **AMD RDNA 3 Microbenchmarking** - Chips and Cheese
- [chipsandcheese.com](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture)

### Tier 3: TPU & Others

📝 **The Rise of Pallas: Custom TPU Kernels** - Towards Data Science 🔥
- [towardsdatascience.com](https://towardsdatascience.com/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a/)
- JAX Pallas for TPU programming

📝 **vLLM TPU: Unified JAX Backend** - vLLM Blog ✨
- [blog.vllm.ai](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)
- 20% throughput improvement via JAX primitives

📝 **Building Production AI on Cloud TPUs with JAX** - Google
- [docs.cloud.google.com](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)

---

## 8. Production Inference Systems

### Tier 1: Core Systems

💻 **vLLM** - [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) 🔥
- PagedAttention, continuous batching, high throughput

💻 **SGLang** - [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) 🔥
- RadixAttention, structured generation, prefix caching

💻 **TensorRT-LLM** - [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- NVIDIA's optimized inference library

📝 **Accelerating Transformers with cuDNN 9** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/)
- Fused attention, Graph API

### Tier 2: Continuous Batching

📄 **Orca: Distributed Serving with Iteration-Level Scheduling** - OSDI 2022 🔥
- [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu)
- Original continuous batching paper, 36.9x throughput

📝 **Continuous Batching from First Principles** - Hugging Face
- [huggingface.co/blog](https://huggingface.co/blog/continuous_batching)
- Clear explanation of dynamic batching

📝 **Achieve 23x LLM Inference Throughput** - Anyscale
- [anyscale.com/blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- vLLM performance analysis

### Tier 3: Speculative Decoding

📄 **Medusa: Simple Framework for Accelerating LLM Generation**
- Multiple heads for parallel draft tokens

📄 **EAGLE: Speculative Sampling with Draft Model**
- Autoregressive draft prediction

📝 **Speculative Decoding Overview** - vLLM Docs
- [docs.vllm.ai](https://docs.vllm.ai)
- Implementation in vLLM

---

## 9. LLM-Generated Kernels

### Tier 1: Benchmarks & Models

📄 **KernelBench: Can LLMs Write Efficient GPU Kernels?** - Stanford 🔥
- [arxiv.org/abs/2502.10517](https://arxiv.org/abs/2502.10517)
- 250 PyTorch workloads, fast_p metric

💻 **KernelLLM** - Meta/Facebook ✨
- [huggingface.co/facebook/KernelLLM](https://huggingface.co/facebook/KernelLLM)
- 8B model trained on 25k PyTorch→Triton pairs, beats GPT-4o

📄 **TritonBench** - [arxiv.org/abs/2502.14752](https://arxiv.org/abs/2502.14752) ✨
- 184 real-world Triton operators from GitHub

### Tier 2: Agentic Approaches

📝 **The AI CUDA Engineer** - Sakana AI ✨
- [sakana.ai/ai-cuda-engineer](https://sakana.ai/ai-cuda-engineer/)
- Evolutionary optimization, 10-100x speedups (with caveats about benchmark gaming)

📄 **AlphaEvolve** - Google DeepMind ✨
- [deepmind.google](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- 32.5% FlashAttention speedup, 23% GEMM speedup

📄 **Kevin: Multi-Turn RL for CUDA Kernels** - [arxiv.org/abs/2507.11948](https://arxiv.org/abs/2507.11948) ✨
- First multi-turn RL model, 82% correctness (vs 56% base)

📄 **CUDA-L1: Contrastive RL for CUDA Optimization** - [arxiv.org/abs/2507.14111](https://arxiv.org/abs/2507.14111) ✨
- 3.12x average speedup on KernelBench

### Tier 3: Research Papers

📄 **EvoEngineer: Automated CUDA Kernel Evolution** - [arxiv.org/abs/2510.03760](https://arxiv.org/abs/2510.03760)

📄 **QiMeng-Kernel: Macro-Thinking Micro-Coding for GPU Kernels** - [arxiv.org/abs/2511.20100](https://arxiv.org/abs/2511.20100)

📄 **CUDA-LLM: LLMs Can Write Efficient CUDA Kernels** - [arxiv.org/abs/2506.09092](https://arxiv.org/abs/2506.09092)

📝 **GEAK: Triton Kernel AI Agent** - AMD ROCm ✨
- [rocm.blogs.amd.com](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html)
- 51% accuracy, 1.81x speedup on MI300X

---

## 10. Distributed & Multi-GPU

### Tier 1: Communication Primitives

💻 **NVIDIA NCCL** - [github.com/NVIDIA/nccl](https://github.com/NVIDIA/nccl)
- Collective communication: all-reduce, all-gather, broadcast

📝 **Fast Multi-GPU Collectives with NCCL** - NVIDIA
- [developer.nvidia.com/blog](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/)
- Ring, tree algorithms, topology-aware optimization

📄 **Demystifying NCCL** - [arxiv.org/abs/2507.04786](https://arxiv.org/abs/2507.04786)
- In-depth analysis of GPU communication protocols

📄 **Collective Communication for 100k+ GPUs** - Meta NCCLX
- [arxiv.org/abs/2510.20171](https://arxiv.org/abs/2510.20171)
- Scaling to massive clusters

### Tier 2: Parallelism Strategies

📄 **Megatron-LM: Training Multi-Billion Parameter Models** - NVIDIA 🔥
- [arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- Tensor parallelism, pipeline parallelism

💻 **Megatron-LM** - [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Up to 47% MFU on H100 clusters

📝 **Large Scale Tensor Parallel Training** - PyTorch Tutorial
- [pytorch.org/tutorials](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- Native TP support in PyTorch

💻 **Horovod** - [github.com/horovod/horovod](https://github.com/horovod/horovod)
- Ring-allreduce distributed training, 90% scaling efficiency

### Tier 3: Kernel Fusion

📝 **Kernel Fusion in CUDA** - vrushankdes.ai
- [vrushankdes.ai](https://www.vrushankdes.ai/diffusion-policy-inference-optimization/part-vi---kernel-fusion-in-cuda)
- Vertical vs horizontal fusion, U-Net optimization

📄 **Automatic Horizontal Fusion for GPU Kernels** - CMU
- [cs.toronto.edu](https://www.cs.toronto.edu/ecosystem/papers/CGO_22/Horizontal_Fusion.pdf)
- 12-55% speedup via parallel kernel execution

---

## 11. The Big Picture

### Industry Analysis

📝 **The CUDA Moat Debate** - Various authors
- Is NVIDIA's software ecosystem an insurmountable advantage?

📝 **GPU Poor: Why Cloud Costs Matter** - Various
- Economic analysis of GPU compute

### Practitioner Blogs & Substacks

📝 **Michal Pitr - From Scratch** - [michalpitr.substack.com](https://michalpitr.substack.com)
- GPU programming, inference optimization

📝 **cudaforfun Substack** - [cudaforfun.substack.com](https://cudaforfun.substack.com)
- cuBLAS-level kernel development

📝 **Lei Mao's Log Book** - [leimao.github.io](https://leimao.github.io)
- CUTLASS, CUDA optimization deep dives

📝 **Aleksa Gordić's Blog** - [aleksagordic.com/blog](https://www.aleksagordic.com/blog)
- Ex-DeepMind, GPU architecture and matmul

### Communities

🎥 **GPU Mode Discord** - [discord.gg/gpumode](https://discord.gg/gpumode) 🔥
- 23k+ members, weekly lectures, kernel leaderboard

💻 **GPU Mode Resource Stream** - [github.com/gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream)
- Curated CUDA/GPU learning materials

---

## Contributing

Contributions welcome! Please ensure resources meet our quality criteria:
- ✅ Primary sources (papers, official docs)
- ✅ Practitioner blogs with real implementation insights
- ✅ Active maintenance or timeless fundamentals
- ❌ Surface-level tutorials
- ❌ AI-generated content without human verification

## License

MIT
