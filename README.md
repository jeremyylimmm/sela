# sela

An ahead-of-time compiler for tensor graphs, written in Rust.

Sela lets you describe a tensor computation as a graph, then compiles the whole
graph up front into optimized native code before you feed it any data. Rather
than interpreting operations one at a time, the entire graph is lowered to a
single kernel, handed to a backend compiler, and loaded back in as a shared
library ready to run.

Today the only backend is **CPU** — the graph is emitted as C and built with
your system compiler (`cc -O2 -march=native`). The architecture is designed so
that GPU backends (CUDA, Metal compute, etc.) can be added as additional
targets without touching the front-end or IR.

> ⚠️ Early and experimental. The API is unstable, the op set is small, and there
> are sharp edges. This is a learning/research project, not a production library.

## How it works

```
   Builder  ──►   IR graph   ──►   Target.compile   ──►   Graph
  (front-end)   (ops, shapes,      (lower + codegen)    (runnable,
                 strides)                                 read/write tensors)
```

1. **Build.** You construct a computation with `Builder`, chaining tensor ops.
   Each op records its shape and strides. Views like `transpose`, `broadcast`,
   and `view` are *lazy* — they only rewrite strides, no data moves. Call
   `.expose()` on any node to get a `Handle` you can later read or write.
2. **Lower to IR.** `builder.build()` freezes the graph into an immutable `IR`
   of nodes (op + inputs + shape + strides).
3. **Compile.** A `Target` walks the graph and emits a backend kernel for every
   exposed node. The CPU target generates C, compiles it to a `.so` with `cc`,
   and loads it via `libloading`. Pure view ops (broadcast, transpose, view)
   compile to zero code — they just alias the underlying buffer with adjusted
   strides.
4. **Run.** The returned `Graph` owns the tensor buffers. You write inputs by
   handle, call `run()` to execute the compiled kernel, and read results back.

## Example

```rust
use sela::{
    Builder, shape, target,
    target::{Graph, Target},
};

fn main() {
    let builder = Builder::new();

    // Declare input tensors.
    let a = builder.tensor(shape![12]);
    let b = builder.tensor(shape![2, 3]);
    let d = builder.tensor(shape![2, 2]);

    // Batched matmul plus a bias: (a · bᵀ) + d.
    // transpose, broadcast, and view are free (lazy) views.
    let c = a.view(shape![2, 2, 3])
        .dot(b.transpose(0, 1).broadcast(shape![2, 3, 2]))
        .to_contiguous()
        + d.broadcast(shape![2, 2, 2]).to_contiguous();

    // Expose the nodes you want to read or write, getting a Handle for each.
    let a_handle = a.expose();
    let b_handle = b.expose();
    let d_handle = d.expose();
    let c_handle = c.expose();

    let ir = builder.build();

    // Optional: dump the graph as Graphviz DOT.
    println!("{}", ir.graphviz());

    // Compile the graph for the CPU.
    let target = target::cpu::CPUTarget::new();
    let mut graph = target.compile(ir);

    // Fill inputs with flat f32 slices, run, read outputs back.
    graph.write_tensor(a_handle, &[2.0, 6.0, 4.0, 1.0, 9.0, 7.0, 4.0, 8.0, 4.0, 4.0, 2.0, 9.0]);
    graph.write_tensor(b_handle, &[3.0, 4.0, 11.0, 2.0, 10.0, 8.0]);
    graph.write_tensor(d_handle, &[0.0, 0.0, 0.0, 0.0]);

    graph.run();

    let out = graph.read_tensor(c_handle).to_vec();
    println!("{out:?}");
}
```

Run the bundled demo (a batched matmul) with:

```sh
cargo run
```

## Operations

| Op                     | Builder API                        | Notes                                  |
| ---------------------- | ---------------------------------- | -------------------------------------- |
| Declare tensor         | `builder.tensor(shape![...])`      | Dense strides                          |
| Add / Sub              | `a + b`, `a - b`                   | Elementwise, matching shapes           |
| Elementwise mul / div  | `a.elementwise_mul(b)` / `_div`    | Elementwise, matching shapes           |
| Matmul / dot           | `a.dot(b)`                         | Batched; contracts last two dims       |
| Transpose              | `a.transpose(d0, d1)`              | Lazy view (swaps strides)              |
| Broadcast              | `a.broadcast(shape![...])`         | Lazy view (zero strides)               |
| View                   | `a.view(shape![...])`              | Lazy reshape (rewrites strides)        |
| Materialize            | `a.to_contiguous()`                | Copies a view into a dense buffer      |
| Expose handle          | `a.expose()`                       | Returns a `Handle` for read/write      |

All tensors are `f32`. Shapes are built with the `shape![...]` macro. Reads and
writes go through a `Handle` obtained from `expose()`, using flat `&[f32]`
slices in row-major order.

## Design notes

- **Lazy views.** `transpose`, `broadcast`, and `view` never copy. They produce
  new IR nodes whose strides encode the reindexing, and the codegen aliases the
  source buffer. Broadcasting sets the broadcast dims to stride 0. Use
  `to_contiguous()` when you need the view materialized into a dense buffer.
- **One kernel per graph.** The graph compiles into a single C `run()`
  function; intermediate tensors are allocated as flat buffers and indexed by
  stride.
- **Handles.** Call `expose()` on a node to get a `Handle`. After compiling,
  `write_tensor(handle, &[f32])` and `read_tensor(handle) -> &[f32]` move data
  in and out of the graph's buffers as flat row-major slices.
- **Pluggable backends.** A backend implements the `Target` trait
  (`compile(ir) -> impl Graph`) and the `Graph` trait (`run`, `read_tensor`,
  `write_tensor`). CPU is the reference implementation.

## Building

Requires a Rust toolchain (edition 2024) and a C compiler (`cc`) on `PATH`.

```sh
cargo build
cargo run       # runs the demo in src/main.rs
```

Compiled kernels are written to `graph/build_<timestamp>.c` and `.so`.

## Roadmap

- [ ] More ops (reductions, activations, reshape, concat)
- [ ] Graph optimizations (fusion, common-subexpression elimination, constant folding)
- [ ] GPU backends: CUDA and Metal compute
- [ ] A safer, higher-level tensor API
- [ ] Autodiff

## License

Licensed under the [MIT License](LICENSE).
