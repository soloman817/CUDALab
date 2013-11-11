(**
# Getting Started
*)

(**
First, reference `Alea.CUDA.dll`.
*)
#I @"..\..\packages\Alea.cuBase\lib\net40"
#r "Alea.CUDA.dll"

(**
Then open some namespaces:

- `Microsoft.FSharp.Quotations` - We will use `Expr` type from this module, for we will do quotation
                                  splicing.
- `Alea.CUDA` - The core library of Alea.cuBase.
- `Alea.CUDA.Utilities` - We will use `Compiler` module as a shortcut to the compiler.
*)
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities

(**
The next step is to define a template by `cuda` workflow. We create a function here, you give
a unary function by quotation and then generate a template. This argument `transform` is of
type `Expr<float -> float>`.
*)
let template transform = cuda {
    // Define a kernel by a lambda quotation.
    // The lambda must return a unit.
    let! kernel =
        <@ fun (n:int) (input:deviceptr<float>) (output:deviceptr<float>) ->
            let start = blockIdx.x * blockDim.x + threadIdx.x
            let stride = gridDim.x * blockDim.x
            let mutable i = start
            while i < n do
                // Use quotation splicing operator "%" to apply
                // the unary transform function.
                output.[i] <- (%transform) input.[i]
                i <- i + stride @>
        // Use a workflow statement to define this kernel
        |> Compiler.DefineKernel

    // Define the entry point.
    return Entry(fun program ->
        let worker = program.Worker
        // Apply kernel resource with a program to get kernel runtime.
        let kernel = program.Apply kernel

        // Define a run function.
        let run (input:float[]) =
            let n = input.Length

            // Device memory is IDisposable, easy to use with "use" keyword.
            use input = worker.Malloc(input)
            use output = worker.Malloc(n)

            // For simplicity we use a fixed block size.
            let blockSize = 128
            let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
            // We tend to partition data so that each SM could handle
            // 16 blocks to hide the memory latency.
            // For more detail, please reference "SM Occupancy".
            let gridSize = min (numSm * 16) (divup n blockSize)
            // Now we know the launch shape, could create launching parameter.
            let lp = LaunchParam(gridSize, blockSize)

            // Create two CUDA events to record the kernel execution time.
            use start = worker.CreateEvent()
            use stop = worker.CreateEvent()
            worker.Synchronize()
            start.Record()

            // Now launch the kernel.
            kernel.Launch lp n input.Ptr output.Ptr

            // Record stop event and get the time.
            stop.Record()
            stop.Synchronize()
            let msec = Event.ElapsedMilliseconds(start, stop)

            // Gathering data from device.
            let output = output.Gather()

            // Return output and time.
            output, msec

        // return the run function
        run ) }

(**
A test function, accept a unary function for CPU and a unary function quotation for GPU.
*)
let test cpuTransform gpuTransform tol =
    // Apply unary function quotation to get a template.
    // Then load it into a program.
    // Program is disposble, could use "use" keyword.
    let template = template gpuTransform
    use program = template |> Compiler.load Worker.Default

    // Create random input data.
    let n = 1 <<< 24
    let rng = System.Random(42)
    let input = Array.init n (fun _ -> rng.NextDouble())

    // Calculate on CPU with single threads. To make the time
    // more accurate, we don't count the time for creating the
    // array.
    let cpuResults, cpuTime =
        let output = Array.zeroCreate n
        let watch = System.Diagnostics.Stopwatch.StartNew()
        for i = 0 to n - 1 do output.[i] <- cpuTransform input.[i]
        watch.Stop()
        output, watch.Elapsed.TotalMilliseconds

    // Run the program to get GPU result and time.
    let gpuResults, gpuTime = program.Run input

    // Verify the results.
    let error = ref 0
    (cpuResults, gpuResults) ||> Array.iter2 (fun cpuResult gpuResult ->
        if abs (cpuResult - gpuResult) > tol then error := !error + 1)

    // Print out final results.
    printfn "n=%d:  (CPU %10.6f ms)  (GPU %10.6f ms)  (Error %d)"
            n cpuTime gpuTime !error

(** Print the GPU device name. *)
printfn "GPU: %s" Worker.Default.Device.Name

(** A simple test for function sin. *)
test sin <@ sin @> 1e-12

(** Another test. *)
test (fun x -> (pown (sin x) 2) * (pown (cos x) 2))
     <@ fun x -> (pown (sin x) 2) * (pown (cos x) 2) @>
     1e-12
