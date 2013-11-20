(**
# Getting Started

This example shows how to code a basic kernel using Alea.cuBase. The example 
creates a template which uses a unary function (given as a quotation) and dynamically 
generates a transform kernel which applies the unary function to a large array.

A lot of tasks in large scale data analysis or simulation are very similar in concept:
the function to be applied is not known until runtime or has to be tuned for the 
specific data set to be processes. This is a perfect application of the dynamic 
compilation capabilities of Alea.cuBase. 
*)

(**
Alea.cuBase is shipped as a single assembly `Alea.CUDA.dll`. It supports both 32 bit and 64 bit
applications. It can be conveniently installed from the _NuGet Gallery_. In this script, the first 
step is to reference the assembly `Alea.CUDA.dll`.
*)
#I @"..\packages\Alea.cuBase\lib\net40"
#r "Alea.CUDA.dll"

(**
Then open some namespaces:

- `Alea.CUDA` - Core namespace of Alea.cuBase.
- `Alea.CUDA.Utilities` - Some useful utilities.
*)
open Alea.CUDA
open Alea.CUDA.Utilities

(**
Code a kernel quotation generating function. It accepts an unary function expression
as input and uses quotation splicing operator `%` to apply the function inside the kernel quotation.
*)
let kernel transform =
    <@ fun (n:int) (input:deviceptr<float>) (output:deviceptr<float>) ->
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < n do
            // Use quotation splicing operator "%" to apply
            // the unary transform function.
            output.[i] <- (%transform) input.[i]
            i <- i + stride @>

(**
Code a template generating function with the `cuda` workflow. The above kernel quotation 
is used in the template to define a kernel resource. The template returns an `Entry` object, 
This _entry point_ provides the necessary host code to bring the data to the GPU, 
execute the calulation on GPU in the proper way and return the results back to the caller. 
*)
let template transform = cuda {
    // Define a kernel by a lambda quotation.
    // The lambda must return a unit.
    let! kernel = transform |> kernel |> Compiler.DefineKernel

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
Write a test function. It accepts an unary function for CPU and a corresponding implementation for the GPU,
provides as a unary function expression.
*)
let test cpuTransform gpuTransform tol =
    // Apply unary function quotation to get a template.
    // Then load it into a program.
    // A program is disposble, so we create a use binding, instead of a let binding.
    let template = template gpuTransform
    use program = template |> Compiler.load Worker.Default

    // Create random input data.
    let n = 1 <<< 24
    let rng = System.Random(42)
    let input = Array.init n (fun _ -> rng.NextDouble())

    // Calculate on CPU with a single thread. To make the time 
    // measuremnt more accurate, we don't count the time for creating 
    // the array. We also use a for loop construct which is faster 
    // than any Array.map or Array.iter constructs. 
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
    printfn "n=%d:  (CPU %12.6f ms)  (GPU %10.6f ms)  (Error %d)"
            n cpuTime gpuTime !error

(** Print the GPU device name. *)
printfn "GPU: %s" Worker.Default.Device.Name

(** A simple test for the function $ \sin(x) $. *)
test sin <@ sin @> 1e-12

(** Another test for a more complicated function $ \sin(x)^2 + \cos(x)^2 $. *)
test (fun x -> (pown (sin x) 2) * (pown (cos x) 2))
     <@ fun x -> (pown (sin x) 2) * (pown (cos x) 2) @>
     1e-12
