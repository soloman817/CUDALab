(**
# Performance Test

When developing CUDA kernels, we usually care about two things:

- Correctness
- Performance

Alea.cuBase provides some utilities useful utilities in the module `Alea.CUDA.Utilities.TestUtil`
to simplify testing and performance measurement of GPU algorithms.
.
*)

#I @"..\packages\Alea.cuBase\lib\net40"
#r "Alea.CUDA.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities

(** Use default worker to do GPU test. *)
let worker = Worker.Default

(**
Each test implements `TestUtil.ITest<'InputT, 'OutputT>`. You need to implement:

- `Name`: given the input (of type `InputT`) and the number of iterations, it returns a description on this test;
- `Run`: given the input (of type `InputT`) and the number of iterations, it returns the output (of type `OutputT`)
         and a performance string. Both the output and the performance string are optional. If output is `None`,
         then no verification for the data has been done. If performance is `None`, then no string will be printed
         out in the performance column of the final report. The value `iters` defines how many
         iterations are required in order to calculate the performance as a average of the calculation times.
- Interface `IDisposable`: for a GPU test, some unmanaged resources, such as the program instance, may need to disposed.

Note that each test is a factory function. Calling a test with the transform function
it returns a function `unit -> ITest<'InputT, 'OutputT>`. Later in the runner, the test has to be registered 
as factory function.

The first test is a simple single threaded CPU test. The example can be extended in 
a straightforward manner to a multi-thread version using all the avaiable CPU cores.
*)
let cpuSingleThread transform () =
    let run (input:'T[]) (iters:int) =
        let n = input.Length
        let output = Array.zeroCreate<'T> n

        // Use the utility function tictoc to get the result and 
        // timespan. The signature is:
        // val tictoc : (unit -> 'T) -> 'T * TimeSpan
        let _, timespan = TestUtil.tictoc (fun _ ->
            for iter = 1 to iters do
                for i = 0 to n - 1 do
                    output.[i] <- transform input.[i])

        // Calculate the performance as the average time.
        // Performance is just a string, so that it is possible to return 
        // more advanced performance measurement, such as thoughput, bandwidth, etc.
        // or even algorithm specific figures. 
        let msecs = timespan.TotalMilliseconds
        let msec = msecs / (float iters)
        let performance = sprintf "%12.6f ms" msec

        Some(output), Some(performance)

    { new TestUtil.ITest<'T[], 'T[]> with
        member this.Name input iters = "[CPU] Single Thread"
        member this.Run input iters = run input iters
      interface IDisposable with member this.Dispose() = () }

(**
The GPU test is similar to the CPU test. It explores different combination of block and grid sizes.
This time we implement the `IDisposable` interface to dispose the program explicitly. Actually, if you 
don't dispose the program explicitly, it will also be disposed by GC, but it is a good habit to dispose
unmanaged resource explicitly.
*)
let gpu transform (blocks:int) (threads:int) () =
    let template = cuda {
        let! kernel =
            <@ fun (n:int) (input:deviceptr<'T>) (output:deviceptr<'T>) ->
                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x
                let mutable i = start
                while i < n do
                    output.[i] <- (%transform) input.[i]
                    i <- i + stride @>
            |> Compiler.DefineKernel

        return Entry(fun program ->
            let worker = program.Worker
            let kernel = program.Apply kernel
            
            let run (input:'T[]) (iters:int) =
                fun () ->
                    let n = input.Length

                    use input = worker.Malloc<'T>(input)
                    use output = worker.Malloc<'T>(n)

                    let lp = LaunchParam(blocks, threads)

                    // Use event to record the time of kernel execution
                    use start = worker.CreateEvent()
                    use stop = worker.CreateEvent()
                    worker.Synchronize()
                    start.Record()
                    for iter = 1 to iters do
                        kernel.Launch lp n input.Ptr output.Ptr
                    stop.Record()
                    stop.Synchronize()

                    let msecs = Event.ElapsedMilliseconds(start, stop)
                    let msec = msecs / (float iters)
                    let performance = sprintf "%12.6f ms" msec
                    let output = output.Gather()

                    Some(output), Some(performance)
                // By using worker.Eval, you put all function run inside the
                // CUDA context worker thread, without any threading switching
                // this is good for performance test.
                |> worker.Eval

            run ) }

    let program = template |> Compiler.load worker

    { new TestUtil.ITest<'T[], 'T[]> with
        member this.Name input iters = sprintf "[GPU] (%3dx%3d)" blocks threads
        member this.Run input iters = program.Run input iters
      interface IDisposable with member this.Dispose() = program.Dispose() }

(**
To run the tests we need to implement a `TestUtil.IRunner<'InputT, 'OutputT>` class for the set of tests to be run:

- `Description`: returns a description string for this test runner;
- `Baseline`: one test must be set as the baseline, then its output will be considered as 
              the expected value. The results of the other tests are compared to
              these expected values.
- `Tests`: a list of functions `unit -> ITest<'InputT, 'OutputT>`, some of them can be a splitter. 
           The elements in the list are factory functions to generate the tester. 
           The runner will create the tests with the factory functions and finally dispose them.
- `Verify`: an optional function, which, if provided, will do the verification.
*)
let runner description cpuTransform gpuTransform tol =
    { new TestUtil.IRunner<float[], float[]> with
        member this.Description input iters = description

        member this.Baseline = cpuSingleThread cpuTransform

        member this.Tests =
            [
                TestUtil.splitter ""

                gpu gpuTransform  64 128
                gpu gpuTransform 128 128
                gpu gpuTransform 192 128
                gpu gpuTransform 256 128
                
                TestUtil.splitter ""

                gpu gpuTransform  64 192
                gpu gpuTransform 128 192
                gpu gpuTransform 192 192
                gpu gpuTransform 256 192

                TestUtil.splitter ""

                gpu gpuTransform  64 256
                gpu gpuTransform 128 256
                gpu gpuTransform 192 256
                gpu gpuTransform 256 256
            ]

        member this.Verify =
            let verify (output:float[]) (baseline:float[]) =
                (output, baseline)
                ||> Array.exists2 (fun x y -> abs (x - y) > tol)
                |> not
            Some verify }

(**
Now we create two test runner and run them.
*)                       
let n = (1 <<< 24)
let iters = 10
let input() = Array.init n (TestUtil.genRandomDouble 0.0 1.0)

let runner1 = runner "transform(sin)" sin <@ sin @> 1e-12
let runner2 = runner "transform(sin^2 * cos^2)"
                     (fun x -> (pown (sin x) 2) * (pown (cos x) 2))
                     <@ fun x -> (pown (sin x) 2) * (pown (cos x) 2) @>
                     1e-12

TestUtil.run runner1 iters (input())
printfn ""
TestUtil.run runner2 iters (input())