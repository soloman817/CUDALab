(**
# Performance Test

When developing CUDA kernels, we usually care about two things:

- Correctness
- Performance

Alea.cuBase provides some utilities to help you do these things easier through
`Alea.CUDA.Utilities.TestUtil` module.
*)

#I @"..\..\packages\Alea.cuBase\lib\net40"
#r "Alea.CUDA.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities

(** Use default worker to do GPU test. *)
let worker = Worker.Default

(**
Each test implements `TestUtil.ITest<'InputT, 'OutputT>`. You need to implement:

- `Name`: given the input (of `InputT`) and number of iterations, return a description on this test;
- `Run`: given the input (of `InputT`) and number of iterations, return the output ( of `OutputT`)
         and a performance string. Both output and performance string are optional. If output is `None`,
         then no verification for the data. If performance is `None`, then no string will be printed
         out in the performance column of the final report. the `iters` represents how many
         iterations you need to performance this operation to finally get a average performance.
- Interface `IDisposable`: for GPU test, it need to dispose some unmanaged stuff, like the program
                           itself.

You should also noticed that each test is a test factory function. After given the transform function
it became `unit -> ITest<'InputT, 'OutputT>`, this is because later in the runner, we must register
test as factory function.

The first test we use a simple CPU single thread test. You can also code another CPU multiple thread 
version, to match your CPU cores.
*)
let cpuSingleThread transform () =
    let run (input:'T[]) (iters:int) =
        let n = input.Length
        let output = Array.zeroCreate<'T> n

        // Use a utility function tictoc to get the result and 
        // timespace, its signature is:
        // val tictoc : (unit -> 'T) -> 'T * TimeSpan
        let _, timespan = TestUtil.tictoc (fun _ ->
            for iter = 1 to iters do
                for i = 0 to n - 1 do
                    output.[i] <- transform input.[i])

        // Calculate the performance by its average time.
        // performance is just a string, you can calculate
        // more advanced measurement, like thoughput, bandwidth, etc.
        let msecs = timespan.TotalMilliseconds
        let msec = msecs / (float iters)
        let performance = sprintf "%12.6f ms" msec

        Some(output), Some(performance)

    { new TestUtil.ITest<'T[], 'T[]> with
        member this.Name input iters = "[CPU] Single Thread"
        member this.Run input iters = run input iters
      interface IDisposable with member this.Dispose() = () }

(**
The GPU test is similar to CPU test, you can create it via different combination of blocks and threads.
Also you need implement the `IDisposable` interface to dispose the program explicitly (actually if you 
don't dispose the program explicitly it will also be disposed by GC, but it is a good habit to dispose
unmanaged resource explicitly).
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
You need implement `TestUtil.IRunner<'InputT, 'OutputT>` for a set of tests to be run:

- `Description`: return a description string for this test runner;
- `Baseline`: one test must be set as baseline, then its output will be considered as 
              the expected value, which will be used to compare with output generated from
              other tests.
- `Tests`: a list of `unit -> ITest<'InputT, 'OutputT>`. You can add splitter there. 
           Elements in the list are factory functions to generate the tester, this is
           because tests might implement their `IDisposable` interface, so the runner
           will create those test and finally dispose them.
- `Verify`: an optional function, if you provide it, then it will be used to do verification.
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