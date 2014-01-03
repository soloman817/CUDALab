#I @"..\packages\Alea.cuBase\lib\net40"
#r "Alea.CUDA.dll"

open System
open Alea.CUDA
open Alea.CUDA.Utilities

// TODO: 1
// Write an test example to show the benifit of using blob. The idea is, if you have many input, 
// malloc GPU memory once will be more efficient than malloc multiple times. That is the lazy behavor 
// of blob. We need to write a test (with the test framework 
// http://www.aleacubase.com/cudalab/performance_test.html) to malloc a 100 MB GPU memory. We can do 
// it by 1) malloc 100 MB once; or 2) malloc 10 MB for 10 times; or 3) malloc 1 MB for 100 times. 
// Then we can have proof of value of blob.

let doprint = true
//let doprint = false
//let logger : TimingLogger option = TimingLogger("Blob") |> Some
let logger : TimingLogger option = None
let worker = Worker.Default

let KB = 1024
let MB = KB*KB

let gpuMalloc (partitions:int) () =
    let run (sizeInMB:int) (iters:int) =
        worker.Eval <| fun _ ->
            let times = Array.zeroCreate<float> iters
            let size = sizeInMB * MB / partitions

            for iter = 0 to iters - 1 do
                // the target of this test, is to collect partitions number of pointers
                let ptrs = Array.zeroCreate<deviceptr<byte>> partitions

                // we need store the device memory returned from malloc, otherwise
                // it will be disposed immediately, which makes your timing not
                // accurate, because there will be memfree api call then.
                let dmems = Array.init<DeviceMemory<byte> option> partitions (fun _ -> None)

                // now we measure time of the malloc
                let _, timespan = Timing.tictoc (fun _ ->
                    for i = 0 to partitions - 1 do
                        let dmem = worker.Malloc<byte>(size)
                        dmems.[i] <- Some dmem // need store it, otherwise GC will call memfree
                        ptrs.[i] <- dmem.Ptr )

                // now record the time
                times.[iter] <- timespan.TotalMilliseconds

                // IMPORTANT, need to dispose memory explicitly, if we let GC do the job, it
                // might effect the next test iteration (because GC is in another thread, you
                // cannot control when it will trigger the memfree
                dmems |> Array.iter (Option.iter (fun dmem -> dmem.Dispose()))

            let time = times |> Array.average
            let performance = sprintf "%12.6f ms" time

            None, Some(performance)

    // input type is int (the sizeInMB), output type is unit, because we dont
    // verify the result
    { new TestUtil.ITest<int, unit> with
        member this.Name sizeInMB iters = sprintf "Malloc %dMBx%d" (sizeInMB / partitions) partitions
        member this.Run sizeInMB iters = run sizeInMB iters
      interface IDisposable with member this.Dispose() = () }

let gpuBlob (partitions:int) () =
    let run (sizeInMB:int) (iters:int) =
        worker.Eval <| fun _ ->
            let times = Array.zeroCreate<float> iters 
            let size = sizeInMB * MB / partitions

            for iter = 0 to iters - 1 do
                // blob must be disposed in each iteration, otherwise
                // it will hold all old memories, and here inside one
                // iteration, we can use 'use' keyword to auotmatically
                // call its dispose().
                use blob = logger |> function
                    | Some logger -> new Blob(worker, logger)
                    | None -> new Blob(worker)

                let bmems = Array.init<BlobArray<byte> option> partitions (fun _ -> None)
                let ptrs = Array.zeroCreate<deviceptr<byte>> partitions

                let _, timespan = Timing.tictoc (fun _ ->
                    for i = 0 to partitions - 1 do
                        bmems.[i] <- blob.CreateArray<byte>(size) |> Some
                    for i = 0 to partitions - 1 do
                        ptrs.[i] <- bmems.[i].Value.Ptr )    

                times.[iter] <- timespan.TotalMilliseconds
                
                // no need to dipose blob, because blob use 'use' binding.
                // it will automatically disposed here.                

            let time = times |> Array.average
            let performance = sprintf "%12.6f ms" time
                    
            None, Some(performance)

    { new TestUtil.ITest<int, unit> with
        member this.Name sizeInMB iters = sprintf "Blob %dMBx%d" (sizeInMB / partitions) partitions
        member this.Run sizeInMB iters = run sizeInMB iters
      interface IDisposable with member this.Dispose() = () }

let runner =
    { new TestUtil.IRunner<int, unit> with
        member this.Description input iters = "malloc efficiency"

        // we use baseline to warmup the gpu
        member this.Baseline = gpuMalloc 1

        member this.Tests =
            [
                TestUtil.splitter ""

                gpuMalloc 1
                gpuMalloc 10
                gpuMalloc 25
                gpuMalloc 50
                gpuMalloc 75
                gpuMalloc 100
                gpuMalloc 150
                gpuMalloc 300

                TestUtil.splitter ""

                gpuBlob 1
                gpuBlob 10
                gpuBlob 25
                gpuBlob 50
                gpuBlob 75
                gpuBlob 100
                gpuBlob 150
                gpuBlob 300
            ]

        // no verify
        member this.Verify = None }

let iters = 100
let sizeInMB = 300
TestUtil.run runner iters sizeInMB
logger |> Option.iter (fun logger -> logger.DumpLogs())

