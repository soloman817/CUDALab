(**
# Heat PDE

*)

#I @"..\packages\SharpDX\lib\net40"
#I @"..\packages\SharpDX.Direct3D9\lib\net40"
#I @"..\packages\SharpDX.RawInput\lib\net40"
#I @"..\packages\Alea.cuBase\lib\net40"
#I @"..\packages\Alea.cuBase.Direct3D9\lib\net40"
#r "SharpDX.dll"
#r "SharpDX.Direct3D9.dll"
#r "SharpDX.RawInput.dll"
#r "Alea.CUDA.dll"
#r "Alea.CUDA.Direct3D9.dll"

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open SharpDX
open SharpDX.Direct3D9
open SharpDX.Windows
open Alea.CUDA
open Alea.CUDA.Utilities
open Alea.CUDA.Direct3D9

(**
Create a homogeneous grid between a and b of n points.
*)
let inline homogeneousGrid (real:RealTraits<'T>) (n:int) (a:'T) (b:'T) =
    let dx = (b - a) / (real.Of (n - 1))
    let x = Array.init n (fun i -> a + (real.Of i) * dx)
    x, dx

(**
Create an exponentially grid up to tstop of step size not larger than dt,
with nc condensing points in the first interval.
*)
let inline exponentiallyCondensedGrid (real:RealTraits<'T>) (nc:int) (tstart:'T) (tstop:'T) (dt:'T) =
    if abs (tstop - tstart) < __epsilon() then
        [|tstart|]
    else
        let n = int(ceil (tstop-tstart)/dt)
        let dt' = (tstop-tstart) / (real.Of n)
        let dt'' = dt' / (real.Of (1 <<< (nc+1)))
        let tg1 = [0..nc] |> Seq.map (fun n -> tstart + (real.Of (1 <<< n))*dt'')
        let tg2 = [1..n] |> Seq.map (fun n -> tstart + (real.Of n)*dt')
        Seq.concat [Seq.singleton tstart; tg1; tg2] |> Seq.toArray

(**
Parallel tridiagonal linear system solver. The algorithm is implemented according to 
[this paper](http://www.cse.uiuc.edu/courses/cs554/notes/09_tridiagonal.pdf).

Optimized version for n <= max number of threads per block.
   
- `n`      the dimension of the tridiagonal system, must fit into one block
- `l`      lower diagonal
- `d`      diagonal
- `u`      upper diagonal
- `h`      right hand side and solution at exit
*)
[<ReflectedDefinition>]
let inline solveTriDiag n (l:deviceptr<'T>) (d:deviceptr<'T>) (u:deviceptr<'T>) (h:deviceptr<'T>) =
    let rank = threadIdx.x

    let mutable ltemp = 0G
    let mutable utemp = 0G
    let mutable htemp = 0G
        
    let mutable span = 1
    while span < n do
              
        if rank < n then
            if rank - span >= 0 then
                ltemp <- if d.[rank - span] <> 0G then -l.[rank] / d.[rank - span] else 0G
            else
                ltemp <- 0G
            if rank + span < n then
                utemp <- if d.[rank + span] <> 0G then -u.[rank] / d.[rank + span] else 0G
            else
                utemp <- 0G
            htemp <- h.[rank]
            
        __syncthreads()

        if rank < n then    
            if rank - span >= 0 then              
                d.[rank] <- d.[rank] + ltemp * u.[rank - span]
                htemp <- htemp + ltemp * h.[rank - span]
                ltemp <-ltemp * l.[rank - span]
                
            if rank + span < n then               
                d.[rank] <- d.[rank] + utemp * l.[rank + span]
                htemp <- htemp + utemp * h.[rank + span]
                utemp <- utemp * u.[rank + span]
                           
        __syncthreads()
            
        if rank < n then
            l.[rank] <- ltemp
            u.[rank] <- utemp
            h.[rank] <- htemp

        __syncthreads()

        span <- 2*span
               
    if rank < n then
        h.[rank] <- h.[rank] / d.[rank]

(**
Solves ny-2 systems of dimension nx in the x-coordinate direction.
*)
[<ReflectedDefinition>]
let inline xSweep (boundary:'T -> 'T -> 'T -> 'T) (sourceFunction:'T -> 'T -> 'T -> 'T)
                  (nx:int) (ny:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (Cx:'T) (Cy:'T)
                  (dt:'T) (t0:'T) (t1:'T) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =
    let shared = __shared__.Extern<'T>()
    let h = shared
    let d = h + nx
    let l = d + nx
    let u = l + nx

    let mutable xi = 0G
    let mutable yj = 0G

    let mstride = ny

    let mutable j = blockIdx.x
    while j < ny do  
        yj <- y.[j]

        if j = 0 || j = ny-1 then

            let mutable i = threadIdx.x
            while i < nx do  
                xi <- x.[i]
                u1.[i*mstride+j] <- boundary t1 xi yj 
                i <- i + blockDim.x

            __syncthreads()

        else

            let mutable i = threadIdx.x
            while i < nx do
                xi <- x.[i]

                if i = 0 then
                    d.[i] <- 1G
                    u.[i] <- 0G
                    h.[i] <- boundary t1 xi yj
                else if i = nx-1 then
                    l.[i] <- 0G
                    d.[i] <- 1G
                    h.[i] <- boundary t1 xi yj
                else
                    l.[i] <- -Cx
                    d.[i] <- 2G + 2G*Cx
                    u.[i] <- -Cx
                    h.[i] <- 2G*u0.[i*mstride+j] +
                             Cy*(u0.[i*mstride+(j-1)] - 2G*u0.[i*mstride+j] + u0.[i*mstride+(j+1)]) +
                             dt*(sourceFunction t1 xi yj)

                i <- i + blockDim.x

            __syncthreads()

            solveTriDiag nx l d u h

            i <- threadIdx.x
            while i < nx do  
                u1.[i*mstride+j] <- h.[i]
                i <- i + blockDim.x

            __syncthreads()

        j <- j + gridDim.x

(**
Solves nx-2 systems of dimension ny in the y-coordinate direction.
*)
[<ReflectedDefinition>]
let inline ySweep (boundary:'T -> 'T -> 'T -> 'T) (sourceFunction:'T -> 'T -> 'T -> 'T)
                  (nx:int) (ny:int) (x:deviceptr<'T>) (y:deviceptr<'T>) (Cx:'T) (Cy:'T)
                  (dt:'T) (t0:'T) (t1:'T) (u0:deviceptr<'T>) (u1:deviceptr<'T>) =
    let shared = __shared__.Extern<'T>()
    let h = shared
    let d = h + ny
    let l = d + ny
    let u = l + ny

    let mutable xi = 0G
    let mutable yj = 0G

    let mstride = ny

    let mutable i = blockIdx.x
    while i < nx do

        xi <- x.[i]

        if i = 0 || i = nx-1 then

            let mutable j = threadIdx.x
            while j < ny do
                yj <- y.[j]
                u1.[i*mstride+j] <- boundary t1 xi yj
                j <- j + blockDim.x

            __syncthreads()
        
        else

            let mutable j = threadIdx.x
            while j < ny do  
                yj <- y.[j]

                if j = 0 then
                    d.[j] <- 1G
                    u.[j] <- 0G
                    h.[j] <- boundary t1 xi yj
                else if j = ny-1 then
                    l.[j] <- 0G
                    d.[j] <- 1G
                    h.[j] <- boundary t1 xi yj
                else
                    l.[j] <- -Cy
                    d.[j] <- 2G + 2G*Cy
                    u.[j] <- -Cy
                    h.[j] <- 2G*u0.[i*mstride+j] +
                             Cx*(u0.[(i-1)*mstride+j] - 2G*u0.[i*mstride+j] + u0.[(i+1)*mstride+j]) +
                             dt*(sourceFunction t1 xi yj)

                j <- j + blockDim.x

            __syncthreads()

            solveTriDiag ny l d u h

            j <- threadIdx.x
            while j < ny do 
                u1.[i*mstride+j] <- h.[j]
                j <- j + blockDim.x

            __syncthreads()

        i <- i + gridDim.x

let inline kernels (real:RealTraits<'T>)
                   (initCondExpr:Expr<'T -> 'T -> 'T -> 'T>) 
                   (boundaryExpr:Expr<'T -> 'T -> 'T -> 'T>) 
                   (sourceExpr:Expr<'T -> 'T -> 'T -> 'T>) = cuda {

    let! initCondKernel =     
        <@ fun nx ny t (x:deviceptr<'T>) (y:deviceptr<'T>) (u:deviceptr<'T>) ->
            let initCond = %initCondExpr
            let i = blockIdx.x*blockDim.x + threadIdx.x
            let j = blockIdx.y*blockDim.y + threadIdx.y
            let mstride = ny
            if i < nx && j < ny then u.[i*mstride+j] <- initCond t x.[i] y.[j] @> |> Compiler.DefineKernel

    let! xSweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->     
            let boundary = %boundaryExpr
            let source = %sourceExpr     
            xSweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    let! ySweepKernel =     
        <@ fun nx ny (x:deviceptr<'T>) (y:deviceptr<'T>) Cx Cy dt t0 t1 (u0:deviceptr<'T>) (u1:deviceptr<'T>) ->          
            let boundary = %boundaryExpr
            let source = % sourceExpr     
            ySweep boundary source nx ny x y Cx Cy dt t0 t1 u0 u1 @> |> Compiler.DefineKernel

    return initCondKernel, xSweepKernel, ySweepKernel }

type Example =
    {
        Name : string
        InitCond  : Expr<float -> float -> float -> float>
        Boundary  : Expr<float -> float -> float -> float>
        Source    : Expr<float -> float -> float -> float>
        LoopTime  : float
        TimeRatio : float
        ValSpace  : AxisSpace<float>
    }

    member this.Real = RealTraits.Real64
    member this.Kernels = kernels this.Real this.InitCond this.Boundary this.Source

let example1 =
    {
        Name      = "exp(-t) * sin(pi*x) * cos(pi*y)"
        InitCond  = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
        Boundary  = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) @>
        Source    = <@ fun t x y -> exp(-t) * sin(__pi()*x) * cos(__pi()*y) * (2.0*__pi()*__pi() - 1.0) @>
        LoopTime  = 8000.0
        TimeRatio = 8.0
        ValSpace = { MinValue = -1.0; MaxValue = 1.0; Ratio = 2.0 }  
    }

let example2 =
    {
        Name      = "heat box (instable solution)"
        InitCond  = <@ fun t x y -> if x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 then 1.0 else 0.0 @>
        Boundary  = <@ fun t x y -> 0.0 @>
        Source    = <@ fun t x y -> 0.0 @>
        LoopTime  = 5000.0
        TimeRatio = 0.03
        ValSpace  = { MinValue = -0.13; MaxValue = 1.0; Ratio = 1.3 }  
    }

let example3 =
    let sigma1 = 0.04
    let sigma2 = 0.04
    let sigma3 = 0.04
    {
        Name      = "heat gauss"
        InitCond  =  <@ fun t x y -> 1.0/3.0*exp (-((x-0.2)*(x-0.2) + (y-0.2)*(y-0.2))/(2.0*sigma1*sigma1)) / (sigma1*sigma1*2.0*__pi()) +
                                     1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.8)*(y-0.8))/(2.0*sigma2*sigma2)) / (sigma2*sigma2*2.0*__pi()) +
                                     1.0/3.0*exp (-((x-0.8)*(x-0.8) + (y-0.2)*(y-0.2))/(2.0*sigma3*sigma3)) / (sigma3*sigma3*2.0*__pi()) @>
        Boundary  = <@ fun t x y -> 0.0 @>
        Source    = <@ fun t x y -> 0.0 @>
        LoopTime  = 8000.0
        TimeRatio = 0.005
        ValSpace = { MinValue = 0.0; MaxValue = 35.0; Ratio = 1.0 }  
    }

let examples = [| example1; example2; example3 |]

[<STAThread>]
let visual() =
    let example =
        printfn "Please choose the equation:"
        examples |> Array.iteri (fun i example -> printfn "(%d) %s" i example.Name)
        printf "Please choose: "
        let selection = int32(Console.Read()) - 48
        examples.[selection]

    let nx = 512
    let ny = 512
    let tstop = 1.0
    let diffusionCoeff = 1.0
    let tstart = 0.0
    let xMin = 0.0
    let xMax = 1.0
    let yMin = 0.0
    let yMax = 1.0
    let dt = 0.01

    let plotter = cuda {
        let! initCondKernel, xSweepKernel, ySweepKernel = example.Kernels

        return Entry(fun program (ctx:ApplicationContext) ->
            let worker = program.Worker
            let initCondKernel = program.Apply initCondKernel
            let xSweepKernel = program.Apply xSweepKernel
            let ySweepKernel = program.Apply ySweepKernel
            let real = example.Real

            let timeGrid = exponentiallyCondensedGrid real 5 tstart tstop dt
            let xgrid, dx = homogeneousGrid real nx xMin xMax
            let ygrid, dy = homogeneousGrid real ny yMin yMax
            let nu = nx * ny
            let lp0 = LaunchParam(dim3(divup nx 16, divup ny 16), dim3(16, 16))
            let lpx = LaunchParam(ny, nx, 4*nx*sizeof<float>)
            let lpy = LaunchParam(nx, ny, 4*ny*sizeof<float>)

            use x = worker.Malloc(xgrid)
            use y = worker.Malloc(ygrid)
            use u0 = worker.Malloc<float>(nu)
            use u1 = worker.Malloc<float>(nu)

            let initCondKernelFunc = initCondKernel.Launch lp0 
            let xSweepKernelFunc = xSweepKernel.Launch lpx
            let ySweepKernelFunc = ySweepKernel.Launch lpy

            let step t0 t1 =
                let dt = t1 - t0
                if dt > 0.0 then
                    //printfn "t1(%f) - t0(%f) = dt(%f)" t1 t0 dt
                    let Cx = diffusionCoeff * dt / (dx * dx)
                    let Cy = diffusionCoeff * dt / (dy * dy)
                    xSweepKernelFunc nx ny x.Ptr y.Ptr Cx Cy dt t0 (t0 + __half() * dt) u0.Ptr u1.Ptr
                    ySweepKernelFunc nx ny x.Ptr y.Ptr Cx Cy dt (t0 + __half() * dt) t1 u1.Ptr u0.Ptr

            let t0 = ref -10.0
            let maxu = ref Double.NegativeInfinity
            let minu = ref Double.PositiveInfinity

            let frame (time:float) =
                let result =
                    let t1 = time / example.LoopTime * example.TimeRatio
                    if !t0 < 0.0 || !t0 > t1 then
                        // a new loop
                        t0 := 0.0
                        initCondKernelFunc nx ny tstart x.Ptr y.Ptr u0.Ptr
                        step !t0 t1
                        t0 := t1
                        u0.Ptr, None
                    else
                        // a step
                        step !t0 t1
                        t0 := t1
                        u0.Ptr, None

                // just to check the max value
                if false then
                    let u = u0.Gather()
                    let maxu' = u |> Array.max
                    let minu' = u |> Array.min
                    if maxu' > !maxu then
                        maxu := maxu'
                        printfn "maxu: %f" !maxu
                    if minu' < !minu then
                        minu := minu'
                        printfn "minu: %f" !minu

                result

            let param : SurfacePlotter.AnimationParam<float> =
                {
                    Order = MatrixStorageOrder.RowMajor
                    RowLength = nx
                    ColLength = ny
                    RowSpace = { MinValue = xMin; MaxValue = xMax; Ratio = 1.0 }
                    ColSpace = { MinValue = yMin; MaxValue = yMax; Ratio = 1.0 }
                    ValSpace = Some example.ValSpace
                    RowPtr = x.Ptr
                    ColPtr = y.Ptr
                    Frame = frame
                    LoopTime = Some example.LoopTime
                }

            SurfacePlotter.animationLoop ctx param ) }

    let param : ApplicationParam =
        {
            CUDADevice = Device.Devices.[0]
            FormTitle = "Heat PDE"
            DrawingSize = Drawing.Size(800, 800)
        }

    let loop ctx =
        printf "Compiling ... "
        use plotter = plotter |> Compiler.load ctx.Worker
        printfn "[OK]"
        plotter.Run ctx

    let app = Application(param, loop)
    app.Start()

if fsi.CommandLineArgs.Length = 2 && fsi.CommandLineArgs.[1] = "-visual" then
    visual()
else
    printfn "Usage:"
    printfn "> fsi -O \"04.Heat PDE.fsx\" -visual"