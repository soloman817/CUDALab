[<AutoOpen>]
module CUDALab.D3D9.PlotterCommon

open System
open System.Runtime.InteropServices
open System.Threading
open Microsoft.FSharp.Quotations
open SharpDX
open SharpDX.Windows
open SharpDX.Direct3D
open SharpDX.Direct3D9
open Alea.CUDA
open Alea.CUDA.Utilities
open CUDALab.D3D9

[<Struct;Align(16)>]
type Vector4 =
    val x : float32
    val y : float32
    val z : float32
    val w : float32

    [<ReflectedDefinition>]
    new (x, y, z, w) = { x = x; y = y; z = z; w = w }

    override this.ToString() = sprintf "(%f,%f,%f,%f)" this.x this.y this.z this.w
    
[<Struct;Align(16)>]
type Vertex =
    val position : Vector4
    val color : Vector4

    [<ReflectedDefinition>]
    new (position, color) = { position = position; color = color }

    override this.ToString() = sprintf "[Position%O,Color%O]" this.position this.color

[<Record>]
type AxisSpace<'T> =
    {
        MinValue : 'T
        MaxValue : 'T
        Ratio : 'T
    }

let createVertexBuffer (ctx:Context) (elements:int) =
    new VertexBuffer(ctx.D3D9Device, __sizeof<Vertex>() * elements, Usage.WriteOnly, VertexFormat.None, Pool.Default)

let createVertexDeclaration (ctx:Context) =
    let ves = [| VertexElement(0s,  0s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Position, 0uy)
                 VertexElement(0s, 16s, DeclarationType.Float4, DeclarationMethod.Default, DeclarationUsage.Color,    0uy)
                 VertexElement.VertexDeclarationEnd |]
    new VertexDeclaration(ctx.D3D9Device, ves)

let transform2d (transform:Expr<int -> int -> 'T -> 'U>) = cuda {

    let kernel (transform:Expr<int -> int -> 'T -> 'U>) =
        <@ fun (majors:int) (minors:int) (inputs:deviceptr<'T>) (outputs:deviceptr<'U>) ->
            let minorStart = blockIdx.x * blockDim.x + threadIdx.x
            let majorStart = blockIdx.y * blockDim.y + threadIdx.y
            
            let minorStride = gridDim.x * blockDim.x
            let majorStride = gridDim.y * blockDim.y

            let mutable major = majorStart
            while major < majors do
                let mutable minor = minorStart
                while minor < minors do
                    let i = major * minors + minor
                    outputs.[i] <- (%transform) major minor inputs.[i]
                    minor <- minor + minorStride
                major <- major + majorStride @>

    let! kernelRowMajor = transform |> kernel |> Compiler.DefineKernel
    let! kernelColMajor = <@ fun c r v -> (%transform) r c v @> |> kernel |> Compiler.DefineKernel

    return (fun (program:Program) ->
        let worker = program.Worker
        let kernelRowMajor = program.Apply kernelRowMajor
        let kernelColMajor = program.Apply kernelColMajor

        let lp =
            let blockSize = dim3(32, 8)
            let gridSize = dim3(16, 16)
            LaunchParam(gridSize, blockSize)

        let run (order:Array2DStorageOrder) (rows:int) (cols:int) (inputs:deviceptr<'T>) (outputs:deviceptr<'U>) =
            order |> function
            | Array2DStorageOrder.RawMajor -> kernelRowMajor.Launch lp rows cols inputs outputs
            | Array2DStorageOrder.ColMajor -> kernelColMajor.Launch lp cols rows inputs outputs

        run ) }

[<ReflectedDefinition>]
let inline mapColor (value:'T) (minv:'T) (maxv:'T) =
    let mapB (level:'T) = max 0G (cos (level * __pi()))
    let mapG (level:'T) = sin (level * __pi())
    let mapR (level:'T) = max 0.0 (-(cos (level * __pi())))
    let level = (value - minv) / (maxv - minv)
    Vector4(float32(mapR level), float32(mapG level), float32(mapB level), 1.0f)

type RenderType =
    | Mesh
    | Point
