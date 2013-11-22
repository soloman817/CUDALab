module CUDALab.D3D9.SurfacePlotter

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open System.Threading
open System.Diagnostics
open SharpDX
open SharpDX.Windows
open SharpDX.Multimedia
open SharpDX.RawInput
open SharpDX.Direct3D
open SharpDX.Direct3D9
open Alea.CUDA
open Alea.CUDA.Utilities
open CUDALab.D3D9

type AnimationParam<'T> =
    {
        Order : Array2DStorageOrder
        Rows : int
        Cols : int
        RowSpace : AxisSpace<'T>
        ColSpace : AxisSpace<'T>
        RowPtr : deviceptr<'T>
        ColPtr : deviceptr<'T>
        Frame : float -> deviceptr<'T> * (AxisSpace<'T> option)
        LoopTime : float option
    }

    member this.Elements = this.Rows * this.Cols

let inline fill (real:RealTraits<'T>) (vbRes:CUgraphicsResource) (rows:int) (cols:int) = cuda {
    let! cRows = Compiler.DefineConstantArray<'T>(rows)
    let! cCols = Compiler.DefineConstantArray<'T>(cols)
    let! cRowSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()
    let! cColSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()
    let! cValSpace = Compiler.DefineConstantVariable<AxisSpace<'T>>()

    let transform =
        <@ fun (r:int) (c:int) (v:'T) ->
            let colSpace = cColSpace.Value |> __unbox
            let rowSpace = cRowSpace.Value |> __unbox
            let valSpace = cValSpace.Value |> __unbox

            let x = (cCols.[c] - colSpace.MinValue) / (colSpace.MaxValue - colSpace.MinValue) * colSpace.Ratio - colSpace.Ratio / 2G |> float32
            let z = (cRows.[r] - rowSpace.MinValue) / (rowSpace.MaxValue - rowSpace.MinValue) * rowSpace.Ratio - rowSpace.Ratio / 2G |> float32
            let y = (v - valSpace.MinValue) / (valSpace.MaxValue - valSpace.MinValue) * valSpace.Ratio - valSpace.Ratio / 2G |> float32

            let position = Vector4(x, y, z, 1.0f)
            let color = mapColor v valSpace.MinValue valSpace.MaxValue

            Vertex(position, color) @>

    let! transform = transform |> transform2d

    return Entry(fun program order (rowPtr:deviceptr<'T>) (colPtr:deviceptr<'T>) rowSpace colSpace ->
        let worker = program.Worker
        let transform = transform program order rows cols
        let cRows = program.Apply cRows
        let cCols = program.Apply cCols
        let cRowSpace = program.Apply cRowSpace
        let cColSpace = program.Apply cColSpace
        let cValSpace = program.Apply cValSpace

        worker.Eval <| fun _ ->
            cuSafeCall(cuMemcpyDtoD(cRows.Handle, rowPtr.Handle, sizeof<'T> * rows |> nativeint))
            cuSafeCall(cuMemcpyDtoD(cCols.Handle, colPtr.Handle, sizeof<'T> * cols |> nativeint))

        cRowSpace.Scatter(rowSpace)
        cColSpace.Scatter(colSpace)

        let run (valSpace:AxisSpace<'T> option) (inputs:deviceptr<'T>) =
            worker.Eval <| fun _ ->
                valSpace |> Option.iter (fun valSpace -> cValSpace.Scatter(valSpace))

                let mutable vbRes = vbRes
                cuSafeCall(cuGraphicsMapResources(1u, &&vbRes, 0n))

                let mutable vbPtr = 0n
                let mutable vbSize = 0n
                cuSafeCall(cuGraphicsResourceGetMappedPointer(&&vbPtr, &&vbSize, vbRes))

                let vb = deviceptr<Vertex>(vbPtr)
                transform inputs vb
                    
                cuSafeCall(cuGraphicsUnmapResources(1u, &&vbRes, 0n))

        run ) }

let createPointIndexBuffer (ctx:Context) =
    new IndexBuffer(ctx.D3D9Device, sizeof<int>, Usage.WriteOnly, Pool.Managed, false)

//let createMeshIndexBuffer (context:Context) (order:Util.MatrixStorageOrder) (rows:int) (cols:int) = pcalc {
//    let initIB = context.Worker.LoadPModule(Kernels.initMeshIBIRM.Value).Invoke
//    let ib = new IndexBuffer(context.D3D9Device, sizeof<int> * rows * cols * 2, Usage.WriteOnly, Pool.Default, false)
//    let ibRes = context.RegisterGraphicsResource(ib)
//    do! initIB order rows cols ibRes
//    context.UnregisterGraphicsResource(ibRes)
//    return ib }

let renderingLoop (ctx:Context) (vd:VertexDeclaration) (vb:VertexBuffer) (order:Array2DStorageOrder) (rows:int) (cols:int) (hook:Stopwatch -> unit) (renderType:RenderType) =
    let elements = rows * cols

    use ib = renderType |> function
        | RenderType.Mesh -> failwith "TODO" //createMeshIndexBuffer ctx order rows cols
        | RenderType.Point -> createPointIndexBuffer ctx

    let eye = Vector3(0.0f, 2.0f, -2.0f)
    let lookat = Vector3(0.0f, 0.0f, 0.0f)
    let up = Vector3(0.0f, 1.0f, 0.0f)

    let view = Matrix.LookAtLH(eye, lookat, up)
    let proj = Matrix.PerspectiveFovLH(Math.PI * 0.25 |> float32, 1.0f, 1.0f, 100.0f)
    let world = ref (Matrix.RotationY(Math.PI * 0.25 |> float32))

    ctx.D3D9Device.SetTransform(TransformState.View, view)
    ctx.D3D9Device.SetTransform(TransformState.Projection, proj)
    ctx.D3D9Device.SetRenderState(RenderState.Lighting, false)

    ctx.D3D9Device.Indices <- ib
    ctx.D3D9Device.VertexDeclaration <- vd
    ctx.D3D9Device.SetStreamSource(0, vb, 0, sizeof<Vertex>)

    let isMouseLeftButtonDown = ref false
    RawInputDevice.RegisterDevice(UsagePage.Generic, UsageId.GenericMouse, DeviceFlags.None)
    RawInputDevice.MouseInput.Add(fun args ->
        //printfn "(x,y):(%d,%d) Buttons: %A State: %A Wheel: %A" args.X args.Y args.ButtonFlags args.Mode args.WheelDelta
        if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonDown) <> 0u then isMouseLeftButtonDown := true
        if uint32(args.ButtonFlags &&& MouseButtonFlags.LeftButtonUp) <> 0u then isMouseLeftButtonDown := false

        if !isMouseLeftButtonDown && args.X <> 0 then
            let r = float(-args.X) / 150.0 * Math.PI * 0.25 |> float32
            world := Matrix.Multiply(!world, Matrix.RotationY(r))

        if !isMouseLeftButtonDown && args.Y <> 0 then
            let r = float(-args.Y) / 150.0 * Math.PI * 0.25 |> float32
            world := Matrix.Multiply(!world, Matrix.RotationX(r))

        match args.WheelDelta with
        | delta when delta > 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(1.01f))
        | delta when delta < 0 -> world := Matrix.Multiply(!world, Matrix.Scaling(0.99f))
        | _ -> ())

    let clock = System.Diagnostics.Stopwatch.StartNew()

    let render () = 
        hook clock

        ctx.D3D9Device.Clear(ClearFlags.Target ||| ClearFlags.ZBuffer, ColorBGRA(0uy, 40uy, 100uy, 0uy), 1.0f, 0)
        ctx.D3D9Device.BeginScene()

        ctx.D3D9Device.SetTransform(TransformState.World, world)

        match renderType with
        | RenderType.Point -> ctx.D3D9Device.DrawPrimitives(PrimitiveType.PointList, 0, elements)

        | RenderType.Mesh ->
            failwith "TODO"
//            match order with
//            | Util.RowMajorOrder ->
//                for r = 0 to rows - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, cols, r * cols, cols - 1)
//                for c = 0 to cols - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, rows, elements + c * rows, rows - 1)
//            | Util.ColMajorOrder ->
//                for c = 0 to cols - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, rows, c * rows, rows - 1)
//                for r = 0 to rows - 1 do context.D3D9Device.DrawIndexedPrimitive(PrimitiveType.LineStrip, 0, 0, cols, elements + r * cols, cols - 1)

        ctx.D3D9Device.EndScene()
        ctx.D3D9Device.Present()

    RenderLoop.Run(ctx.Form, RenderLoop.RenderCallback(render))

let inline animationLoop (real:RealTraits<'T>) (param:AnimationParam<'T>) (ctx:Context) =
    use vb = createVertexBuffer ctx param.Elements
    use vd = createVertexDeclaration ctx

    let vbRes = ctx.RegisterGraphicsResource(vb)

    try
        use program = fill real vbRes param.Rows param.Cols |> Compiler.load ctx.Worker
        let fill = program.Run param.Order param.RowPtr param.ColPtr param.RowSpace param.ColSpace

        let hook (clock:Stopwatch) =
            let time = param.LoopTime |> function
                | None -> clock.Elapsed.TotalMilliseconds
                | Some loopTime ->
                    if loopTime <= 0.0 then clock.Elapsed.TotalMilliseconds
                    else
                        let time = clock.Elapsed.TotalMilliseconds
                        if time > loopTime then clock.Restart()
                        clock.Elapsed.TotalMilliseconds
            let valuePtr, valSpace = param.Frame time
            fill valSpace valuePtr

        renderingLoop ctx vd vb param.Order param.Rows param.Cols hook RenderType.Point

    finally
        ctx.UnregisterGraphicsResource(vbRes)


