[<AutoOpen>]
module CUDALab.D3D9.Application

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.InteropServices
open System.Threading
open SharpDX
open SharpDX.Windows
open SharpDX.Direct3D
open SharpDX.Direct3D9
open Alea.CUDA

[<DllImport("nvcuda.dll", EntryPoint="cuD3D9CtxCreate_v2", CallingConvention=CallingConvention.StdCall)>]
extern CUresult cuD3D9CtxCreate(CUcontext* pCtx, CUdevice* pCudaDevice, uint32 Flags, nativeint pD3DDevice);

[<DllImport("nvcuda.dll", EntryPoint="cuGraphicsD3D9RegisterResource", CallingConvention=CallingConvention.StdCall)>]
extern CUresult cuGraphicsD3D9RegisterResource (CUgraphicsResource* pCudaResource, nativeint pD3DResource, uint32  Flags);

let registerGraphicsResource (worker:Worker) (d3d9Res:CppObject) =
    worker.Eval <| fun _ ->
        let mutable cudaRes = 0n
        cuSafeCall(cuGraphicsD3D9RegisterResource(&&cudaRes, d3d9Res.NativePointer, 0u))
        cudaRes

let unregisterGraphicsResource (worker:Worker) (cudaRes:nativeint) =
    worker.Eval <| fun _ ->
        cuSafeCall(cuGraphicsUnregisterResource(cudaRes))

type CUDADevice = Alea.CUDA.Engine.Device
type D3D9Device = SharpDX.Direct3D9.Device
type RawInputDevice = SharpDX.RawInput.Device

type Param =
    {
        CUDADevice : CUDADevice
        FormTitle : string
        DrawingSize : Drawing.Size
    }

    static member Create(cudaDevice) =
        {
            CUDADevice = cudaDevice
            FormTitle = "NoName"
            DrawingSize = Drawing.Size(800, 600)
        }

type Context =
    {
        Form : RenderForm
        D3D9Device : D3D9Device
        CUDADevice : CUDADevice
        Worker : Worker
    }

    member this.RegisterGraphicsResource(d3d9Res:CppObject) = registerGraphicsResource this.Worker d3d9Res
    member this.UnregisterGraphicsResource(cudaRes:nativeint) = unregisterGraphicsResource this.Worker cudaRes

type Application(param:Param, loop:Context -> unit) =
    let proc() =
        use form = new RenderForm(Text = param.FormTitle, ClientSize = param.DrawingSize)

        let cudaDevice = param.CUDADevice

        use d3d9Device = new D3D9Device(new Direct3D(),
                                        cudaDevice.ID,
                                        DeviceType.Hardware,
                                        form.Handle,
                                        CreateFlags.HardwareVertexProcessing,
                                        PresentParameters(form.ClientSize.Width, form.ClientSize.Height))

        use worker =
            let generate() =
                let mutable ctx = 0n
                let mutable dev = -1
                cuSafeCall(cuD3D9CtxCreate(&&ctx, &&dev, 0u, d3d9Device.NativePointer))
                if dev <> cudaDevice.ID then printfn "warning: returned dev is %d, but you require %d" dev cudaDevice.ID
                let dev = Device.DeviceDict.[dev]
                dev, ctx
            Worker.Create(generate)

        let context =
            {
                Form = form
                D3D9Device = d3d9Device
                CUDADevice = cudaDevice
                Worker = worker
            }

        loop context

    member this.Start(?forceNewThread:bool, ?waitForStop:bool) =
        let forceNewThread = defaultArg forceNewThread false
        let waitForStop = defaultArg waitForStop true
        match forceNewThread with
        | true ->
            let thread = Thread(proc)
            thread.SetApartmentState(ApartmentState.STA)
            thread.Start()
            if waitForStop then thread.Join()
        | false ->
            match Thread.CurrentThread.GetApartmentState() with
            | ApartmentState.STA -> proc()
            | _ -> 
                let thread = Thread(proc)
                thread.SetApartmentState(ApartmentState.STA)
                thread.Start()
                if waitForStop then thread.Join()

