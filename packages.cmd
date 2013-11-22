@echo off
if not exist packages\FAKE\tools\FAKE.exe (
	.nuget\nuget.exe install FAKE -OutputDirectory packages -ExcludeVersion -Prerelease
)
rem if not exist packages\FSharp.Formatting\lib\net40\FSharp.Literate.dll (
rem 	.nuget\nuget.exe install FSharp.Formatting -OutputDirectory packages -ExcludeVersion -Prerelease
rem )
if not exist packages\Alea.cuBase\lib\net40\Alea.CUDA.dll (
	.nuget\nuget.exe install Alea.cuBase -OutputDirectory packages -ExcludeVersion -Prerelease
)
if not exist packages\SharpDX.Direct3D9\lib\net40\SharpDX.Direct3D9.dll (
	.nuget\nuget.exe install SharpDX.Direct3D9 -OutputDirectory packages -ExcludeVersion -Prerelease
)
if not exist packages\SharpDX.RawInput\lib\net40\SharpDX.RawInput.dll (
	.nuget\nuget.exe install SharpDX.RawInput -OutputDirectory packages -ExcludeVersion -Prerelease
)
