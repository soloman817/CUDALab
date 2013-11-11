@echo off
if not exist packages\FAKE\tools\FAKE.exe (
	.nuget\nuget.exe install FAKE -OutputDirectory packages -ExcludeVersion -Prerelease
)
if not exist packages\FSharp.Formatting\lib\net40\FSharp.Literate.dll (
	.nuget\nuget.exe install FSharp.Formatting -OutputDirectory packages -ExcludeVersion -Prerelease
)
if not exist packages\Alea.cuBase\lib\net40\Alea.CUDA.dll (
	.nuget\nuget.exe install Alea.cuBase -OutputDirectory packages -ExcludeVersion -Prerelease
)
packages\FAKE\tools\FAKE.exe build.fsx %*
pause