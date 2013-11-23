#I @"packages\FAKE\tools"
//#I @"packages\FSharp.Formatting\lib\net40"
//#I @"packages\RazorEngine\lib\net40"
// Currently I used a customized FSharp.Formatting library to support mathjax
#I @"misc\fsformatting"

#r "FakeLib.dll"
#r "RazorEngine.dll"
#r "FSharp.CodeFormat.dll"
#r "FSharp.Literate.dll"

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open FSharp.Literate
open Fake

Environment.CurrentDirectory <- __SOURCE_DIRECTORY__

let projectName = "CUDA Lab"
let templateDir = "templates" @@ "default"
let imagesDir = "images"
let sourceDir = "source"
let outputDir = "output"

let (@@) a b = Path.Combine(a, b)

let htmlEncode (text:string) =
    text.Replace("&", "&amp;")
        .Replace("<", "&lt;")
        .Replace(">", "&gt;")
        .Replace("\"", "&quot;")

let normalizeDocumentName (name:string) =   
    if name = "README" then 0, "Index", "index"
    else
        let idx = name.IndexOf('.')
        let order = name.Substring(0, idx)
        let order = Int32.Parse(order)
        let name = name.Substring(idx + 1, name.Length - idx - 1)
        let filename = name
        let filename = filename.Replace(' ', '_')
        let filename = filename.Replace('.', '_')
        let filename = filename.Replace(",", "")
        let filename = filename.Replace("#", "sharp")
        let filename = filename.ToLower()
        order, name, filename

type [<AbstractClass>] Document(parent:Document option, srcPath:string) =
    member this.IsRoot = parent.IsNone
    member this.Parent = match parent with Some parent -> parent | None -> failwith "This is root doc"
    member this.SrcPath = srcPath
    abstract DstPath : string
    abstract Prefix : string
    abstract Name : string
    abstract UrlName : string
    abstract Order : int
    default this.DstPath = failwith "DstPath not set"
    default this.Prefix = failwith "Prefix not set"
    default this.Order = failwith "Order not set"
    abstract Dump : unit -> unit
    abstract Build : unit -> unit

type Folder(parent:Document option, srcPath:string) =
    inherit Document(parent, srcPath)

    let prefix, name, order = parent |> function
        | None -> "", projectName, 0
        | Some(parent) ->
            let order, name, filename = Path.GetFileName(srcPath) |> normalizeDocumentName
            let prefix = sprintf "%s%s-" parent.Prefix filename
            prefix, name, order

    let urlname = sprintf "%sindex.html" prefix

    let documents = List<Document>()

    member this.AddDocument(doc) = documents.Add(doc)
    member this.Documents = documents |> Seq.toArray |> Array.sortBy (fun doc -> doc.Order)

    override this.Prefix = prefix
    override this.Order = order
    override this.Name = name
    override this.UrlName = urlname
    override this.Dump() = this.Documents |> Array.iter (fun doc -> doc.Dump())
    override this.Build() = documents |> Seq.iter (fun doc -> doc.Build())   

    member this.GenNavList(urlroot:string, child:Document) =
        let strs = List<string>()

        if not this.IsRoot then
            let parent = this.Parent :?> Folder
            strs.Add(parent.GenNavList(urlroot, this))

        strs.Add(sprintf "<li class=\"nav-header\">%s</li>" this.Name)

        this.Documents |> Array.iter (fun doc -> doc.Name |> function
            | "Index" | "index" -> ()
            | name when name = child.Name -> strs.Add(sprintf "<li class=\"active\"><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name)
            | name -> strs.Add(sprintf "<li><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name))

        strs |> String.concat "\n"

    member this.GenIndex(urlroot:string, child:Document) =
        let strs = List<string>()

        if child.Order = 0 then
            strs.Add("<ul>")
            this.Documents |> Array.iter (fun doc -> doc.Name |> function
                | "Index" | "index" -> ()
                | name -> strs.Add(sprintf "<li><a href=\"%s%s\">%s</a></li>" urlroot doc.UrlName doc.Name))
            strs.Add("</ul>")

        strs |> String.concat "\n"

type [<AbstractClass>] Page(parent:Document option, srcPath:string) =
    inherit Document(parent, srcPath)

    let ext = Path.GetExtension(srcPath)
    
    let order, name, filename = Path.GetFileNameWithoutExtension(srcPath) |> normalizeDocumentName
    
    let dstPath = parent |> function
        | None -> outputDir @@ (sprintf "%s.html" filename)
        | Some(parent) -> outputDir @@ (sprintf "%s%s.html" parent.Prefix filename)

    let urlname = Path.GetFileName(dstPath)

    override this.DstPath = dstPath
    override this.Order = order    
    override this.Name = name
    override this.UrlName = urlname
    override this.Dump() = printfn "%s -> %s" srcPath dstPath

type MarkdownPage(parent:Document option, srcPath:string) =
    inherit Page(parent, srcPath)

    let templatePath = templateDir @@ "template.html"

    override this.Build() =
        printfn "Generating %s ..." this.UrlName
        let projectInfo =
            [ "project-name", projectName
              "nav-list", ((this.Parent :?> Folder).GenNavList("", this))
              "index", ((this.Parent :?> Folder).GenIndex("", this))
              "script-output", "" ]
        Literate.ProcessMarkdown(this.SrcPath, templatePath, this.DstPath, OutputKind.Html, replacements = projectInfo, lineNumbers = true)

type ScriptPage(parent:Document option, srcPath:string) =
    inherit Page(parent, srcPath)

    let templatePath = templateDir @@ "template.html"

    override this.Build() =
        printfn "Generating %s ..." this.UrlName

        let exitcode, stdout, stderr =
            use p = new Process()
            p.StartInfo.FileName <- "fsi"
            p.StartInfo.Arguments <- sprintf "-O \"%s\"" (Path.GetFileName(srcPath))
            p.StartInfo.WorkingDirectory <- (Path.GetDirectoryName(srcPath))
            p.StartInfo.CreateNoWindow <- false
            p.StartInfo.UseShellExecute <- false
            p.StartInfo.RedirectStandardOutput <- true
            p.StartInfo.RedirectStandardError <- true
            if not (p.Start()) then failwithf "Fail to run %s" srcPath
            let stdout = p.StandardOutput.ReadToEnd()
            let stderr = p.StandardError.ReadToEnd()
            p.WaitForExit()
            p.ExitCode, stdout, stderr

        if exitcode <> 0 then
            printfn "%s" stderr
            failwithf "Fail to run %s" srcPath
        let scriptOutput = sprintf "<h2>Script Output</h2><pre lang=\"text\">%s</pre>" (htmlEncode(stdout))

        let projectInfo =
            [ "project-name", projectName
              "nav-list", ((this.Parent :?> Folder).GenNavList("", this))
              "index", ((this.Parent :?> Folder).GenIndex("", this))
              "script-output", scriptOutput ]
        Literate.ProcessScriptFile(this.SrcPath, templatePath, this.DstPath, OutputKind.Html, replacements = projectInfo, lineNumbers = true)

let createDocument() =
    let rec create (parent:Document option) (srcDir:string) =
        let folder = Folder(parent, srcDir)
        let parent = folder :> Document |> Some

        for file in (Directory.GetFiles(srcDir)) do
            let srcPath = file
            let doc =
                let ext = Path.GetExtension(srcPath)
                match ext with
                | ".md" -> MarkdownPage(parent, srcPath) :> Document
                | ".fsx" -> ScriptPage(parent, srcPath) :> Document
                | ext -> failwithf "Unkown ext %s" ext
            folder.AddDocument(doc)

        for dir in (Directory.GetDirectories(srcDir)) do
            let srcPath = dir
            let doc = create parent srcPath
            folder.AddDocument(doc)

        folder :> Document

    let rootdoc = create None sourceDir
    (rootdoc :?> Folder).AddDocument(MarkdownPage(Some rootdoc, "README.md"))
    rootdoc

//let packages = [ "Alea.cuBase"; "Alea.cuBase.Direct3D9" ]
//
//Target "RestorePackages" (fun _ ->
//    let setParams param =
//        { param with Sources = [ "http://nuget.aleacubase.com/nuget" ]
//                     ExcludeVersion = true
//                     IncludePreRelease = true }
//    packages
//    |> List.iter (RestorePackageId setParams))

Target "RestorePackages" DoNothing

Target "CleanDocs" (fun _ ->
    CleanDirs [ outputDir ])

Target "BuildDocs" (fun _ ->
    CopyDir (outputDir @@ "style") (templateDir @@ "style") (fun _ -> true)
    CopyDir (outputDir @@ "images") imagesDir (fun _ -> true)
    CopyFile (outputDir @@ "favicon.ico") (templateDir @@ "favicon.ico")
    let rootdoc = createDocument()
    rootdoc.Build())

"CleanDocs"
    ==> "BuildDocs"

RunTargetOrDefault "BuildDocs"