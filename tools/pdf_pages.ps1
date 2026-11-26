# Renders PDF pages to PNG with the Windows PDF renderer (Windows.Data.Pdf), for checking a layout.
#   powershell -File tools\pdf_pages.ps1 -Pdf x.pdf -Out folder -Pages "1,2,5" [-Width 900]
param([string]$Pdf, [string]$Out, [string]$Pages = "1", [int]$Width = 900)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(); $task.Result }
function AwaitAction($op) { $task = $asTaskAction.Invoke($null, @($op)); $task.Wait() }
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync((Resolve-Path $Pdf).Path)) ([Windows.Storage.StorageFile])
$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
New-Item -ItemType Directory -Force $Out | Out-Null
$list = @()
foreach ($p in $Pages.Split(",")) {
    if ($p -match "^(\d+)-(\d+)$") { $list += [int]$Matches[1]..[int]$Matches[2] } elseif ($p -eq "all") { $list += 1..$doc.PageCount } else { $list += [int]$p }
}
foreach ($n in $list) {
    if ($n -lt 1 -or $n -gt $doc.PageCount) { continue }
    $page = $doc.GetPage($n - 1)
    $opt = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $opt.DestinationWidth = $Width
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAction ($page.RenderToStreamAsync($stream, $opt))
    $net = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream.GetInputStreamAt(0))
    $fs = [System.IO.File]::Create((Join-Path $Out ("page_{0:D3}.png" -f $n)))
    $net.CopyTo($fs); $fs.Close(); $net.Close()
    $page.Dispose()
}
Write-Output ("pages=" + $doc.PageCount)
