# Lays out an HTML manuscript made by tools/build_publications.py in Word (COM) and saves .docx and .pdf.
#   -Page B5 : 176 x 250 mm, 1.5 cm margins (thesis);  -Page A4 : 2.5 cm margins, line numbers (papers)
param([string]$Html, [string]$Docx, [string]$Pdf, [string]$Page = "B5", [string]$Kind = "thesis")
$ErrorActionPreference = "Stop"
$miss = [Type]::Missing
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$w.Options.CheckSpellingAsYouType = $false
$w.Options.CheckGrammarAsYouType = $false

function Find-Each($doc, [string]$text, [bool]$wild, [scriptblock]$action) {
    $n = 0
    while ($true) {
        $r = $doc.Content
        $f = $r.Find
        $f.ClearFormatting()
        $f.Text = $text
        $f.MatchWildcards = $wild
        $f.Forward = $true
        $f.Wrap = 0
        if (-not $f.Execute()) { break }
        & $action $r
        $n++
        if ($n -gt 3000) { break }
    }
    return $n
}

try {
    $d = $w.Documents.Open($Html, $false, $false)
    $d.SaveAs([ref]$Docx, [ref]12)
    $w.ActiveWindow.View.Type = 3
    $ps = $d.PageSetup
    if ($Page -eq "B5") {
        $ps.PageWidth = $w.MillimetersToPoints(176); $ps.PageHeight = $w.MillimetersToPoints(250)
        $m = 1.5
    } else {
        $ps.PageWidth = $w.MillimetersToPoints(210); $ps.PageHeight = $w.MillimetersToPoints(297)
        $m = 2.5
    }
    foreach ($k in "TopMargin", "BottomMargin", "LeftMargin", "RightMargin") { $ps.$k = $w.CentimetersToPoints($m) }
    $ps.HeaderDistance = $w.CentimetersToPoints(0.7)
    $ps.FooterDistance = $w.CentimetersToPoints(0.7)

    # styles
    $normal = $d.Styles.Item(-1)
    $normal.Font.Name = "Times New Roman"; $normal.Font.Size = 11
    for ($i = 0; $i -lt 4; $i++) {
        $s = $d.Styles.Item(-2 - $i)
        $s.Font.Name = "Times New Roman"; $s.Font.Color = -16777216
        $s.ParagraphFormat.KeepWithNext = $true
    }
    $fc = $d.Styles.Add("FigCap", 1)
    $fc.Font.Name = "Times New Roman"; $fc.Font.Size = 10; $fc.Font.Bold = $false
    $fc.ParagraphFormat.Alignment = 1; $fc.ParagraphFormat.SpaceBefore = 2; $fc.ParagraphFormat.SpaceAfter = 9
    $fc.ParagraphFormat.LineSpacingRule = 0
    $tc = $d.Styles.Add("TabCap", 1)
    $tc.Font.Name = "Times New Roman"; $tc.Font.Size = 10; $tc.Font.Bold = $false
    $tc.ParagraphFormat.Alignment = 1; $tc.ParagraphFormat.SpaceBefore = 8; $tc.ParagraphFormat.SpaceAfter = 3
    $tc.ParagraphFormat.KeepWithNext = $true; $tc.ParagraphFormat.LineSpacingRule = 0

    for ($i = 0; $i -lt 4; $i++) {
        $r = $d.Content
        $f = $r.Find
        $f.ClearFormatting(); $f.Replacement.ClearFormatting()
        $f.Text = ""; $f.Replacement.Text = ""
        $f.Style = $d.Styles.Item(-2 - $i)
        $f.Replacement.ParagraphFormat.KeepWithNext = $true
        if ($i -eq 0) { $f.Replacement.ParagraphFormat.Alignment = 1 } else { $f.Replacement.ParagraphFormat.Alignment = 0 }
        $f.Format = $true
        $null = $f.Execute($miss, $miss, $miss, $miss, $miss, $miss, $true, 1, $true, $miss, 2)
    }
    $null = Find-Each $d "@@FC@@" $false { param($r) $r.Paragraphs.Item(1).Style = "FigCap"; $r.Text = "" }
    $null = Find-Each $d "@@TC@@" $false { param($r) $r.Paragraphs.Item(1).Style = "TabCap"; $r.Text = "" }

    # pictures and equations
    $nimg = Find-Each $d "\[\[IMG:*\]\]" $true {
        param($r)
        $t = $r.Text
        $body = $t.Substring(6, $t.Length - 8)
        $cut = $body.LastIndexOf("|")
        $path = $body.Substring(0, $cut).Replace("/", "\")
        $wd = [double]$body.Substring($cut + 1)
        $r.Text = ""
        $pic = $d.InlineShapes.AddPicture($path, $false, $true, $r)
        $pic.LockAspectRatio = -1
        $pic.Width = $wd
    }

    # tables: data tables full width with repeated header; equation tables without borders
    foreach ($t in $d.Tables) {
        $isEq = ($t.Rows.Count -eq 1 -and $t.Columns.Count -eq 3 -and $t.Cell(1, 1).Range.Text.Length -le 2)
        if ($isEq) {
            $t.Borders.Enable = 0
            $t.Rows.AllowBreakAcrossPages = 0
        } else {
            $t.AutoFitBehavior(1)
            $t.AutoFitBehavior(2)
            $t.Rows.Item(1).HeadingFormat = -1
            if ($t.Rows.Count -le 16) {
                for ($q = 1; $q -lt $t.Rows.Count; $q++) { $t.Rows.Item($q).Range.ParagraphFormat.KeepWithNext = $true }
            } else {
                $t.Rows.Item(1).Range.ParagraphFormat.KeepWithNext = $true
            }
            $t.Range.ParagraphFormat.SpaceAfter = 0
            $t.Range.ParagraphFormat.Alignment = 0
            try { $nx = $t.Range.Next(4, 1); if ($nx -ne $null) { $nx.ParagraphFormat.SpaceBefore = 8 } } catch {}
        }
    }

    # contents and lists
    $null = Find-Each $d "[[TOC]]" $false { param($r) $r.Text = ""; $null = $d.Fields.Add($r, -1, 'TOC \o "1-3" \h \z \u', $false) }
    $null = Find-Each $d "[[LOF]]" $false { param($r) $r.Text = ""; $null = $d.TablesOfContents.Add($r, $false, 1, 9, $false, $miss, $true, $true, "FigCap,1", $true, $true, $false) }
    $null = Find-Each $d "[[LOT]]" $false { param($r) $r.Text = ""; $null = $d.TablesOfContents.Add($r, $false, 1, 9, $false, $miss, $true, $true, "TabCap,1", $true, $true, $false) }
    $nsec = Find-Each $d "[[SECTION]]" $false { param($r) $r.Text = ""; $r.InsertBreak(3) }
    $null = Find-Each $d "[[PAGEBREAK]]" $false { param($r) $r.Text = ""; $r.InsertBreak(7) }

    # page numbers
    $s1 = $d.Sections.Item(1)
    if ($Kind -eq "thesis") {
        $s1.PageSetup.DifferentFirstPageHeaderFooter = -1
        $pn = $s1.Footers.Item(1).PageNumbers
        $pn.NumberStyle = 2
        $null = $pn.Add(1, $false)
        for ($i = 2; $i -le $d.Sections.Count; $i++) {
            $sec = $d.Sections.Item($i)
            $sec.PageSetup.DifferentFirstPageHeaderFooter = 0
            $ft = $sec.Footers.Item(1)
            $ft.LinkToPrevious = $false
            $ft.PageNumbers.NumberStyle = 0
            if ($i -eq 2) { $ft.PageNumbers.RestartNumberingAtSection = $true; $ft.PageNumbers.StartingNumber = 1 }
            $null = $ft.PageNumbers.Add(1, $true)
        }
    } elseif ($Kind -eq "letter") {
        $null = $s1.Footers.Item(1).PageNumbers.Add(1, $false)
    } else {
        $null = $s1.Footers.Item(1).PageNumbers.Add(1, $true)
        $ps.LineNumbering.Active = $true
        $ps.LineNumbering.RestartMode = 2
        $ps.LineNumbering.CountBy = 1
        $ps.LineNumbering.DistanceFromText = 14
    }
    foreach ($toc in $d.TablesOfContents) { $toc.Update() }
    foreach ($toc in $d.TablesOfContents) { $toc.Range.Font.Name = "Times New Roman"; $toc.Range.Font.Size = 10.5 }
    $d.Repaginate()
    foreach ($toc in $d.TablesOfContents) { $toc.UpdatePageNumbers() }
    $pages = $d.ComputeStatistics(2)
    $d.Save()
    $d.ExportAsFixedFormat($Pdf, 17)
    $d.Close([ref]0)
    Write-Output "OK pages=$pages images=$nimg sections=$nsec"
} catch {
    Write-Output ("ERROR: " + $_.Exception.Message + " at " + $_.InvocationInfo.ScriptLineNumber)
    try { $d.Close([ref]0) } catch {}
} finally {
    try { $w.Quit() } catch {}
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($w) | Out-Null
}
