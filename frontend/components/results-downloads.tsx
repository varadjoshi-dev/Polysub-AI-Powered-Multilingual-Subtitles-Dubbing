"use client"

import { useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Download, Languages, FileText, AudioLines } from "lucide-react"

export type JobData = {
  id: string
  filename: string
  size: number
  langs: { code: string; name: string }[]
  opts: { generateSrt: boolean; enableTts: boolean; enableRealtime: boolean }
}

function createSrt(langCode: string) {
  return `1
00:00:00,000  00:00:03,000
[${langCode}] Subtitle preview line 1

2
00:00:03,000  00:00:06,000
[${langCode}] Subtitle preview line 2
`
}

function downloadBlob(content: string | Blob, filename: string, type = "text/plain") {
  const blob = typeof content === "string" ? new Blob([content], { type }) : content
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export function ResultsDownloads({ job }: { job: JobData }) {
  const hasSrt = job.opts.generateSrt
  const hasTts = job.opts.enableTts
  const hasRealtime = job.opts.enableRealtime

  const zipAll = async () => {
    // Placeholder "bundle" manifest; replace with JSZip when wiring backend
    const manifest = [
      `PolySub Job: ${job.id}`,
      `File: ${job.filename}`,
      `Languages: ${job.langs.map((l) => l.code).join(", ")}`,
      `Includes: ${[hasSrt && "SRT", hasTts && "Dubbed", hasRealtime && "Embedded"].filter(Boolean).join(", ")}`,
    ].join("\n")
    downloadBlob(manifest, `polysub-${job.id}-all.txt`)
  }

  const sizeMB = useMemo(() => (job.size / (1024 * 1024)).toFixed(1), [job.size])

  return (
    <section className="space-y-6">
      <div className="rounded-xl border bg-background/60 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold">Outputs</h3>
            <p className="text-xs text-foreground/60">
              Source: {job.filename} • {sizeMB} MB
            </p>
          </div>
          <Button
            onClick={zipAll}
            className="bg-gradient-to-r from-sky-600 to-teal-500 text-white hover:from-sky-700 hover:to-teal-600"
          >
            <Download className="mr-2 h-4 w-4" /> Download All
          </Button>
        </div>
      </div>

      {hasSrt && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-sky-700" />
            <h4 className="text-sm font-medium">Subtitle files (.srt)</h4>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
            {job.langs.map((l) => (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">SRT • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => downloadBlob(createSrt(l.code), `subtitles-${l.code}.srt`, "text/plain")}
                >
                  <Download className="mr-2 h-3 w-3" /> Download
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {hasTts && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <AudioLines className="h-4 w-4 text-sky-700" />
            <h4 className="text-sm font-medium">Dubbed video (TTS voices)</h4>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
            {job.langs.map((l) => (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">MP4 • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    downloadBlob("[placeholder] Dubbed MP4 not generated in demo", `dubbed-${l.code}.txt`, "text/plain")
                  }
                >
                  <Download className="mr-2 h-3 w-3" /> Download
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {hasRealtime && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Languages className="h-4 w-4 text-sky-700" />
            <h4 className="text-sm font-medium">Embedded subtitles video</h4>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
            {job.langs.map((l) => (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">MP4 • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() =>
                    downloadBlob(
                      "[placeholder] Embedded (burned-in) MP4 not generated in demo",
                      `embedded-${l.code}.txt`,
                      "text/plain",
                    )
                  }
                >
                  <Download className="mr-2 h-3 w-3" /> Download
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
