"use client"

import { useMemo , useState} from "react"
import { Button } from "@/components/ui/button"
import { Download, Languages, FileText, AudioLines, Loader2 } from "lucide-react"

export type JobData = {
  id: string
  filename: string
  size: number
  langs: { code: string; name: string }[]
  opts: { generateSrt: boolean; enableTts: boolean; enableRealtime: boolean }
  all_outputs?: {
      lang: string;
      srt: string;
      subs_video: string;
      tts_audio: string;
      dubbed_video: string;
    }[]
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

  const [downloading, setDownloading] = useState<string | null>(null);


   const langOutputs = useMemo(() => {
       if (!job.all_outputs) return job.langs.map(l => ({ ...l, output: undefined }));

       // Create a map: { NLLB_CODE: outputObject }
       const outputMap = new Map(job.all_outputs.map(out => [out.lang, out]));

          // Map the selected languages (job.langs) to include the output paths
          return job.langs.map(l => {
            const outputCode = l.code === 'hi' ? 'hin_Deva' : l.code;

            return {
              ...l,
              // Use the determined outputCode for lookup
              output: outputMap.get(outputCode),
            }
          });
        }, [job.langs, job.all_outputs]);

    /**
     * Fetches the file from the Flask server and triggers the download.
     * @param filePath The full path of the file on the server (e.g., processed/job_id/file.mp4)
     * @param suggestedName The name to save the file as locally
     * @param downloadKey A unique key (e.g., langCode-fileType) for managing the loading state
     */
    const handleDownload = async (filePath: string, suggestedName: string, downloadKey: string) => {
      if (!filePath) return;

      setDownloading(downloadKey);

      try {
        const url = `http://localhost:5000/api/download?path=${encodeURIComponent(filePath)}`;
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`Failed to download file: ${response.statusText}`);
        }

        const blob = await response.blob();

        // Use the existing helper to force the browser to save the blob
        downloadBlob(blob, suggestedName, blob.type);

      } catch (error) {
        console.error("Download error:", error);
        alert("Failed to download the file. Check console for details.");
      } finally {
        setDownloading(null);
      }
    };

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
            {langOutputs.map((l) => {
                const fileType = 'srt';
                const downloadKey = `${l.code}-${fileType}`;
                const filePath = l.output?.srt || '';
                const suggestedName = `${job.filename.split('.')[0]}_${l.code}.${fileType}`;

              return (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">SRT • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDownload(filePath, suggestedName, downloadKey)}
                                              disabled={!l.output?.srt || downloading === downloadKey}
                >
                  {downloading === downloadKey ? (
                    <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  ) : (
                    <Download className="mr-2 h-3 w-3" />
                  )}
                  Download
                </Button>
              </div>
              )
            })}
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
            {langOutputs.map((l) => {
              const fileType = 'mp4';
              const downloadKey = `${l.code}-tts`;
              const filePath = l.output?.dubbed_video || '';
              const suggestedName = `${job.filename.split('.')[0]}_${l.code}_dubbed.${fileType}`;

              return (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">MP4 • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDownload(filePath, suggestedName, downloadKey)}
                                      disabled={!l.output?.dubbed_video || downloading === downloadKey}
                  >
                  {downloading === downloadKey ? (
                    <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  ) : (
                    <Download className="mr-2 h-3 w-3" />
                  )}
                  Download
                </Button>
              </div>
            )
          })}
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
            {langOutputs.map((l) => {

              const fileType = 'mp4';
              const downloadKey = `${l.code}-embedded`;
              const filePath = l.output?.subs_video || '';
              const suggestedName = `${job.filename.split('.')[0]}_${l.code}_embedded.${fileType}`;

              return (
              <div key={l.code} className="flex items-center justify-between rounded-lg border bg-background px-3 py-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{l.name}</p>
                  <p className="text-xs text-foreground/60">MP4 • {l.code}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDownload(filePath, suggestedName, downloadKey)}
                                      disabled={!l.output?.subs_video || downloading === downloadKey}
                  >
                  {downloading === downloadKey ? (
                    <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  ) : (
                    <Download className="mr-2 h-3 w-3" />
                  )}
                  Download
                </Button>
              </div>
            )
          })}
          </div>
        </div>
      )}
    </section>
  )
}
