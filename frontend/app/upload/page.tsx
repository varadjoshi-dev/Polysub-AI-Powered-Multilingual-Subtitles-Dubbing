"use client"

import type React from "react"

import { useEffect, useMemo, useState } from "react"
import { SiteHeader } from "@/components/site-header"
import { FileUploader, type SelectedFile } from "@/components/file-uploader"
import { LanguageMultiSelect, type Language } from "@/components/language-multi-select"
import { Button } from "@/components/ui/button"
import { useRouter, useSearchParams } from "next/navigation"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"

export default function UploadPage() {
  const router = useRouter()
  const demo = useSearchParams().get("demo")
  const [file, setFile] = useState<SelectedFile | null>(null)
  const [langs, setLangs] = useState<Language[]>([])
  const [generateSrt, setGenerateSrt] = useState(true)
  const [enableTts, setEnableTts] = useState(false)
  const [enableRealtime, setEnableRealtime] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (demo) {
      setLangs([
        { code: "en", name: "English", popular: true },
        { code: "hi", name: "Hindi", popular: true },
      ])
      setGenerateSrt(true)
      setEnableTts(true)
      setEnableRealtime(true)
    }
  }, [demo])

  const canSubmit = useMemo(() => {
    return (!file || !!file.file) && !file?.error && langs.length > 0 && (generateSrt || enableTts || enableRealtime)
  }, [file, langs, generateSrt, enableTts, enableRealtime])

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    setSubmitting(true)

    const uploadedJob = JSON.parse(sessionStorage.getItem("uploadedFiles") || "{}")

        if (!uploadedJob.path) {
            console.error("No uploaded file found in sessionStorage")
            return
        }

      const payload = {
             filePath: uploadedJob.path,
             jobId: uploadedJob.id,
             enableTts,       // from state
             generateSrt,     // from state
             enableRealtime,  // from state
             langs            // from state
         }
     try {
        const res = await fetch("http://localhost:5000/api/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })

        if (!res.ok) throw new Error("Failed to start job")

        const job = await res.json()

        sessionStorage.setItem("polysub_job", JSON.stringify(job))

        console.log("job response:", job);
        // Navigate with jobId
           router.push(`/processing?job=${job.id}`)
         } catch (err) {
           console.error(err)
           alert("Something went wrong while starting processing.")
         } finally {
           setSubmitting(false)
         }
  }

  return (
    <main className="min-h-dvh">
      <SiteHeader />
      <div className="mx-auto max-w-3xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Upload Video</h1>
        <p className="mt-1 text-sm text-foreground/70">
          Upload a video, choose languages and options, and start processing.
        </p>

        <form onSubmit={onSubmit} className="mt-6 space-y-8">
          <section className="space-y-3">
            <Label className="text-sm font-medium">Video file</Label>
            <FileUploader value={file} onChange={setFile} />
          </section>

          <section className="space-y-3">
            <Label className="text-sm font-medium">Languages</Label>
            <LanguageMultiSelect value={langs} onChange={setLangs} />
            <p className="text-xs text-foreground/60">150+ supported languages</p>
          </section>

          <section className="space-y-3">
            <Label className="text-sm font-medium">Subtitle options</Label>
            <div className="space-y-3 rounded-xl border bg-background/60 p-4">
              <div className="flex items-start gap-3">
                <Checkbox id="srt" checked={generateSrt} onCheckedChange={(v) => setGenerateSrt(Boolean(v))} />
                <div>
                  <Label htmlFor="srt">Generate subtitles (.srt)</Label>
                  <p className="text-xs text-foreground/60">Create a subtitle file for each selected language.</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Checkbox id="tts" checked={enableTts} onCheckedChange={(v) => setEnableTts(Boolean(v))} />
                <div>
                  <Label htmlFor="tts">Enable TTS Dubbing</Label>
                  <p className="text-xs text-foreground/60">
                    Generate AI voice-over in selected languages with natural neural voices.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Checkbox
                  id="realtime"
                  checked={enableRealtime}
                  onCheckedChange={(v) => setEnableRealtime(Boolean(v))}
                />
                <div>
                  <Label htmlFor="realtime">Enable Real-Time Subtitles</Label>
                  <p className="text-xs text-foreground/60">Generate embedded subtitles for uploaded video.</p>
                </div>
              </div>
            </div>
          </section>

          <div className="flex items-center gap-3">
            <Button
              type="submit"
              disabled={!canSubmit || submitting}
              className="bg-gradient-to-r from-sky-600 to-teal-500 text-white shadow-md hover:from-sky-700 hover:to-teal-600 disabled:opacity-50"
            >
              🚀 Start Processing
            </Button>
            <span className="text-xs text-foreground/60">We never store your files.</span>
          </div>
        </form>
      </div>
    </main>
  )
}
