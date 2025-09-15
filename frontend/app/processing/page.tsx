"use client"

import type React from "react"

import { useEffect, useMemo, useRef, useState } from "react"
import { SiteHeader } from "@/components/site-header"
import { Button } from "@/components/ui/button"
import { Mic2, Languages, FileText } from "lucide-react"
import { useRouter, useSearchParams } from "next/navigation"

type Step = { key: string; label: string; icon: React.ReactNode }

const steps: Step[] = [
  { key: "asr", label: "Speech Recognition", icon: <Mic2 className="h-4 w-4" /> },
  { key: "translate", label: "Translation", icon: <Languages className="h-4 w-4" /> },
  { key: "subtitle", label: "Subtitle Generation", icon: <FileText className="h-4 w-4" /> },
]

export default function ProcessingPage() {
  const router = useRouter()
  const params = useSearchParams()
  const jobId = params.get("job")

  const [progress, setProgress] = useState(0)
  const [stepIndex, setStepIndex] = useState(0)
  const timerRef = useRef<number | null>(null)

  const eta = useMemo(() => {
    const remain = Math.max(0, 100 - progress)
    return Math.ceil((remain / 10) * 0.6) // fake ETA
  }, [progress])

  useEffect(() => {
    timerRef.current = window.setInterval(() => {
      setProgress((p) => {
        const next = Math.min(100, p + Math.ceil(Math.random() * 8))
        const idx = Math.min(steps.length - 1, Math.floor((next / 100) * steps.length))
        setStepIndex(idx)
        if (next >= 100) {
          if (timerRef.current) clearInterval(timerRef.current)
          setTimeout(() => router.push(`/results?job=${jobId}`), 800)
        }
        return next
      })
    }, 600)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [router, jobId])

  return (
    <main className="min-h-dvh">
      <SiteHeader />
      <div className="mx-auto max-w-2xl px-4 py-10">
        <h1 className="text-2xl font-semibold">Processing your video</h1>
        <p className="mt-1 text-sm text-foreground/70">Hang tight while we process your request.</p>

        <div className="mt-6 space-y-5 rounded-xl border bg-background/60 p-5">
          <ol className="grid gap-3">
            {steps.map((s, i) => {
              const active = i <= stepIndex
              return (
                <li key={s.key} className="flex items-center gap-3">
                  <div
                    className={`flex h-8 w-8 items-center justify-center rounded-full ${active ? "bg-sky-600 text-white" : "bg-muted text-foreground/60"}`}
                  >
                    {s.icon}
                  </div>
                  <span className={`text-sm ${active ? "font-medium" : "text-foreground/70"}`}>{s.label}</span>
                </li>
              )
            })}
          </ol>

          <div className="mt-2">
            <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-600 to-teal-500 transition-all"
                style={{ width: `${progress}%` }}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progress}
                role="progressbar"
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-foreground/70">
              <span>{progress}%</span>
              <span>Est. {eta}s remaining</span>
            </div>
          </div>

          <div className="flex gap-3">
            <Button variant="outline" onClick={() => router.back()}>
              Go Back
            </Button>
            <Button variant="ghost" onClick={() => router.push("/upload")}>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </main>
  )
}
