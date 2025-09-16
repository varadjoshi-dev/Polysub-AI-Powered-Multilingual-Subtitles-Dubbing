"use client"

import { useEffect, useMemo, useState } from "react"
import { Star } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Evaluation({ jobId }: { jobId: string }) {
  const hash = useMemo(() => Array.from(jobId).reduce((a, c) => a + c.charCodeAt(0), 0), [jobId])
  const bleu = 60 + (hash % 25)
  const wer = 5 + (hash % 8)
  const latency = 0.7 + (hash % 10) / 10
  const sync = 80 + (hash % 15)

  const [ratings, setRatings] = useState<{ fluency: number; accuracy: number; timing: number; comment: string }>({
    fluency: 0,
    accuracy: 0,
    timing: 0,
    comment: "",
  })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem(`polysub_feedback_${jobId}`)
    if (saved) setRatings(JSON.parse(saved))
  }, [jobId])

  const submit = () => {
    localStorage.setItem(`polysub_feedback_${jobId}`, JSON.stringify(ratings))
    setSaved(true)
    setTimeout(() => setSaved(false), 1500)
  }

  return (
    <section className="space-y-6">
      {/* <div className="rounded-xl border bg-background/60 p-4">
        <h3 className="text-base font-semibold">Automated Metrics</h3>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          <Metric label="BLEU Score" value={`${bleu}`} />
          <Metric label="Word Error Rate" value={`${wer}`} unit="%" />
          <Metric label="Latency" value={`${latency.toFixed(1)}`} unit="× video length" />
          <Metric label="Subtitle Sync Score" value={`${sync}`} />
        </div>
      </div> */}

      <div className="rounded-xl border bg-background/60 p-4">
        <h3 className="text-base font-semibold">Your Feedback</h3>
        <div className="mt-4 grid gap-6 md:grid-cols-2">
          <StarRating
            label="Fluency"
            value={ratings.fluency}
            onChange={(v) => setRatings((r) => ({ ...r, fluency: v }))}
          />
          <StarRating
            label="Accuracy"
            value={ratings.accuracy}
            onChange={(v) => setRatings((r) => ({ ...r, accuracy: v }))}
          />
          <StarRating
            label="Timing / Sync"
            value={ratings.timing}
            onChange={(v) => setRatings((r) => ({ ...r, timing: v }))}
          />
          <div>
            <label className="text-sm font-medium">Comments (optional)</label>
            <textarea
              className="mt-2 w-full rounded-md border bg-background p-2 text-sm"
              rows={4}
              value={ratings.comment}
              onChange={(e) => setRatings((r) => ({ ...r, comment: e.target.value }))}
              placeholder="Tell us what worked well or what could be improved..."
            />
          </div>
        </div>
        <div className="mt-4">
          <Button
            onClick={submit}
            className="bg-gradient-to-r from-sky-600 to-teal-500 text-white hover:from-sky-700 hover:to-teal-600"
          >
            Submit Feedback
          </Button>
          {saved && <span className="ml-3 text-sm text-foreground/70">Saved! Thank you.</span>}
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="rounded-lg border bg-background p-3">
      <p className="text-xs text-foreground/60">{label}</p>
      <p className="mt-1 text-xl font-semibold">
        {value}
        {unit ? <span className="ml-1 text-sm font-normal text-foreground/60">{unit}</span> : null}
      </p>
    </div>
  )
}

function StarRating({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <p className="text-sm font-medium">{label}</p>
      <div className="mt-2 flex gap-1">
        {[1, 2, 3, 4, 5].map((i) => (
          <button
            key={i}
            aria-label={`${label} ${i} star${i > 1 ? "s" : ""}`}
            className="rounded p-1 hover:bg-muted"
            onClick={() => onChange(i)}
          >
            <Star className={`h-5 w-5 ${i <= value ? "fill-sky-600 text-sky-600" : "text-foreground/40"}`} />
          </button>
        ))}
      </div>
    </div>
  )
}
