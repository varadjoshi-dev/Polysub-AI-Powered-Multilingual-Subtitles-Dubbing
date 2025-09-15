"use client"

import { useEffect, useMemo, useState } from "react"
import { SiteHeader } from "@/components/site-header"
import { useSearchParams, useRouter } from "next/navigation"
import { ResultsDownloads, type JobData } from "@/components/results-downloads"
import { Evaluation } from "@/components/evaluation"
import { Button } from "@/components/ui/button"

export default function ResultsPage() {
  const params = useSearchParams()
  const router = useRouter()
  const jobId = params.get("job") || ""

  const [job, setJob] = useState<JobData | null>(null)

  useEffect(() => {
    const raw = sessionStorage.getItem("polysub_job")
    if (raw) {
      const parsed = JSON.parse(raw)
      if (parsed?.id === jobId || !jobId) setJob(parsed)
    }
  }, [jobId])

  const missing = useMemo(() => !job, [job])

  return (
    <main className="min-h-dvh">
      <SiteHeader />
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Results</h1>
            <p className="text-sm text-foreground/70">Your processed outputs are ready to download.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => router.push("/upload")}>
              Process another
            </Button>
          </div>
        </div>

        {missing ? (
          <div className="mt-8 rounded-xl border bg-background/60 p-6">
            <p className="text-sm">No job found. Start from the upload page.</p>
            <Button className="mt-3" onClick={() => router.push("/upload")}>
              Go to Upload
            </Button>
          </div>
        ) : (
          <div className="mt-8 space-y-10">
            <ResultsDownloads job={job!} />
            <Evaluation jobId={job!.id} />
          </div>
        )}
      </div>
    </main>
  )
}
