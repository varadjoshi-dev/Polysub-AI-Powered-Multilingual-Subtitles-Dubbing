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

  const pollJobStatus = async (jobId: string) => {
      try {
<<<<<<< HEAD
        const res = await fetch(`http://localhost:8000/api/status/${jobId}`)
=======
        const res = await fetch(`http://localhost:5000/api/status/${jobId}`)
>>>>>>> 91e491ad34ad58b255033d0221d066dd19acb66f
        if (!res.ok) throw new Error("Failed to fetch job status")
        const data = await res.json()
        return data
      } catch (err) {
        console.error(err)
        return null
      }
    }

  useEffect(() => {
      const fetchJob = async () => {
        const result = await pollJobStatus(jobId)   // REST call
        if (result) {
          setJob(result)   // Save response in React state
        }
      }

      fetchJob()
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
