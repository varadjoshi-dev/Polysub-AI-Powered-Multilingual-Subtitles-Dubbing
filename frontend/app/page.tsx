"use client"

import type React from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { SiteHeader } from "@/components/site-header"
import { Clapperboard, Languages, Mic2, Radio, Download, FileText } from "lucide-react"
import { motion } from "framer-motion"

export default function Page() {
  return (
    <main className="min-h-dvh">
      <SiteHeader />
      <section className="relative">
        <div className="mx-auto max-w-6xl px-4 py-16 md:py-24">
          <motion.div
            className="grid items-center gap-10 md:grid-cols-2"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          >
            <motion.div
              className="space-y-6"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            >
              <motion.h1
                className="text-balance text-4xl font-bold leading-tight md:text-5xl"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                AI-Powered Multilingual Subtitles & Dubbing
              </motion.h1>
              <motion.p
                className="text-pretty text-base text-foreground/70 md:text-lg"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                Translate, Subtitle, and Dub your videos in 200+ languages instantly.
              </motion.p>
              <motion.div
                className="flex flex-wrap gap-3"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <Link href="/upload">
                  <Button className="bg-gradient-to-r from-sky-600 to-teal-500 text-white shadow-md hover:from-sky-700 hover:to-teal-600">
                    🚀 Upload Video
                  </Button>
                </Link>
                <Link href="/upload?demo=1">
                  <Button variant="secondary">🎬 Try Demo</Button>
                </Link>
                <a href="#learn-more">
                  <Button variant="outline">ℹ️ Learn More</Button>
                </a>
              </motion.div>

              <motion.div
                className="rounded-xl border bg-background/60 p-4 backdrop-blur"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 }}
              >
                <p className="text-sm text-foreground/70">
                  The system leverages Whisper for accurate speech recognition, Meta’s NLLB for multilingual
                  translation, and integrates FFmpeg for media processing. For natural voice output, it uses edge-tts
                  (Microsoft Neural TTS). Delivers fully translated, dubbed, and subtitled video output.
                </p>
              </motion.div>
            </motion.div>

            <motion.div
              className="rounded-2xl border bg-background/60 p-6 shadow-sm backdrop-blur"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
            >
              <div className="aspect-video w-full overflow-hidden rounded-lg border bg-muted">
                <img src="/Final.png" alt="Video preview frame" className="h-full w-full object-cover" />
              </div>
              <div className="mt-4 grid grid-cols-3 gap-3 text-sm text-foreground/80">
                <span className="rounded-md border px-3 py-2 text-center">Hindi</span>
                <span className="rounded-md border px-3 py-2 text-center">Marathi</span>
                <span className="rounded-md border px-3 py-2 text-center">Tamil</span>
                <span className="rounded-md border px-3 py-2 text-center">Bengali</span>
                <span className="rounded-md border px-3 py-2 text-center">English</span>
                <span className="rounded-md border px-3 py-2 text-center">Spanish</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      <section aria-labelledby="features" className="border-t">
        <div className="mx-auto max-w-6xl px-4 py-12" id="learn-more">
          <motion.h2
            id="features"
            className="text-2xl font-semibold"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            Features
          </motion.h2>
          <motion.div
            className="mt-8 grid gap-6 md:grid-cols-3"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <FeatureItem
              icon={<Languages className="h-5 w-5" />}
              title="Multilingual subtitles"
              desc="Generate subtitles in 200+ languages."
            />
            <FeatureItem
              icon={<Mic2 className="h-5 w-5" />}
              title="AI TTS dubbing"
              desc="Natural neural voices for voice-over."
            />
            <FeatureItem
              icon={<Radio className="h-5 w-5" />}
              title="Real-time subtitles"
              desc="Embed captions directly into your video."
            />
            <FeatureItem
              icon={<Clapperboard className="h-5 w-5" />}
              title="Fast & accurate"
              desc="Optimized pipeline for speed and quality."
            />
            <FeatureItem
              icon={<FileText className="h-5 w-5" />}
              title=".srt export"
              desc="Download per language or all-in-one ZIP."
            />
            <FeatureItem
              icon={<Download className="h-5 w-5" />}
              title="Regional languages"
              desc="Great for Hindi, Marathi, Tamil, Bengali, and more."
            />
          </motion.div>
        </div>
      </section>
    </main>
  )
}

function FeatureItem({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode
  title: string
  desc: string
}) {
  return (
    <motion.div
      className="group rounded-xl border bg-background/60 p-5 transition-colors hover:bg-background"
      whileHover={{ y: -2 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-gradient-to-r from-sky-600 to-teal-500 text-white">
          {icon}
        </div>
        <h3 className="text-base font-semibold">{title}</h3>
      </div>
      <p className="mt-3 text-sm text-foreground/70">{desc}</p>
    </motion.div>
  )
}
