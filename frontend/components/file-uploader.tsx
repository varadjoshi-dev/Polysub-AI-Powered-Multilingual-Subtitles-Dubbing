"use client"

import type React from "react"

import { useCallback, useRef, useState } from "react"
import { UploadCloud, X } from "lucide-react"
import { cn } from "@/lib/utils"

const ACCEPTED = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"]
const MAX_SIZE = 500 * 1024 * 1024 // 500MB

export type SelectedFile = { file: File; error?: string }

export function FileUploader({
  onChange,
  value,
}: {
  onChange: (f: SelectedFile | null) => void
  value: SelectedFile | null
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const validate = (file: File): SelectedFile => {
    if (!ACCEPTED.includes(file.type)) return { file, error: "Unsupported format. Use .mp4, .mov, .avi, or .mkv" }
    if (file.size > MAX_SIZE) return { file, error: "File exceeds 500MB limit" }
    return { file }
  }

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files?.[0]
      if (file) {
          const validated = validate(file)
                onChange(validated)
                if (!validated.error) uploadToBackend(validated.file)  // ✅ Send to backend
      }
    },
    [onChange],
  )

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
              const validated = validate(file)
                    onChange(validated)
                    if (!validated.error) uploadToBackend(validated.file)  // ✅ Send to backend
    }
  }

const uploadToBackend = async (file: File) => {
  const formData = new FormData()
  formData.append("file", file)

  try {
    const response = await fetch("http://localhost:5000/api/upload", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error("Upload failed")
    }

    const data = await response.json()
    console.log("✅ File uploaded:", data)
    sessionStorage.setItem("uploadedFiles", JSON.stringify(data))
    return data
  } catch (error) {
    console.error("❌ Upload error:", error)
  }
}

  return (
    <div>
      <div
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors",
          dragOver ? "border-sky-500 bg-sky-500/5" : "hover:bg-muted/30",
        )}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        aria-label="Upload video by clicking or dragging"
      >
        <UploadCloud className="h-6 w-6 text-foreground/80" />
        <div className="space-y-1">
          <p className="text-sm">
            Drag and drop your video here, or <span className="font-medium underline">browse</span>
          </p>
          <p className="text-xs text-foreground/60">Supported: .mp4, .mov, .avi, .mkv — Max 500MB</p>
        </div>
        <input ref={inputRef} type="file" accept={ACCEPTED.join(",")} className="hidden" onChange={onInputChange} />
      </div>

      {value?.file && (
        <div className="mt-3 flex items-center justify-between rounded-lg border bg-background px-3 py-2">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{value.file.name}</p>
            <p className={cn("text-xs", value.error ? "text-red-600" : "text-foreground/60")}>
              {value.error ? value.error : `${(value.file.size / (1024 * 1024)).toFixed(1)} MB`}
            </p>
          </div>
          <button
            className="rounded-md p-1 hover:bg-muted"
            aria-label="Remove file"
            onClick={(e) => {
              e.stopPropagation()
              onChange(null)
              if (inputRef.current) inputRef.current.value = ""
            }}
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  )
}
