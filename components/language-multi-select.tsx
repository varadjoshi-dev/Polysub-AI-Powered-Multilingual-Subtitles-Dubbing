"use client"

import { useMemo, useState } from "react"
import { Check, ChevronDown, Search, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export type Language = { code: string; name: string; popular?: boolean }

const ALL_LANGS: Language[] = [
  { code: "en", name: "English", popular: true },
  { code: "hi", name: "Hindi", popular: true },
  { code: "es", name: "Spanish", popular: true },
  { code: "fr", name: "French", popular: true },
  { code: "mr", name: "Marathi" },
  { code: "ta", name: "Tamil" },
  { code: "bn", name: "Bengali" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese" },
]

export function LanguageMultiSelect({
  value,
  onChange,
}: {
  value: Language[]
  onChange: (val: Language[]) => void
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const arr = [...ALL_LANGS].sort((a, b) => Number(b.popular) - Number(a.popular) || a.name.localeCompare(b.name))
    return !q ? arr : arr.filter((l) => l.name.toLowerCase().includes(q) || l.code.includes(q))
  }, [query])

  const toggle = (lang: Language) => {
    const exists = value.find((l) => l.code === lang.code)
    if (exists) onChange(value.filter((l) => l.code !== lang.code))
    else onChange([...value, lang])
  }

  return (
    <div className="relative">
      <Button
        type="button"
        variant="outline"
        onClick={() => setOpen((s) => !s)}
        className="w-full justify-between"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="truncate">
          {value.length === 0 ? "Select languages (200+ supported)" : value.map((l) => l.name).join(", ")}
        </span>
        <ChevronDown className="ml-2 h-4 w-4" />
      </Button>

      {open && (
        <div className="absolute z-50 mt-2 w-full rounded-lg border bg-popover shadow-md">
          <div className="flex items-center gap-2 border-b p-2">
            <Search className="h-4 w-4 text-foreground/60" />
            <input
              placeholder="Search languages"
              className="w-full bg-transparent text-sm outline-none placeholder:text-foreground/50"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search languages"
            />
            <button aria-label="Close" className="rounded p-1 hover:bg-muted" onClick={() => setOpen(false)}>
              <X className="h-4 w-4" />
            </button>
          </div>
          <ul className="max-h-64 overflow-auto p-2" role="listbox">
            {filtered.map((lang) => {
              const selected = !!value.find((l) => l.code === lang.code)
              return (
                <li
                  key={lang.code}
                  role="option"
                  aria-selected={selected}
                  className={cn(
                    "flex cursor-pointer items-center justify-between rounded-md px-2 py-2 text-sm hover:bg-muted",
                    selected && "bg-muted",
                  )}
                  onClick={() => toggle(lang)}
                >
                  <div className="flex items-center gap-2">
                    {selected ? <Check className="h-4 w-4 text-sky-600" /> : <span className="h-4 w-4" />}
                    <span>{lang.name}</span>
                    {lang.popular && (
                      <span className="rounded bg-sky-600/10 px-1.5 py-0.5 text-xs text-sky-700">Popular</span>
                    )}
                  </div>
                  <span className="text-xs text-foreground/50">{lang.code}</span>
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {value.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {value.map((l) => (
            <span key={l.code} className="inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs">
              {l.name}
              <button
                aria-label={`Remove ${l.name}`}
                className="rounded p-0.5 hover:bg-muted"
                onClick={() => onChange(value.filter((x) => x.code !== l.code))}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
