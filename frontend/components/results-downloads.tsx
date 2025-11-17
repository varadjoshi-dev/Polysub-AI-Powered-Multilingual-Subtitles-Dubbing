"use client"

import { useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Download, Languages, FileText, AudioLines,Loader2 } from "lucide-react"

const ISO_TO_NLLB_MAP: { [key: string]: string } = {
    "hi": "hin_Deva", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym", "or": "ory_Orya",
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "af": "afr_Latn", "am": "amh_Ethi", "as": "asm_Beng", "ast": "ast_Latn", "az": "azj_Latn",
    "ba": "bak_Cyrl", "be": "bel_Cyrl", "bem": "bem_Latn", "bg": "bul_Cyrl", "bho": "bho_Deva",
    "bm": "bam_Latn", "bo": "bod_Tibt", "bs": "bos_Latn", "ca": "cat_Latn", "ceb": "ceb_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "dv": "div_Thaa", "dz": "dzo_Tibt",
    "el": "ell_Grek", "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fil": "fil_Latn", "fj": "fij_Latn", "fo": "fao_Latn", "fy": "fry_Latn", "ga": "gle_Latn",
    "gd": "gla_Latn", "gl": "glg_Latn", "gn": "grn_Latn", "ha": "hau_Latn", "haw": "haw_Latn",
    "he": "heb_Hebr", "hr": "hrv_Latn", "ht": "hat_Latn", "hu": "hun_Latn", "hy": "hye_Armn",
    "id": "ind_Latn", "ig": "ibo_Latn", "ilo": "ilo_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "jv": "jav_Latn", "kab": "kab_Latn", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kns": "kik_Latn", "rw": "kin_Latn", "ky": "kir_Cyrl", "ku": "ckb_Arab", "lb": "ltz_Latn",
    "lg": "lug_Latn", "ln": "lin_Latn", "lo": "lao_Laoo", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mai": "mai_Deva", "mg": "plt_Latn", "mk": "mkd_Cyrl", "mn": "khk_Cyrl", "mni": "mni_Beng",
    "mos": "mos_Latn", "ms": "msa_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "ne": "npi_Deva",
    "nl": "nld_Latn", "nn": "nno_Latn", "no": "nob_Latn", "ny": "nya_Latn", "oc": "oci_Latn",
    "om": "gaz_Latn", "pap": "pap_Latn", "pl": "pol_Latn", "ps": "pbt_Arab", "qu": "quy_Latn",
    "ro": "ron_Latn", "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sm": "smo_Latn", "sn": "sna_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "ss": "ssw_Latn", "st": "sot_Latn", "su": "sun_Latn",
    "sv": "swe_Latn", "sw": "swh_Latn", "taq": "taq_Latn", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "ti": "tir_Ethi", "tk": "tuk_Latn", "tn": "tsn_Latn", "to": "ton_Latn", "tr": "tur_Latn",
    "ts": "tso_Latn", "tum": "tum_Latn", "tw": "twi_Latn", "tzm": "tzm_Tfng", "ug": "uig_Arab",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn", "wo": "wol_Latn",
    "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "yue": "yue_Hant", "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant", "zu": "zul_Latn",
    "ak": "aka_Latn", "ee": "ewe_Latn", "knj": "kan_Latn", "tl": "tgl_Latn", "kg": "kon_Latn",
    "kr": "kau_Latn",
}

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

const getNllbCode = (shortCode: string): string => {
  // If the short code is already NLLB format (e.g., 'eng_Latn'), return it.
  if (shortCode.includes('_')) {
    return shortCode;
  }
  // Otherwise, look it up in the map or fall back to the original short code.
  return ISO_TO_NLLB_MAP[shortCode] || shortCode;
};

export function ResultsDownloads({ job }: { job: JobData }) {
  const hasSrt = job.opts.generateSrt
  const hasTts = job.opts.enableTts
  const hasRealtime = job.opts.enableRealtime

    const [downloading, setDownloading] = useState<string | null>(null);


   const langOutputs = useMemo(() => {
       if (!job.all_outputs || !job.langs) return []; // Defensive check

       // Create a map: { NLLB_CODE: outputObject }
       const outputMap = new Map((job.all_outputs || []).map(out => [out.lang, out]));

          // Map the selected languages (job.langs) to include the output paths
          return job.langs.map(l => {
            // FIX: Use the mapping function to convert the short code (l.code) to the NLLB key
            const outputCode = getNllbCode(l.code); 

            return {
              ...l,
              // The lookup now uses the consistently formatted NLLB code
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
        const url = http://localhost:5000/api/download?path=${encodeURIComponent(filePath)};
        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(Failed to download file: ${response.statusText});
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
      PolySub Job: ${job.id},
      File: ${job.filename},
      Languages: ${job.langs.map((l) => l.code).join(", ")},
      Includes: ${[hasSrt && "SRT", hasTts && "Dubbed", hasRealtime && "Embedded"].filter(Boolean).join(", ")},
    ].join("\n")
    downloadBlob(manifest, polysub-${job.id}-all.txt)
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
