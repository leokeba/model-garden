<script lang="ts">
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    type Dataset = {
        name: string;
        path: string;
        size: number;
        examples: number;
        format: string;
        created_at: string;
        modified_at?: string;
        metadata?: Record<string, any>;
    };

    type PreviewImage = {
        src: string;
        label: string;
    };

    interface Props {
        dataset: Dataset | null;
        previewData: any[];
        loading: boolean;
        onClose: () => void;
    }

    const IMAGE_KEYS = [
        "image",
        "image_url",
        "image_base64",
        "images",
        "vision",
        "img",
        "picture",
        "imageuri",
        "image_path",
        "imagepath",
        "imagedata",
    ];

    let { dataset, previewData, loading, onClose }: Props = $props();
    let zoomedImage: PreviewImage | null = $state(null);

    function isImageKey(key: string): boolean {
        const lower = key.toLowerCase();
        return IMAGE_KEYS.some((candidate) => lower.includes(candidate));
    }

    function isLikelyBase64(value: string): boolean {
        const trimmed = value.trim();
        return /^[A-Za-z0-9+/=]+$/.test(trimmed) && trimmed.length > 100;
    }

    function toImageSrc(value: any): string | null {
        if (!value) return null;
        if (typeof value === "string") {
            const trimmed = value.trim();
            if (trimmed === "") return null;
            if (/^data:image\//i.test(trimmed)) return trimmed;
            if (/^https?:\/\//i.test(trimmed)) return trimmed;
            if (trimmed.startsWith("/")) return trimmed;
            // Relative path with common image extensions
            if (/\.(png|jpg|jpeg|gif|webp|bmp|svg)$/i.test(trimmed)) {
                const normalized = trimmed.replace(/^\.\//, "");
                return `${window.location.origin}/${normalized}`;
            }
            if (isLikelyBase64(trimmed)) return `data:image/png;base64,${trimmed}`;
            return null;
        }

        if (typeof value === "object") {
            const candidate = value.url || value.uri || value.image || value.data;
            return typeof candidate === "string" ? toImageSrc(candidate) : null;
        }

        return null;
    }

    function collectImages(sample: any): PreviewImage[] {
        const results: PreviewImage[] = [];
        const seen = new Set<string>();

        function visit(node: any, label: string) {
            if (node === null || node === undefined) return;

            // Direct image source
            const direct = toImageSrc(node);
            if (direct) {
                if (!seen.has(direct)) {
                    seen.add(direct);
                    results.push({ src: direct, label });
                }
                return;
            }

            if (Array.isArray(node)) {
                node.forEach((item, index) => visit(item, `${label}${node.length > 1 ? ` #${index + 1}` : ""}`));
                return;
            }

            if (typeof node === "object") {
                // OpenAI-style content entries: { type: "image_url", image_url: { url: "data:..." } }
                if (node.type === "image_url" && node.image_url) {
                    const src = toImageSrc(node.image_url.url || node.image_url);
                    if (src && !seen.has(src)) {
                        seen.add(src);
                        results.push({ src, label: node.type });
                    }
                }

                // Walk object fields
                Object.entries(node).forEach(([key, value]) => {
                    const keyLabel = label ? `${label}/${key}` : key;
                    if (isImageKey(key)) {
                        visit(value, key);
                    } else {
                        visit(value, keyLabel);
                    }
                });
            }
        }

        visit(sample, "image");
        return results.slice(0, 8);
    }

    function normalizeText(value: any): string {
        if (value === null || value === undefined) return "";
        if (typeof value === "string") return value;
        if (typeof value === "number" || typeof value === "boolean") return value.toString();
        if (Array.isArray(value)) return value.map((v) => normalizeText(v)).join("\n");
        return JSON.stringify(value, null, 2);
    }

    function extractTextSections(sample: any): { label: string; value: string }[] {
        if (!sample || typeof sample !== "object") return [];

        // Support OpenAI message-style schemas
        if (Array.isArray(sample.messages)) {
            const collected = sample.messages
                .flatMap((msg: any) => {
                    if (!Array.isArray(msg?.content)) return [];
                    return msg.content
                        .filter((part: any) => part?.type === "text" && part.text)
                        .map((part: any, idx: number) => ({ label: `${msg.role || "message"}#${idx + 1}`, value: normalizeText(part.text) }));
                })
                .filter(Boolean);
            if (collected.length) return collected.slice(0, 8);
        }

        const preferredOrder = [
            "instruction",
            "input",
            "text",
            "prompt",
            "question",
            "context",
            "output",
            "answer",
            "response",
        ];
        const seen = new Set<string>();
        const sections: { label: string; value: string }[] = [];

        preferredOrder.forEach((key) => {
            if (key in sample) {
                sections.push({ label: key, value: normalizeText(sample[key]) });
                seen.add(key);
            }
        });

        Object.entries(sample).forEach(([key, value]) => {
            if (seen.has(key) || isImageKey(key)) return;
            sections.push({ label: key, value: normalizeText(value) });
        });

        return sections.slice(0, 8);
    }

    function getPreviewSections(sample: any): {
        images: PreviewImage[];
        textSections: { label: string; value: string }[];
    } {
        return {
            images: collectImages(sample),
            textSections: extractTextSections(sample),
        };
    }
</script>

{#if dataset}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    >
        <Card class="max-w-5xl w-full max-h-[88vh] flex flex-col shadow-2xl">
            <div class="p-6 border-b border-gray-200 bg-gray-50/60">
                <div class="flex justify-between items-start gap-3">
                    <div>
                        <div class="flex items-center gap-3 flex-wrap">
                            <h2 class="text-2xl font-bold text-gray-900">
                                {dataset.name}
                            </h2>
                            <span class="text-xs px-3 py-1 rounded-full bg-primary-50 text-primary-700 border border-primary-100">Previewing first 10</span>
                        </div>
                        <p class="text-sm text-gray-600 mt-1 break-all">
                            {dataset.path}
                        </p>
                    </div>
                    <button
                        onclick={onClose}
                        class="text-gray-400 hover:text-gray-600"
                    >
                        ✕
                    </button>
                </div>
            </div>

            <div class="flex-1 overflow-y-auto p-6 bg-white">
                {#if loading}
                    <div class="flex justify-center items-center h-40">
                        <div
                            class="animate-spin rounded-full h-10 w-10 border-[3px] border-primary-200 border-t-primary-600"
                        ></div>
                    </div>
                {:else if previewData.length === 0}
                    <p class="text-center text-gray-500 py-10">
                        No data to preview
                    </p>
                {:else}
                    <div class="space-y-4">
                        {#each previewData as sample, index}
                            {@const sections = getPreviewSections(sample)}
                            <div class="border border-gray-200 rounded-2xl shadow-sm overflow-hidden bg-gray-50">
                                <div class="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-white">
                                    <div class="flex items-center gap-3">
                                        <div class="h-9 w-9 rounded-xl bg-primary-50 text-primary-700 font-semibold flex items-center justify-center">
                                            {index + 1}
                                        </div>
                                        <div>
                                            <p class="text-sm font-semibold text-gray-900">Sample {index + 1}</p>
                                            <p class="text-xs text-gray-500">{sections.images.length ? "Includes images" : "Text only"}</p>
                                        </div>
                                    </div>
                                    <div class="text-xs text-gray-500">
                                        {Object.keys(sample || {}).length} fields
                                    </div>
                                </div>

                                <div class="grid md:grid-cols-12 gap-4 p-4">
                                    <div class="space-y-3 md:col-span-7">
                                        {#if sections.textSections.length}
                                            {#each sections.textSections as section}
                                                <div class="bg-white rounded-xl border border-gray-200 px-3 py-2 shadow-inner">
                                                    <div class="text-[11px] uppercase tracking-wide text-gray-500 mb-1">{section.label}</div>
                                                    <div class="text-sm text-gray-900 whitespace-pre-wrap leading-relaxed">
                                                        {section.value}
                                                    </div>
                                                </div>
                                            {/each}
                                        {:else}
                                            <p class="text-sm text-gray-500">No textual fields detected.</p>
                                        {/if}
                                    </div>

                                    {#if sections.images.length}
                                        <div class="md:col-span-5">
                                            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-1 gap-3 h-full">
                                                {#each sections.images as image}
                                                    <button
                                                        class="relative overflow-hidden rounded-xl border border-gray-200 bg-white group h-full w-full"
                                                        onclick={() => (zoomedImage = image)}
                                                        aria-label={`View ${image.label}`}
                                                    >
                                                        <div class="aspect-[3/4] bg-gray-100 flex items-center justify-center">
                                                            <img
                                                                src={image.src}
                                                                alt={image.label}
                                                                class="object-contain h-full w-full transition-transform duration-300 group-hover:scale-[1.02]"
                                                                loading="lazy"
                                                            />
                                                        </div>
                                                        <div class="absolute bottom-2 left-2 text-[11px] px-2 py-1 rounded-full bg-black/60 text-white">
                                                            {image.label}
                                                        </div>
                                                        <div class="absolute top-2 right-2 text-[11px] px-2 py-1 rounded-full bg-white/80 text-gray-700 border border-gray-200">
                                                            Tap to zoom
                                                        </div>
                                                    </button>
                                                {/each}
                                            </div>
                                        </div>
                                    {:else}
                                        <div class="flex items-center justify-center rounded-xl border border-dashed border-gray-300 bg-white text-sm text-gray-500 md:col-span-5">
                                            No images detected in this sample
                                        </div>
                                    {/if}
                                </div>

                                <details class="bg-white border-t border-gray-200 px-4 py-3 text-sm text-gray-600">
                                    <summary class="cursor-pointer text-gray-700 font-medium">Raw JSON</summary>
                                    <pre class="mt-3 text-xs text-gray-800 bg-gray-50 rounded-lg p-3 overflow-x-auto">{JSON.stringify(
                                        sample,
                                        null,
                                        2,
                                    )}</pre>
                                </details>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>

            <div class="p-6 border-t border-gray-200 bg-white">
                <Button onclick={onClose} variant="secondary" fullWidth>
                    Close
                </Button>
            </div>
        </Card>
    </div>

    {#if zoomedImage}
        <div class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" role="dialog" aria-label="Image preview">
            <div class="relative max-w-5xl w-full bg-white rounded-2xl shadow-2xl overflow-hidden">
                <button
                    class="absolute top-3 right-3 text-gray-600 hover:text-gray-900 bg-white/80 rounded-full h-10 w-10 flex items-center justify-center shadow"
                    onclick={() => (zoomedImage = null)}
                    aria-label="Close image preview"
                >
                    ✕
                </button>
                <div class="w-full bg-gray-100">
                    <img
                        src={zoomedImage.src}
                        alt={zoomedImage.label}
                        class="max-h-[80vh] w-full object-contain"
                    />
                </div>
                <div class="p-3 text-sm text-gray-700 border-t border-gray-200">
                    {zoomedImage.label}
                </div>
            </div>
        </div>
    {/if}
{/if}
