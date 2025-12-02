<script lang="ts">
    interface Props {
        qualityMode: boolean;
        loadIn16bit: boolean;
        loadIn8bit: boolean;
        onQualityModeChange: (value: boolean) => void;
        onLoadIn16bitChange: (value: boolean) => void;
        onLoadIn8bitChange: (value: boolean) => void;
    }

    let {
        qualityMode,
        loadIn16bit,
        loadIn8bit,
        onQualityModeChange,
        onLoadIn16bitChange,
        onLoadIn8bitChange,
    }: Props = $props();
</script>

<div>
    <h3
        class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2"
    >
        🎯 Quality Settings
    </h3>

    <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg mb-4">
        <div class="flex items-start gap-3">
            <div class="flex-shrink-0">
                <svg
                    class="w-5 h-5 text-blue-600 mt-0.5"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                >
                    <path
                        fill-rule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                        clip-rule="evenodd"
                    />
                </svg>
            </div>
            <div class="flex-1">
                <p class="text-sm text-blue-800 font-medium mb-1">
                    Quality vs Memory Tradeoff
                </p>
                <p class="text-xs text-blue-700">
                    Default settings prioritize memory efficiency. Enable
                    quality mode or adjust individual settings for better
                    accuracy at the cost of 2-4x more VRAM.
                </p>
            </div>
        </div>
    </div>

    <div class="space-y-4">
        <!-- Quality Mode Toggle -->
        <div
            class="p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg"
        >
            <div class="flex items-start justify-between">
                <div class="flex-1">
                    <div class="flex items-center gap-3 mb-2">
                        <input
                            type="checkbox"
                            id="quality_mode"
                            checked={qualityMode}
                            onchange={(e) =>
                                onQualityModeChange(e.currentTarget.checked)}
                            class="h-5 w-5 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                        />
                        <label
                            for="quality_mode"
                            class="text-base font-semibold text-gray-900"
                        >
                            🏆 Quality Mode (Recommended for Production)
                        </label>
                    </div>
                    <p class="text-sm text-gray-700 ml-8">
                        Automatically enables 16-bit precision, better
                        optimizer, and optimized settings for maximum accuracy.
                    </p>
                    <div
                        class="mt-3 ml-8 p-3 bg-white border border-purple-100 rounded-lg"
                    >
                        <p class="text-xs font-medium text-purple-900 mb-2">
                            Quality mode includes:
                        </p>
                        <ul class="text-xs text-gray-600 space-y-1">
                            <li>✓ 16-bit precision (better than 4-bit)</li>
                            <li>
                                ✓ Standard gradient checkpointing (better than
                                "unsloth")
                            </li>
                            <li>
                                ✓ AdamW optimizer (better than 8-bit version)
                            </li>
                            <li>✓ RSLoRA for ranks ≥ 32</li>
                            <li>⚠️ Requires ~4x more VRAM</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- Manual Precision Controls -->
        <div class="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <h4 class="text-sm font-semibold text-gray-900 mb-3">
                Manual Precision Settings
            </h4>
            <p class="text-xs text-gray-600 mb-3">
                Override individual settings (quality mode will take precedence
                if enabled)
            </p>

            <div class="space-y-3">
                <div class="flex items-start gap-3">
                    <input
                        type="checkbox"
                        id="load_in_16bit"
                        checked={loadIn16bit}
                        onchange={(e) =>
                            onLoadIn16bitChange(e.currentTarget.checked)}
                        disabled={qualityMode}
                        class="h-4 w-4 mt-0.5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded disabled:opacity-50"
                    />
                    <div class="flex-1">
                        <label
                            for="load_in_16bit"
                            class="text-sm font-medium text-gray-700"
                        >
                            Load in 16-bit precision
                        </label>
                        <p class="text-xs text-gray-500 mt-0.5">
                            Best quality, uses 4x more VRAM than 4-bit
                        </p>
                    </div>
                </div>

                <div class="flex items-start gap-3">
                    <input
                        type="checkbox"
                        id="load_in_8bit"
                        checked={loadIn8bit}
                        onchange={(e) =>
                            onLoadIn8bitChange(e.currentTarget.checked)}
                        disabled={qualityMode || loadIn16bit}
                        class="h-4 w-4 mt-0.5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded disabled:opacity-50"
                    />
                    <div class="flex-1">
                        <label
                            for="load_in_8bit"
                            class="text-sm font-medium text-gray-700"
                        >
                            Load in 8-bit precision
                        </label>
                        <p class="text-xs text-gray-500 mt-0.5">
                            Balanced quality/memory, uses 2x more VRAM than
                            4-bit
                        </p>
                    </div>
                </div>

                {#if !qualityMode && !loadIn16bit && !loadIn8bit}
                    <div
                        class="text-xs text-gray-600 bg-blue-50 border border-blue-100 rounded px-3 py-2"
                    >
                        ℹ️ Using default 4-bit quantization (most memory
                        efficient)
                    </div>
                {/if}
            </div>
        </div>
    </div>
</div>
