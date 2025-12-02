<script lang="ts">
    interface Props {
        saveMethod: string;
        onSaveMethodChange: (value: string) => void;
    }

    let { saveMethod, onSaveMethodChange }: Props = $props();
</script>

<div>
    <h3 class="text-lg font-semibold text-gray-900 mb-4">Model Save Options</h3>

    <div>
        <label
            for="save_method"
            class="block text-sm font-medium text-gray-700 mb-2"
        >
            Save Method
        </label>
        <select
            id="save_method"
            value={saveMethod}
            onchange={(e) => onSaveMethodChange(e.currentTarget.value)}
            class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
        >
            <option value="merged_16bit"
                >Save Merged Model (16-bit) - Recommended</option
            >
            <option value="merged_4bit"
                >Save Merged Model (4-bit) - Smaller Size</option
            >
            <option value="lora">Save LoRA Adapters Only - Advanced</option>
        </select>
        <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="text-sm text-blue-800">
                {#if saveMethod === "merged_16bit"}
                    <strong>✅ Merged 16-bit (Recommended):</strong> Full model with
                    LoRA weights merged using Unsloth. Creates split files for vLLM
                    compatibility.
                {:else if saveMethod === "merged_4bit"}
                    <strong>📦 Merged 4-bit:</strong> Full model with LoRA weights
                    merged in 4-bit quantized format. Smaller file size.
                {:else}
                    <strong>🔧 LoRA Adapters Only (Advanced):</strong> Saves only
                    the adapter weights. Requires the base model to load.
                {/if}
            </p>
        </div>
    </div>
</div>
